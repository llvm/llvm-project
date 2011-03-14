//===--- SemaExceptionSpec.cpp - C++ Exception Specifications ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Sema routines for C++ exception specification testing.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {

static const FunctionProtoType *GetUnderlyingFunction(QualType T)
{
  if (const PointerType *PtrTy = T->getAs<PointerType>())
    T = PtrTy->getPointeeType();
  else if (const ReferenceType *RefTy = T->getAs<ReferenceType>())
    T = RefTy->getPointeeType();
  else if (const MemberPointerType *MPTy = T->getAs<MemberPointerType>())
    T = MPTy->getPointeeType();
  return T->getAs<FunctionProtoType>();
}

/// CheckSpecifiedExceptionType - Check if the given type is valid in an
/// exception specification. Incomplete types, or pointers to incomplete types
/// other than void are not allowed.
bool Sema::CheckSpecifiedExceptionType(QualType T, const SourceRange &Range) {

  // This check (and the similar one below) deals with issue 437, that changes
  // C++ 9.2p2 this way:
  // Within the class member-specification, the class is regarded as complete
  // within function bodies, default arguments, exception-specifications, and
  // constructor ctor-initializers (including such things in nested classes).
  if (T->isRecordType() && T->getAs<RecordType>()->isBeingDefined())
    return false;
    
  // C++ 15.4p2: A type denoted in an exception-specification shall not denote
  //   an incomplete type.
  if (RequireCompleteType(Range.getBegin(), T,
      PDiag(diag::err_incomplete_in_exception_spec) << /*direct*/0 << Range))
    return true;

  // C++ 15.4p2: A type denoted in an exception-specification shall not denote
  //   an incomplete type a pointer or reference to an incomplete type, other
  //   than (cv) void*.
  int kind;
  if (const PointerType* IT = T->getAs<PointerType>()) {
    T = IT->getPointeeType();
    kind = 1;
  } else if (const ReferenceType* IT = T->getAs<ReferenceType>()) {
    T = IT->getPointeeType();
    kind = 2;
  } else
    return false;

  // Again as before
  if (T->isRecordType() && T->getAs<RecordType>()->isBeingDefined())
    return false;
    
  if (!T->isVoidType() && RequireCompleteType(Range.getBegin(), T,
      PDiag(diag::err_incomplete_in_exception_spec) << kind << Range))
    return true;

  return false;
}

/// CheckDistantExceptionSpec - Check if the given type is a pointer or pointer
/// to member to a function with an exception specification. This means that
/// it is invalid to add another level of indirection.
bool Sema::CheckDistantExceptionSpec(QualType T) {
  if (const PointerType *PT = T->getAs<PointerType>())
    T = PT->getPointeeType();
  else if (const MemberPointerType *PT = T->getAs<MemberPointerType>())
    T = PT->getPointeeType();
  else
    return false;

  const FunctionProtoType *FnT = T->getAs<FunctionProtoType>();
  if (!FnT)
    return false;

  return FnT->hasExceptionSpec();
}

bool Sema::CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New) {
  bool MissingExceptionSpecification = false;
  bool MissingEmptyExceptionSpecification = false;
  if (!CheckEquivalentExceptionSpec(PDiag(diag::err_mismatched_exception_spec),
                                    PDiag(diag::note_previous_declaration),
                                    Old->getType()->getAs<FunctionProtoType>(),
                                    Old->getLocation(),
                                    New->getType()->getAs<FunctionProtoType>(),
                                    New->getLocation(),
                                    &MissingExceptionSpecification,
                                    &MissingEmptyExceptionSpecification))
    return false;

  // The failure was something other than an empty exception
  // specification; return an error.
  if (!MissingExceptionSpecification && !MissingEmptyExceptionSpecification)
    return true;

  const FunctionProtoType *NewProto 
    = New->getType()->getAs<FunctionProtoType>();

  // The new function declaration is only missing an empty exception
  // specification "throw()". If the throw() specification came from a
  // function in a system header that has C linkage, just add an empty
  // exception specification to the "new" declaration. This is an
  // egregious workaround for glibc, which adds throw() specifications
  // to many libc functions as an optimization. Unfortunately, that
  // optimization isn't permitted by the C++ standard, so we're forced
  // to work around it here.
  if (MissingEmptyExceptionSpecification && NewProto &&
      (Old->getLocation().isInvalid() ||
       Context.getSourceManager().isInSystemHeader(Old->getLocation())) &&
      Old->isExternC()) {
    FunctionProtoType::ExtProtoInfo EPI = NewProto->getExtProtoInfo();
    EPI.HasExceptionSpec = true;
    EPI.HasAnyExceptionSpec = false;
    EPI.NumExceptions = 0;
    QualType NewType = Context.getFunctionType(NewProto->getResultType(),
                                               NewProto->arg_type_begin(),
                                               NewProto->getNumArgs(),
                                               EPI);
    New->setType(NewType);
    return false;
  }

  if (MissingExceptionSpecification && NewProto) {
    const FunctionProtoType *OldProto
      = Old->getType()->getAs<FunctionProtoType>();

    FunctionProtoType::ExtProtoInfo EPI = NewProto->getExtProtoInfo();
    EPI.HasExceptionSpec = OldProto->hasExceptionSpec();
    EPI.HasAnyExceptionSpec = OldProto->hasAnyExceptionSpec();
    EPI.NumExceptions = OldProto->getNumExceptions();
    EPI.Exceptions = OldProto->exception_begin();

    // Update the type of the function with the appropriate exception
    // specification.
    QualType NewType = Context.getFunctionType(NewProto->getResultType(),
                                               NewProto->arg_type_begin(),
                                               NewProto->getNumArgs(),
                                               EPI);
    New->setType(NewType);

    // If exceptions are disabled, suppress the warning about missing
    // exception specifications for new and delete operators.
    if (!getLangOptions().CXXExceptions) {
      switch (New->getDeclName().getCXXOverloadedOperator()) {
      case OO_New:
      case OO_Array_New:
      case OO_Delete:
      case OO_Array_Delete:
        if (New->getDeclContext()->isTranslationUnit())
          return false;
        break;

      default:
        break;
      }
    } 

    // Warn about the lack of exception specification.
    llvm::SmallString<128> ExceptionSpecString;
    llvm::raw_svector_ostream OS(ExceptionSpecString);
    OS << "throw(";
    bool OnFirstException = true;
    for (FunctionProtoType::exception_iterator E = OldProto->exception_begin(),
                                            EEnd = OldProto->exception_end();
         E != EEnd;
         ++E) {
      if (OnFirstException)
        OnFirstException = false;
      else
        OS << ", ";
      
      OS << E->getAsString(Context.PrintingPolicy);
    }
    OS << ")";
    OS.flush();

    SourceLocation AfterParenLoc;
    if (TypeSourceInfo *TSInfo = New->getTypeSourceInfo()) {
      TypeLoc TL = TSInfo->getTypeLoc().IgnoreParens();
      if (const FunctionTypeLoc *FTLoc = dyn_cast<FunctionTypeLoc>(&TL))
        AfterParenLoc = PP.getLocForEndOfToken(FTLoc->getRParenLoc());
    }

    if (AfterParenLoc.isInvalid())
      Diag(New->getLocation(), diag::warn_missing_exception_specification)
        << New << OS.str();
    else {
      // FIXME: This will get more complicated with C++0x
      // late-specified return types.
      Diag(New->getLocation(), diag::warn_missing_exception_specification)
        << New << OS.str()
        << FixItHint::CreateInsertion(AfterParenLoc, " " + OS.str().str());
    }

    if (!Old->getLocation().isInvalid())
      Diag(Old->getLocation(), diag::note_previous_declaration);

    return false;    
  }

  Diag(New->getLocation(), diag::err_mismatched_exception_spec);
  Diag(Old->getLocation(), diag::note_previous_declaration);
  return true;
}

/// CheckEquivalentExceptionSpec - Check if the two types have equivalent
/// exception specifications. Exception specifications are equivalent if
/// they allow exactly the same set of exception types. It does not matter how
/// that is achieved. See C++ [except.spec]p2.
bool Sema::CheckEquivalentExceptionSpec(
    const FunctionProtoType *Old, SourceLocation OldLoc,
    const FunctionProtoType *New, SourceLocation NewLoc) {
  return CheckEquivalentExceptionSpec(
                                    PDiag(diag::err_mismatched_exception_spec),
                                      PDiag(diag::note_previous_declaration),
                                      Old, OldLoc, New, NewLoc);
}

/// CheckEquivalentExceptionSpec - Check if the two types have equivalent
/// exception specifications. Exception specifications are equivalent if
/// they allow exactly the same set of exception types. It does not matter how
/// that is achieved. See C++ [except.spec]p2.
bool Sema::CheckEquivalentExceptionSpec(const PartialDiagnostic &DiagID, 
                                        const PartialDiagnostic & NoteID,
                                        const FunctionProtoType *Old, 
                                        SourceLocation OldLoc,
                                        const FunctionProtoType *New, 
                                        SourceLocation NewLoc,
                                        bool *MissingExceptionSpecification,
                                     bool *MissingEmptyExceptionSpecification)  {
  // Just completely ignore this under -fno-exceptions.
  if (!getLangOptions().CXXExceptions)
    return false;

  if (MissingExceptionSpecification)
    *MissingExceptionSpecification = false;

  if (MissingEmptyExceptionSpecification)
    *MissingEmptyExceptionSpecification = false;

  bool OldAny = !Old->hasExceptionSpec() || Old->hasAnyExceptionSpec();
  bool NewAny = !New->hasExceptionSpec() || New->hasAnyExceptionSpec();
  if (getLangOptions().Microsoft) {
    // Treat throw(whatever) as throw(...) to be compatible with MS headers.
    if (New->hasExceptionSpec() && New->getNumExceptions() > 0)
      NewAny = true;
    if (Old->hasExceptionSpec() && Old->getNumExceptions() > 0)
      OldAny = true;
  }

  if (OldAny && NewAny)
    return false;
  if (OldAny || NewAny) {
    if (MissingExceptionSpecification && Old->hasExceptionSpec() &&
        !New->hasExceptionSpec()) {
      // The old type has an exception specification of some sort, but
      // the new type does not.
      *MissingExceptionSpecification = true;

      if (MissingEmptyExceptionSpecification && 
          !Old->hasAnyExceptionSpec() && Old->getNumExceptions() == 0) {
        // The old type has a throw() exception specification and the
        // new type has no exception specification, and the caller asked
        // to handle this itself.
        *MissingEmptyExceptionSpecification = true;
      }

      return true;
    }

    Diag(NewLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(OldLoc, NoteID);
    return true;
  }

  bool Success = true;
  // Both have a definite exception spec. Collect the first set, then compare
  // to the second.
  llvm::SmallPtrSet<CanQualType, 8> OldTypes, NewTypes;
  for (FunctionProtoType::exception_iterator I = Old->exception_begin(),
       E = Old->exception_end(); I != E; ++I)
    OldTypes.insert(Context.getCanonicalType(*I).getUnqualifiedType());

  for (FunctionProtoType::exception_iterator I = New->exception_begin(),
       E = New->exception_end(); I != E && Success; ++I) {
    CanQualType TypePtr = Context.getCanonicalType(*I).getUnqualifiedType();
    if(OldTypes.count(TypePtr))
      NewTypes.insert(TypePtr);
    else
      Success = false;
  }

  Success = Success && OldTypes.size() == NewTypes.size();

  if (Success) {
    return false;
  }
  Diag(NewLoc, DiagID);
  if (NoteID.getDiagID() != 0)
    Diag(OldLoc, NoteID);
  return true;
}

/// CheckExceptionSpecSubset - Check whether the second function type's
/// exception specification is a subset (or equivalent) of the first function
/// type. This is used by override and pointer assignment checks.
bool Sema::CheckExceptionSpecSubset(
    const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
    const FunctionProtoType *Superset, SourceLocation SuperLoc,
    const FunctionProtoType *Subset, SourceLocation SubLoc) {

  // Just auto-succeed under -fno-exceptions.
  if (!getLangOptions().CXXExceptions)
    return false;

  // FIXME: As usual, we could be more specific in our error messages, but
  // that better waits until we've got types with source locations.

  if (!SubLoc.isValid())
    SubLoc = SuperLoc;

  // If superset contains everything, we're done.
  if (!Superset->hasExceptionSpec() || Superset->hasAnyExceptionSpec())
    return CheckParamExceptionSpec(NoteID, Superset, SuperLoc, Subset, SubLoc);

  // It does not. If the subset contains everything, we've failed.
  if (!Subset->hasExceptionSpec() || Subset->hasAnyExceptionSpec()) {
    Diag(SubLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(SuperLoc, NoteID);
    return true;
  }

  // Neither contains everything. Do a proper comparison.
  for (FunctionProtoType::exception_iterator SubI = Subset->exception_begin(),
       SubE = Subset->exception_end(); SubI != SubE; ++SubI) {
    // Take one type from the subset.
    QualType CanonicalSubT = Context.getCanonicalType(*SubI);
    // Unwrap pointers and references so that we can do checks within a class
    // hierarchy. Don't unwrap member pointers; they don't have hierarchy
    // conversions on the pointee.
    bool SubIsPointer = false;
    if (const ReferenceType *RefTy = CanonicalSubT->getAs<ReferenceType>())
      CanonicalSubT = RefTy->getPointeeType();
    if (const PointerType *PtrTy = CanonicalSubT->getAs<PointerType>()) {
      CanonicalSubT = PtrTy->getPointeeType();
      SubIsPointer = true;
    }
    bool SubIsClass = CanonicalSubT->isRecordType();
    CanonicalSubT = CanonicalSubT.getLocalUnqualifiedType();

    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);

    bool Contained = false;
    // Make sure it's in the superset.
    for (FunctionProtoType::exception_iterator SuperI =
           Superset->exception_begin(), SuperE = Superset->exception_end();
         SuperI != SuperE; ++SuperI) {
      QualType CanonicalSuperT = Context.getCanonicalType(*SuperI);
      // SubT must be SuperT or derived from it, or pointer or reference to
      // such types.
      if (const ReferenceType *RefTy = CanonicalSuperT->getAs<ReferenceType>())
        CanonicalSuperT = RefTy->getPointeeType();
      if (SubIsPointer) {
        if (const PointerType *PtrTy = CanonicalSuperT->getAs<PointerType>())
          CanonicalSuperT = PtrTy->getPointeeType();
        else {
          continue;
        }
      }
      CanonicalSuperT = CanonicalSuperT.getLocalUnqualifiedType();
      // If the types are the same, move on to the next type in the subset.
      if (CanonicalSubT == CanonicalSuperT) {
        Contained = true;
        break;
      }

      // Otherwise we need to check the inheritance.
      if (!SubIsClass || !CanonicalSuperT->isRecordType())
        continue;

      Paths.clear();
      if (!IsDerivedFrom(CanonicalSubT, CanonicalSuperT, Paths))
        continue;

      if (Paths.isAmbiguous(Context.getCanonicalType(CanonicalSuperT)))
        continue;

      // Do this check from a context without privileges.
      switch (CheckBaseClassAccess(SourceLocation(),
                                   CanonicalSuperT, CanonicalSubT,
                                   Paths.front(),
                                   /*Diagnostic*/ 0,
                                   /*ForceCheck*/ true,
                                   /*ForceUnprivileged*/ true)) {
      case AR_accessible: break;
      case AR_inaccessible: continue;
      case AR_dependent:
        llvm_unreachable("access check dependent for unprivileged context");
        break;
      case AR_delayed:
        llvm_unreachable("access check delayed in non-declaration");
        break;
      }

      Contained = true;
      break;
    }
    if (!Contained) {
      Diag(SubLoc, DiagID);
      if (NoteID.getDiagID() != 0)
        Diag(SuperLoc, NoteID);
      return true;
    }
  }
  // We've run half the gauntlet.
  return CheckParamExceptionSpec(NoteID, Superset, SuperLoc, Subset, SubLoc);
}

static bool CheckSpecForTypesEquivalent(Sema &S,
    const PartialDiagnostic &DiagID, const PartialDiagnostic & NoteID,
    QualType Target, SourceLocation TargetLoc,
    QualType Source, SourceLocation SourceLoc)
{
  const FunctionProtoType *TFunc = GetUnderlyingFunction(Target);
  if (!TFunc)
    return false;
  const FunctionProtoType *SFunc = GetUnderlyingFunction(Source);
  if (!SFunc)
    return false;

  return S.CheckEquivalentExceptionSpec(DiagID, NoteID, TFunc, TargetLoc,
                                        SFunc, SourceLoc);
}

/// CheckParamExceptionSpec - Check if the parameter and return types of the
/// two functions have equivalent exception specs. This is part of the
/// assignment and override compatibility check. We do not check the parameters
/// of parameter function pointers recursively, as no sane programmer would
/// even be able to write such a function type.
bool Sema::CheckParamExceptionSpec(const PartialDiagnostic & NoteID,
    const FunctionProtoType *Target, SourceLocation TargetLoc,
    const FunctionProtoType *Source, SourceLocation SourceLoc)
{
  if (CheckSpecForTypesEquivalent(*this,
                           PDiag(diag::err_deep_exception_specs_differ) << 0, 
                                  PDiag(),
                                  Target->getResultType(), TargetLoc,
                                  Source->getResultType(), SourceLoc))
    return true;

  // We shouldn't even be testing this unless the arguments are otherwise
  // compatible.
  assert(Target->getNumArgs() == Source->getNumArgs() &&
         "Functions have different argument counts.");
  for (unsigned i = 0, E = Target->getNumArgs(); i != E; ++i) {
    if (CheckSpecForTypesEquivalent(*this,
                           PDiag(diag::err_deep_exception_specs_differ) << 1, 
                                    PDiag(),
                                    Target->getArgType(i), TargetLoc,
                                    Source->getArgType(i), SourceLoc))
      return true;
  }
  return false;
}

bool Sema::CheckExceptionSpecCompatibility(Expr *From, QualType ToType)
{
  // First we check for applicability.
  // Target type must be a function, function pointer or function reference.
  const FunctionProtoType *ToFunc = GetUnderlyingFunction(ToType);
  if (!ToFunc)
    return false;

  // SourceType must be a function or function pointer.
  const FunctionProtoType *FromFunc = GetUnderlyingFunction(From->getType());
  if (!FromFunc)
    return false;

  // Now we've got the correct types on both sides, check their compatibility.
  // This means that the source of the conversion can only throw a subset of
  // the exceptions of the target, and any exception specs on arguments or
  // return types must be equivalent.
  return CheckExceptionSpecSubset(PDiag(diag::err_incompatible_exception_specs),
                                  PDiag(), ToFunc, 
                                  From->getSourceRange().getBegin(),
                                  FromFunc, SourceLocation());
}

bool Sema::CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
                                                const CXXMethodDecl *Old) {
  return CheckExceptionSpecSubset(PDiag(diag::err_override_exception_spec),
                                  PDiag(diag::note_overridden_virtual_function),
                                  Old->getType()->getAs<FunctionProtoType>(),
                                  Old->getLocation(),
                                  New->getType()->getAs<FunctionProtoType>(),
                                  New->getLocation());
}

} // end namespace clang

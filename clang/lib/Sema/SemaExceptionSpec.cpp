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
#include "llvm/ADT/SmallString.h"

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
                          diag::err_incomplete_in_exception_spec,
                          /*direct*/0, Range))
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
    
  if (!T->isVoidType() &&
      RequireCompleteType(Range.getBegin(), T,
                          diag::err_incomplete_in_exception_spec, kind, Range))
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

const FunctionProtoType *
Sema::ResolveExceptionSpec(SourceLocation Loc, const FunctionProtoType *FPT) {
  if (!isUnresolvedExceptionSpec(FPT->getExceptionSpecType()))
    return FPT;

  FunctionDecl *SourceDecl = FPT->getExceptionSpecDecl();
  const FunctionProtoType *SourceFPT =
      SourceDecl->getType()->castAs<FunctionProtoType>();

  // If the exception specification has already been resolved, just return it.
  if (!isUnresolvedExceptionSpec(SourceFPT->getExceptionSpecType()))
    return SourceFPT;

  // Compute or instantiate the exception specification now.
  if (FPT->getExceptionSpecType() == EST_Unevaluated)
    EvaluateImplicitExceptionSpec(Loc, cast<CXXMethodDecl>(SourceDecl));
  else
    InstantiateExceptionSpec(Loc, SourceDecl);

  return SourceDecl->getType()->castAs<FunctionProtoType>();
}

/// Determine whether a function has an implicitly-generated exception
/// specification.
static bool hasImplicitExceptionSpec(FunctionDecl *Decl) {
  if (!isa<CXXDestructorDecl>(Decl) &&
      Decl->getDeclName().getCXXOverloadedOperator() != OO_Delete &&
      Decl->getDeclName().getCXXOverloadedOperator() != OO_Array_Delete)
    return false;

  // If the user didn't declare the function, its exception specification must
  // be implicit.
  if (!Decl->getTypeSourceInfo())
    return true;

  const FunctionProtoType *Ty =
    Decl->getTypeSourceInfo()->getType()->getAs<FunctionProtoType>();
  return !Ty->hasExceptionSpec();
}

bool Sema::CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New) {
  OverloadedOperatorKind OO = New->getDeclName().getCXXOverloadedOperator();
  bool IsOperatorNew = OO == OO_New || OO == OO_Array_New;
  bool MissingExceptionSpecification = false;
  bool MissingEmptyExceptionSpecification = false;
  unsigned DiagID = diag::err_mismatched_exception_spec;
  if (getLangOpts().MicrosoftExt)
    DiagID = diag::warn_mismatched_exception_spec; 

  // Check the types as written: they must match before any exception
  // specification adjustment is applied.
  if (!CheckEquivalentExceptionSpec(
        PDiag(DiagID), PDiag(diag::note_previous_declaration),
        Old->getType()->getAs<FunctionProtoType>(), Old->getLocation(),
        New->getType()->getAs<FunctionProtoType>(), New->getLocation(),
        &MissingExceptionSpecification, &MissingEmptyExceptionSpecification,
        /*AllowNoexceptAllMatchWithNoSpec=*/true, IsOperatorNew)) {
    // C++11 [except.spec]p4 [DR1492]:
    //   If a declaration of a function has an implicit
    //   exception-specification, other declarations of the function shall
    //   not specify an exception-specification.
    if (getLangOpts().CPlusPlus0x &&
        hasImplicitExceptionSpec(Old) != hasImplicitExceptionSpec(New)) {
      Diag(New->getLocation(), diag::ext_implicit_exception_spec_mismatch)
        << hasImplicitExceptionSpec(Old);
      if (!Old->getLocation().isInvalid())
        Diag(Old->getLocation(), diag::note_previous_declaration);
    }
    return false;
  }

  // The failure was something other than an empty exception
  // specification; return an error.
  if (!MissingExceptionSpecification && !MissingEmptyExceptionSpecification)
    return true;

  const FunctionProtoType *NewProto =
    New->getType()->getAs<FunctionProtoType>();

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
    EPI.ExceptionSpecType = EST_DynamicNone;
    QualType NewType = Context.getFunctionType(NewProto->getResultType(),
                                               NewProto->arg_type_begin(),
                                               NewProto->getNumArgs(),
                                               EPI);
    New->setType(NewType);
    return false;
  }

  if (MissingExceptionSpecification && NewProto) {
    const FunctionProtoType *OldProto =
      Old->getType()->getAs<FunctionProtoType>();

    FunctionProtoType::ExtProtoInfo EPI = NewProto->getExtProtoInfo();
    EPI.ExceptionSpecType = OldProto->getExceptionSpecType();
    if (EPI.ExceptionSpecType == EST_Dynamic) {
      EPI.NumExceptions = OldProto->getNumExceptions();
      EPI.Exceptions = OldProto->exception_begin();
    } else if (EPI.ExceptionSpecType == EST_ComputedNoexcept) {
      // FIXME: We can't just take the expression from the old prototype. It
      // likely contains references to the old prototype's parameters.
    }

    // Update the type of the function with the appropriate exception
    // specification.
    QualType NewType = Context.getFunctionType(NewProto->getResultType(),
                                               NewProto->arg_type_begin(),
                                               NewProto->getNumArgs(),
                                               EPI);
    New->setType(NewType);

    // If exceptions are disabled, suppress the warning about missing
    // exception specifications for new and delete operators.
    if (!getLangOpts().CXXExceptions) {
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
    SmallString<128> ExceptionSpecString;
    llvm::raw_svector_ostream OS(ExceptionSpecString);
    switch (OldProto->getExceptionSpecType()) {
    case EST_DynamicNone:
      OS << "throw()";
      break;

    case EST_Dynamic: {
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
        
        OS << E->getAsString(getPrintingPolicy());
      }
      OS << ")";
      break;
    }

    case EST_BasicNoexcept:
      OS << "noexcept";
      break;

    case EST_ComputedNoexcept:
      OS << "noexcept(";
      OldProto->getNoexceptExpr()->printPretty(OS, 0, getPrintingPolicy());
      OS << ")";
      break;

    default:
      llvm_unreachable("This spec type is compatible with none.");
    }
    OS.flush();

    SourceLocation FixItLoc;
    if (TypeSourceInfo *TSInfo = New->getTypeSourceInfo()) {
      TypeLoc TL = TSInfo->getTypeLoc().IgnoreParens();
      if (const FunctionTypeLoc *FTLoc = dyn_cast<FunctionTypeLoc>(&TL))
        FixItLoc = PP.getLocForEndOfToken(FTLoc->getLocalRangeEnd());
    }

    if (FixItLoc.isInvalid())
      Diag(New->getLocation(), diag::warn_missing_exception_specification)
        << New << OS.str();
    else {
      // FIXME: This will get more complicated with C++0x
      // late-specified return types.
      Diag(New->getLocation(), diag::warn_missing_exception_specification)
        << New << OS.str()
        << FixItHint::CreateInsertion(FixItLoc, " " + OS.str().str());
    }

    if (!Old->getLocation().isInvalid())
      Diag(Old->getLocation(), diag::note_previous_declaration);

    return false;    
  }

  Diag(New->getLocation(), DiagID);
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
  unsigned DiagID = diag::err_mismatched_exception_spec;
  if (getLangOpts().MicrosoftExt)
    DiagID = diag::warn_mismatched_exception_spec; 
  return CheckEquivalentExceptionSpec(PDiag(DiagID),
                                      PDiag(diag::note_previous_declaration),
                                      Old, OldLoc, New, NewLoc);
}

/// CheckEquivalentExceptionSpec - Check if the two types have compatible
/// exception specifications. See C++ [except.spec]p3.
///
/// \return \c false if the exception specifications match, \c true if there is
/// a problem. If \c true is returned, either a diagnostic has already been
/// produced or \c *MissingExceptionSpecification is set to \c true.
bool Sema::CheckEquivalentExceptionSpec(const PartialDiagnostic &DiagID,
                                        const PartialDiagnostic & NoteID,
                                        const FunctionProtoType *Old,
                                        SourceLocation OldLoc,
                                        const FunctionProtoType *New,
                                        SourceLocation NewLoc,
                                        bool *MissingExceptionSpecification,
                                        bool*MissingEmptyExceptionSpecification,
                                        bool AllowNoexceptAllMatchWithNoSpec,
                                        bool IsOperatorNew) {
  // Just completely ignore this under -fno-exceptions.
  if (!getLangOpts().CXXExceptions)
    return false;

  if (MissingExceptionSpecification)
    *MissingExceptionSpecification = false;

  if (MissingEmptyExceptionSpecification)
    *MissingEmptyExceptionSpecification = false;

  Old = ResolveExceptionSpec(NewLoc, Old);
  if (!Old)
    return false;
  New = ResolveExceptionSpec(NewLoc, New);
  if (!New)
    return false;

  // C++0x [except.spec]p3: Two exception-specifications are compatible if:
  //   - both are non-throwing, regardless of their form,
  //   - both have the form noexcept(constant-expression) and the constant-
  //     expressions are equivalent,
  //   - both are dynamic-exception-specifications that have the same set of
  //     adjusted types.
  //
  // C++0x [except.spec]p12: An exception-specifcation is non-throwing if it is
  //   of the form throw(), noexcept, or noexcept(constant-expression) where the
  //   constant-expression yields true.
  //
  // C++0x [except.spec]p4: If any declaration of a function has an exception-
  //   specifier that is not a noexcept-specification allowing all exceptions,
  //   all declarations [...] of that function shall have a compatible
  //   exception-specification.
  //
  // That last point basically means that noexcept(false) matches no spec.
  // It's considered when AllowNoexceptAllMatchWithNoSpec is true.

  ExceptionSpecificationType OldEST = Old->getExceptionSpecType();
  ExceptionSpecificationType NewEST = New->getExceptionSpecType();

  assert(!isUnresolvedExceptionSpec(OldEST) &&
         !isUnresolvedExceptionSpec(NewEST) &&
         "Shouldn't see unknown exception specifications here");

  // Shortcut the case where both have no spec.
  if (OldEST == EST_None && NewEST == EST_None)
    return false;

  FunctionProtoType::NoexceptResult OldNR = Old->getNoexceptSpec(Context);
  FunctionProtoType::NoexceptResult NewNR = New->getNoexceptSpec(Context);
  if (OldNR == FunctionProtoType::NR_BadNoexcept ||
      NewNR == FunctionProtoType::NR_BadNoexcept)
    return false;

  // Dependent noexcept specifiers are compatible with each other, but nothing
  // else.
  // One noexcept is compatible with another if the argument is the same
  if (OldNR == NewNR &&
      OldNR != FunctionProtoType::NR_NoNoexcept &&
      NewNR != FunctionProtoType::NR_NoNoexcept)
    return false;
  if (OldNR != NewNR &&
      OldNR != FunctionProtoType::NR_NoNoexcept &&
      NewNR != FunctionProtoType::NR_NoNoexcept) {
    Diag(NewLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(OldLoc, NoteID);
    return true;
  }

  // The MS extension throw(...) is compatible with itself.
  if (OldEST == EST_MSAny && NewEST == EST_MSAny)
    return false;

  // It's also compatible with no spec.
  if ((OldEST == EST_None && NewEST == EST_MSAny) ||
      (OldEST == EST_MSAny && NewEST == EST_None))
    return false;

  // It's also compatible with noexcept(false).
  if (OldEST == EST_MSAny && NewNR == FunctionProtoType::NR_Throw)
    return false;
  if (NewEST == EST_MSAny && OldNR == FunctionProtoType::NR_Throw)
    return false;

  // As described above, noexcept(false) matches no spec only for functions.
  if (AllowNoexceptAllMatchWithNoSpec) {
    if (OldEST == EST_None && NewNR == FunctionProtoType::NR_Throw)
      return false;
    if (NewEST == EST_None && OldNR == FunctionProtoType::NR_Throw)
      return false;
  }

  // Any non-throwing specifications are compatible.
  bool OldNonThrowing = OldNR == FunctionProtoType::NR_Nothrow ||
                        OldEST == EST_DynamicNone;
  bool NewNonThrowing = NewNR == FunctionProtoType::NR_Nothrow ||
                        NewEST == EST_DynamicNone;
  if (OldNonThrowing && NewNonThrowing)
    return false;

  // As a special compatibility feature, under C++0x we accept no spec and
  // throw(std::bad_alloc) as equivalent for operator new and operator new[].
  // This is because the implicit declaration changed, but old code would break.
  if (getLangOpts().CPlusPlus0x && IsOperatorNew) {
    const FunctionProtoType *WithExceptions = 0;
    if (OldEST == EST_None && NewEST == EST_Dynamic)
      WithExceptions = New;
    else if (OldEST == EST_Dynamic && NewEST == EST_None)
      WithExceptions = Old;
    if (WithExceptions && WithExceptions->getNumExceptions() == 1) {
      // One has no spec, the other throw(something). If that something is
      // std::bad_alloc, all conditions are met.
      QualType Exception = *WithExceptions->exception_begin();
      if (CXXRecordDecl *ExRecord = Exception->getAsCXXRecordDecl()) {
        IdentifierInfo* Name = ExRecord->getIdentifier();
        if (Name && Name->getName() == "bad_alloc") {
          // It's called bad_alloc, but is it in std?
          DeclContext* DC = ExRecord->getDeclContext();
          DC = DC->getEnclosingNamespaceContext();
          if (NamespaceDecl* NS = dyn_cast<NamespaceDecl>(DC)) {
            IdentifierInfo* NSName = NS->getIdentifier();
            DC = DC->getParent();
            if (NSName && NSName->getName() == "std" &&
                DC->getEnclosingNamespaceContext()->isTranslationUnit()) {
              return false;
            }
          }
        }
      }
    }
  }

  // At this point, the only remaining valid case is two matching dynamic
  // specifications. We return here unless both specifications are dynamic.
  if (OldEST != EST_Dynamic || NewEST != EST_Dynamic) {
    if (MissingExceptionSpecification && Old->hasExceptionSpec() &&
        !New->hasExceptionSpec()) {
      // The old type has an exception specification of some sort, but
      // the new type does not.
      *MissingExceptionSpecification = true;

      if (MissingEmptyExceptionSpecification && OldNonThrowing) {
        // The old type has a throw() or noexcept(true) exception specification
        // and the new type has no exception specification, and the caller asked
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

  assert(OldEST == EST_Dynamic && NewEST == EST_Dynamic &&
      "Exception compatibility logic error: non-dynamic spec slipped through.");

  bool Success = true;
  // Both have a dynamic exception spec. Collect the first set, then compare
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
  if (!getLangOpts().CXXExceptions)
    return false;

  // FIXME: As usual, we could be more specific in our error messages, but
  // that better waits until we've got types with source locations.

  if (!SubLoc.isValid())
    SubLoc = SuperLoc;

  // Resolve the exception specifications, if needed.
  Superset = ResolveExceptionSpec(SuperLoc, Superset);
  if (!Superset)
    return false;
  Subset = ResolveExceptionSpec(SubLoc, Subset);
  if (!Subset)
    return false;

  ExceptionSpecificationType SuperEST = Superset->getExceptionSpecType();

  // If superset contains everything, we're done.
  if (SuperEST == EST_None || SuperEST == EST_MSAny)
    return CheckParamExceptionSpec(NoteID, Superset, SuperLoc, Subset, SubLoc);

  // If there are dependent noexcept specs, assume everything is fine. Unlike
  // with the equivalency check, this is safe in this case, because we don't
  // want to merge declarations. Checks after instantiation will catch any
  // omissions we make here.
  // We also shortcut checking if a noexcept expression was bad.

  FunctionProtoType::NoexceptResult SuperNR =Superset->getNoexceptSpec(Context);
  if (SuperNR == FunctionProtoType::NR_BadNoexcept ||
      SuperNR == FunctionProtoType::NR_Dependent)
    return false;

  // Another case of the superset containing everything.
  if (SuperNR == FunctionProtoType::NR_Throw)
    return CheckParamExceptionSpec(NoteID, Superset, SuperLoc, Subset, SubLoc);

  ExceptionSpecificationType SubEST = Subset->getExceptionSpecType();

  assert(!isUnresolvedExceptionSpec(SuperEST) &&
         !isUnresolvedExceptionSpec(SubEST) &&
         "Shouldn't see unknown exception specifications here");

  // It does not. If the subset contains everything, we've failed.
  if (SubEST == EST_None || SubEST == EST_MSAny) {
    Diag(SubLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(SuperLoc, NoteID);
    return true;
  }

  FunctionProtoType::NoexceptResult SubNR = Subset->getNoexceptSpec(Context);
  if (SubNR == FunctionProtoType::NR_BadNoexcept ||
      SubNR == FunctionProtoType::NR_Dependent)
    return false;

  // Another case of the subset containing everything.
  if (SubNR == FunctionProtoType::NR_Throw) {
    Diag(SubLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(SuperLoc, NoteID);
    return true;
  }

  // If the subset contains nothing, we're done.
  if (SubEST == EST_DynamicNone || SubNR == FunctionProtoType::NR_Nothrow)
    return CheckParamExceptionSpec(NoteID, Superset, SuperLoc, Subset, SubLoc);

  // Otherwise, if the superset contains nothing, we've failed.
  if (SuperEST == EST_DynamicNone || SuperNR == FunctionProtoType::NR_Nothrow) {
    Diag(SubLoc, DiagID);
    if (NoteID.getDiagID() != 0)
      Diag(SuperLoc, NoteID);
    return true;
  }

  assert(SuperEST == EST_Dynamic && SubEST == EST_Dynamic &&
         "Exception spec subset: non-dynamic case slipped through.");

  // Neither contains everything or nothing. Do a proper comparison.
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
      case AR_delayed:
        llvm_unreachable("access check delayed in non-declaration");
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
  if (getLangOpts().CPlusPlus0x && isa<CXXDestructorDecl>(New)) {
    // Don't check uninstantiated template destructors at all. We can only
    // synthesize correct specs after the template is instantiated.
    if (New->getParent()->isDependentType())
      return false;
    if (New->getParent()->isBeingDefined()) {
      // The destructor might be updated once the definition is finished. So
      // remember it and check later.
      DelayedDestructorExceptionSpecChecks.push_back(std::make_pair(
        cast<CXXDestructorDecl>(New), cast<CXXDestructorDecl>(Old)));
      return false;
    }
  }
  unsigned DiagID = diag::err_override_exception_spec;
  if (getLangOpts().MicrosoftExt)
    DiagID = diag::warn_override_exception_spec;
  return CheckExceptionSpecSubset(PDiag(DiagID),
                                  PDiag(diag::note_overridden_virtual_function),
                                  Old->getType()->getAs<FunctionProtoType>(),
                                  Old->getLocation(),
                                  New->getType()->getAs<FunctionProtoType>(),
                                  New->getLocation());
}

static CanThrowResult canSubExprsThrow(Sema &S, const Expr *CE) {
  Expr *E = const_cast<Expr*>(CE);
  CanThrowResult R = CT_Cannot;
  for (Expr::child_range I = E->children(); I && R != CT_Can; ++I)
    R = mergeCanThrow(R, S.canThrow(cast<Expr>(*I)));
  return R;
}

static CanThrowResult canCalleeThrow(Sema &S, const Expr *E,
                                           const Decl *D,
                                           bool NullThrows = true) {
  if (!D)
    return NullThrows ? CT_Can : CT_Cannot;

  // See if we can get a function type from the decl somehow.
  const ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD) // If we have no clue what we're calling, assume the worst.
    return CT_Can;

  // As an extension, we assume that __attribute__((nothrow)) functions don't
  // throw.
  if (isa<FunctionDecl>(D) && D->hasAttr<NoThrowAttr>())
    return CT_Cannot;

  QualType T = VD->getType();
  const FunctionProtoType *FT;
  if ((FT = T->getAs<FunctionProtoType>())) {
  } else if (const PointerType *PT = T->getAs<PointerType>())
    FT = PT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const ReferenceType *RT = T->getAs<ReferenceType>())
    FT = RT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const MemberPointerType *MT = T->getAs<MemberPointerType>())
    FT = MT->getPointeeType()->getAs<FunctionProtoType>();
  else if (const BlockPointerType *BT = T->getAs<BlockPointerType>())
    FT = BT->getPointeeType()->getAs<FunctionProtoType>();

  if (!FT)
    return CT_Can;

  FT = S.ResolveExceptionSpec(E->getLocStart(), FT);
  if (!FT)
    return CT_Can;

  return FT->isNothrow(S.Context) ? CT_Cannot : CT_Can;
}

static CanThrowResult canDynamicCastThrow(const CXXDynamicCastExpr *DC) {
  if (DC->isTypeDependent())
    return CT_Dependent;

  if (!DC->getTypeAsWritten()->isReferenceType())
    return CT_Cannot;

  if (DC->getSubExpr()->isTypeDependent())
    return CT_Dependent;

  return DC->getCastKind() == clang::CK_Dynamic? CT_Can : CT_Cannot;
}

static CanThrowResult canTypeidThrow(Sema &S, const CXXTypeidExpr *DC) {
  if (DC->isTypeOperand())
    return CT_Cannot;

  Expr *Op = DC->getExprOperand();
  if (Op->isTypeDependent())
    return CT_Dependent;

  const RecordType *RT = Op->getType()->getAs<RecordType>();
  if (!RT)
    return CT_Cannot;

  if (!cast<CXXRecordDecl>(RT->getDecl())->isPolymorphic())
    return CT_Cannot;

  if (Op->Classify(S.Context).isPRValue())
    return CT_Cannot;

  return CT_Can;
}

CanThrowResult Sema::canThrow(const Expr *E) {
  // C++ [expr.unary.noexcept]p3:
  //   [Can throw] if in a potentially-evaluated context the expression would
  //   contain:
  switch (E->getStmtClass()) {
  case Expr::CXXThrowExprClass:
    //   - a potentially evaluated throw-expression
    return CT_Can;

  case Expr::CXXDynamicCastExprClass: {
    //   - a potentially evaluated dynamic_cast expression dynamic_cast<T>(v),
    //     where T is a reference type, that requires a run-time check
    CanThrowResult CT = canDynamicCastThrow(cast<CXXDynamicCastExpr>(E));
    if (CT == CT_Can)
      return CT;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

  case Expr::CXXTypeidExprClass:
    //   - a potentially evaluated typeid expression applied to a glvalue
    //     expression whose type is a polymorphic class type
    return canTypeidThrow(*this, cast<CXXTypeidExpr>(E));

    //   - a potentially evaluated call to a function, member function, function
    //     pointer, or member function pointer that does not have a non-throwing
    //     exception-specification
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
  case Expr::UserDefinedLiteralClass: {
    const CallExpr *CE = cast<CallExpr>(E);
    CanThrowResult CT;
    if (E->isTypeDependent())
      CT = CT_Dependent;
    else if (isa<CXXPseudoDestructorExpr>(CE->getCallee()->IgnoreParens()))
      CT = CT_Cannot;
    else
      CT = canCalleeThrow(*this, E, CE->getCalleeDecl());
    if (CT == CT_Can)
      return CT;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

  case Expr::CXXConstructExprClass:
  case Expr::CXXTemporaryObjectExprClass: {
    CanThrowResult CT = canCalleeThrow(*this, E,
        cast<CXXConstructExpr>(E)->getConstructor());
    if (CT == CT_Can)
      return CT;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

  case Expr::LambdaExprClass: {
    const LambdaExpr *Lambda = cast<LambdaExpr>(E);
    CanThrowResult CT = CT_Cannot;
    for (LambdaExpr::capture_init_iterator Cap = Lambda->capture_init_begin(),
                                        CapEnd = Lambda->capture_init_end();
         Cap != CapEnd; ++Cap)
      CT = mergeCanThrow(CT, canThrow(*Cap));
    return CT;
  }

  case Expr::CXXNewExprClass: {
    CanThrowResult CT;
    if (E->isTypeDependent())
      CT = CT_Dependent;
    else
      CT = canCalleeThrow(*this, E, cast<CXXNewExpr>(E)->getOperatorNew());
    if (CT == CT_Can)
      return CT;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

  case Expr::CXXDeleteExprClass: {
    CanThrowResult CT;
    QualType DTy = cast<CXXDeleteExpr>(E)->getDestroyedType();
    if (DTy.isNull() || DTy->isDependentType()) {
      CT = CT_Dependent;
    } else {
      CT = canCalleeThrow(*this, E,
                          cast<CXXDeleteExpr>(E)->getOperatorDelete());
      if (const RecordType *RT = DTy->getAs<RecordType>()) {
        const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
        CT = mergeCanThrow(CT, canCalleeThrow(*this, E, RD->getDestructor()));
      }
      if (CT == CT_Can)
        return CT;
    }
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

  case Expr::CXXBindTemporaryExprClass: {
    // The bound temporary has to be destroyed again, which might throw.
    CanThrowResult CT = canCalleeThrow(*this, E,
      cast<CXXBindTemporaryExpr>(E)->getTemporary()->getDestructor());
    if (CT == CT_Can)
      return CT;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

    // ObjC message sends are like function calls, but never have exception
    // specs.
  case Expr::ObjCMessageExprClass:
  case Expr::ObjCPropertyRefExprClass:
  case Expr::ObjCSubscriptRefExprClass:
    return CT_Can;

    // All the ObjC literals that are implemented as calls are
    // potentially throwing unless we decide to close off that
    // possibility.
  case Expr::ObjCArrayLiteralClass:
  case Expr::ObjCDictionaryLiteralClass:
  case Expr::ObjCBoxedExprClass:
    return CT_Can;

    // Many other things have subexpressions, so we have to test those.
    // Some are simple:
  case Expr::ConditionalOperatorClass:
  case Expr::CompoundLiteralExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::CXXDefaultArgExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::DesignatedInitExprClass:
  case Expr::ExprWithCleanupsClass:
  case Expr::ExtVectorElementExprClass:
  case Expr::InitListExprClass:
  case Expr::MemberExprClass:
  case Expr::ObjCIsaExprClass:
  case Expr::ObjCIvarRefExprClass:
  case Expr::ParenExprClass:
  case Expr::ParenListExprClass:
  case Expr::ShuffleVectorExprClass:
  case Expr::VAArgExprClass:
    return canSubExprsThrow(*this, E);

    // Some might be dependent for other reasons.
  case Expr::ArraySubscriptExprClass:
  case Expr::BinaryOperatorClass:
  case Expr::CompoundAssignOperatorClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::ImplicitCastExprClass:
  case Expr::MaterializeTemporaryExprClass:
  case Expr::UnaryOperatorClass: {
    CanThrowResult CT = E->isTypeDependent() ? CT_Dependent : CT_Cannot;
    return mergeCanThrow(CT, canSubExprsThrow(*this, E));
  }

    // FIXME: We should handle StmtExpr, but that opens a MASSIVE can of worms.
  case Expr::StmtExprClass:
    return CT_Can;

  case Expr::ChooseExprClass:
    if (E->isTypeDependent() || E->isValueDependent())
      return CT_Dependent;
    return canThrow(cast<ChooseExpr>(E)->getChosenSubExpr(Context));

  case Expr::GenericSelectionExprClass:
    if (cast<GenericSelectionExpr>(E)->isResultDependent())
      return CT_Dependent;
    return canThrow(cast<GenericSelectionExpr>(E)->getResultExpr());

    // Some expressions are always dependent.
  case Expr::CXXDependentScopeMemberExprClass:
  case Expr::CXXUnresolvedConstructExprClass:
  case Expr::DependentScopeDeclRefExprClass:
    return CT_Dependent;

  case Expr::AsTypeExprClass:
  case Expr::BinaryConditionalOperatorClass:
  case Expr::BlockExprClass:
  case Expr::CUDAKernelCallExprClass:
  case Expr::DeclRefExprClass:
  case Expr::ObjCBridgedCastExprClass:
  case Expr::ObjCIndirectCopyRestoreExprClass:
  case Expr::ObjCProtocolExprClass:
  case Expr::ObjCSelectorExprClass:
  case Expr::OffsetOfExprClass:
  case Expr::PackExpansionExprClass:
  case Expr::PseudoObjectExprClass:
  case Expr::SubstNonTypeTemplateParmExprClass:
  case Expr::SubstNonTypeTemplateParmPackExprClass:
  case Expr::FunctionParmPackExprClass:
  case Expr::UnaryExprOrTypeTraitExprClass:
  case Expr::UnresolvedLookupExprClass:
  case Expr::UnresolvedMemberExprClass:
    // FIXME: Can any of the above throw?  If so, when?
    return CT_Cannot;

  case Expr::AddrLabelExprClass:
  case Expr::ArrayTypeTraitExprClass:
  case Expr::AtomicExprClass:
  case Expr::BinaryTypeTraitExprClass:
  case Expr::TypeTraitExprClass:
  case Expr::CXXBoolLiteralExprClass:
  case Expr::CXXNoexceptExprClass:
  case Expr::CXXNullPtrLiteralExprClass:
  case Expr::CXXPseudoDestructorExprClass:
  case Expr::CXXScalarValueInitExprClass:
  case Expr::CXXThisExprClass:
  case Expr::CXXUuidofExprClass:
  case Expr::CharacterLiteralClass:
  case Expr::ExpressionTraitExprClass:
  case Expr::FloatingLiteralClass:
  case Expr::GNUNullExprClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::ImplicitValueInitExprClass:
  case Expr::IntegerLiteralClass:
  case Expr::ObjCEncodeExprClass:
  case Expr::ObjCStringLiteralClass:
  case Expr::ObjCBoolLiteralExprClass:
  case Expr::OpaqueValueExprClass:
  case Expr::PredefinedExprClass:
  case Expr::SizeOfPackExprClass:
  case Expr::StringLiteralClass:
  case Expr::UnaryTypeTraitExprClass:
    // These expressions can never throw.
    return CT_Cannot;

#define STMT(CLASS, PARENT) case Expr::CLASS##Class:
#define STMT_RANGE(Base, First, Last)
#define LAST_STMT_RANGE(BASE, FIRST, LAST)
#define EXPR(CLASS, PARENT)
#define ABSTRACT_STMT(STMT)
#include "clang/AST/StmtNodes.inc"
  case Expr::NoStmtClass:
    llvm_unreachable("Invalid class for expression");
  }
  llvm_unreachable("Bogus StmtClass");
}

} // end namespace clang

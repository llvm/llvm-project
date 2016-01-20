//===--- SemaExpr.cpp - Semantic Analysis for Expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "TreeTransform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprOpenMP.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/AnalysisBasedWarnings.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Designator.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaFixItUtils.h"
#include "clang/Sema/Template.h"
#include "llvm/Support/ConvertUTF.h"
using namespace clang;
using namespace sema;

/// \brief Determine whether the use of this declaration is valid, without
/// emitting diagnostics.
bool Sema::CanUseDecl(NamedDecl *D) {
  // See if this is an auto-typed variable whose initializer we are parsing.
  if (ParsingInitForAutoVars.count(D))
    return false;

  // See if this is a deleted function.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isDeleted())
      return false;

    // If the function has a deduced return type, and we can't deduce it,
    // then we can't use it either.
    if (getLangOpts().CPlusPlus14 && FD->getReturnType()->isUndeducedType() &&
        DeduceReturnType(FD, SourceLocation(), /*Diagnose*/ false))
      return false;
  }

  // See if this function is unavailable.
  if (D->getAvailability() == AR_Unavailable &&
      cast<Decl>(CurContext)->getAvailability() != AR_Unavailable)
    return false;

  return true;
}

static void DiagnoseUnusedOfDecl(Sema &S, NamedDecl *D, SourceLocation Loc) {
  // Warn if this is used but marked unused.
  if (D->hasAttr<UnusedAttr>()) {
    const Decl *DC = cast_or_null<Decl>(S.getCurObjCLexicalContext());
    if (DC && !DC->hasAttr<UnusedAttr>())
      S.Diag(Loc, diag::warn_used_but_marked_unused) << D->getDeclName();
  }
}

static bool HasRedeclarationWithoutAvailabilityInCategory(const Decl *D) {
  const auto *OMD = dyn_cast<ObjCMethodDecl>(D);
  if (!OMD)
    return false;
  const ObjCInterfaceDecl *OID = OMD->getClassInterface();
  if (!OID)
    return false;

  for (const ObjCCategoryDecl *Cat : OID->visible_categories())
    if (ObjCMethodDecl *CatMeth =
            Cat->getMethod(OMD->getSelector(), OMD->isInstanceMethod()))
      if (!CatMeth->hasAttr<AvailabilityAttr>())
        return true;
  return false;
}

static AvailabilityResult
DiagnoseAvailabilityOfDecl(Sema &S, NamedDecl *D, SourceLocation Loc,
                           const ObjCInterfaceDecl *UnknownObjCClass,
                           bool ObjCPropertyAccess) {
  // See if this declaration is unavailable or deprecated.
  std::string Message;
  AvailabilityResult Result = D->getAvailability(&Message);

  // For typedefs, if the typedef declaration appears available look
  // to the underlying type to see if it is more restrictive.
  while (const TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    if (Result == AR_Available) {
      if (const TagType *TT = TD->getUnderlyingType()->getAs<TagType>()) {
        D = TT->getDecl();
        Result = D->getAvailability(&Message);
        continue;
      }
    }
    break;
  }
    
  // Forward class declarations get their attributes from their definition.
  if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(D)) {
    if (IDecl->getDefinition()) {
      D = IDecl->getDefinition();
      Result = D->getAvailability(&Message);
    }
  }

  if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D))
    if (Result == AR_Available) {
      const DeclContext *DC = ECD->getDeclContext();
      if (const EnumDecl *TheEnumDecl = dyn_cast<EnumDecl>(DC))
        Result = TheEnumDecl->getAvailability(&Message);
    }

  const ObjCPropertyDecl *ObjCPDecl = nullptr;
  if (Result == AR_Deprecated || Result == AR_Unavailable ||
      AR_NotYetIntroduced) {
    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
      if (const ObjCPropertyDecl *PD = MD->findPropertyDecl()) {
        AvailabilityResult PDeclResult = PD->getAvailability(nullptr);
        if (PDeclResult == Result)
          ObjCPDecl = PD;
      }
    }
  }
  
  switch (Result) {
    case AR_Available:
      break;

    case AR_Deprecated:
      if (S.getCurContextAvailability() != AR_Deprecated)
        S.EmitAvailabilityWarning(Sema::AD_Deprecation,
                                  D, Message, Loc, UnknownObjCClass, ObjCPDecl,
                                  ObjCPropertyAccess);
      break;

    case AR_NotYetIntroduced: {
      // Don't do this for enums, they can't be redeclared.
      if (isa<EnumConstantDecl>(D) || isa<EnumDecl>(D))
        break;
 
      bool Warn = !D->getAttr<AvailabilityAttr>()->isInherited();
      // Objective-C method declarations in categories are not modelled as
      // redeclarations, so manually look for a redeclaration in a category
      // if necessary.
      if (Warn && HasRedeclarationWithoutAvailabilityInCategory(D))
        Warn = false;
      // In general, D will point to the most recent redeclaration. However,
      // for `@class A;` decls, this isn't true -- manually go through the
      // redecl chain in that case.
      if (Warn && isa<ObjCInterfaceDecl>(D))
        for (Decl *Redecl = D->getMostRecentDecl(); Redecl && Warn;
             Redecl = Redecl->getPreviousDecl())
          if (!Redecl->hasAttr<AvailabilityAttr>() ||
              Redecl->getAttr<AvailabilityAttr>()->isInherited())
            Warn = false;
 
      if (Warn)
        S.EmitAvailabilityWarning(Sema::AD_Partial, D, Message, Loc,
                                  UnknownObjCClass, ObjCPDecl,
                                  ObjCPropertyAccess);
      break;
    }

    case AR_Unavailable:
      if (S.getCurContextAvailability() != AR_Unavailable)
        S.EmitAvailabilityWarning(Sema::AD_Unavailable,
                                  D, Message, Loc, UnknownObjCClass, ObjCPDecl,
                                  ObjCPropertyAccess);
      break;

    }
    return Result;
}

/// \brief Emit a note explaining that this function is deleted.
void Sema::NoteDeletedFunction(FunctionDecl *Decl) {
  assert(Decl->isDeleted());

  CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Decl);

  if (Method && Method->isDeleted() && Method->isDefaulted()) {
    // If the method was explicitly defaulted, point at that declaration.
    if (!Method->isImplicit())
      Diag(Decl->getLocation(), diag::note_implicitly_deleted);

    // Try to diagnose why this special member function was implicitly
    // deleted. This might fail, if that reason no longer applies.
    CXXSpecialMember CSM = getSpecialMember(Method);
    if (CSM != CXXInvalid)
      ShouldDeleteSpecialMember(Method, CSM, /*Diagnose=*/true);

    return;
  }

  if (CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(Decl)) {
    if (CXXConstructorDecl *BaseCD =
            const_cast<CXXConstructorDecl*>(CD->getInheritedConstructor())) {
      Diag(Decl->getLocation(), diag::note_inherited_deleted_here);
      if (BaseCD->isDeleted()) {
        NoteDeletedFunction(BaseCD);
      } else {
        // FIXME: An explanation of why exactly it can't be inherited
        // would be nice.
        Diag(BaseCD->getLocation(), diag::note_cannot_inherit);
      }
      return;
    }
  }

  Diag(Decl->getLocation(), diag::note_availability_specified_here)
    << Decl << true;
}

/// \brief Determine whether a FunctionDecl was ever declared with an
/// explicit storage class.
static bool hasAnyExplicitStorageClass(const FunctionDecl *D) {
  for (auto I : D->redecls()) {
    if (I->getStorageClass() != SC_None)
      return true;
  }
  return false;
}

/// \brief Check whether we're in an extern inline function and referring to a
/// variable or function with internal linkage (C11 6.7.4p3).
///
/// This is only a warning because we used to silently accept this code, but
/// in many cases it will not behave correctly. This is not enabled in C++ mode
/// because the restriction language is a bit weaker (C++11 [basic.def.odr]p6)
/// and so while there may still be user mistakes, most of the time we can't
/// prove that there are errors.
static void diagnoseUseOfInternalDeclInInlineFunction(Sema &S,
                                                      const NamedDecl *D,
                                                      SourceLocation Loc) {
  // This is disabled under C++; there are too many ways for this to fire in
  // contexts where the warning is a false positive, or where it is technically
  // correct but benign.
  if (S.getLangOpts().CPlusPlus)
    return;

  // Check if this is an inlined function or method.
  FunctionDecl *Current = S.getCurFunctionDecl();
  if (!Current)
    return;
  if (!Current->isInlined())
    return;
  if (!Current->isExternallyVisible())
    return;

  // Check if the decl has internal linkage.
  if (D->getFormalLinkage() != InternalLinkage)
    return;

  // Downgrade from ExtWarn to Extension if
  //  (1) the supposedly external inline function is in the main file,
  //      and probably won't be included anywhere else.
  //  (2) the thing we're referencing is a pure function.
  //  (3) the thing we're referencing is another inline function.
  // This last can give us false negatives, but it's better than warning on
  // wrappers for simple C library functions.
  const FunctionDecl *UsedFn = dyn_cast<FunctionDecl>(D);
  bool DowngradeWarning = S.getSourceManager().isInMainFile(Loc);
  if (!DowngradeWarning && UsedFn)
    DowngradeWarning = UsedFn->isInlined() || UsedFn->hasAttr<ConstAttr>();

  S.Diag(Loc, DowngradeWarning ? diag::ext_internal_in_extern_inline_quiet
                               : diag::ext_internal_in_extern_inline)
    << /*IsVar=*/!UsedFn << D;

  S.MaybeSuggestAddingStaticToDecl(Current);

  S.Diag(D->getCanonicalDecl()->getLocation(), diag::note_entity_declared_at)
      << D;
}

void Sema::MaybeSuggestAddingStaticToDecl(const FunctionDecl *Cur) {
  const FunctionDecl *First = Cur->getFirstDecl();

  // Suggest "static" on the function, if possible.
  if (!hasAnyExplicitStorageClass(First)) {
    SourceLocation DeclBegin = First->getSourceRange().getBegin();
    Diag(DeclBegin, diag::note_convert_inline_to_static)
      << Cur << FixItHint::CreateInsertion(DeclBegin, "static ");
  }
}

/// \brief Determine whether the use of this declaration is valid, and
/// emit any corresponding diagnostics.
///
/// This routine diagnoses various problems with referencing
/// declarations that can occur when using a declaration. For example,
/// it might warn if a deprecated or unavailable declaration is being
/// used, or produce an error (and return true) if a C++0x deleted
/// function is being used.
///
/// \returns true if there was an error (this declaration cannot be
/// referenced), false otherwise.
///
bool Sema::DiagnoseUseOfDecl(NamedDecl *D, SourceLocation Loc,
                             const ObjCInterfaceDecl *UnknownObjCClass,
                             bool ObjCPropertyAccess) {
  if (getLangOpts().CPlusPlus && isa<FunctionDecl>(D)) {
    // If there were any diagnostics suppressed by template argument deduction,
    // emit them now.
    auto Pos = SuppressedDiagnostics.find(D->getCanonicalDecl());
    if (Pos != SuppressedDiagnostics.end()) {
      for (const PartialDiagnosticAt &Suppressed : Pos->second)
        Diag(Suppressed.first, Suppressed.second);

      // Clear out the list of suppressed diagnostics, so that we don't emit
      // them again for this specialization. However, we don't obsolete this
      // entry from the table, because we want to avoid ever emitting these
      // diagnostics again.
      Pos->second.clear();
    }

    // C++ [basic.start.main]p3:
    //   The function 'main' shall not be used within a program.
    if (cast<FunctionDecl>(D)->isMain())
      Diag(Loc, diag::ext_main_used);
  }

  // See if this is an auto-typed variable whose initializer we are parsing.
  if (ParsingInitForAutoVars.count(D)) {
    const AutoType *AT = cast<VarDecl>(D)->getType()->getContainedAutoType();

    Diag(Loc, diag::err_auto_variable_cannot_appear_in_own_initializer)
      << D->getDeclName() << (unsigned)AT->getKeyword();
    return true;
  }

  // See if this is a deleted function.
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->isDeleted()) {
      Diag(Loc, diag::err_deleted_function_use);
      NoteDeletedFunction(FD);
      return true;
    }

    // If the function has a deduced return type, and we can't deduce it,
    // then we can't use it either.
    if (getLangOpts().CPlusPlus14 && FD->getReturnType()->isUndeducedType() &&
        DeduceReturnType(FD, Loc))
      return true;
  }
  DiagnoseAvailabilityOfDecl(*this, D, Loc, UnknownObjCClass,
                             ObjCPropertyAccess);

  DiagnoseUnusedOfDecl(*this, D, Loc);

  diagnoseUseOfInternalDeclInInlineFunction(*this, D, Loc);

  return false;
}

/// \brief Retrieve the message suffix that should be added to a
/// diagnostic complaining about the given function being deleted or
/// unavailable.
std::string Sema::getDeletedOrUnavailableSuffix(const FunctionDecl *FD) {
  std::string Message;
  if (FD->getAvailability(&Message))
    return ": " + Message;

  return std::string();
}

/// DiagnoseSentinelCalls - This routine checks whether a call or
/// message-send is to a declaration with the sentinel attribute, and
/// if so, it checks that the requirements of the sentinel are
/// satisfied.
void Sema::DiagnoseSentinelCalls(NamedDecl *D, SourceLocation Loc,
                                 ArrayRef<Expr *> Args) {
  const SentinelAttr *attr = D->getAttr<SentinelAttr>();
  if (!attr)
    return;

  // The number of formal parameters of the declaration.
  unsigned numFormalParams;

  // The kind of declaration.  This is also an index into a %select in
  // the diagnostic.
  enum CalleeType { CT_Function, CT_Method, CT_Block } calleeType;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    numFormalParams = MD->param_size();
    calleeType = CT_Method;
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    numFormalParams = FD->param_size();
    calleeType = CT_Function;
  } else if (isa<VarDecl>(D)) {
    QualType type = cast<ValueDecl>(D)->getType();
    const FunctionType *fn = nullptr;
    if (const PointerType *ptr = type->getAs<PointerType>()) {
      fn = ptr->getPointeeType()->getAs<FunctionType>();
      if (!fn) return;
      calleeType = CT_Function;
    } else if (const BlockPointerType *ptr = type->getAs<BlockPointerType>()) {
      fn = ptr->getPointeeType()->castAs<FunctionType>();
      calleeType = CT_Block;
    } else {
      return;
    }

    if (const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(fn)) {
      numFormalParams = proto->getNumParams();
    } else {
      numFormalParams = 0;
    }
  } else {
    return;
  }

  // "nullPos" is the number of formal parameters at the end which
  // effectively count as part of the variadic arguments.  This is
  // useful if you would prefer to not have *any* formal parameters,
  // but the language forces you to have at least one.
  unsigned nullPos = attr->getNullPos();
  assert((nullPos == 0 || nullPos == 1) && "invalid null position on sentinel");
  numFormalParams = (nullPos > numFormalParams ? 0 : numFormalParams - nullPos);

  // The number of arguments which should follow the sentinel.
  unsigned numArgsAfterSentinel = attr->getSentinel();

  // If there aren't enough arguments for all the formal parameters,
  // the sentinel, and the args after the sentinel, complain.
  if (Args.size() < numFormalParams + numArgsAfterSentinel + 1) {
    Diag(Loc, diag::warn_not_enough_argument) << D->getDeclName();
    Diag(D->getLocation(), diag::note_sentinel_here) << int(calleeType);
    return;
  }

  // Otherwise, find the sentinel expression.
  Expr *sentinelExpr = Args[Args.size() - numArgsAfterSentinel - 1];
  if (!sentinelExpr) return;
  if (sentinelExpr->isValueDependent()) return;
  if (Context.isSentinelNullExpr(sentinelExpr)) return;

  // Pick a reasonable string to insert.  Optimistically use 'nil', 'nullptr',
  // or 'NULL' if those are actually defined in the context.  Only use
  // 'nil' for ObjC methods, where it's much more likely that the
  // variadic arguments form a list of object pointers.
  SourceLocation MissingNilLoc
    = getLocForEndOfToken(sentinelExpr->getLocEnd());
  std::string NullValue;
  if (calleeType == CT_Method && PP.isMacroDefined("nil"))
    NullValue = "nil";
  else if (getLangOpts().CPlusPlus11)
    NullValue = "nullptr";
  else if (PP.isMacroDefined("NULL"))
    NullValue = "NULL";
  else
    NullValue = "(void*) 0";

  if (MissingNilLoc.isInvalid())
    Diag(Loc, diag::warn_missing_sentinel) << int(calleeType);
  else
    Diag(MissingNilLoc, diag::warn_missing_sentinel) 
      << int(calleeType)
      << FixItHint::CreateInsertion(MissingNilLoc, ", " + NullValue);
  Diag(D->getLocation(), diag::note_sentinel_here) << int(calleeType);
}

SourceRange Sema::getExprRange(Expr *E) const {
  return E ? E->getSourceRange() : SourceRange();
}

//===----------------------------------------------------------------------===//
//  Standard Promotions and Conversions
//===----------------------------------------------------------------------===//

/// DefaultFunctionArrayConversion (C99 6.3.2.1p3, C99 6.3.2.1p4).
ExprResult Sema::DefaultFunctionArrayConversion(Expr *E, bool Diagnose) {
  // Handle any placeholder expressions which made it here.
  if (E->getType()->isPlaceholderType()) {
    ExprResult result = CheckPlaceholderExpr(E);
    if (result.isInvalid()) return ExprError();
    E = result.get();
  }
  
  QualType Ty = E->getType();
  assert(!Ty.isNull() && "DefaultFunctionArrayConversion - missing type");

  if (Ty->isFunctionType()) {
    // If we are here, we are not calling a function but taking
    // its address (which is not allowed in OpenCL v1.0 s6.8.a.3).
    if (getLangOpts().OpenCL) {
      if (Diagnose)
        Diag(E->getExprLoc(), diag::err_opencl_taking_function_address);
      return ExprError();
    }

    if (auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts()))
      if (auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
        if (!checkAddressOfFunctionIsAvailable(FD, Diagnose, E->getExprLoc()))
          return ExprError();

    E = ImpCastExprToType(E, Context.getPointerType(Ty),
                          CK_FunctionToPointerDecay).get();
  } else if (Ty->isArrayType()) {
    // In C90 mode, arrays only promote to pointers if the array expression is
    // an lvalue.  The relevant legalese is C90 6.2.2.1p3: "an lvalue that has
    // type 'array of type' is converted to an expression that has type 'pointer
    // to type'...".  In C99 this was changed to: C99 6.3.2.1p3: "an expression
    // that has type 'array of type' ...".  The relevant change is "an lvalue"
    // (C90) to "an expression" (C99).
    //
    // C++ 4.2p1:
    // An lvalue or rvalue of type "array of N T" or "array of unknown bound of
    // T" can be converted to an rvalue of type "pointer to T".
    //
    if (getLangOpts().C99 || getLangOpts().CPlusPlus || E->isLValue())
      E = ImpCastExprToType(E, Context.getArrayDecayedType(Ty),
                            CK_ArrayToPointerDecay).get();
  }
  return E;
}

static void CheckForNullPointerDereference(Sema &S, Expr *E) {
  // Check to see if we are dereferencing a null pointer.  If so,
  // and if not volatile-qualified, this is undefined behavior that the
  // optimizer will delete, so warn about it.  People sometimes try to use this
  // to get a deterministic trap and are surprised by clang's behavior.  This
  // only handles the pattern "*null", which is a very syntactic check.
  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E->IgnoreParenCasts()))
    if (UO->getOpcode() == UO_Deref &&
        UO->getSubExpr()->IgnoreParenCasts()->
          isNullPointerConstant(S.Context, Expr::NPC_ValueDependentIsNotNull) &&
        !UO->getType().isVolatileQualified()) {
    S.DiagRuntimeBehavior(UO->getOperatorLoc(), UO,
                          S.PDiag(diag::warn_indirection_through_null)
                            << UO->getSubExpr()->getSourceRange());
    S.DiagRuntimeBehavior(UO->getOperatorLoc(), UO,
                        S.PDiag(diag::note_indirection_through_null));
  }
}

static void DiagnoseDirectIsaAccess(Sema &S, const ObjCIvarRefExpr *OIRE,
                                    SourceLocation AssignLoc,
                                    const Expr* RHS) {
  const ObjCIvarDecl *IV = OIRE->getDecl();
  if (!IV)
    return;
  
  DeclarationName MemberName = IV->getDeclName();
  IdentifierInfo *Member = MemberName.getAsIdentifierInfo();
  if (!Member || !Member->isStr("isa"))
    return;
  
  const Expr *Base = OIRE->getBase();
  QualType BaseType = Base->getType();
  if (OIRE->isArrow())
    BaseType = BaseType->getPointeeType();
  if (const ObjCObjectType *OTy = BaseType->getAs<ObjCObjectType>())
    if (ObjCInterfaceDecl *IDecl = OTy->getInterface()) {
      ObjCInterfaceDecl *ClassDeclared = nullptr;
      ObjCIvarDecl *IV = IDecl->lookupInstanceVariable(Member, ClassDeclared);
      if (!ClassDeclared->getSuperClass()
          && (*ClassDeclared->ivar_begin()) == IV) {
        if (RHS) {
          NamedDecl *ObjectSetClass =
            S.LookupSingleName(S.TUScope,
                               &S.Context.Idents.get("object_setClass"),
                               SourceLocation(), S.LookupOrdinaryName);
          if (ObjectSetClass) {
            SourceLocation RHSLocEnd = S.getLocForEndOfToken(RHS->getLocEnd());
            S.Diag(OIRE->getExprLoc(), diag::warn_objc_isa_assign) <<
            FixItHint::CreateInsertion(OIRE->getLocStart(), "object_setClass(") <<
            FixItHint::CreateReplacement(SourceRange(OIRE->getOpLoc(),
                                                     AssignLoc), ",") <<
            FixItHint::CreateInsertion(RHSLocEnd, ")");
          }
          else
            S.Diag(OIRE->getLocation(), diag::warn_objc_isa_assign);
        } else {
          NamedDecl *ObjectGetClass =
            S.LookupSingleName(S.TUScope,
                               &S.Context.Idents.get("object_getClass"),
                               SourceLocation(), S.LookupOrdinaryName);
          if (ObjectGetClass)
            S.Diag(OIRE->getExprLoc(), diag::warn_objc_isa_use) <<
            FixItHint::CreateInsertion(OIRE->getLocStart(), "object_getClass(") <<
            FixItHint::CreateReplacement(
                                         SourceRange(OIRE->getOpLoc(),
                                                     OIRE->getLocEnd()), ")");
          else
            S.Diag(OIRE->getLocation(), diag::warn_objc_isa_use);
        }
        S.Diag(IV->getLocation(), diag::note_ivar_decl);
      }
    }
}

ExprResult Sema::DefaultLvalueConversion(Expr *E) {
  // Handle any placeholder expressions which made it here.
  if (E->getType()->isPlaceholderType()) {
    ExprResult result = CheckPlaceholderExpr(E);
    if (result.isInvalid()) return ExprError();
    E = result.get();
  }
  
  // C++ [conv.lval]p1:
  //   A glvalue of a non-function, non-array type T can be
  //   converted to a prvalue.
  if (!E->isGLValue()) return E;

  QualType T = E->getType();
  assert(!T.isNull() && "r-value conversion on typeless expression?");

  // We don't want to throw lvalue-to-rvalue casts on top of
  // expressions of certain types in C++.
  if (getLangOpts().CPlusPlus &&
      (E->getType() == Context.OverloadTy ||
       T->isDependentType() ||
       T->isRecordType()))
    return E;

  // The C standard is actually really unclear on this point, and
  // DR106 tells us what the result should be but not why.  It's
  // generally best to say that void types just doesn't undergo
  // lvalue-to-rvalue at all.  Note that expressions of unqualified
  // 'void' type are never l-values, but qualified void can be.
  if (T->isVoidType())
    return E;

  // OpenCL usually rejects direct accesses to values of 'half' type.
  if (getLangOpts().OpenCL && !getOpenCLOptions().cl_khr_fp16 &&
      T->isHalfType()) {
    Diag(E->getExprLoc(), diag::err_opencl_half_load_store)
      << 0 << T;
    return ExprError();
  }

  CheckForNullPointerDereference(*this, E);
  if (const ObjCIsaExpr *OISA = dyn_cast<ObjCIsaExpr>(E->IgnoreParenCasts())) {
    NamedDecl *ObjectGetClass = LookupSingleName(TUScope,
                                     &Context.Idents.get("object_getClass"),
                                     SourceLocation(), LookupOrdinaryName);
    if (ObjectGetClass)
      Diag(E->getExprLoc(), diag::warn_objc_isa_use) <<
        FixItHint::CreateInsertion(OISA->getLocStart(), "object_getClass(") <<
        FixItHint::CreateReplacement(
                    SourceRange(OISA->getOpLoc(), OISA->getIsaMemberLoc()), ")");
    else
      Diag(E->getExprLoc(), diag::warn_objc_isa_use);
  }
  else if (const ObjCIvarRefExpr *OIRE =
            dyn_cast<ObjCIvarRefExpr>(E->IgnoreParenCasts()))
    DiagnoseDirectIsaAccess(*this, OIRE, SourceLocation(), /* Expr*/nullptr);

  // C++ [conv.lval]p1:
  //   [...] If T is a non-class type, the type of the prvalue is the
  //   cv-unqualified version of T. Otherwise, the type of the
  //   rvalue is T.
  //
  // C99 6.3.2.1p2:
  //   If the lvalue has qualified type, the value has the unqualified
  //   version of the type of the lvalue; otherwise, the value has the
  //   type of the lvalue.
  if (T.hasQualifiers())
    T = T.getUnqualifiedType();

  // Under the MS ABI, lock down the inheritance model now.
  if (T->isMemberPointerType() &&
      Context.getTargetInfo().getCXXABI().isMicrosoft())
    (void)isCompleteType(E->getExprLoc(), T);

  UpdateMarkingForLValueToRValue(E);
  
  // Loading a __weak object implicitly retains the value, so we need a cleanup to 
  // balance that.
  if (getLangOpts().ObjCAutoRefCount &&
      E->getType().getObjCLifetime() == Qualifiers::OCL_Weak)
    ExprNeedsCleanups = true;

  ExprResult Res = ImplicitCastExpr::Create(Context, T, CK_LValueToRValue, E,
                                            nullptr, VK_RValue);

  // C11 6.3.2.1p2:
  //   ... if the lvalue has atomic type, the value has the non-atomic version 
  //   of the type of the lvalue ...
  if (const AtomicType *Atomic = T->getAs<AtomicType>()) {
    T = Atomic->getValueType().getUnqualifiedType();
    Res = ImplicitCastExpr::Create(Context, T, CK_AtomicToNonAtomic, Res.get(),
                                   nullptr, VK_RValue);
  }
  
  return Res;
}

ExprResult Sema::DefaultFunctionArrayLvalueConversion(Expr *E, bool Diagnose) {
  ExprResult Res = DefaultFunctionArrayConversion(E, Diagnose);
  if (Res.isInvalid())
    return ExprError();
  Res = DefaultLvalueConversion(Res.get());
  if (Res.isInvalid())
    return ExprError();
  return Res;
}

/// CallExprUnaryConversions - a special case of an unary conversion
/// performed on a function designator of a call expression.
ExprResult Sema::CallExprUnaryConversions(Expr *E) {
  QualType Ty = E->getType();
  ExprResult Res = E;
  // Only do implicit cast for a function type, but not for a pointer
  // to function type.
  if (Ty->isFunctionType()) {
    Res = ImpCastExprToType(E, Context.getPointerType(Ty),
                            CK_FunctionToPointerDecay).get();
    if (Res.isInvalid())
      return ExprError();
  }
  Res = DefaultLvalueConversion(Res.get());
  if (Res.isInvalid())
    return ExprError();
  return Res.get();
}

/// UsualUnaryConversions - Performs various conversions that are common to most
/// operators (C99 6.3). The conversions of array and function types are
/// sometimes suppressed. For example, the array->pointer conversion doesn't
/// apply if the array is an argument to the sizeof or address (&) operators.
/// In these instances, this routine should *not* be called.
ExprResult Sema::UsualUnaryConversions(Expr *E) {
  // First, convert to an r-value.
  ExprResult Res = DefaultFunctionArrayLvalueConversion(E);
  if (Res.isInvalid())
    return ExprError();
  E = Res.get();

  QualType Ty = E->getType();
  assert(!Ty.isNull() && "UsualUnaryConversions - missing type");

  // Half FP have to be promoted to float unless it is natively supported
  if (Ty->isHalfType() && !getLangOpts().NativeHalfType)
    return ImpCastExprToType(Res.get(), Context.FloatTy, CK_FloatingCast);

  // Try to perform integral promotions if the object has a theoretically
  // promotable type.
  if (Ty->isIntegralOrUnscopedEnumerationType()) {
    // C99 6.3.1.1p2:
    //
    //   The following may be used in an expression wherever an int or
    //   unsigned int may be used:
    //     - an object or expression with an integer type whose integer
    //       conversion rank is less than or equal to the rank of int
    //       and unsigned int.
    //     - A bit-field of type _Bool, int, signed int, or unsigned int.
    //
    //   If an int can represent all values of the original type, the
    //   value is converted to an int; otherwise, it is converted to an
    //   unsigned int. These are called the integer promotions. All
    //   other types are unchanged by the integer promotions.

    QualType PTy = Context.isPromotableBitField(E);
    if (!PTy.isNull()) {
      E = ImpCastExprToType(E, PTy, CK_IntegralCast).get();
      return E;
    }
    if (Ty->isPromotableIntegerType()) {
      QualType PT = Context.getPromotedIntegerType(Ty);
      E = ImpCastExprToType(E, PT, CK_IntegralCast).get();
      return E;
    }
  }
  return E;
}

/// DefaultArgumentPromotion (C99 6.5.2.2p6). Used for function calls that
/// do not have a prototype. Arguments that have type float or __fp16
/// are promoted to double. All other argument types are converted by
/// UsualUnaryConversions().
ExprResult Sema::DefaultArgumentPromotion(Expr *E) {
  QualType Ty = E->getType();
  assert(!Ty.isNull() && "DefaultArgumentPromotion - missing type");

  ExprResult Res = UsualUnaryConversions(E);
  if (Res.isInvalid())
    return ExprError();
  E = Res.get();

  // If this is a 'float' or '__fp16' (CVR qualified or typedef) promote to
  // double.
  const BuiltinType *BTy = Ty->getAs<BuiltinType>();
  if (BTy && (BTy->getKind() == BuiltinType::Half ||
              BTy->getKind() == BuiltinType::Float))
    E = ImpCastExprToType(E, Context.DoubleTy, CK_FloatingCast).get();

  // C++ performs lvalue-to-rvalue conversion as a default argument
  // promotion, even on class types, but note:
  //   C++11 [conv.lval]p2:
  //     When an lvalue-to-rvalue conversion occurs in an unevaluated
  //     operand or a subexpression thereof the value contained in the
  //     referenced object is not accessed. Otherwise, if the glvalue
  //     has a class type, the conversion copy-initializes a temporary
  //     of type T from the glvalue and the result of the conversion
  //     is a prvalue for the temporary.
  // FIXME: add some way to gate this entire thing for correctness in
  // potentially potentially evaluated contexts.
  if (getLangOpts().CPlusPlus && E->isGLValue() && !isUnevaluatedContext()) {
    ExprResult Temp = PerformCopyInitialization(
                       InitializedEntity::InitializeTemporary(E->getType()),
                                                E->getExprLoc(), E);
    if (Temp.isInvalid())
      return ExprError();
    E = Temp.get();
  }

  return E;
}

/// Determine the degree of POD-ness for an expression.
/// Incomplete types are considered POD, since this check can be performed
/// when we're in an unevaluated context.
Sema::VarArgKind Sema::isValidVarArgType(const QualType &Ty) {
  if (Ty->isIncompleteType()) {
    // C++11 [expr.call]p7:
    //   After these conversions, if the argument does not have arithmetic,
    //   enumeration, pointer, pointer to member, or class type, the program
    //   is ill-formed.
    //
    // Since we've already performed array-to-pointer and function-to-pointer
    // decay, the only such type in C++ is cv void. This also handles
    // initializer lists as variadic arguments.
    if (Ty->isVoidType())
      return VAK_Invalid;

    if (Ty->isObjCObjectType())
      return VAK_Invalid;
    return VAK_Valid;
  }

  if (Ty.isCXX98PODType(Context))
    return VAK_Valid;

  // C++11 [expr.call]p7:
  //   Passing a potentially-evaluated argument of class type (Clause 9)
  //   having a non-trivial copy constructor, a non-trivial move constructor,
  //   or a non-trivial destructor, with no corresponding parameter,
  //   is conditionally-supported with implementation-defined semantics.
  if (getLangOpts().CPlusPlus11 && !Ty->isDependentType())
    if (CXXRecordDecl *Record = Ty->getAsCXXRecordDecl())
      if (!Record->hasNonTrivialCopyConstructor() &&
          !Record->hasNonTrivialMoveConstructor() &&
          !Record->hasNonTrivialDestructor())
        return VAK_ValidInCXX11;

  if (getLangOpts().ObjCAutoRefCount && Ty->isObjCLifetimeType())
    return VAK_Valid;

  if (Ty->isObjCObjectType())
    return VAK_Invalid;

  if (getLangOpts().MSVCCompat)
    return VAK_MSVCUndefined;

  // FIXME: In C++11, these cases are conditionally-supported, meaning we're
  // permitted to reject them. We should consider doing so.
  return VAK_Undefined;
}

void Sema::checkVariadicArgument(const Expr *E, VariadicCallType CT) {
  // Don't allow one to pass an Objective-C interface to a vararg.
  const QualType &Ty = E->getType();
  VarArgKind VAK = isValidVarArgType(Ty);

  // Complain about passing non-POD types through varargs.
  switch (VAK) {
  case VAK_ValidInCXX11:
    DiagRuntimeBehavior(
        E->getLocStart(), nullptr,
        PDiag(diag::warn_cxx98_compat_pass_non_pod_arg_to_vararg)
          << Ty << CT);
    // Fall through.
  case VAK_Valid:
    if (Ty->isRecordType()) {
      // This is unlikely to be what the user intended. If the class has a
      // 'c_str' member function, the user probably meant to call that.
      DiagRuntimeBehavior(E->getLocStart(), nullptr,
                          PDiag(diag::warn_pass_class_arg_to_vararg)
                            << Ty << CT << hasCStrMethod(E) << ".c_str()");
    }
    break;

  case VAK_Undefined:
  case VAK_MSVCUndefined:
    DiagRuntimeBehavior(
        E->getLocStart(), nullptr,
        PDiag(diag::warn_cannot_pass_non_pod_arg_to_vararg)
          << getLangOpts().CPlusPlus11 << Ty << CT);
    break;

  case VAK_Invalid:
    if (Ty->isObjCObjectType())
      DiagRuntimeBehavior(
          E->getLocStart(), nullptr,
          PDiag(diag::err_cannot_pass_objc_interface_to_vararg)
            << Ty << CT);
    else
      Diag(E->getLocStart(), diag::err_cannot_pass_to_vararg)
        << isa<InitListExpr>(E) << Ty << CT;
    break;
  }
}

/// DefaultVariadicArgumentPromotion - Like DefaultArgumentPromotion, but
/// will create a trap if the resulting type is not a POD type.
ExprResult Sema::DefaultVariadicArgumentPromotion(Expr *E, VariadicCallType CT,
                                                  FunctionDecl *FDecl) {
  if (const BuiltinType *PlaceholderTy = E->getType()->getAsPlaceholderType()) {
    // Strip the unbridged-cast placeholder expression off, if applicable.
    if (PlaceholderTy->getKind() == BuiltinType::ARCUnbridgedCast &&
        (CT == VariadicMethod ||
         (FDecl && FDecl->hasAttr<CFAuditedTransferAttr>()))) {
      E = stripARCUnbridgedCast(E);

    // Otherwise, do normal placeholder checking.
    } else {
      ExprResult ExprRes = CheckPlaceholderExpr(E);
      if (ExprRes.isInvalid())
        return ExprError();
      E = ExprRes.get();
    }
  }
  
  ExprResult ExprRes = DefaultArgumentPromotion(E);
  if (ExprRes.isInvalid())
    return ExprError();
  E = ExprRes.get();

  // Diagnostics regarding non-POD argument types are
  // emitted along with format string checking in Sema::CheckFunctionCall().
  if (isValidVarArgType(E->getType()) == VAK_Undefined) {
    // Turn this into a trap.
    CXXScopeSpec SS;
    SourceLocation TemplateKWLoc;
    UnqualifiedId Name;
    Name.setIdentifier(PP.getIdentifierInfo("__builtin_trap"),
                       E->getLocStart());
    ExprResult TrapFn = ActOnIdExpression(TUScope, SS, TemplateKWLoc,
                                          Name, true, false);
    if (TrapFn.isInvalid())
      return ExprError();

    ExprResult Call = ActOnCallExpr(TUScope, TrapFn.get(),
                                    E->getLocStart(), None,
                                    E->getLocEnd());
    if (Call.isInvalid())
      return ExprError();

    ExprResult Comma = ActOnBinOp(TUScope, E->getLocStart(), tok::comma,
                                  Call.get(), E);
    if (Comma.isInvalid())
      return ExprError();
    return Comma.get();
  }

  if (!getLangOpts().CPlusPlus &&
      RequireCompleteType(E->getExprLoc(), E->getType(),
                          diag::err_call_incomplete_argument))
    return ExprError();

  return E;
}

/// \brief Converts an integer to complex float type.  Helper function of
/// UsualArithmeticConversions()
///
/// \return false if the integer expression is an integer type and is
/// successfully converted to the complex type.
static bool handleIntegerToComplexFloatConversion(Sema &S, ExprResult &IntExpr,
                                                  ExprResult &ComplexExpr,
                                                  QualType IntTy,
                                                  QualType ComplexTy,
                                                  bool SkipCast) {
  if (IntTy->isComplexType() || IntTy->isRealFloatingType()) return true;
  if (SkipCast) return false;
  if (IntTy->isIntegerType()) {
    QualType fpTy = cast<ComplexType>(ComplexTy)->getElementType();
    IntExpr = S.ImpCastExprToType(IntExpr.get(), fpTy, CK_IntegralToFloating);
    IntExpr = S.ImpCastExprToType(IntExpr.get(), ComplexTy,
                                  CK_FloatingRealToComplex);
  } else {
    assert(IntTy->isComplexIntegerType());
    IntExpr = S.ImpCastExprToType(IntExpr.get(), ComplexTy,
                                  CK_IntegralComplexToFloatingComplex);
  }
  return false;
}

/// \brief Handle arithmetic conversion with complex types.  Helper function of
/// UsualArithmeticConversions()
static QualType handleComplexFloatConversion(Sema &S, ExprResult &LHS,
                                             ExprResult &RHS, QualType LHSType,
                                             QualType RHSType,
                                             bool IsCompAssign) {
  // if we have an integer operand, the result is the complex type.
  if (!handleIntegerToComplexFloatConversion(S, RHS, LHS, RHSType, LHSType,
                                             /*skipCast*/false))
    return LHSType;
  if (!handleIntegerToComplexFloatConversion(S, LHS, RHS, LHSType, RHSType,
                                             /*skipCast*/IsCompAssign))
    return RHSType;

  // This handles complex/complex, complex/float, or float/complex.
  // When both operands are complex, the shorter operand is converted to the
  // type of the longer, and that is the type of the result. This corresponds
  // to what is done when combining two real floating-point operands.
  // The fun begins when size promotion occur across type domains.
  // From H&S 6.3.4: When one operand is complex and the other is a real
  // floating-point type, the less precise type is converted, within it's
  // real or complex domain, to the precision of the other type. For example,
  // when combining a "long double" with a "double _Complex", the
  // "double _Complex" is promoted to "long double _Complex".

  // Compute the rank of the two types, regardless of whether they are complex.
  int Order = S.Context.getFloatingTypeOrder(LHSType, RHSType);

  auto *LHSComplexType = dyn_cast<ComplexType>(LHSType);
  auto *RHSComplexType = dyn_cast<ComplexType>(RHSType);
  QualType LHSElementType =
      LHSComplexType ? LHSComplexType->getElementType() : LHSType;
  QualType RHSElementType =
      RHSComplexType ? RHSComplexType->getElementType() : RHSType;

  QualType ResultType = S.Context.getComplexType(LHSElementType);
  if (Order < 0) {
    // Promote the precision of the LHS if not an assignment.
    ResultType = S.Context.getComplexType(RHSElementType);
    if (!IsCompAssign) {
      if (LHSComplexType)
        LHS =
            S.ImpCastExprToType(LHS.get(), ResultType, CK_FloatingComplexCast);
      else
        LHS = S.ImpCastExprToType(LHS.get(), RHSElementType, CK_FloatingCast);
    }
  } else if (Order > 0) {
    // Promote the precision of the RHS.
    if (RHSComplexType)
      RHS = S.ImpCastExprToType(RHS.get(), ResultType, CK_FloatingComplexCast);
    else
      RHS = S.ImpCastExprToType(RHS.get(), LHSElementType, CK_FloatingCast);
  }
  return ResultType;
}

/// \brief Hande arithmetic conversion from integer to float.  Helper function
/// of UsualArithmeticConversions()
static QualType handleIntToFloatConversion(Sema &S, ExprResult &FloatExpr,
                                           ExprResult &IntExpr,
                                           QualType FloatTy, QualType IntTy,
                                           bool ConvertFloat, bool ConvertInt) {
  if (IntTy->isIntegerType()) {
    if (ConvertInt)
      // Convert intExpr to the lhs floating point type.
      IntExpr = S.ImpCastExprToType(IntExpr.get(), FloatTy,
                                    CK_IntegralToFloating);
    return FloatTy;
  }
     
  // Convert both sides to the appropriate complex float.
  assert(IntTy->isComplexIntegerType());
  QualType result = S.Context.getComplexType(FloatTy);

  // _Complex int -> _Complex float
  if (ConvertInt)
    IntExpr = S.ImpCastExprToType(IntExpr.get(), result,
                                  CK_IntegralComplexToFloatingComplex);

  // float -> _Complex float
  if (ConvertFloat)
    FloatExpr = S.ImpCastExprToType(FloatExpr.get(), result,
                                    CK_FloatingRealToComplex);

  return result;
}

/// \brief Handle arithmethic conversion with floating point types.  Helper
/// function of UsualArithmeticConversions()
static QualType handleFloatConversion(Sema &S, ExprResult &LHS,
                                      ExprResult &RHS, QualType LHSType,
                                      QualType RHSType, bool IsCompAssign) {
  bool LHSFloat = LHSType->isRealFloatingType();
  bool RHSFloat = RHSType->isRealFloatingType();

  // If we have two real floating types, convert the smaller operand
  // to the bigger result.
  if (LHSFloat && RHSFloat) {
    int order = S.Context.getFloatingTypeOrder(LHSType, RHSType);
    if (order > 0) {
      RHS = S.ImpCastExprToType(RHS.get(), LHSType, CK_FloatingCast);
      return LHSType;
    }

    assert(order < 0 && "illegal float comparison");
    if (!IsCompAssign)
      LHS = S.ImpCastExprToType(LHS.get(), RHSType, CK_FloatingCast);
    return RHSType;
  }

  if (LHSFloat) {
    // Half FP has to be promoted to float unless it is natively supported
    if (LHSType->isHalfType() && !S.getLangOpts().NativeHalfType)
      LHSType = S.Context.FloatTy;

    return handleIntToFloatConversion(S, LHS, RHS, LHSType, RHSType,
                                      /*convertFloat=*/!IsCompAssign,
                                      /*convertInt=*/ true);
  }
  assert(RHSFloat);
  return handleIntToFloatConversion(S, RHS, LHS, RHSType, LHSType,
                                    /*convertInt=*/ true,
                                    /*convertFloat=*/!IsCompAssign);
}

typedef ExprResult PerformCastFn(Sema &S, Expr *operand, QualType toType);

namespace {
/// These helper callbacks are placed in an anonymous namespace to
/// permit their use as function template parameters.
ExprResult doIntegralCast(Sema &S, Expr *op, QualType toType) {
  return S.ImpCastExprToType(op, toType, CK_IntegralCast);
}

ExprResult doComplexIntegralCast(Sema &S, Expr *op, QualType toType) {
  return S.ImpCastExprToType(op, S.Context.getComplexType(toType),
                             CK_IntegralComplexCast);
}
}

/// \brief Handle integer arithmetic conversions.  Helper function of
/// UsualArithmeticConversions()
template <PerformCastFn doLHSCast, PerformCastFn doRHSCast>
static QualType handleIntegerConversion(Sema &S, ExprResult &LHS,
                                        ExprResult &RHS, QualType LHSType,
                                        QualType RHSType, bool IsCompAssign) {
  // The rules for this case are in C99 6.3.1.8
  int order = S.Context.getIntegerTypeOrder(LHSType, RHSType);
  bool LHSSigned = LHSType->hasSignedIntegerRepresentation();
  bool RHSSigned = RHSType->hasSignedIntegerRepresentation();
  if (LHSSigned == RHSSigned) {
    // Same signedness; use the higher-ranked type
    if (order >= 0) {
      RHS = (*doRHSCast)(S, RHS.get(), LHSType);
      return LHSType;
    } else if (!IsCompAssign)
      LHS = (*doLHSCast)(S, LHS.get(), RHSType);
    return RHSType;
  } else if (order != (LHSSigned ? 1 : -1)) {
    // The unsigned type has greater than or equal rank to the
    // signed type, so use the unsigned type
    if (RHSSigned) {
      RHS = (*doRHSCast)(S, RHS.get(), LHSType);
      return LHSType;
    } else if (!IsCompAssign)
      LHS = (*doLHSCast)(S, LHS.get(), RHSType);
    return RHSType;
  } else if (S.Context.getIntWidth(LHSType) != S.Context.getIntWidth(RHSType)) {
    // The two types are different widths; if we are here, that
    // means the signed type is larger than the unsigned type, so
    // use the signed type.
    if (LHSSigned) {
      RHS = (*doRHSCast)(S, RHS.get(), LHSType);
      return LHSType;
    } else if (!IsCompAssign)
      LHS = (*doLHSCast)(S, LHS.get(), RHSType);
    return RHSType;
  } else {
    // The signed type is higher-ranked than the unsigned type,
    // but isn't actually any bigger (like unsigned int and long
    // on most 32-bit systems).  Use the unsigned type corresponding
    // to the signed type.
    QualType result =
      S.Context.getCorrespondingUnsignedType(LHSSigned ? LHSType : RHSType);
    RHS = (*doRHSCast)(S, RHS.get(), result);
    if (!IsCompAssign)
      LHS = (*doLHSCast)(S, LHS.get(), result);
    return result;
  }
}

/// \brief Handle conversions with GCC complex int extension.  Helper function
/// of UsualArithmeticConversions()
static QualType handleComplexIntConversion(Sema &S, ExprResult &LHS,
                                           ExprResult &RHS, QualType LHSType,
                                           QualType RHSType,
                                           bool IsCompAssign) {
  const ComplexType *LHSComplexInt = LHSType->getAsComplexIntegerType();
  const ComplexType *RHSComplexInt = RHSType->getAsComplexIntegerType();

  if (LHSComplexInt && RHSComplexInt) {
    QualType LHSEltType = LHSComplexInt->getElementType();
    QualType RHSEltType = RHSComplexInt->getElementType();
    QualType ScalarType =
      handleIntegerConversion<doComplexIntegralCast, doComplexIntegralCast>
        (S, LHS, RHS, LHSEltType, RHSEltType, IsCompAssign);

    return S.Context.getComplexType(ScalarType);
  }

  if (LHSComplexInt) {
    QualType LHSEltType = LHSComplexInt->getElementType();
    QualType ScalarType =
      handleIntegerConversion<doComplexIntegralCast, doIntegralCast>
        (S, LHS, RHS, LHSEltType, RHSType, IsCompAssign);
    QualType ComplexType = S.Context.getComplexType(ScalarType);
    RHS = S.ImpCastExprToType(RHS.get(), ComplexType,
                              CK_IntegralRealToComplex);
 
    return ComplexType;
  }

  assert(RHSComplexInt);

  QualType RHSEltType = RHSComplexInt->getElementType();
  QualType ScalarType =
    handleIntegerConversion<doIntegralCast, doComplexIntegralCast>
      (S, LHS, RHS, LHSType, RHSEltType, IsCompAssign);
  QualType ComplexType = S.Context.getComplexType(ScalarType);
  
  if (!IsCompAssign)
    LHS = S.ImpCastExprToType(LHS.get(), ComplexType,
                              CK_IntegralRealToComplex);
  return ComplexType;
}

/// UsualArithmeticConversions - Performs various conversions that are common to
/// binary operators (C99 6.3.1.8). If both operands aren't arithmetic, this
/// routine returns the first non-arithmetic type found. The client is
/// responsible for emitting appropriate error diagnostics.
QualType Sema::UsualArithmeticConversions(ExprResult &LHS, ExprResult &RHS,
                                          bool IsCompAssign) {
  if (!IsCompAssign) {
    LHS = UsualUnaryConversions(LHS.get());
    if (LHS.isInvalid())
      return QualType();
  }

  RHS = UsualUnaryConversions(RHS.get());
  if (RHS.isInvalid())
    return QualType();

  // For conversion purposes, we ignore any qualifiers.
  // For example, "const float" and "float" are equivalent.
  QualType LHSType =
    Context.getCanonicalType(LHS.get()->getType()).getUnqualifiedType();
  QualType RHSType =
    Context.getCanonicalType(RHS.get()->getType()).getUnqualifiedType();

  // For conversion purposes, we ignore any atomic qualifier on the LHS.
  if (const AtomicType *AtomicLHS = LHSType->getAs<AtomicType>())
    LHSType = AtomicLHS->getValueType();

  // If both types are identical, no conversion is needed.
  if (LHSType == RHSType)
    return LHSType;

  // If either side is a non-arithmetic type (e.g. a pointer), we are done.
  // The caller can deal with this (e.g. pointer + int).
  if (!LHSType->isArithmeticType() || !RHSType->isArithmeticType())
    return QualType();

  // Apply unary and bitfield promotions to the LHS's type.
  QualType LHSUnpromotedType = LHSType;
  if (LHSType->isPromotableIntegerType())
    LHSType = Context.getPromotedIntegerType(LHSType);
  QualType LHSBitfieldPromoteTy = Context.isPromotableBitField(LHS.get());
  if (!LHSBitfieldPromoteTy.isNull())
    LHSType = LHSBitfieldPromoteTy;
  if (LHSType != LHSUnpromotedType && !IsCompAssign)
    LHS = ImpCastExprToType(LHS.get(), LHSType, CK_IntegralCast);

  // If both types are identical, no conversion is needed.
  if (LHSType == RHSType)
    return LHSType;

  // At this point, we have two different arithmetic types.

  // Handle complex types first (C99 6.3.1.8p1).
  if (LHSType->isComplexType() || RHSType->isComplexType())
    return handleComplexFloatConversion(*this, LHS, RHS, LHSType, RHSType,
                                        IsCompAssign);

  // Now handle "real" floating types (i.e. float, double, long double).
  if (LHSType->isRealFloatingType() || RHSType->isRealFloatingType())
    return handleFloatConversion(*this, LHS, RHS, LHSType, RHSType,
                                 IsCompAssign);

  // Handle GCC complex int extension.
  if (LHSType->isComplexIntegerType() || RHSType->isComplexIntegerType())
    return handleComplexIntConversion(*this, LHS, RHS, LHSType, RHSType,
                                      IsCompAssign);

  // Finally, we have two differing integer types.
  return handleIntegerConversion<doIntegralCast, doIntegralCast>
           (*this, LHS, RHS, LHSType, RHSType, IsCompAssign);
}


//===----------------------------------------------------------------------===//
//  Semantic Analysis for various Expression Types
//===----------------------------------------------------------------------===//


ExprResult
Sema::ActOnGenericSelectionExpr(SourceLocation KeyLoc,
                                SourceLocation DefaultLoc,
                                SourceLocation RParenLoc,
                                Expr *ControllingExpr,
                                ArrayRef<ParsedType> ArgTypes,
                                ArrayRef<Expr *> ArgExprs) {
  unsigned NumAssocs = ArgTypes.size();
  assert(NumAssocs == ArgExprs.size());

  TypeSourceInfo **Types = new TypeSourceInfo*[NumAssocs];
  for (unsigned i = 0; i < NumAssocs; ++i) {
    if (ArgTypes[i])
      (void) GetTypeFromParser(ArgTypes[i], &Types[i]);
    else
      Types[i] = nullptr;
  }

  ExprResult ER = CreateGenericSelectionExpr(KeyLoc, DefaultLoc, RParenLoc,
                                             ControllingExpr,
                                             llvm::makeArrayRef(Types, NumAssocs),
                                             ArgExprs);
  delete [] Types;
  return ER;
}

ExprResult
Sema::CreateGenericSelectionExpr(SourceLocation KeyLoc,
                                 SourceLocation DefaultLoc,
                                 SourceLocation RParenLoc,
                                 Expr *ControllingExpr,
                                 ArrayRef<TypeSourceInfo *> Types,
                                 ArrayRef<Expr *> Exprs) {
  unsigned NumAssocs = Types.size();
  assert(NumAssocs == Exprs.size());

  // Decay and strip qualifiers for the controlling expression type, and handle
  // placeholder type replacement. See committee discussion from WG14 DR423.
  ExprResult R = DefaultFunctionArrayLvalueConversion(ControllingExpr);
  if (R.isInvalid())
    return ExprError();
  ControllingExpr = R.get();

  // The controlling expression is an unevaluated operand, so side effects are
  // likely unintended.
  if (ActiveTemplateInstantiations.empty() &&
      ControllingExpr->HasSideEffects(Context, false))
    Diag(ControllingExpr->getExprLoc(),
         diag::warn_side_effects_unevaluated_context);

  bool TypeErrorFound = false,
       IsResultDependent = ControllingExpr->isTypeDependent(),
       ContainsUnexpandedParameterPack
         = ControllingExpr->containsUnexpandedParameterPack();

  for (unsigned i = 0; i < NumAssocs; ++i) {
    if (Exprs[i]->containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;

    if (Types[i]) {
      if (Types[i]->getType()->containsUnexpandedParameterPack())
        ContainsUnexpandedParameterPack = true;

      if (Types[i]->getType()->isDependentType()) {
        IsResultDependent = true;
      } else {
        // C11 6.5.1.1p2 "The type name in a generic association shall specify a
        // complete object type other than a variably modified type."
        unsigned D = 0;
        if (Types[i]->getType()->isIncompleteType())
          D = diag::err_assoc_type_incomplete;
        else if (!Types[i]->getType()->isObjectType())
          D = diag::err_assoc_type_nonobject;
        else if (Types[i]->getType()->isVariablyModifiedType())
          D = diag::err_assoc_type_variably_modified;

        if (D != 0) {
          Diag(Types[i]->getTypeLoc().getBeginLoc(), D)
            << Types[i]->getTypeLoc().getSourceRange()
            << Types[i]->getType();
          TypeErrorFound = true;
        }

        // C11 6.5.1.1p2 "No two generic associations in the same generic
        // selection shall specify compatible types."
        for (unsigned j = i+1; j < NumAssocs; ++j)
          if (Types[j] && !Types[j]->getType()->isDependentType() &&
              Context.typesAreCompatible(Types[i]->getType(),
                                         Types[j]->getType())) {
            Diag(Types[j]->getTypeLoc().getBeginLoc(),
                 diag::err_assoc_compatible_types)
              << Types[j]->getTypeLoc().getSourceRange()
              << Types[j]->getType()
              << Types[i]->getType();
            Diag(Types[i]->getTypeLoc().getBeginLoc(),
                 diag::note_compat_assoc)
              << Types[i]->getTypeLoc().getSourceRange()
              << Types[i]->getType();
            TypeErrorFound = true;
          }
      }
    }
  }
  if (TypeErrorFound)
    return ExprError();

  // If we determined that the generic selection is result-dependent, don't
  // try to compute the result expression.
  if (IsResultDependent)
    return new (Context) GenericSelectionExpr(
        Context, KeyLoc, ControllingExpr, Types, Exprs, DefaultLoc, RParenLoc,
        ContainsUnexpandedParameterPack);

  SmallVector<unsigned, 1> CompatIndices;
  unsigned DefaultIndex = -1U;
  for (unsigned i = 0; i < NumAssocs; ++i) {
    if (!Types[i])
      DefaultIndex = i;
    else if (Context.typesAreCompatible(ControllingExpr->getType(),
                                        Types[i]->getType()))
      CompatIndices.push_back(i);
  }

  // C11 6.5.1.1p2 "The controlling expression of a generic selection shall have
  // type compatible with at most one of the types named in its generic
  // association list."
  if (CompatIndices.size() > 1) {
    // We strip parens here because the controlling expression is typically
    // parenthesized in macro definitions.
    ControllingExpr = ControllingExpr->IgnoreParens();
    Diag(ControllingExpr->getLocStart(), diag::err_generic_sel_multi_match)
      << ControllingExpr->getSourceRange() << ControllingExpr->getType()
      << (unsigned) CompatIndices.size();
    for (unsigned I : CompatIndices) {
      Diag(Types[I]->getTypeLoc().getBeginLoc(),
           diag::note_compat_assoc)
        << Types[I]->getTypeLoc().getSourceRange()
        << Types[I]->getType();
    }
    return ExprError();
  }

  // C11 6.5.1.1p2 "If a generic selection has no default generic association,
  // its controlling expression shall have type compatible with exactly one of
  // the types named in its generic association list."
  if (DefaultIndex == -1U && CompatIndices.size() == 0) {
    // We strip parens here because the controlling expression is typically
    // parenthesized in macro definitions.
    ControllingExpr = ControllingExpr->IgnoreParens();
    Diag(ControllingExpr->getLocStart(), diag::err_generic_sel_no_match)
      << ControllingExpr->getSourceRange() << ControllingExpr->getType();
    return ExprError();
  }

  // C11 6.5.1.1p3 "If a generic selection has a generic association with a
  // type name that is compatible with the type of the controlling expression,
  // then the result expression of the generic selection is the expression
  // in that generic association. Otherwise, the result expression of the
  // generic selection is the expression in the default generic association."
  unsigned ResultIndex =
    CompatIndices.size() ? CompatIndices[0] : DefaultIndex;

  return new (Context) GenericSelectionExpr(
      Context, KeyLoc, ControllingExpr, Types, Exprs, DefaultLoc, RParenLoc,
      ContainsUnexpandedParameterPack, ResultIndex);
}

/// getUDSuffixLoc - Create a SourceLocation for a ud-suffix, given the
/// location of the token and the offset of the ud-suffix within it.
static SourceLocation getUDSuffixLoc(Sema &S, SourceLocation TokLoc,
                                     unsigned Offset) {
  return Lexer::AdvanceToTokenCharacter(TokLoc, Offset, S.getSourceManager(),
                                        S.getLangOpts());
}

/// BuildCookedLiteralOperatorCall - A user-defined literal was found. Look up
/// the corresponding cooked (non-raw) literal operator, and build a call to it.
static ExprResult BuildCookedLiteralOperatorCall(Sema &S, Scope *Scope,
                                                 IdentifierInfo *UDSuffix,
                                                 SourceLocation UDSuffixLoc,
                                                 ArrayRef<Expr*> Args,
                                                 SourceLocation LitEndLoc) {
  assert(Args.size() <= 2 && "too many arguments for literal operator");

  QualType ArgTy[2];
  for (unsigned ArgIdx = 0; ArgIdx != Args.size(); ++ArgIdx) {
    ArgTy[ArgIdx] = Args[ArgIdx]->getType();
    if (ArgTy[ArgIdx]->isArrayType())
      ArgTy[ArgIdx] = S.Context.getArrayDecayedType(ArgTy[ArgIdx]);
  }

  DeclarationName OpName =
    S.Context.DeclarationNames.getCXXLiteralOperatorName(UDSuffix);
  DeclarationNameInfo OpNameInfo(OpName, UDSuffixLoc);
  OpNameInfo.setCXXLiteralOperatorNameLoc(UDSuffixLoc);

  LookupResult R(S, OpName, UDSuffixLoc, Sema::LookupOrdinaryName);
  if (S.LookupLiteralOperator(Scope, R, llvm::makeArrayRef(ArgTy, Args.size()),
                              /*AllowRaw*/false, /*AllowTemplate*/false,
                              /*AllowStringTemplate*/false) == Sema::LOLR_Error)
    return ExprError();

  return S.BuildLiteralOperatorCall(R, OpNameInfo, Args, LitEndLoc);
}

/// ActOnStringLiteral - The specified tokens were lexed as pasted string
/// fragments (e.g. "foo" "bar" L"baz").  The result string has to handle string
/// concatenation ([C99 5.1.1.2, translation phase #6]), so it may come from
/// multiple tokens.  However, the common case is that StringToks points to one
/// string.
///
ExprResult
Sema::ActOnStringLiteral(ArrayRef<Token> StringToks, Scope *UDLScope) {
  assert(!StringToks.empty() && "Must have at least one string!");

  StringLiteralParser Literal(StringToks, PP);
  if (Literal.hadError)
    return ExprError();

  SmallVector<SourceLocation, 4> StringTokLocs;
  for (const Token &Tok : StringToks)
    StringTokLocs.push_back(Tok.getLocation());

  QualType CharTy = Context.CharTy;
  StringLiteral::StringKind Kind = StringLiteral::Ascii;
  if (Literal.isWide()) {
    CharTy = Context.getWideCharType();
    Kind = StringLiteral::Wide;
  } else if (Literal.isUTF8()) {
    Kind = StringLiteral::UTF8;
  } else if (Literal.isUTF16()) {
    CharTy = Context.Char16Ty;
    Kind = StringLiteral::UTF16;
  } else if (Literal.isUTF32()) {
    CharTy = Context.Char32Ty;
    Kind = StringLiteral::UTF32;
  } else if (Literal.isPascal()) {
    CharTy = Context.UnsignedCharTy;
  }

  QualType CharTyConst = CharTy;
  // A C++ string literal has a const-qualified element type (C++ 2.13.4p1).
  if (getLangOpts().CPlusPlus || getLangOpts().ConstStrings)
    CharTyConst.addConst();

  // Get an array type for the string, according to C99 6.4.5.  This includes
  // the nul terminator character as well as the string length for pascal
  // strings.
  QualType StrTy = Context.getConstantArrayType(CharTyConst,
                                 llvm::APInt(32, Literal.GetNumStringChars()+1),
                                 ArrayType::Normal, 0);

  // OpenCL v1.1 s6.5.3: a string literal is in the constant address space.
  if (getLangOpts().OpenCL) {
    StrTy = Context.getAddrSpaceQualType(StrTy, LangAS::opencl_constant);
  }

  // Pass &StringTokLocs[0], StringTokLocs.size() to factory!
  StringLiteral *Lit = StringLiteral::Create(Context, Literal.GetString(),
                                             Kind, Literal.Pascal, StrTy,
                                             &StringTokLocs[0],
                                             StringTokLocs.size());
  if (Literal.getUDSuffix().empty())
    return Lit;

  // We're building a user-defined literal.
  IdentifierInfo *UDSuffix = &Context.Idents.get(Literal.getUDSuffix());
  SourceLocation UDSuffixLoc =
    getUDSuffixLoc(*this, StringTokLocs[Literal.getUDSuffixToken()],
                   Literal.getUDSuffixOffset());

  // Make sure we're allowed user-defined literals here.
  if (!UDLScope)
    return ExprError(Diag(UDSuffixLoc, diag::err_invalid_string_udl));

  // C++11 [lex.ext]p5: The literal L is treated as a call of the form
  //   operator "" X (str, len)
  QualType SizeType = Context.getSizeType();

  DeclarationName OpName =
    Context.DeclarationNames.getCXXLiteralOperatorName(UDSuffix);
  DeclarationNameInfo OpNameInfo(OpName, UDSuffixLoc);
  OpNameInfo.setCXXLiteralOperatorNameLoc(UDSuffixLoc);

  QualType ArgTy[] = {
    Context.getArrayDecayedType(StrTy), SizeType
  };

  LookupResult R(*this, OpName, UDSuffixLoc, LookupOrdinaryName);
  switch (LookupLiteralOperator(UDLScope, R, ArgTy,
                                /*AllowRaw*/false, /*AllowTemplate*/false,
                                /*AllowStringTemplate*/true)) {

  case LOLR_Cooked: {
    llvm::APInt Len(Context.getIntWidth(SizeType), Literal.GetNumStringChars());
    IntegerLiteral *LenArg = IntegerLiteral::Create(Context, Len, SizeType,
                                                    StringTokLocs[0]);
    Expr *Args[] = { Lit, LenArg };

    return BuildLiteralOperatorCall(R, OpNameInfo, Args, StringTokLocs.back());
  }

  case LOLR_StringTemplate: {
    TemplateArgumentListInfo ExplicitArgs;

    unsigned CharBits = Context.getIntWidth(CharTy);
    bool CharIsUnsigned = CharTy->isUnsignedIntegerType();
    llvm::APSInt Value(CharBits, CharIsUnsigned);

    TemplateArgument TypeArg(CharTy);
    TemplateArgumentLocInfo TypeArgInfo(Context.getTrivialTypeSourceInfo(CharTy));
    ExplicitArgs.addArgument(TemplateArgumentLoc(TypeArg, TypeArgInfo));

    for (unsigned I = 0, N = Lit->getLength(); I != N; ++I) {
      Value = Lit->getCodeUnit(I);
      TemplateArgument Arg(Context, Value, CharTy);
      TemplateArgumentLocInfo ArgInfo;
      ExplicitArgs.addArgument(TemplateArgumentLoc(Arg, ArgInfo));
    }
    return BuildLiteralOperatorCall(R, OpNameInfo, None, StringTokLocs.back(),
                                    &ExplicitArgs);
  }
  case LOLR_Raw:
  case LOLR_Template:
    llvm_unreachable("unexpected literal operator lookup result");
  case LOLR_Error:
    return ExprError();
  }
  llvm_unreachable("unexpected literal operator lookup result");
}

ExprResult
Sema::BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
                       SourceLocation Loc,
                       const CXXScopeSpec *SS) {
  DeclarationNameInfo NameInfo(D->getDeclName(), Loc);
  return BuildDeclRefExpr(D, Ty, VK, NameInfo, SS);
}

/// BuildDeclRefExpr - Build an expression that references a
/// declaration that does not require a closure capture.
ExprResult
Sema::BuildDeclRefExpr(ValueDecl *D, QualType Ty, ExprValueKind VK,
                       const DeclarationNameInfo &NameInfo,
                       const CXXScopeSpec *SS, NamedDecl *FoundD,
                       const TemplateArgumentListInfo *TemplateArgs) {
  if (getLangOpts().CUDA)
    if (const FunctionDecl *Caller = dyn_cast<FunctionDecl>(CurContext))
      if (const FunctionDecl *Callee = dyn_cast<FunctionDecl>(D)) {
        if (CheckCUDATarget(Caller, Callee)) {
          Diag(NameInfo.getLoc(), diag::err_ref_bad_target)
            << IdentifyCUDATarget(Callee) << D->getIdentifier()
            << IdentifyCUDATarget(Caller);
          Diag(D->getLocation(), diag::note_previous_decl)
            << D->getIdentifier();
          return ExprError();
        }
      }

  bool RefersToCapturedVariable =
      isa<VarDecl>(D) &&
      NeedToCaptureVariable(cast<VarDecl>(D), NameInfo.getLoc());

  DeclRefExpr *E;
  if (isa<VarTemplateSpecializationDecl>(D)) {
    VarTemplateSpecializationDecl *VarSpec =
        cast<VarTemplateSpecializationDecl>(D);

    E = DeclRefExpr::Create(Context, SS ? SS->getWithLocInContext(Context)
                                        : NestedNameSpecifierLoc(),
                            VarSpec->getTemplateKeywordLoc(), D,
                            RefersToCapturedVariable, NameInfo.getLoc(), Ty, VK,
                            FoundD, TemplateArgs);
  } else {
    assert(!TemplateArgs && "No template arguments for non-variable"
                            " template specialization references");
    E = DeclRefExpr::Create(Context, SS ? SS->getWithLocInContext(Context)
                                        : NestedNameSpecifierLoc(),
                            SourceLocation(), D, RefersToCapturedVariable,
                            NameInfo, Ty, VK, FoundD);
  }

  MarkDeclRefReferenced(E);

  if (getLangOpts().ObjCWeak && isa<VarDecl>(D) &&
      Ty.getObjCLifetime() == Qualifiers::OCL_Weak &&
      !Diags.isIgnored(diag::warn_arc_repeated_use_of_weak, E->getLocStart()))
      recordUseOfEvaluatedWeak(E);

  // Just in case we're building an illegal pointer-to-member.
  FieldDecl *FD = dyn_cast<FieldDecl>(D);
  if (FD && FD->isBitField())
    E->setObjectKind(OK_BitField);

  return E;
}

/// Decomposes the given name into a DeclarationNameInfo, its location, and
/// possibly a list of template arguments.
///
/// If this produces template arguments, it is permitted to call
/// DecomposeTemplateName.
///
/// This actually loses a lot of source location information for
/// non-standard name kinds; we should consider preserving that in
/// some way.
void
Sema::DecomposeUnqualifiedId(const UnqualifiedId &Id,
                             TemplateArgumentListInfo &Buffer,
                             DeclarationNameInfo &NameInfo,
                             const TemplateArgumentListInfo *&TemplateArgs) {
  if (Id.getKind() == UnqualifiedId::IK_TemplateId) {
    Buffer.setLAngleLoc(Id.TemplateId->LAngleLoc);
    Buffer.setRAngleLoc(Id.TemplateId->RAngleLoc);

    ASTTemplateArgsPtr TemplateArgsPtr(Id.TemplateId->getTemplateArgs(),
                                       Id.TemplateId->NumArgs);
    translateTemplateArguments(TemplateArgsPtr, Buffer);

    TemplateName TName = Id.TemplateId->Template.get();
    SourceLocation TNameLoc = Id.TemplateId->TemplateNameLoc;
    NameInfo = Context.getNameForTemplate(TName, TNameLoc);
    TemplateArgs = &Buffer;
  } else {
    NameInfo = GetNameFromUnqualifiedId(Id);
    TemplateArgs = nullptr;
  }
}

static void emitEmptyLookupTypoDiagnostic(
    const TypoCorrection &TC, Sema &SemaRef, const CXXScopeSpec &SS,
    DeclarationName Typo, SourceLocation TypoLoc, ArrayRef<Expr *> Args,
    unsigned DiagnosticID, unsigned DiagnosticSuggestID) {
  DeclContext *Ctx =
      SS.isEmpty() ? nullptr : SemaRef.computeDeclContext(SS, false);
  if (!TC) {
    // Emit a special diagnostic for failed member lookups.
    // FIXME: computing the declaration context might fail here (?)
    if (Ctx)
      SemaRef.Diag(TypoLoc, diag::err_no_member) << Typo << Ctx
                                                 << SS.getRange();
    else
      SemaRef.Diag(TypoLoc, DiagnosticID) << Typo;
    return;
  }

  std::string CorrectedStr = TC.getAsString(SemaRef.getLangOpts());
  bool DroppedSpecifier =
      TC.WillReplaceSpecifier() && Typo.getAsString() == CorrectedStr;
  unsigned NoteID = TC.getCorrectionDeclAs<ImplicitParamDecl>()
                        ? diag::note_implicit_param_decl
                        : diag::note_previous_decl;
  if (!Ctx)
    SemaRef.diagnoseTypo(TC, SemaRef.PDiag(DiagnosticSuggestID) << Typo,
                         SemaRef.PDiag(NoteID));
  else
    SemaRef.diagnoseTypo(TC, SemaRef.PDiag(diag::err_no_member_suggest)
                                 << Typo << Ctx << DroppedSpecifier
                                 << SS.getRange(),
                         SemaRef.PDiag(NoteID));
}

/// Diagnose an empty lookup.
///
/// \return false if new lookup candidates were found
bool
Sema::DiagnoseEmptyLookup(Scope *S, CXXScopeSpec &SS, LookupResult &R,
                          std::unique_ptr<CorrectionCandidateCallback> CCC,
                          TemplateArgumentListInfo *ExplicitTemplateArgs,
                          ArrayRef<Expr *> Args, TypoExpr **Out) {
  DeclarationName Name = R.getLookupName();

  unsigned diagnostic = diag::err_undeclared_var_use;
  unsigned diagnostic_suggest = diag::err_undeclared_var_use_suggest;
  if (Name.getNameKind() == DeclarationName::CXXOperatorName ||
      Name.getNameKind() == DeclarationName::CXXLiteralOperatorName ||
      Name.getNameKind() == DeclarationName::CXXConversionFunctionName) {
    diagnostic = diag::err_undeclared_use;
    diagnostic_suggest = diag::err_undeclared_use_suggest;
  }

  // If the original lookup was an unqualified lookup, fake an
  // unqualified lookup.  This is useful when (for example) the
  // original lookup would not have found something because it was a
  // dependent name.
  DeclContext *DC = SS.isEmpty() ? CurContext : nullptr;
  while (DC) {
    if (isa<CXXRecordDecl>(DC)) {
      LookupQualifiedName(R, DC);

      if (!R.empty()) {
        // Don't give errors about ambiguities in this lookup.
        R.suppressDiagnostics();

        // During a default argument instantiation the CurContext points
        // to a CXXMethodDecl; but we can't apply a this-> fixit inside a
        // function parameter list, hence add an explicit check.
        bool isDefaultArgument = !ActiveTemplateInstantiations.empty() &&
                              ActiveTemplateInstantiations.back().Kind ==
            ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation;
        CXXMethodDecl *CurMethod = dyn_cast<CXXMethodDecl>(CurContext);
        bool isInstance = CurMethod &&
                          CurMethod->isInstance() &&
                          DC == CurMethod->getParent() && !isDefaultArgument;

        // Give a code modification hint to insert 'this->'.
        // TODO: fixit for inserting 'Base<T>::' in the other cases.
        // Actually quite difficult!
        if (getLangOpts().MSVCCompat)
          diagnostic = diag::ext_found_via_dependent_bases_lookup;
        if (isInstance) {
          Diag(R.getNameLoc(), diagnostic) << Name
            << FixItHint::CreateInsertion(R.getNameLoc(), "this->");
          CheckCXXThisCapture(R.getNameLoc());
        } else {
          Diag(R.getNameLoc(), diagnostic) << Name;
        }

        // Do we really want to note all of these?
        for (NamedDecl *D : R)
          Diag(D->getLocation(), diag::note_dependent_var_use);

        // Return true if we are inside a default argument instantiation
        // and the found name refers to an instance member function, otherwise
        // the function calling DiagnoseEmptyLookup will try to create an
        // implicit member call and this is wrong for default argument.
        if (isDefaultArgument && ((*R.begin())->isCXXInstanceMember())) {
          Diag(R.getNameLoc(), diag::err_member_call_without_object);
          return true;
        }

        // Tell the callee to try to recover.
        return false;
      }

      R.clear();
    }

    // In Microsoft mode, if we are performing lookup from within a friend
    // function definition declared at class scope then we must set
    // DC to the lexical parent to be able to search into the parent
    // class.
    if (getLangOpts().MSVCCompat && isa<FunctionDecl>(DC) &&
        cast<FunctionDecl>(DC)->getFriendObjectKind() &&
        DC->getLexicalParent()->isRecord())
      DC = DC->getLexicalParent();
    else
      DC = DC->getParent();
  }

  // We didn't find anything, so try to correct for a typo.
  TypoCorrection Corrected;
  if (S && Out) {
    SourceLocation TypoLoc = R.getNameLoc();
    assert(!ExplicitTemplateArgs &&
           "Diagnosing an empty lookup with explicit template args!");
    *Out = CorrectTypoDelayed(
        R.getLookupNameInfo(), R.getLookupKind(), S, &SS, std::move(CCC),
        [=](const TypoCorrection &TC) {
          emitEmptyLookupTypoDiagnostic(TC, *this, SS, Name, TypoLoc, Args,
                                        diagnostic, diagnostic_suggest);
        },
        nullptr, CTK_ErrorRecovery);
    if (*Out)
      return true;
  } else if (S && (Corrected =
                       CorrectTypo(R.getLookupNameInfo(), R.getLookupKind(), S,
                                   &SS, std::move(CCC), CTK_ErrorRecovery))) {
    std::string CorrectedStr(Corrected.getAsString(getLangOpts()));
    bool DroppedSpecifier =
        Corrected.WillReplaceSpecifier() && Name.getAsString() == CorrectedStr;
    R.setLookupName(Corrected.getCorrection());

    bool AcceptableWithRecovery = false;
    bool AcceptableWithoutRecovery = false;
    NamedDecl *ND = Corrected.getFoundDecl();
    if (ND) {
      if (Corrected.isOverloaded()) {
        OverloadCandidateSet OCS(R.getNameLoc(),
                                 OverloadCandidateSet::CSK_Normal);
        OverloadCandidateSet::iterator Best;
        for (NamedDecl *CD : Corrected) {
          if (FunctionTemplateDecl *FTD =
                   dyn_cast<FunctionTemplateDecl>(CD))
            AddTemplateOverloadCandidate(
                FTD, DeclAccessPair::make(FTD, AS_none), ExplicitTemplateArgs,
                Args, OCS);
          else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(CD))
            if (!ExplicitTemplateArgs || ExplicitTemplateArgs->size() == 0)
              AddOverloadCandidate(FD, DeclAccessPair::make(FD, AS_none),
                                   Args, OCS);
        }
        switch (OCS.BestViableFunction(*this, R.getNameLoc(), Best)) {
        case OR_Success:
          ND = Best->FoundDecl;
          Corrected.setCorrectionDecl(ND);
          break;
        default:
          // FIXME: Arbitrarily pick the first declaration for the note.
          Corrected.setCorrectionDecl(ND);
          break;
        }
      }
      R.addDecl(ND);
      if (getLangOpts().CPlusPlus && ND->isCXXClassMember()) {
        CXXRecordDecl *Record = nullptr;
        if (Corrected.getCorrectionSpecifier()) {
          const Type *Ty = Corrected.getCorrectionSpecifier()->getAsType();
          Record = Ty->getAsCXXRecordDecl();
        }
        if (!Record)
          Record = cast<CXXRecordDecl>(
              ND->getDeclContext()->getRedeclContext());
        R.setNamingClass(Record);
      }

      auto *UnderlyingND = ND->getUnderlyingDecl();
      AcceptableWithRecovery = isa<ValueDecl>(UnderlyingND) ||
                               isa<FunctionTemplateDecl>(UnderlyingND);
      // FIXME: If we ended up with a typo for a type name or
      // Objective-C class name, we're in trouble because the parser
      // is in the wrong place to recover. Suggest the typo
      // correction, but don't make it a fix-it since we're not going
      // to recover well anyway.
      AcceptableWithoutRecovery =
          isa<TypeDecl>(UnderlyingND) || isa<ObjCInterfaceDecl>(UnderlyingND);
    } else {
      // FIXME: We found a keyword. Suggest it, but don't provide a fix-it
      // because we aren't able to recover.
      AcceptableWithoutRecovery = true;
    }

    if (AcceptableWithRecovery || AcceptableWithoutRecovery) {
      unsigned NoteID = Corrected.getCorrectionDeclAs<ImplicitParamDecl>()
                            ? diag::note_implicit_param_decl
                            : diag::note_previous_decl;
      if (SS.isEmpty())
        diagnoseTypo(Corrected, PDiag(diagnostic_suggest) << Name,
                     PDiag(NoteID), AcceptableWithRecovery);
      else
        diagnoseTypo(Corrected, PDiag(diag::err_no_member_suggest)
                                  << Name << computeDeclContext(SS, false)
                                  << DroppedSpecifier << SS.getRange(),
                     PDiag(NoteID), AcceptableWithRecovery);

      // Tell the callee whether to try to recover.
      return !AcceptableWithRecovery;
    }
  }
  R.clear();

  // Emit a special diagnostic for failed member lookups.
  // FIXME: computing the declaration context might fail here (?)
  if (!SS.isEmpty()) {
    Diag(R.getNameLoc(), diag::err_no_member)
      << Name << computeDeclContext(SS, false)
      << SS.getRange();
    return true;
  }

  // Give up, we can't recover.
  Diag(R.getNameLoc(), diagnostic) << Name;
  return true;
}

/// In Microsoft mode, if we are inside a template class whose parent class has
/// dependent base classes, and we can't resolve an unqualified identifier, then
/// assume the identifier is a member of a dependent base class.  We can only
/// recover successfully in static methods, instance methods, and other contexts
/// where 'this' is available.  This doesn't precisely match MSVC's
/// instantiation model, but it's close enough.
static Expr *
recoverFromMSUnqualifiedLookup(Sema &S, ASTContext &Context,
                               DeclarationNameInfo &NameInfo,
                               SourceLocation TemplateKWLoc,
                               const TemplateArgumentListInfo *TemplateArgs) {
  // Only try to recover from lookup into dependent bases in static methods or
  // contexts where 'this' is available.
  QualType ThisType = S.getCurrentThisType();
  const CXXRecordDecl *RD = nullptr;
  if (!ThisType.isNull())
    RD = ThisType->getPointeeType()->getAsCXXRecordDecl();
  else if (auto *MD = dyn_cast<CXXMethodDecl>(S.CurContext))
    RD = MD->getParent();
  if (!RD || !RD->hasAnyDependentBases())
    return nullptr;

  // Diagnose this as unqualified lookup into a dependent base class.  If 'this'
  // is available, suggest inserting 'this->' as a fixit.
  SourceLocation Loc = NameInfo.getLoc();
  auto DB = S.Diag(Loc, diag::ext_undeclared_unqual_id_with_dependent_base);
  DB << NameInfo.getName() << RD;

  if (!ThisType.isNull()) {
    DB << FixItHint::CreateInsertion(Loc, "this->");
    return CXXDependentScopeMemberExpr::Create(
        Context, /*This=*/nullptr, ThisType, /*IsArrow=*/true,
        /*Op=*/SourceLocation(), NestedNameSpecifierLoc(), TemplateKWLoc,
        /*FirstQualifierInScope=*/nullptr, NameInfo, TemplateArgs);
  }

  // Synthesize a fake NNS that points to the derived class.  This will
  // perform name lookup during template instantiation.
  CXXScopeSpec SS;
  auto *NNS =
      NestedNameSpecifier::Create(Context, nullptr, true, RD->getTypeForDecl());
  SS.MakeTrivial(Context, NNS, SourceRange(Loc, Loc));
  return DependentScopeDeclRefExpr::Create(
      Context, SS.getWithLocInContext(Context), TemplateKWLoc, NameInfo,
      TemplateArgs);
}

ExprResult
Sema::ActOnIdExpression(Scope *S, CXXScopeSpec &SS,
                        SourceLocation TemplateKWLoc, UnqualifiedId &Id,
                        bool HasTrailingLParen, bool IsAddressOfOperand,
                        std::unique_ptr<CorrectionCandidateCallback> CCC,
                        bool IsInlineAsmIdentifier, Token *KeywordReplacement) {
  assert(!(IsAddressOfOperand && HasTrailingLParen) &&
         "cannot be direct & operand and have a trailing lparen");
  if (SS.isInvalid())
    return ExprError();

  TemplateArgumentListInfo TemplateArgsBuffer;

  // Decompose the UnqualifiedId into the following data.
  DeclarationNameInfo NameInfo;
  const TemplateArgumentListInfo *TemplateArgs;
  DecomposeUnqualifiedId(Id, TemplateArgsBuffer, NameInfo, TemplateArgs);

  DeclarationName Name = NameInfo.getName();
  IdentifierInfo *II = Name.getAsIdentifierInfo();
  SourceLocation NameLoc = NameInfo.getLoc();

  // C++ [temp.dep.expr]p3:
  //   An id-expression is type-dependent if it contains:
  //     -- an identifier that was declared with a dependent type,
  //        (note: handled after lookup)
  //     -- a template-id that is dependent,
  //        (note: handled in BuildTemplateIdExpr)
  //     -- a conversion-function-id that specifies a dependent type,
  //     -- a nested-name-specifier that contains a class-name that
  //        names a dependent type.
  // Determine whether this is a member of an unknown specialization;
  // we need to handle these differently.
  bool DependentID = false;
  if (Name.getNameKind() == DeclarationName::CXXConversionFunctionName &&
      Name.getCXXNameType()->isDependentType()) {
    DependentID = true;
  } else if (SS.isSet()) {
    if (DeclContext *DC = computeDeclContext(SS, false)) {
      if (RequireCompleteDeclContext(SS, DC))
        return ExprError();
    } else {
      DependentID = true;
    }
  }

  if (DependentID)
    return ActOnDependentIdExpression(SS, TemplateKWLoc, NameInfo,
                                      IsAddressOfOperand, TemplateArgs);

  // Perform the required lookup.
  LookupResult R(*this, NameInfo, 
                 (Id.getKind() == UnqualifiedId::IK_ImplicitSelfParam) 
                  ? LookupObjCImplicitSelfParam : LookupOrdinaryName);
  if (TemplateArgs) {
    // Lookup the template name again to correctly establish the context in
    // which it was found. This is really unfortunate as we already did the
    // lookup to determine that it was a template name in the first place. If
    // this becomes a performance hit, we can work harder to preserve those
    // results until we get here but it's likely not worth it.
    bool MemberOfUnknownSpecialization;
    LookupTemplateName(R, S, SS, QualType(), /*EnteringContext=*/false,
                       MemberOfUnknownSpecialization);
    
    if (MemberOfUnknownSpecialization ||
        (R.getResultKind() == LookupResult::NotFoundInCurrentInstantiation))
      return ActOnDependentIdExpression(SS, TemplateKWLoc, NameInfo,
                                        IsAddressOfOperand, TemplateArgs);
  } else {
    bool IvarLookupFollowUp = II && !SS.isSet() && getCurMethodDecl();
    LookupParsedName(R, S, &SS, !IvarLookupFollowUp);

    // If the result might be in a dependent base class, this is a dependent 
    // id-expression.
    if (R.getResultKind() == LookupResult::NotFoundInCurrentInstantiation)
      return ActOnDependentIdExpression(SS, TemplateKWLoc, NameInfo,
                                        IsAddressOfOperand, TemplateArgs);

    // If this reference is in an Objective-C method, then we need to do
    // some special Objective-C lookup, too.
    if (IvarLookupFollowUp) {
      ExprResult E(LookupInObjCMethod(R, S, II, true));
      if (E.isInvalid())
        return ExprError();

      if (Expr *Ex = E.getAs<Expr>())
        return Ex;
    }
  }

  if (R.isAmbiguous())
    return ExprError();

  // This could be an implicitly declared function reference (legal in C90,
  // extension in C99, forbidden in C++).
  if (R.empty() && HasTrailingLParen && II && !getLangOpts().CPlusPlus) {
    NamedDecl *D = ImplicitlyDefineFunction(NameLoc, *II, S);
    if (D) R.addDecl(D);
  }

  // Determine whether this name might be a candidate for
  // argument-dependent lookup.
  bool ADL = UseArgumentDependentLookup(SS, R, HasTrailingLParen);

  if (R.empty() && !ADL) {
    if (SS.isEmpty() && getLangOpts().MSVCCompat) {
      if (Expr *E = recoverFromMSUnqualifiedLookup(*this, Context, NameInfo,
                                                   TemplateKWLoc, TemplateArgs))
        return E;
    }

    // Don't diagnose an empty lookup for inline assembly.
    if (IsInlineAsmIdentifier)
      return ExprError();

    // If this name wasn't predeclared and if this is not a function
    // call, diagnose the problem.
    TypoExpr *TE = nullptr;
    auto DefaultValidator = llvm::make_unique<CorrectionCandidateCallback>(
        II, SS.isValid() ? SS.getScopeRep() : nullptr);
    DefaultValidator->IsAddressOfOperand = IsAddressOfOperand;
    assert((!CCC || CCC->IsAddressOfOperand == IsAddressOfOperand) &&
           "Typo correction callback misconfigured");
    if (CCC) {
      // Make sure the callback knows what the typo being diagnosed is.
      CCC->setTypoName(II);
      if (SS.isValid())
        CCC->setTypoNNS(SS.getScopeRep());
    }
    if (DiagnoseEmptyLookup(S, SS, R,
                            CCC ? std::move(CCC) : std::move(DefaultValidator),
                            nullptr, None, &TE)) {
      if (TE && KeywordReplacement) {
        auto &State = getTypoExprState(TE);
        auto BestTC = State.Consumer->getNextCorrection();
        if (BestTC.isKeyword()) {
          auto *II = BestTC.getCorrectionAsIdentifierInfo();
          if (State.DiagHandler)
            State.DiagHandler(BestTC);
          KeywordReplacement->startToken();
          KeywordReplacement->setKind(II->getTokenID());
          KeywordReplacement->setIdentifierInfo(II);
          KeywordReplacement->setLocation(BestTC.getCorrectionRange().getBegin());
          // Clean up the state associated with the TypoExpr, since it has
          // now been diagnosed (without a call to CorrectDelayedTyposInExpr).
          clearDelayedTypo(TE);
          // Signal that a correction to a keyword was performed by returning a
          // valid-but-null ExprResult.
          return (Expr*)nullptr;
        }
        State.Consumer->resetCorrectionStream();
      }
      return TE ? TE : ExprError();
    }

    assert(!R.empty() &&
           "DiagnoseEmptyLookup returned false but added no results");

    // If we found an Objective-C instance variable, let
    // LookupInObjCMethod build the appropriate expression to
    // reference the ivar.
    if (ObjCIvarDecl *Ivar = R.getAsSingle<ObjCIvarDecl>()) {
      R.clear();
      ExprResult E(LookupInObjCMethod(R, S, Ivar->getIdentifier()));
      // In a hopelessly buggy code, Objective-C instance variable
      // lookup fails and no expression will be built to reference it.
      if (!E.isInvalid() && !E.get())
        return ExprError();
      return E;
    }
  }

  // This is guaranteed from this point on.
  assert(!R.empty() || ADL);

  // Check whether this might be a C++ implicit instance member access.
  // C++ [class.mfct.non-static]p3:
  //   When an id-expression that is not part of a class member access
  //   syntax and not used to form a pointer to member is used in the
  //   body of a non-static member function of class X, if name lookup
  //   resolves the name in the id-expression to a non-static non-type
  //   member of some class C, the id-expression is transformed into a
  //   class member access expression using (*this) as the
  //   postfix-expression to the left of the . operator.
  //
  // But we don't actually need to do this for '&' operands if R
  // resolved to a function or overloaded function set, because the
  // expression is ill-formed if it actually works out to be a
  // non-static member function:
  //
  // C++ [expr.ref]p4:
  //   Otherwise, if E1.E2 refers to a non-static member function. . .
  //   [t]he expression can be used only as the left-hand operand of a
  //   member function call.
  //
  // There are other safeguards against such uses, but it's important
  // to get this right here so that we don't end up making a
  // spuriously dependent expression if we're inside a dependent
  // instance method.
  if (!R.empty() && (*R.begin())->isCXXClassMember()) {
    bool MightBeImplicitMember;
    if (!IsAddressOfOperand)
      MightBeImplicitMember = true;
    else if (!SS.isEmpty())
      MightBeImplicitMember = false;
    else if (R.isOverloadedResult())
      MightBeImplicitMember = false;
    else if (R.isUnresolvableResult())
      MightBeImplicitMember = true;
    else
      MightBeImplicitMember = isa<FieldDecl>(R.getFoundDecl()) ||
                              isa<IndirectFieldDecl>(R.getFoundDecl()) ||
                              isa<MSPropertyDecl>(R.getFoundDecl());

    if (MightBeImplicitMember)
      return BuildPossibleImplicitMemberExpr(SS, TemplateKWLoc,
                                             R, TemplateArgs, S);
  }

  if (TemplateArgs || TemplateKWLoc.isValid()) {

    // In C++1y, if this is a variable template id, then check it
    // in BuildTemplateIdExpr().
    // The single lookup result must be a variable template declaration.
    if (Id.getKind() == UnqualifiedId::IK_TemplateId && Id.TemplateId &&
        Id.TemplateId->Kind == TNK_Var_template) {
      assert(R.getAsSingle<VarTemplateDecl>() &&
             "There should only be one declaration found.");
    }

    return BuildTemplateIdExpr(SS, TemplateKWLoc, R, ADL, TemplateArgs);
  }

  return BuildDeclarationNameExpr(SS, R, ADL);
}

/// BuildQualifiedDeclarationNameExpr - Build a C++ qualified
/// declaration name, generally during template instantiation.
/// There's a large number of things which don't need to be done along
/// this path.
ExprResult Sema::BuildQualifiedDeclarationNameExpr(
    CXXScopeSpec &SS, const DeclarationNameInfo &NameInfo,
    bool IsAddressOfOperand, const Scope *S, TypeSourceInfo **RecoveryTSI) {
  DeclContext *DC = computeDeclContext(SS, false);
  if (!DC)
    return BuildDependentDeclRefExpr(SS, /*TemplateKWLoc=*/SourceLocation(),
                                     NameInfo, /*TemplateArgs=*/nullptr);

  if (RequireCompleteDeclContext(SS, DC))
    return ExprError();

  LookupResult R(*this, NameInfo, LookupOrdinaryName);
  LookupQualifiedName(R, DC);

  if (R.isAmbiguous())
    return ExprError();

  if (R.getResultKind() == LookupResult::NotFoundInCurrentInstantiation)
    return BuildDependentDeclRefExpr(SS, /*TemplateKWLoc=*/SourceLocation(),
                                     NameInfo, /*TemplateArgs=*/nullptr);

  if (R.empty()) {
    Diag(NameInfo.getLoc(), diag::err_no_member)
      << NameInfo.getName() << DC << SS.getRange();
    return ExprError();
  }

  if (const TypeDecl *TD = R.getAsSingle<TypeDecl>()) {
    // Diagnose a missing typename if this resolved unambiguously to a type in
    // a dependent context.  If we can recover with a type, downgrade this to
    // a warning in Microsoft compatibility mode.
    unsigned DiagID = diag::err_typename_missing;
    if (RecoveryTSI && getLangOpts().MSVCCompat)
      DiagID = diag::ext_typename_missing;
    SourceLocation Loc = SS.getBeginLoc();
    auto D = Diag(Loc, DiagID);
    D << SS.getScopeRep() << NameInfo.getName().getAsString()
      << SourceRange(Loc, NameInfo.getEndLoc());

    // Don't recover if the caller isn't expecting us to or if we're in a SFINAE
    // context.
    if (!RecoveryTSI)
      return ExprError();

    // Only issue the fixit if we're prepared to recover.
    D << FixItHint::CreateInsertion(Loc, "typename ");

    // Recover by pretending this was an elaborated type.
    QualType Ty = Context.getTypeDeclType(TD);
    TypeLocBuilder TLB;
    TLB.pushTypeSpec(Ty).setNameLoc(NameInfo.getLoc());

    QualType ET = getElaboratedType(ETK_None, SS, Ty);
    ElaboratedTypeLoc QTL = TLB.push<ElaboratedTypeLoc>(ET);
    QTL.setElaboratedKeywordLoc(SourceLocation());
    QTL.setQualifierLoc(SS.getWithLocInContext(Context));

    *RecoveryTSI = TLB.getTypeSourceInfo(Context, ET);

    return ExprEmpty();
  }

  // Defend against this resolving to an implicit member access. We usually
  // won't get here if this might be a legitimate a class member (we end up in
  // BuildMemberReferenceExpr instead), but this can be valid if we're forming
  // a pointer-to-member or in an unevaluated context in C++11.
  if (!R.empty() && (*R.begin())->isCXXClassMember() && !IsAddressOfOperand)
    return BuildPossibleImplicitMemberExpr(SS,
                                           /*TemplateKWLoc=*/SourceLocation(),
                                           R, /*TemplateArgs=*/nullptr, S);

  return BuildDeclarationNameExpr(SS, R, /* ADL */ false);
}

/// LookupInObjCMethod - The parser has read a name in, and Sema has
/// detected that we're currently inside an ObjC method.  Perform some
/// additional lookup.
///
/// Ideally, most of this would be done by lookup, but there's
/// actually quite a lot of extra work involved.
///
/// Returns a null sentinel to indicate trivial success.
ExprResult
Sema::LookupInObjCMethod(LookupResult &Lookup, Scope *S,
                         IdentifierInfo *II, bool AllowBuiltinCreation) {
  SourceLocation Loc = Lookup.getNameLoc();
  ObjCMethodDecl *CurMethod = getCurMethodDecl();
  
  // Check for error condition which is already reported.
  if (!CurMethod)
    return ExprError();

  // There are two cases to handle here.  1) scoped lookup could have failed,
  // in which case we should look for an ivar.  2) scoped lookup could have
  // found a decl, but that decl is outside the current instance method (i.e.
  // a global variable).  In these two cases, we do a lookup for an ivar with
  // this name, if the lookup sucedes, we replace it our current decl.

  // If we're in a class method, we don't normally want to look for
  // ivars.  But if we don't find anything else, and there's an
  // ivar, that's an error.
  bool IsClassMethod = CurMethod->isClassMethod();

  bool LookForIvars;
  if (Lookup.empty())
    LookForIvars = true;
  else if (IsClassMethod)
    LookForIvars = false;
  else
    LookForIvars = (Lookup.isSingleResult() &&
                    Lookup.getFoundDecl()->isDefinedOutsideFunctionOrMethod());
  ObjCInterfaceDecl *IFace = nullptr;
  if (LookForIvars) {
    IFace = CurMethod->getClassInterface();
    ObjCInterfaceDecl *ClassDeclared;
    ObjCIvarDecl *IV = nullptr;
    if (IFace && (IV = IFace->lookupInstanceVariable(II, ClassDeclared))) {
      // Diagnose using an ivar in a class method.
      if (IsClassMethod)
        return ExprError(Diag(Loc, diag::error_ivar_use_in_class_method)
                         << IV->getDeclName());

      // If we're referencing an invalid decl, just return this as a silent
      // error node.  The error diagnostic was already emitted on the decl.
      if (IV->isInvalidDecl())
        return ExprError();

      // Check if referencing a field with __attribute__((deprecated)).
      if (DiagnoseUseOfDecl(IV, Loc))
        return ExprError();

      // Diagnose the use of an ivar outside of the declaring class.
      if (IV->getAccessControl() == ObjCIvarDecl::Private &&
          !declaresSameEntity(ClassDeclared, IFace) &&
          !getLangOpts().DebuggerSupport)
        Diag(Loc, diag::error_private_ivar_access) << IV->getDeclName();

      // FIXME: This should use a new expr for a direct reference, don't
      // turn this into Self->ivar, just return a BareIVarExpr or something.
      IdentifierInfo &II = Context.Idents.get("self");
      UnqualifiedId SelfName;
      SelfName.setIdentifier(&II, SourceLocation());
      SelfName.setKind(UnqualifiedId::IK_ImplicitSelfParam);
      CXXScopeSpec SelfScopeSpec;
      SourceLocation TemplateKWLoc;
      ExprResult SelfExpr = ActOnIdExpression(S, SelfScopeSpec, TemplateKWLoc,
                                              SelfName, false, false);
      if (SelfExpr.isInvalid())
        return ExprError();

      SelfExpr = DefaultLvalueConversion(SelfExpr.get());
      if (SelfExpr.isInvalid())
        return ExprError();

      MarkAnyDeclReferenced(Loc, IV, true);

      ObjCMethodFamily MF = CurMethod->getMethodFamily();
      if (MF != OMF_init && MF != OMF_dealloc && MF != OMF_finalize &&
          !IvarBacksCurrentMethodAccessor(IFace, CurMethod, IV))
        Diag(Loc, diag::warn_direct_ivar_access) << IV->getDeclName();

      ObjCIvarRefExpr *Result = new (Context)
          ObjCIvarRefExpr(IV, IV->getUsageType(SelfExpr.get()->getType()), Loc,
                          IV->getLocation(), SelfExpr.get(), true, true);

      if (getLangOpts().ObjCAutoRefCount) {
        if (IV->getType().getObjCLifetime() == Qualifiers::OCL_Weak) {
          if (!Diags.isIgnored(diag::warn_arc_repeated_use_of_weak, Loc))
            recordUseOfEvaluatedWeak(Result);
        }
        if (CurContext->isClosure())
          Diag(Loc, diag::warn_implicitly_retains_self)
            << FixItHint::CreateInsertion(Loc, "self->");
      }
      
      return Result;
    }
  } else if (CurMethod->isInstanceMethod()) {
    // We should warn if a local variable hides an ivar.
    if (ObjCInterfaceDecl *IFace = CurMethod->getClassInterface()) {
      ObjCInterfaceDecl *ClassDeclared;
      if (ObjCIvarDecl *IV = IFace->lookupInstanceVariable(II, ClassDeclared)) {
        if (IV->getAccessControl() != ObjCIvarDecl::Private ||
            declaresSameEntity(IFace, ClassDeclared))
          Diag(Loc, diag::warn_ivar_use_hidden) << IV->getDeclName();
      }
    }
  } else if (Lookup.isSingleResult() &&
             Lookup.getFoundDecl()->isDefinedOutsideFunctionOrMethod()) {
    // If accessing a stand-alone ivar in a class method, this is an error.
    if (const ObjCIvarDecl *IV = dyn_cast<ObjCIvarDecl>(Lookup.getFoundDecl()))
      return ExprError(Diag(Loc, diag::error_ivar_use_in_class_method)
                       << IV->getDeclName());
  }

  if (Lookup.empty() && II && AllowBuiltinCreation) {
    // FIXME. Consolidate this with similar code in LookupName.
    if (unsigned BuiltinID = II->getBuiltinID()) {
      if (!(getLangOpts().CPlusPlus &&
            Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID))) {
        NamedDecl *D = LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID,
                                           S, Lookup.isForRedeclaration(),
                                           Lookup.getNameLoc());
        if (D) Lookup.addDecl(D);
      }
    }
  }
  // Sentinel value saying that we didn't do anything special.
  return ExprResult((Expr *)nullptr);
}

/// \brief Cast a base object to a member's actual type.
///
/// Logically this happens in three phases:
///
/// * First we cast from the base type to the naming class.
///   The naming class is the class into which we were looking
///   when we found the member;  it's the qualifier type if a
///   qualifier was provided, and otherwise it's the base type.
///
/// * Next we cast from the naming class to the declaring class.
///   If the member we found was brought into a class's scope by
///   a using declaration, this is that class;  otherwise it's
///   the class declaring the member.
///
/// * Finally we cast from the declaring class to the "true"
///   declaring class of the member.  This conversion does not
///   obey access control.
ExprResult
Sema::PerformObjectMemberConversion(Expr *From,
                                    NestedNameSpecifier *Qualifier,
                                    NamedDecl *FoundDecl,
                                    NamedDecl *Member) {
  CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(Member->getDeclContext());
  if (!RD)
    return From;

  QualType DestRecordType;
  QualType DestType;
  QualType FromRecordType;
  QualType FromType = From->getType();
  bool PointerConversions = false;
  if (isa<FieldDecl>(Member)) {
    DestRecordType = Context.getCanonicalType(Context.getTypeDeclType(RD));

    if (FromType->getAs<PointerType>()) {
      DestType = Context.getPointerType(DestRecordType);
      FromRecordType = FromType->getPointeeType();
      PointerConversions = true;
    } else {
      DestType = DestRecordType;
      FromRecordType = FromType;
    }
  } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Member)) {
    if (Method->isStatic())
      return From;

    DestType = Method->getThisType(Context);
    DestRecordType = DestType->getPointeeType();

    if (FromType->getAs<PointerType>()) {
      FromRecordType = FromType->getPointeeType();
      PointerConversions = true;
    } else {
      FromRecordType = FromType;
      DestType = DestRecordType;
    }
  } else {
    // No conversion necessary.
    return From;
  }

  if (DestType->isDependentType() || FromType->isDependentType())
    return From;

  // If the unqualified types are the same, no conversion is necessary.
  if (Context.hasSameUnqualifiedType(FromRecordType, DestRecordType))
    return From;

  SourceRange FromRange = From->getSourceRange();
  SourceLocation FromLoc = FromRange.getBegin();

  ExprValueKind VK = From->getValueKind();

  // C++ [class.member.lookup]p8:
  //   [...] Ambiguities can often be resolved by qualifying a name with its
  //   class name.
  //
  // If the member was a qualified name and the qualified referred to a
  // specific base subobject type, we'll cast to that intermediate type
  // first and then to the object in which the member is declared. That allows
  // one to resolve ambiguities in, e.g., a diamond-shaped hierarchy such as:
  //
  //   class Base { public: int x; };
  //   class Derived1 : public Base { };
  //   class Derived2 : public Base { };
  //   class VeryDerived : public Derived1, public Derived2 { void f(); };
  //
  //   void VeryDerived::f() {
  //     x = 17; // error: ambiguous base subobjects
  //     Derived1::x = 17; // okay, pick the Base subobject of Derived1
  //   }
  if (Qualifier && Qualifier->getAsType()) {
    QualType QType = QualType(Qualifier->getAsType(), 0);
    assert(QType->isRecordType() && "lookup done with non-record type");

    QualType QRecordType = QualType(QType->getAs<RecordType>(), 0);

    // In C++98, the qualifier type doesn't actually have to be a base
    // type of the object type, in which case we just ignore it.
    // Otherwise build the appropriate casts.
    if (IsDerivedFrom(FromLoc, FromRecordType, QRecordType)) {
      CXXCastPath BasePath;
      if (CheckDerivedToBaseConversion(FromRecordType, QRecordType,
                                       FromLoc, FromRange, &BasePath))
        return ExprError();

      if (PointerConversions)
        QType = Context.getPointerType(QType);
      From = ImpCastExprToType(From, QType, CK_UncheckedDerivedToBase,
                               VK, &BasePath).get();

      FromType = QType;
      FromRecordType = QRecordType;

      // If the qualifier type was the same as the destination type,
      // we're done.
      if (Context.hasSameUnqualifiedType(FromRecordType, DestRecordType))
        return From;
    }
  }

  bool IgnoreAccess = false;

  // If we actually found the member through a using declaration, cast
  // down to the using declaration's type.
  //
  // Pointer equality is fine here because only one declaration of a
  // class ever has member declarations.
  if (FoundDecl->getDeclContext() != Member->getDeclContext()) {
    assert(isa<UsingShadowDecl>(FoundDecl));
    QualType URecordType = Context.getTypeDeclType(
                           cast<CXXRecordDecl>(FoundDecl->getDeclContext()));

    // We only need to do this if the naming-class to declaring-class
    // conversion is non-trivial.
    if (!Context.hasSameUnqualifiedType(FromRecordType, URecordType)) {
      assert(IsDerivedFrom(FromLoc, FromRecordType, URecordType));
      CXXCastPath BasePath;
      if (CheckDerivedToBaseConversion(FromRecordType, URecordType,
                                       FromLoc, FromRange, &BasePath))
        return ExprError();

      QualType UType = URecordType;
      if (PointerConversions)
        UType = Context.getPointerType(UType);
      From = ImpCastExprToType(From, UType, CK_UncheckedDerivedToBase,
                               VK, &BasePath).get();
      FromType = UType;
      FromRecordType = URecordType;
    }

    // We don't do access control for the conversion from the
    // declaring class to the true declaring class.
    IgnoreAccess = true;
  }

  CXXCastPath BasePath;
  if (CheckDerivedToBaseConversion(FromRecordType, DestRecordType,
                                   FromLoc, FromRange, &BasePath,
                                   IgnoreAccess))
    return ExprError();

  return ImpCastExprToType(From, DestType, CK_UncheckedDerivedToBase,
                           VK, &BasePath);
}

bool Sema::UseArgumentDependentLookup(const CXXScopeSpec &SS,
                                      const LookupResult &R,
                                      bool HasTrailingLParen) {
  // Only when used directly as the postfix-expression of a call.
  if (!HasTrailingLParen)
    return false;

  // Never if a scope specifier was provided.
  if (SS.isSet())
    return false;

  // Only in C++ or ObjC++.
  if (!getLangOpts().CPlusPlus)
    return false;

  // Turn off ADL when we find certain kinds of declarations during
  // normal lookup:
  for (NamedDecl *D : R) {
    // C++0x [basic.lookup.argdep]p3:
    //     -- a declaration of a class member
    // Since using decls preserve this property, we check this on the
    // original decl.
    if (D->isCXXClassMember())
      return false;

    // C++0x [basic.lookup.argdep]p3:
    //     -- a block-scope function declaration that is not a
    //        using-declaration
    // NOTE: we also trigger this for function templates (in fact, we
    // don't check the decl type at all, since all other decl types
    // turn off ADL anyway).
    if (isa<UsingShadowDecl>(D))
      D = cast<UsingShadowDecl>(D)->getTargetDecl();
    else if (D->getLexicalDeclContext()->isFunctionOrMethod())
      return false;

    // C++0x [basic.lookup.argdep]p3:
    //     -- a declaration that is neither a function or a function
    //        template
    // And also for builtin functions.
    if (isa<FunctionDecl>(D)) {
      FunctionDecl *FDecl = cast<FunctionDecl>(D);

      // But also builtin functions.
      if (FDecl->getBuiltinID() && FDecl->isImplicit())
        return false;
    } else if (!isa<FunctionTemplateDecl>(D))
      return false;
  }

  return true;
}


/// Diagnoses obvious problems with the use of the given declaration
/// as an expression.  This is only actually called for lookups that
/// were not overloaded, and it doesn't promise that the declaration
/// will in fact be used.
static bool CheckDeclInExpr(Sema &S, SourceLocation Loc, NamedDecl *D) {
  if (isa<TypedefNameDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_typedef) << D->getDeclName();
    return true;
  }

  if (isa<ObjCInterfaceDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_interface) << D->getDeclName();
    return true;
  }

  if (isa<NamespaceDecl>(D)) {
    S.Diag(Loc, diag::err_unexpected_namespace) << D->getDeclName();
    return true;
  }

  return false;
}

ExprResult Sema::BuildDeclarationNameExpr(const CXXScopeSpec &SS,
                                          LookupResult &R, bool NeedsADL,
                                          bool AcceptInvalidDecl) {
  // If this is a single, fully-resolved result and we don't need ADL,
  // just build an ordinary singleton decl ref.
  if (!NeedsADL && R.isSingleResult() && !R.getAsSingle<FunctionTemplateDecl>())
    return BuildDeclarationNameExpr(SS, R.getLookupNameInfo(), R.getFoundDecl(),
                                    R.getRepresentativeDecl(), nullptr,
                                    AcceptInvalidDecl);

  // We only need to check the declaration if there's exactly one
  // result, because in the overloaded case the results can only be
  // functions and function templates.
  if (R.isSingleResult() &&
      CheckDeclInExpr(*this, R.getNameLoc(), R.getFoundDecl()))
    return ExprError();

  // Otherwise, just build an unresolved lookup expression.  Suppress
  // any lookup-related diagnostics; we'll hash these out later, when
  // we've picked a target.
  R.suppressDiagnostics();

  UnresolvedLookupExpr *ULE
    = UnresolvedLookupExpr::Create(Context, R.getNamingClass(),
                                   SS.getWithLocInContext(Context),
                                   R.getLookupNameInfo(),
                                   NeedsADL, R.isOverloadedResult(),
                                   R.begin(), R.end());

  return ULE;
}

/// \brief Complete semantic analysis for a reference to the given declaration.
ExprResult Sema::BuildDeclarationNameExpr(
    const CXXScopeSpec &SS, const DeclarationNameInfo &NameInfo, NamedDecl *D,
    NamedDecl *FoundD, const TemplateArgumentListInfo *TemplateArgs,
    bool AcceptInvalidDecl) {
  assert(D && "Cannot refer to a NULL declaration");
  assert(!isa<FunctionTemplateDecl>(D) &&
         "Cannot refer unambiguously to a function template");

  SourceLocation Loc = NameInfo.getLoc();
  if (CheckDeclInExpr(*this, Loc, D))
    return ExprError();

  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(D)) {
    // Specifically diagnose references to class templates that are missing
    // a template argument list.
    Diag(Loc, diag::err_template_decl_ref) << (isa<VarTemplateDecl>(D) ? 1 : 0)
                                           << Template << SS.getRange();
    Diag(Template->getLocation(), diag::note_template_decl_here);
    return ExprError();
  }

  // Make sure that we're referring to a value.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD) {
    Diag(Loc, diag::err_ref_non_value)
      << D << SS.getRange();
    Diag(D->getLocation(), diag::note_declared_at);
    return ExprError();
  }

  // Check whether this declaration can be used. Note that we suppress
  // this check when we're going to perform argument-dependent lookup
  // on this function name, because this might not be the function
  // that overload resolution actually selects.
  if (DiagnoseUseOfDecl(VD, Loc))
    return ExprError();

  // Only create DeclRefExpr's for valid Decl's.
  if (VD->isInvalidDecl() && !AcceptInvalidDecl)
    return ExprError();

  // Handle members of anonymous structs and unions.  If we got here,
  // and the reference is to a class member indirect field, then this
  // must be the subject of a pointer-to-member expression.
  if (IndirectFieldDecl *indirectField = dyn_cast<IndirectFieldDecl>(VD))
    if (!indirectField->isCXXClassMember())
      return BuildAnonymousStructUnionMemberReference(SS, NameInfo.getLoc(),
                                                      indirectField);

  {
    QualType type = VD->getType();
    ExprValueKind valueKind = VK_RValue;

    switch (D->getKind()) {
    // Ignore all the non-ValueDecl kinds.
#define ABSTRACT_DECL(kind)
#define VALUE(type, base)
#define DECL(type, base) \
    case Decl::type:
#include "clang/AST/DeclNodes.inc"
      llvm_unreachable("invalid value decl kind");

    // These shouldn't make it here.
    case Decl::ObjCAtDefsField:
    case Decl::ObjCIvar:
      llvm_unreachable("forming non-member reference to ivar?");

    // Enum constants are always r-values and never references.
    // Unresolved using declarations are dependent.
    case Decl::EnumConstant:
    case Decl::UnresolvedUsingValue:
      valueKind = VK_RValue;
      break;

    // Fields and indirect fields that got here must be for
    // pointer-to-member expressions; we just call them l-values for
    // internal consistency, because this subexpression doesn't really
    // exist in the high-level semantics.
    case Decl::Field:
    case Decl::IndirectField:
      assert(getLangOpts().CPlusPlus &&
             "building reference to field in C?");

      // These can't have reference type in well-formed programs, but
      // for internal consistency we do this anyway.
      type = type.getNonReferenceType();
      valueKind = VK_LValue;
      break;

    // Non-type template parameters are either l-values or r-values
    // depending on the type.
    case Decl::NonTypeTemplateParm: {
      if (const ReferenceType *reftype = type->getAs<ReferenceType>()) {
        type = reftype->getPointeeType();
        valueKind = VK_LValue; // even if the parameter is an r-value reference
        break;
      }

      // For non-references, we need to strip qualifiers just in case
      // the template parameter was declared as 'const int' or whatever.
      valueKind = VK_RValue;
      type = type.getUnqualifiedType();
      break;
    }

    case Decl::Var:
    case Decl::VarTemplateSpecialization:
    case Decl::VarTemplatePartialSpecialization:
      // In C, "extern void blah;" is valid and is an r-value.
      if (!getLangOpts().CPlusPlus &&
          !type.hasQualifiers() &&
          type->isVoidType()) {
        valueKind = VK_RValue;
        break;
      }
      // fallthrough

    case Decl::ImplicitParam:
    case Decl::ParmVar: {
      // These are always l-values.
      valueKind = VK_LValue;
      type = type.getNonReferenceType();

      // FIXME: Does the addition of const really only apply in
      // potentially-evaluated contexts? Since the variable isn't actually
      // captured in an unevaluated context, it seems that the answer is no.
      if (!isUnevaluatedContext()) {
        QualType CapturedType = getCapturedDeclRefType(cast<VarDecl>(VD), Loc);
        if (!CapturedType.isNull())
          type = CapturedType;
      }
      
      break;
    }
        
    case Decl::Function: {
      if (unsigned BID = cast<FunctionDecl>(VD)->getBuiltinID()) {
        if (!Context.BuiltinInfo.isPredefinedLibFunction(BID)) {
          type = Context.BuiltinFnTy;
          valueKind = VK_RValue;
          break;
        }
      }

      const FunctionType *fty = type->castAs<FunctionType>();

      // If we're referring to a function with an __unknown_anytype
      // result type, make the entire expression __unknown_anytype.
      if (fty->getReturnType() == Context.UnknownAnyTy) {
        type = Context.UnknownAnyTy;
        valueKind = VK_RValue;
        break;
      }

      // Functions are l-values in C++.
      if (getLangOpts().CPlusPlus) {
        valueKind = VK_LValue;
        break;
      }
      
      // C99 DR 316 says that, if a function type comes from a
      // function definition (without a prototype), that type is only
      // used for checking compatibility. Therefore, when referencing
      // the function, we pretend that we don't have the full function
      // type.
      if (!cast<FunctionDecl>(VD)->hasPrototype() &&
          isa<FunctionProtoType>(fty))
        type = Context.getFunctionNoProtoType(fty->getReturnType(),
                                              fty->getExtInfo());

      // Functions are r-values in C.
      valueKind = VK_RValue;
      break;
    }

    case Decl::MSProperty:
      valueKind = VK_LValue;
      break;

    case Decl::CXXMethod:
      // If we're referring to a method with an __unknown_anytype
      // result type, make the entire expression __unknown_anytype.
      // This should only be possible with a type written directly.
      if (const FunctionProtoType *proto
            = dyn_cast<FunctionProtoType>(VD->getType()))
        if (proto->getReturnType() == Context.UnknownAnyTy) {
          type = Context.UnknownAnyTy;
          valueKind = VK_RValue;
          break;
        }

      // C++ methods are l-values if static, r-values if non-static.
      if (cast<CXXMethodDecl>(VD)->isStatic()) {
        valueKind = VK_LValue;
        break;
      }
      // fallthrough

    case Decl::CXXConversion:
    case Decl::CXXDestructor:
    case Decl::CXXConstructor:
      valueKind = VK_RValue;
      break;
    }

    return BuildDeclRefExpr(VD, type, valueKind, NameInfo, &SS, FoundD,
                            TemplateArgs);
  }
}

static void ConvertUTF8ToWideString(unsigned CharByteWidth, StringRef Source,
                                    SmallString<32> &Target) {
  Target.resize(CharByteWidth * (Source.size() + 1));
  char *ResultPtr = &Target[0];
  const UTF8 *ErrorPtr;
  bool success = ConvertUTF8toWide(CharByteWidth, Source, ResultPtr, ErrorPtr);
  (void)success;
  assert(success);
  Target.resize(ResultPtr - &Target[0]);
}

ExprResult Sema::BuildPredefinedExpr(SourceLocation Loc,
                                     PredefinedExpr::IdentType IT) {
  // Pick the current block, lambda, captured statement or function.
  Decl *currentDecl = nullptr;
  if (const BlockScopeInfo *BSI = getCurBlock())
    currentDecl = BSI->TheDecl;
  else if (const LambdaScopeInfo *LSI = getCurLambda())
    currentDecl = LSI->CallOperator;
  else if (const CapturedRegionScopeInfo *CSI = getCurCapturedRegion())
    currentDecl = CSI->TheCapturedDecl;
  else
    currentDecl = getCurFunctionOrMethodDecl();

  if (!currentDecl) {
    Diag(Loc, diag::ext_predef_outside_function);
    currentDecl = Context.getTranslationUnitDecl();
  }

  QualType ResTy;
  StringLiteral *SL = nullptr;
  if (cast<DeclContext>(currentDecl)->isDependentContext())
    ResTy = Context.DependentTy;
  else {
    // Pre-defined identifiers are of type char[x], where x is the length of
    // the string.
    auto Str = PredefinedExpr::ComputeName(IT, currentDecl);
    unsigned Length = Str.length();

    llvm::APInt LengthI(32, Length + 1);
    if (IT == PredefinedExpr::LFunction) {
      ResTy = Context.WideCharTy.withConst();
      SmallString<32> RawChars;
      ConvertUTF8ToWideString(Context.getTypeSizeInChars(ResTy).getQuantity(),
                              Str, RawChars);
      ResTy = Context.getConstantArrayType(ResTy, LengthI, ArrayType::Normal,
                                           /*IndexTypeQuals*/ 0);
      SL = StringLiteral::Create(Context, RawChars, StringLiteral::Wide,
                                 /*Pascal*/ false, ResTy, Loc);
    } else {
      ResTy = Context.CharTy.withConst();
      ResTy = Context.getConstantArrayType(ResTy, LengthI, ArrayType::Normal,
                                           /*IndexTypeQuals*/ 0);
      SL = StringLiteral::Create(Context, Str, StringLiteral::Ascii,
                                 /*Pascal*/ false, ResTy, Loc);
    }
  }

  return new (Context) PredefinedExpr(Loc, ResTy, IT, SL);
}

ExprResult Sema::ActOnPredefinedExpr(SourceLocation Loc, tok::TokenKind Kind) {
  PredefinedExpr::IdentType IT;

  switch (Kind) {
  default: llvm_unreachable("Unknown simple primary expr!");
  case tok::kw___func__: IT = PredefinedExpr::Func; break; // [C99 6.4.2.2]
  case tok::kw___FUNCTION__: IT = PredefinedExpr::Function; break;
  case tok::kw___FUNCDNAME__: IT = PredefinedExpr::FuncDName; break; // [MS]
  case tok::kw___FUNCSIG__: IT = PredefinedExpr::FuncSig; break; // [MS]
  case tok::kw_L__FUNCTION__: IT = PredefinedExpr::LFunction; break;
  case tok::kw___PRETTY_FUNCTION__: IT = PredefinedExpr::PrettyFunction; break;
  }

  return BuildPredefinedExpr(Loc, IT);
}

ExprResult Sema::ActOnCharacterConstant(const Token &Tok, Scope *UDLScope) {
  SmallString<16> CharBuffer;
  bool Invalid = false;
  StringRef ThisTok = PP.getSpelling(Tok, CharBuffer, &Invalid);
  if (Invalid)
    return ExprError();

  CharLiteralParser Literal(ThisTok.begin(), ThisTok.end(), Tok.getLocation(),
                            PP, Tok.getKind());
  if (Literal.hadError())
    return ExprError();

  QualType Ty;
  if (Literal.isWide())
    Ty = Context.WideCharTy; // L'x' -> wchar_t in C and C++.
  else if (Literal.isUTF16())
    Ty = Context.Char16Ty; // u'x' -> char16_t in C11 and C++11.
  else if (Literal.isUTF32())
    Ty = Context.Char32Ty; // U'x' -> char32_t in C11 and C++11.
  else if (!getLangOpts().CPlusPlus || Literal.isMultiChar())
    Ty = Context.IntTy;   // 'x' -> int in C, 'wxyz' -> int in C++.
  else
    Ty = Context.CharTy;  // 'x' -> char in C++

  CharacterLiteral::CharacterKind Kind = CharacterLiteral::Ascii;
  if (Literal.isWide())
    Kind = CharacterLiteral::Wide;
  else if (Literal.isUTF16())
    Kind = CharacterLiteral::UTF16;
  else if (Literal.isUTF32())
    Kind = CharacterLiteral::UTF32;
  else if (Literal.isUTF8())
    Kind = CharacterLiteral::UTF8;

  Expr *Lit = new (Context) CharacterLiteral(Literal.getValue(), Kind, Ty,
                                             Tok.getLocation());

  if (Literal.getUDSuffix().empty())
    return Lit;

  // We're building a user-defined literal.
  IdentifierInfo *UDSuffix = &Context.Idents.get(Literal.getUDSuffix());
  SourceLocation UDSuffixLoc =
    getUDSuffixLoc(*this, Tok.getLocation(), Literal.getUDSuffixOffset());

  // Make sure we're allowed user-defined literals here.
  if (!UDLScope)
    return ExprError(Diag(UDSuffixLoc, diag::err_invalid_character_udl));

  // C++11 [lex.ext]p6: The literal L is treated as a call of the form
  //   operator "" X (ch)
  return BuildCookedLiteralOperatorCall(*this, UDLScope, UDSuffix, UDSuffixLoc,
                                        Lit, Tok.getLocation());
}

ExprResult Sema::ActOnIntegerConstant(SourceLocation Loc, uint64_t Val) {
  unsigned IntSize = Context.getTargetInfo().getIntWidth();
  return IntegerLiteral::Create(Context, llvm::APInt(IntSize, Val),
                                Context.IntTy, Loc);
}

static Expr *BuildFloatingLiteral(Sema &S, NumericLiteralParser &Literal,
                                  QualType Ty, SourceLocation Loc) {
  const llvm::fltSemantics &Format = S.Context.getFloatTypeSemantics(Ty);

  using llvm::APFloat;
  APFloat Val(Format);

  APFloat::opStatus result = Literal.GetFloatValue(Val);

  // Overflow is always an error, but underflow is only an error if
  // we underflowed to zero (APFloat reports denormals as underflow).
  if ((result & APFloat::opOverflow) ||
      ((result & APFloat::opUnderflow) && Val.isZero())) {
    unsigned diagnostic;
    SmallString<20> buffer;
    if (result & APFloat::opOverflow) {
      diagnostic = diag::warn_float_overflow;
      APFloat::getLargest(Format).toString(buffer);
    } else {
      diagnostic = diag::warn_float_underflow;
      APFloat::getSmallest(Format).toString(buffer);
    }

    S.Diag(Loc, diagnostic)
      << Ty
      << StringRef(buffer.data(), buffer.size());
  }

  bool isExact = (result == APFloat::opOK);
  return FloatingLiteral::Create(S.Context, Val, isExact, Ty, Loc);
}

bool Sema::CheckLoopHintExpr(Expr *E, SourceLocation Loc) {
  assert(E && "Invalid expression");

  if (E->isValueDependent())
    return false;

  QualType QT = E->getType();
  if (!QT->isIntegerType() || QT->isBooleanType() || QT->isCharType()) {
    Diag(E->getExprLoc(), diag::err_pragma_loop_invalid_argument_type) << QT;
    return true;
  }

  llvm::APSInt ValueAPS;
  ExprResult R = VerifyIntegerConstantExpression(E, &ValueAPS);

  if (R.isInvalid())
    return true;

  bool ValueIsPositive = ValueAPS.isStrictlyPositive();
  if (!ValueIsPositive || ValueAPS.getActiveBits() > 31) {
    Diag(E->getExprLoc(), diag::err_pragma_loop_invalid_argument_value)
        << ValueAPS.toString(10) << ValueIsPositive;
    return true;
  }

  return false;
}

ExprResult Sema::ActOnNumericConstant(const Token &Tok, Scope *UDLScope) {
  // Fast path for a single digit (which is quite common).  A single digit
  // cannot have a trigraph, escaped newline, radix prefix, or suffix.
  if (Tok.getLength() == 1) {
    const char Val = PP.getSpellingOfSingleCharacterNumericConstant(Tok);
    return ActOnIntegerConstant(Tok.getLocation(), Val-'0');
  }

  SmallString<128> SpellingBuffer;
  // NumericLiteralParser wants to overread by one character.  Add padding to
  // the buffer in case the token is copied to the buffer.  If getSpelling()
  // returns a StringRef to the memory buffer, it should have a null char at
  // the EOF, so it is also safe.
  SpellingBuffer.resize(Tok.getLength() + 1);

  // Get the spelling of the token, which eliminates trigraphs, etc.
  bool Invalid = false;
  StringRef TokSpelling = PP.getSpelling(Tok, SpellingBuffer, &Invalid);
  if (Invalid)
    return ExprError();

  NumericLiteralParser Literal(TokSpelling, Tok.getLocation(), PP);
  if (Literal.hadError)
    return ExprError();

  if (Literal.hasUDSuffix()) {
    // We're building a user-defined literal.
    IdentifierInfo *UDSuffix = &Context.Idents.get(Literal.getUDSuffix());
    SourceLocation UDSuffixLoc =
      getUDSuffixLoc(*this, Tok.getLocation(), Literal.getUDSuffixOffset());

    // Make sure we're allowed user-defined literals here.
    if (!UDLScope)
      return ExprError(Diag(UDSuffixLoc, diag::err_invalid_numeric_udl));

    QualType CookedTy;
    if (Literal.isFloatingLiteral()) {
      // C++11 [lex.ext]p4: If S contains a literal operator with parameter type
      // long double, the literal is treated as a call of the form
      //   operator "" X (f L)
      CookedTy = Context.LongDoubleTy;
    } else {
      // C++11 [lex.ext]p3: If S contains a literal operator with parameter type
      // unsigned long long, the literal is treated as a call of the form
      //   operator "" X (n ULL)
      CookedTy = Context.UnsignedLongLongTy;
    }

    DeclarationName OpName =
      Context.DeclarationNames.getCXXLiteralOperatorName(UDSuffix);
    DeclarationNameInfo OpNameInfo(OpName, UDSuffixLoc);
    OpNameInfo.setCXXLiteralOperatorNameLoc(UDSuffixLoc);

    SourceLocation TokLoc = Tok.getLocation();

    // Perform literal operator lookup to determine if we're building a raw
    // literal or a cooked one.
    LookupResult R(*this, OpName, UDSuffixLoc, LookupOrdinaryName);
    switch (LookupLiteralOperator(UDLScope, R, CookedTy,
                                  /*AllowRaw*/true, /*AllowTemplate*/true,
                                  /*AllowStringTemplate*/false)) {
    case LOLR_Error:
      return ExprError();

    case LOLR_Cooked: {
      Expr *Lit;
      if (Literal.isFloatingLiteral()) {
        Lit = BuildFloatingLiteral(*this, Literal, CookedTy, Tok.getLocation());
      } else {
        llvm::APInt ResultVal(Context.getTargetInfo().getLongLongWidth(), 0);
        if (Literal.GetIntegerValue(ResultVal))
          Diag(Tok.getLocation(), diag::err_integer_literal_too_large)
              << /* Unsigned */ 1;
        Lit = IntegerLiteral::Create(Context, ResultVal, CookedTy,
                                     Tok.getLocation());
      }
      return BuildLiteralOperatorCall(R, OpNameInfo, Lit, TokLoc);
    }

    case LOLR_Raw: {
      // C++11 [lit.ext]p3, p4: If S contains a raw literal operator, the
      // literal is treated as a call of the form
      //   operator "" X ("n")
      unsigned Length = Literal.getUDSuffixOffset();
      QualType StrTy = Context.getConstantArrayType(
          Context.CharTy.withConst(), llvm::APInt(32, Length + 1),
          ArrayType::Normal, 0);
      Expr *Lit = StringLiteral::Create(
          Context, StringRef(TokSpelling.data(), Length), StringLiteral::Ascii,
          /*Pascal*/false, StrTy, &TokLoc, 1);
      return BuildLiteralOperatorCall(R, OpNameInfo, Lit, TokLoc);
    }

    case LOLR_Template: {
      // C++11 [lit.ext]p3, p4: Otherwise (S contains a literal operator
      // template), L is treated as a call fo the form
      //   operator "" X <'c1', 'c2', ... 'ck'>()
      // where n is the source character sequence c1 c2 ... ck.
      TemplateArgumentListInfo ExplicitArgs;
      unsigned CharBits = Context.getIntWidth(Context.CharTy);
      bool CharIsUnsigned = Context.CharTy->isUnsignedIntegerType();
      llvm::APSInt Value(CharBits, CharIsUnsigned);
      for (unsigned I = 0, N = Literal.getUDSuffixOffset(); I != N; ++I) {
        Value = TokSpelling[I];
        TemplateArgument Arg(Context, Value, Context.CharTy);
        TemplateArgumentLocInfo ArgInfo;
        ExplicitArgs.addArgument(TemplateArgumentLoc(Arg, ArgInfo));
      }
      return BuildLiteralOperatorCall(R, OpNameInfo, None, TokLoc,
                                      &ExplicitArgs);
    }
    case LOLR_StringTemplate:
      llvm_unreachable("unexpected literal operator lookup result");
    }
  }

  Expr *Res;

  if (Literal.isFloatingLiteral()) {
    QualType Ty;
    if (Literal.isFloat)
      Ty = Context.FloatTy;
    else if (!Literal.isLong)
      Ty = Context.DoubleTy;
    else
      Ty = Context.LongDoubleTy;

    Res = BuildFloatingLiteral(*this, Literal, Ty, Tok.getLocation());

    if (Ty == Context.DoubleTy) {
      if (getLangOpts().SinglePrecisionConstants) {
        Res = ImpCastExprToType(Res, Context.FloatTy, CK_FloatingCast).get();
      } else if (getLangOpts().OpenCL &&
                 !((getLangOpts().OpenCLVersion >= 120) ||
                   getOpenCLOptions().cl_khr_fp64)) {
        Diag(Tok.getLocation(), diag::warn_double_const_requires_fp64);
        Res = ImpCastExprToType(Res, Context.FloatTy, CK_FloatingCast).get();
      }
    }
  } else if (!Literal.isIntegerLiteral()) {
    return ExprError();
  } else {
    QualType Ty;

    // 'long long' is a C99 or C++11 feature.
    if (!getLangOpts().C99 && Literal.isLongLong) {
      if (getLangOpts().CPlusPlus)
        Diag(Tok.getLocation(),
             getLangOpts().CPlusPlus11 ?
             diag::warn_cxx98_compat_longlong : diag::ext_cxx11_longlong);
      else
        Diag(Tok.getLocation(), diag::ext_c99_longlong);
    }

    // Get the value in the widest-possible width.
    unsigned MaxWidth = Context.getTargetInfo().getIntMaxTWidth();
    llvm::APInt ResultVal(MaxWidth, 0);

    if (Literal.GetIntegerValue(ResultVal)) {
      // If this value didn't fit into uintmax_t, error and force to ull.
      Diag(Tok.getLocation(), diag::err_integer_literal_too_large)
          << /* Unsigned */ 1;
      Ty = Context.UnsignedLongLongTy;
      assert(Context.getTypeSize(Ty) == ResultVal.getBitWidth() &&
             "long long is not intmax_t?");
    } else {
      // If this value fits into a ULL, try to figure out what else it fits into
      // according to the rules of C99 6.4.4.1p5.

      // Octal, Hexadecimal, and integers with a U suffix are allowed to
      // be an unsigned int.
      bool AllowUnsigned = Literal.isUnsigned || Literal.getRadix() != 10;

      // Check from smallest to largest, picking the smallest type we can.
      unsigned Width = 0;

      // Microsoft specific integer suffixes are explicitly sized.
      if (Literal.MicrosoftInteger) {
        if (Literal.MicrosoftInteger == 8 && !Literal.isUnsigned) {
          Width = 8;
          Ty = Context.CharTy;
        } else {
          Width = Literal.MicrosoftInteger;
          Ty = Context.getIntTypeForBitwidth(Width,
                                             /*Signed=*/!Literal.isUnsigned);
        }
      }

      if (Ty.isNull() && !Literal.isLong && !Literal.isLongLong) {
        // Are int/unsigned possibilities?
        unsigned IntSize = Context.getTargetInfo().getIntWidth();

        // Does it fit in a unsigned int?
        if (ResultVal.isIntN(IntSize)) {
          // Does it fit in a signed int?
          if (!Literal.isUnsigned && ResultVal[IntSize-1] == 0)
            Ty = Context.IntTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedIntTy;
          Width = IntSize;
        }
      }

      // Are long/unsigned long possibilities?
      if (Ty.isNull() && !Literal.isLongLong) {
        unsigned LongSize = Context.getTargetInfo().getLongWidth();

        // Does it fit in a unsigned long?
        if (ResultVal.isIntN(LongSize)) {
          // Does it fit in a signed long?
          if (!Literal.isUnsigned && ResultVal[LongSize-1] == 0)
            Ty = Context.LongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongTy;
          // Check according to the rules of C90 6.1.3.2p5. C++03 [lex.icon]p2
          // is compatible.
          else if (!getLangOpts().C99 && !getLangOpts().CPlusPlus11) {
            const unsigned LongLongSize =
                Context.getTargetInfo().getLongLongWidth();
            Diag(Tok.getLocation(),
                 getLangOpts().CPlusPlus
                     ? Literal.isLong
                           ? diag::warn_old_implicitly_unsigned_long_cxx
                           : /*C++98 UB*/ diag::
                                 ext_old_implicitly_unsigned_long_cxx
                     : diag::warn_old_implicitly_unsigned_long)
                << (LongLongSize > LongSize ? /*will have type 'long long'*/ 0
                                            : /*will be ill-formed*/ 1);
            Ty = Context.UnsignedLongTy;
          }
          Width = LongSize;
        }
      }

      // Check long long if needed.
      if (Ty.isNull()) {
        unsigned LongLongSize = Context.getTargetInfo().getLongLongWidth();

        // Does it fit in a unsigned long long?
        if (ResultVal.isIntN(LongLongSize)) {
          // Does it fit in a signed long long?
          // To be compatible with MSVC, hex integer literals ending with the
          // LL or i64 suffix are always signed in Microsoft mode.
          if (!Literal.isUnsigned && (ResultVal[LongLongSize-1] == 0 ||
              (getLangOpts().MicrosoftExt && Literal.isLongLong)))
            Ty = Context.LongLongTy;
          else if (AllowUnsigned)
            Ty = Context.UnsignedLongLongTy;
          Width = LongLongSize;
        }
      }

      // If we still couldn't decide a type, we probably have something that
      // does not fit in a signed long long, but has no U suffix.
      if (Ty.isNull()) {
        Diag(Tok.getLocation(), diag::ext_integer_literal_too_large_for_signed);
        Ty = Context.UnsignedLongLongTy;
        Width = Context.getTargetInfo().getLongLongWidth();
      }

      if (ResultVal.getBitWidth() != Width)
        ResultVal = ResultVal.trunc(Width);
    }
    Res = IntegerLiteral::Create(Context, ResultVal, Ty, Tok.getLocation());
  }

  // If this is an imaginary literal, create the ImaginaryLiteral wrapper.
  if (Literal.isImaginary)
    Res = new (Context) ImaginaryLiteral(Res,
                                        Context.getComplexType(Res->getType()));

  return Res;
}

ExprResult Sema::ActOnParenExpr(SourceLocation L, SourceLocation R, Expr *E) {
  assert(E && "ActOnParenExpr() missing expr");
  return new (Context) ParenExpr(L, R, E);
}

static bool CheckVecStepTraitOperandType(Sema &S, QualType T,
                                         SourceLocation Loc,
                                         SourceRange ArgRange) {
  // [OpenCL 1.1 6.11.12] "The vec_step built-in function takes a built-in
  // scalar or vector data type argument..."
  // Every built-in scalar type (OpenCL 1.1 6.1.1) is either an arithmetic
  // type (C99 6.2.5p18) or void.
  if (!(T->isArithmeticType() || T->isVoidType() || T->isVectorType())) {
    S.Diag(Loc, diag::err_vecstep_non_scalar_vector_type)
      << T << ArgRange;
    return true;
  }

  assert((T->isVoidType() || !T->isIncompleteType()) &&
         "Scalar types should always be complete");
  return false;
}

static bool CheckExtensionTraitOperandType(Sema &S, QualType T,
                                           SourceLocation Loc,
                                           SourceRange ArgRange,
                                           UnaryExprOrTypeTrait TraitKind) {
  // Invalid types must be hard errors for SFINAE in C++.
  if (S.LangOpts.CPlusPlus)
    return true;

  // C99 6.5.3.4p1:
  if (T->isFunctionType() &&
      (TraitKind == UETT_SizeOf || TraitKind == UETT_AlignOf)) {
    // sizeof(function)/alignof(function) is allowed as an extension.
    S.Diag(Loc, diag::ext_sizeof_alignof_function_type)
      << TraitKind << ArgRange;
    return false;
  }

  // Allow sizeof(void)/alignof(void) as an extension, unless in OpenCL where
  // this is an error (OpenCL v1.1 s6.3.k)
  if (T->isVoidType()) {
    unsigned DiagID = S.LangOpts.OpenCL ? diag::err_opencl_sizeof_alignof_type
                                        : diag::ext_sizeof_alignof_void_type;
    S.Diag(Loc, DiagID) << TraitKind << ArgRange;
    return false;
  }

  return true;
}

static bool CheckObjCTraitOperandConstraints(Sema &S, QualType T,
                                             SourceLocation Loc,
                                             SourceRange ArgRange,
                                             UnaryExprOrTypeTrait TraitKind) {
  // Reject sizeof(interface) and sizeof(interface<proto>) if the
  // runtime doesn't allow it.
  if (!S.LangOpts.ObjCRuntime.allowsSizeofAlignof() && T->isObjCObjectType()) {
    S.Diag(Loc, diag::err_sizeof_nonfragile_interface)
      << T << (TraitKind == UETT_SizeOf)
      << ArgRange;
    return true;
  }

  return false;
}

/// \brief Check whether E is a pointer from a decayed array type (the decayed
/// pointer type is equal to T) and emit a warning if it is.
static void warnOnSizeofOnArrayDecay(Sema &S, SourceLocation Loc, QualType T,
                                     Expr *E) {
  // Don't warn if the operation changed the type.
  if (T != E->getType())
    return;

  // Now look for array decays.
  ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E);
  if (!ICE || ICE->getCastKind() != CK_ArrayToPointerDecay)
    return;

  S.Diag(Loc, diag::warn_sizeof_array_decay) << ICE->getSourceRange()
                                             << ICE->getType()
                                             << ICE->getSubExpr()->getType();
}

/// \brief Check the constraints on expression operands to unary type expression
/// and type traits.
///
/// Completes any types necessary and validates the constraints on the operand
/// expression. The logic mostly mirrors the type-based overload, but may modify
/// the expression as it completes the type for that expression through template
/// instantiation, etc.
bool Sema::CheckUnaryExprOrTypeTraitOperand(Expr *E,
                                            UnaryExprOrTypeTrait ExprKind) {
  QualType ExprTy = E->getType();
  assert(!ExprTy->isReferenceType());

  if (ExprKind == UETT_VecStep)
    return CheckVecStepTraitOperandType(*this, ExprTy, E->getExprLoc(),
                                        E->getSourceRange());

  // Whitelist some types as extensions
  if (!CheckExtensionTraitOperandType(*this, ExprTy, E->getExprLoc(),
                                      E->getSourceRange(), ExprKind))
    return false;

  // 'alignof' applied to an expression only requires the base element type of
  // the expression to be complete. 'sizeof' requires the expression's type to
  // be complete (and will attempt to complete it if it's an array of unknown
  // bound).
  if (ExprKind == UETT_AlignOf) {
    if (RequireCompleteType(E->getExprLoc(),
                            Context.getBaseElementType(E->getType()),
                            diag::err_sizeof_alignof_incomplete_type, ExprKind,
                            E->getSourceRange()))
      return true;
  } else {
    if (RequireCompleteExprType(E, diag::err_sizeof_alignof_incomplete_type,
                                ExprKind, E->getSourceRange()))
      return true;
  }

  // Completing the expression's type may have changed it.
  ExprTy = E->getType();
  assert(!ExprTy->isReferenceType());

  if (ExprTy->isFunctionType()) {
    Diag(E->getExprLoc(), diag::err_sizeof_alignof_function_type)
      << ExprKind << E->getSourceRange();
    return true;
  }

  // The operand for sizeof and alignof is in an unevaluated expression context,
  // so side effects could result in unintended consequences.
  if ((ExprKind == UETT_SizeOf || ExprKind == UETT_AlignOf) &&
      ActiveTemplateInstantiations.empty() && E->HasSideEffects(Context, false))
    Diag(E->getExprLoc(), diag::warn_side_effects_unevaluated_context);

  if (CheckObjCTraitOperandConstraints(*this, ExprTy, E->getExprLoc(),
                                       E->getSourceRange(), ExprKind))
    return true;

  if (ExprKind == UETT_SizeOf) {
    if (DeclRefExpr *DeclRef = dyn_cast<DeclRefExpr>(E->IgnoreParens())) {
      if (ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(DeclRef->getFoundDecl())) {
        QualType OType = PVD->getOriginalType();
        QualType Type = PVD->getType();
        if (Type->isPointerType() && OType->isArrayType()) {
          Diag(E->getExprLoc(), diag::warn_sizeof_array_param)
            << Type << OType;
          Diag(PVD->getLocation(), diag::note_declared_at);
        }
      }
    }

    // Warn on "sizeof(array op x)" and "sizeof(x op array)", where the array
    // decays into a pointer and returns an unintended result. This is most
    // likely a typo for "sizeof(array) op x".
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E->IgnoreParens())) {
      warnOnSizeofOnArrayDecay(*this, BO->getOperatorLoc(), BO->getType(),
                               BO->getLHS());
      warnOnSizeofOnArrayDecay(*this, BO->getOperatorLoc(), BO->getType(),
                               BO->getRHS());
    }
  }

  return false;
}

/// \brief Check the constraints on operands to unary expression and type
/// traits.
///
/// This will complete any types necessary, and validate the various constraints
/// on those operands.
///
/// The UsualUnaryConversions() function is *not* called by this routine.
/// C99 6.3.2.1p[2-4] all state:
///   Except when it is the operand of the sizeof operator ...
///
/// C++ [expr.sizeof]p4
///   The lvalue-to-rvalue, array-to-pointer, and function-to-pointer
///   standard conversions are not applied to the operand of sizeof.
///
/// This policy is followed for all of the unary trait expressions.
bool Sema::CheckUnaryExprOrTypeTraitOperand(QualType ExprType,
                                            SourceLocation OpLoc,
                                            SourceRange ExprRange,
                                            UnaryExprOrTypeTrait ExprKind) {
  if (ExprType->isDependentType())
    return false;

  // C++ [expr.sizeof]p2:
  //     When applied to a reference or a reference type, the result
  //     is the size of the referenced type.
  // C++11 [expr.alignof]p3:
  //     When alignof is applied to a reference type, the result
  //     shall be the alignment of the referenced type.
  if (const ReferenceType *Ref = ExprType->getAs<ReferenceType>())
    ExprType = Ref->getPointeeType();

  // C11 6.5.3.4/3, C++11 [expr.alignof]p3:
  //   When alignof or _Alignof is applied to an array type, the result
  //   is the alignment of the element type.
  if (ExprKind == UETT_AlignOf || ExprKind == UETT_OpenMPRequiredSimdAlign)
    ExprType = Context.getBaseElementType(ExprType);

  if (ExprKind == UETT_VecStep)
    return CheckVecStepTraitOperandType(*this, ExprType, OpLoc, ExprRange);

  // Whitelist some types as extensions
  if (!CheckExtensionTraitOperandType(*this, ExprType, OpLoc, ExprRange,
                                      ExprKind))
    return false;

  if (RequireCompleteType(OpLoc, ExprType,
                          diag::err_sizeof_alignof_incomplete_type,
                          ExprKind, ExprRange))
    return true;

  if (ExprType->isFunctionType()) {
    Diag(OpLoc, diag::err_sizeof_alignof_function_type)
      << ExprKind << ExprRange;
    return true;
  }

  if (CheckObjCTraitOperandConstraints(*this, ExprType, OpLoc, ExprRange,
                                       ExprKind))
    return true;

  return false;
}

static bool CheckAlignOfExpr(Sema &S, Expr *E) {
  E = E->IgnoreParens();

  // Cannot know anything else if the expression is dependent.
  if (E->isTypeDependent())
    return false;

  if (E->getObjectKind() == OK_BitField) {
    S.Diag(E->getExprLoc(), diag::err_sizeof_alignof_typeof_bitfield)
       << 1 << E->getSourceRange();
    return true;
  }

  ValueDecl *D = nullptr;
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    D = DRE->getDecl();
  } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    D = ME->getMemberDecl();
  }

  // If it's a field, require the containing struct to have a
  // complete definition so that we can compute the layout.
  //
  // This can happen in C++11 onwards, either by naming the member
  // in a way that is not transformed into a member access expression
  // (in an unevaluated operand, for instance), or by naming the member
  // in a trailing-return-type.
  //
  // For the record, since __alignof__ on expressions is a GCC
  // extension, GCC seems to permit this but always gives the
  // nonsensical answer 0.
  //
  // We don't really need the layout here --- we could instead just
  // directly check for all the appropriate alignment-lowing
  // attributes --- but that would require duplicating a lot of
  // logic that just isn't worth duplicating for such a marginal
  // use-case.
  if (FieldDecl *FD = dyn_cast_or_null<FieldDecl>(D)) {
    // Fast path this check, since we at least know the record has a
    // definition if we can find a member of it.
    if (!FD->getParent()->isCompleteDefinition()) {
      S.Diag(E->getExprLoc(), diag::err_alignof_member_of_incomplete_type)
        << E->getSourceRange();
      return true;
    }

    // Otherwise, if it's a field, and the field doesn't have
    // reference type, then it must have a complete type (or be a
    // flexible array member, which we explicitly want to
    // white-list anyway), which makes the following checks trivial.
    if (!FD->getType()->isReferenceType())
      return false;
  }

  return S.CheckUnaryExprOrTypeTraitOperand(E, UETT_AlignOf);
}

bool Sema::CheckVecStepExpr(Expr *E) {
  E = E->IgnoreParens();

  // Cannot know anything else if the expression is dependent.
  if (E->isTypeDependent())
    return false;

  return CheckUnaryExprOrTypeTraitOperand(E, UETT_VecStep);
}

/// \brief Build a sizeof or alignof expression given a type operand.
ExprResult
Sema::CreateUnaryExprOrTypeTraitExpr(TypeSourceInfo *TInfo,
                                     SourceLocation OpLoc,
                                     UnaryExprOrTypeTrait ExprKind,
                                     SourceRange R) {
  if (!TInfo)
    return ExprError();

  QualType T = TInfo->getType();

  if (!T->isDependentType() &&
      CheckUnaryExprOrTypeTraitOperand(T, OpLoc, R, ExprKind))
    return ExprError();

  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return new (Context) UnaryExprOrTypeTraitExpr(
      ExprKind, TInfo, Context.getSizeType(), OpLoc, R.getEnd());
}

/// \brief Build a sizeof or alignof expression given an expression
/// operand.
ExprResult
Sema::CreateUnaryExprOrTypeTraitExpr(Expr *E, SourceLocation OpLoc,
                                     UnaryExprOrTypeTrait ExprKind) {
  ExprResult PE = CheckPlaceholderExpr(E);
  if (PE.isInvalid()) 
    return ExprError();

  E = PE.get();
  
  // Verify that the operand is valid.
  bool isInvalid = false;
  if (E->isTypeDependent()) {
    // Delay type-checking for type-dependent expressions.
  } else if (ExprKind == UETT_AlignOf) {
    isInvalid = CheckAlignOfExpr(*this, E);
  } else if (ExprKind == UETT_VecStep) {
    isInvalid = CheckVecStepExpr(E);
  } else if (ExprKind == UETT_OpenMPRequiredSimdAlign) {
      Diag(E->getExprLoc(), diag::err_openmp_default_simd_align_expr);
      isInvalid = true;
  } else if (E->refersToBitField()) {  // C99 6.5.3.4p1.
    Diag(E->getExprLoc(), diag::err_sizeof_alignof_typeof_bitfield) << 0;
    isInvalid = true;
  } else {
    isInvalid = CheckUnaryExprOrTypeTraitOperand(E, UETT_SizeOf);
  }

  if (isInvalid)
    return ExprError();

  if (ExprKind == UETT_SizeOf && E->getType()->isVariableArrayType()) {
    PE = TransformToPotentiallyEvaluated(E);
    if (PE.isInvalid()) return ExprError();
    E = PE.get();
  }

  // C99 6.5.3.4p4: the type (an unsigned integer type) is size_t.
  return new (Context) UnaryExprOrTypeTraitExpr(
      ExprKind, E, Context.getSizeType(), OpLoc, E->getSourceRange().getEnd());
}

/// ActOnUnaryExprOrTypeTraitExpr - Handle @c sizeof(type) and @c sizeof @c
/// expr and the same for @c alignof and @c __alignof
/// Note that the ArgRange is invalid if isType is false.
ExprResult
Sema::ActOnUnaryExprOrTypeTraitExpr(SourceLocation OpLoc,
                                    UnaryExprOrTypeTrait ExprKind, bool IsType,
                                    void *TyOrEx, SourceRange ArgRange) {
  // If error parsing type, ignore.
  if (!TyOrEx) return ExprError();

  if (IsType) {
    TypeSourceInfo *TInfo;
    (void) GetTypeFromParser(ParsedType::getFromOpaquePtr(TyOrEx), &TInfo);
    return CreateUnaryExprOrTypeTraitExpr(TInfo, OpLoc, ExprKind, ArgRange);
  }

  Expr *ArgEx = (Expr *)TyOrEx;
  ExprResult Result = CreateUnaryExprOrTypeTraitExpr(ArgEx, OpLoc, ExprKind);
  return Result;
}

static QualType CheckRealImagOperand(Sema &S, ExprResult &V, SourceLocation Loc,
                                     bool IsReal) {
  if (V.get()->isTypeDependent())
    return S.Context.DependentTy;

  // _Real and _Imag are only l-values for normal l-values.
  if (V.get()->getObjectKind() != OK_Ordinary) {
    V = S.DefaultLvalueConversion(V.get());
    if (V.isInvalid())
      return QualType();
  }

  // These operators return the element type of a complex type.
  if (const ComplexType *CT = V.get()->getType()->getAs<ComplexType>())
    return CT->getElementType();

  // Otherwise they pass through real integer and floating point types here.
  if (V.get()->getType()->isArithmeticType())
    return V.get()->getType();

  // Test for placeholders.
  ExprResult PR = S.CheckPlaceholderExpr(V.get());
  if (PR.isInvalid()) return QualType();
  if (PR.get() != V.get()) {
    V = PR;
    return CheckRealImagOperand(S, V, Loc, IsReal);
  }

  // Reject anything else.
  S.Diag(Loc, diag::err_realimag_invalid_type) << V.get()->getType()
    << (IsReal ? "__real" : "__imag");
  return QualType();
}



ExprResult
Sema::ActOnPostfixUnaryOp(Scope *S, SourceLocation OpLoc,
                          tok::TokenKind Kind, Expr *Input) {
  UnaryOperatorKind Opc;
  switch (Kind) {
  default: llvm_unreachable("Unknown unary op!");
  case tok::plusplus:   Opc = UO_PostInc; break;
  case tok::minusminus: Opc = UO_PostDec; break;
  }

  // Since this might is a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Input);
  if (Result.isInvalid()) return ExprError();
  Input = Result.get();

  return BuildUnaryOp(S, OpLoc, Opc, Input);
}

/// \brief Diagnose if arithmetic on the given ObjC pointer is illegal.
///
/// \return true on error
static bool checkArithmeticOnObjCPointer(Sema &S,
                                         SourceLocation opLoc,
                                         Expr *op) {
  assert(op->getType()->isObjCObjectPointerType());
  if (S.LangOpts.ObjCRuntime.allowsPointerArithmetic() &&
      !S.LangOpts.ObjCSubscriptingLegacyRuntime)
    return false;

  S.Diag(opLoc, diag::err_arithmetic_nonfragile_interface)
    << op->getType()->castAs<ObjCObjectPointerType>()->getPointeeType()
    << op->getSourceRange();
  return true;
}

static bool isMSPropertySubscriptExpr(Sema &S, Expr *Base) {
  auto *BaseNoParens = Base->IgnoreParens();
  if (auto *MSProp = dyn_cast<MSPropertyRefExpr>(BaseNoParens))
    return MSProp->getPropertyDecl()->getType()->isArrayType();
  return isa<MSPropertySubscriptExpr>(BaseNoParens);
}

ExprResult
Sema::ActOnArraySubscriptExpr(Scope *S, Expr *base, SourceLocation lbLoc,
                              Expr *idx, SourceLocation rbLoc) {
  if (base && !base->getType().isNull() &&
      base->getType()->isSpecificPlaceholderType(BuiltinType::OMPArraySection))
    return ActOnOMPArraySectionExpr(base, lbLoc, idx, SourceLocation(),
                                    /*Length=*/nullptr, rbLoc);

  // Since this might be a postfix expression, get rid of ParenListExprs.
  if (isa<ParenListExpr>(base)) {
    ExprResult result = MaybeConvertParenListExprToParenExpr(S, base);
    if (result.isInvalid()) return ExprError();
    base = result.get();
  }

  // Handle any non-overload placeholder types in the base and index
  // expressions.  We can't handle overloads here because the other
  // operand might be an overloadable type, in which case the overload
  // resolution for the operator overload should get the first crack
  // at the overload.
  bool IsMSPropertySubscript = false;
  if (base->getType()->isNonOverloadPlaceholderType()) {
    IsMSPropertySubscript = isMSPropertySubscriptExpr(*this, base);
    if (!IsMSPropertySubscript) {
      ExprResult result = CheckPlaceholderExpr(base);
      if (result.isInvalid())
        return ExprError();
      base = result.get();
    }
  }
  if (idx->getType()->isNonOverloadPlaceholderType()) {
    ExprResult result = CheckPlaceholderExpr(idx);
    if (result.isInvalid()) return ExprError();
    idx = result.get();
  }

  // Build an unanalyzed expression if either operand is type-dependent.
  if (getLangOpts().CPlusPlus &&
      (base->isTypeDependent() || idx->isTypeDependent())) {
    return new (Context) ArraySubscriptExpr(base, idx, Context.DependentTy,
                                            VK_LValue, OK_Ordinary, rbLoc);
  }

  // MSDN, property (C++)
  // https://msdn.microsoft.com/en-us/library/yhfk0thd(v=vs.120).aspx
  // This attribute can also be used in the declaration of an empty array in a
  // class or structure definition. For example:
  // __declspec(property(get=GetX, put=PutX)) int x[];
  // The above statement indicates that x[] can be used with one or more array
  // indices. In this case, i=p->x[a][b] will be turned into i=p->GetX(a, b),
  // and p->x[a][b] = i will be turned into p->PutX(a, b, i);
  if (IsMSPropertySubscript) {
    // Build MS property subscript expression if base is MS property reference
    // or MS property subscript.
    return new (Context) MSPropertySubscriptExpr(
        base, idx, Context.PseudoObjectTy, VK_LValue, OK_Ordinary, rbLoc);
  }

  // Use C++ overloaded-operator rules if either operand has record
  // type.  The spec says to do this if either type is *overloadable*,
  // but enum types can't declare subscript operators or conversion
  // operators, so there's nothing interesting for overload resolution
  // to do if there aren't any record types involved.
  //
  // ObjC pointers have their own subscripting logic that is not tied
  // to overload resolution and so should not take this path.
  if (getLangOpts().CPlusPlus &&
      (base->getType()->isRecordType() ||
       (!base->getType()->isObjCObjectPointerType() &&
        idx->getType()->isRecordType()))) {
    return CreateOverloadedArraySubscriptExpr(lbLoc, rbLoc, base, idx);
  }

  return CreateBuiltinArraySubscriptExpr(base, lbLoc, idx, rbLoc);
}

ExprResult Sema::ActOnOMPArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                          Expr *LowerBound,
                                          SourceLocation ColonLoc, Expr *Length,
                                          SourceLocation RBLoc) {
  if (Base->getType()->isPlaceholderType() &&
      !Base->getType()->isSpecificPlaceholderType(
          BuiltinType::OMPArraySection)) {
    ExprResult Result = CheckPlaceholderExpr(Base);
    if (Result.isInvalid())
      return ExprError();
    Base = Result.get();
  }
  if (LowerBound && LowerBound->getType()->isNonOverloadPlaceholderType()) {
    ExprResult Result = CheckPlaceholderExpr(LowerBound);
    if (Result.isInvalid())
      return ExprError();
    LowerBound = Result.get();
  }
  if (Length && Length->getType()->isNonOverloadPlaceholderType()) {
    ExprResult Result = CheckPlaceholderExpr(Length);
    if (Result.isInvalid())
      return ExprError();
    Length = Result.get();
  }

  // Build an unanalyzed expression if either operand is type-dependent.
  if (Base->isTypeDependent() ||
      (LowerBound &&
       (LowerBound->isTypeDependent() || LowerBound->isValueDependent())) ||
      (Length && (Length->isTypeDependent() || Length->isValueDependent()))) {
    return new (Context)
        OMPArraySectionExpr(Base, LowerBound, Length, Context.DependentTy,
                            VK_LValue, OK_Ordinary, ColonLoc, RBLoc);
  }

  // Perform default conversions.
  QualType OriginalTy = OMPArraySectionExpr::getBaseOriginalType(Base);
  QualType ResultTy;
  if (OriginalTy->isAnyPointerType()) {
    ResultTy = OriginalTy->getPointeeType();
  } else if (OriginalTy->isArrayType()) {
    ResultTy = OriginalTy->getAsArrayTypeUnsafe()->getElementType();
  } else {
    return ExprError(
        Diag(Base->getExprLoc(), diag::err_omp_typecheck_section_value)
        << Base->getSourceRange());
  }
  // C99 6.5.2.1p1
  if (LowerBound) {
    auto Res = PerformOpenMPImplicitIntegerConversion(LowerBound->getExprLoc(),
                                                      LowerBound);
    if (Res.isInvalid())
      return ExprError(Diag(LowerBound->getExprLoc(),
                            diag::err_omp_typecheck_section_not_integer)
                       << 0 << LowerBound->getSourceRange());
    LowerBound = Res.get();

    if (LowerBound->getType()->isSpecificBuiltinType(BuiltinType::Char_S) ||
        LowerBound->getType()->isSpecificBuiltinType(BuiltinType::Char_U))
      Diag(LowerBound->getExprLoc(), diag::warn_omp_section_is_char)
          << 0 << LowerBound->getSourceRange();
  }
  if (Length) {
    auto Res =
        PerformOpenMPImplicitIntegerConversion(Length->getExprLoc(), Length);
    if (Res.isInvalid())
      return ExprError(Diag(Length->getExprLoc(),
                            diag::err_omp_typecheck_section_not_integer)
                       << 1 << Length->getSourceRange());
    Length = Res.get();

    if (Length->getType()->isSpecificBuiltinType(BuiltinType::Char_S) ||
        Length->getType()->isSpecificBuiltinType(BuiltinType::Char_U))
      Diag(Length->getExprLoc(), diag::warn_omp_section_is_char)
          << 1 << Length->getSourceRange();
  }

  // C99 6.5.2.1p1: "shall have type "pointer to *object* type". Similarly,
  // C++ [expr.sub]p1: The type "T" shall be a completely-defined object
  // type. Note that functions are not objects, and that (in C99 parlance)
  // incomplete types are not object types.
  if (ResultTy->isFunctionType()) {
    Diag(Base->getExprLoc(), diag::err_omp_section_function_type)
        << ResultTy << Base->getSourceRange();
    return ExprError();
  }

  if (RequireCompleteType(Base->getExprLoc(), ResultTy,
                          diag::err_omp_section_incomplete_type, Base))
    return ExprError();

  if (LowerBound) {
    llvm::APSInt LowerBoundValue;
    if (LowerBound->EvaluateAsInt(LowerBoundValue, Context)) {
      // OpenMP 4.0, [2.4 Array Sections]
      // The lower-bound and length must evaluate to non-negative integers.
      if (LowerBoundValue.isNegative()) {
        Diag(LowerBound->getExprLoc(), diag::err_omp_section_negative)
            << 0 << LowerBoundValue.toString(/*Radix=*/10, /*Signed=*/true)
            << LowerBound->getSourceRange();
        return ExprError();
      }
    }
  }

  if (Length) {
    llvm::APSInt LengthValue;
    if (Length->EvaluateAsInt(LengthValue, Context)) {
      // OpenMP 4.0, [2.4 Array Sections]
      // The lower-bound and length must evaluate to non-negative integers.
      if (LengthValue.isNegative()) {
        Diag(Length->getExprLoc(), diag::err_omp_section_negative)
            << 1 << LengthValue.toString(/*Radix=*/10, /*Signed=*/true)
            << Length->getSourceRange();
        return ExprError();
      }
    }
  } else if (ColonLoc.isValid() &&
             (OriginalTy.isNull() || (!OriginalTy->isConstantArrayType() &&
                                      !OriginalTy->isVariableArrayType()))) {
    // OpenMP 4.0, [2.4 Array Sections]
    // When the size of the array dimension is not known, the length must be
    // specified explicitly.
    Diag(ColonLoc, diag::err_omp_section_length_undefined)
        << (!OriginalTy.isNull() && OriginalTy->isArrayType());
    return ExprError();
  }

  return new (Context)
      OMPArraySectionExpr(Base, LowerBound, Length, Context.OMPArraySectionTy,
                          VK_LValue, OK_Ordinary, ColonLoc, RBLoc);
}

ExprResult
Sema::CreateBuiltinArraySubscriptExpr(Expr *Base, SourceLocation LLoc,
                                      Expr *Idx, SourceLocation RLoc) {
  Expr *LHSExp = Base;
  Expr *RHSExp = Idx;

  // Perform default conversions.
  if (!LHSExp->getType()->getAs<VectorType>()) {
    ExprResult Result = DefaultFunctionArrayLvalueConversion(LHSExp);
    if (Result.isInvalid())
      return ExprError();
    LHSExp = Result.get();
  }
  ExprResult Result = DefaultFunctionArrayLvalueConversion(RHSExp);
  if (Result.isInvalid())
    return ExprError();
  RHSExp = Result.get();

  QualType LHSTy = LHSExp->getType(), RHSTy = RHSExp->getType();
  ExprValueKind VK = VK_LValue;
  ExprObjectKind OK = OK_Ordinary;

  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely equivalent
  // to the expression *((e1)+(e2)). This means the array "Base" may actually be
  // in the subscript position. As a result, we need to derive the array base
  // and index from the expression types.
  Expr *BaseExpr, *IndexExpr;
  QualType ResultType;
  if (LHSTy->isDependentType() || RHSTy->isDependentType()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = Context.DependentTy;
  } else if (const PointerType *PTy = LHSTy->getAs<PointerType>()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const ObjCObjectPointerType *PTy =
               LHSTy->getAs<ObjCObjectPointerType>()) {
    BaseExpr = LHSExp;
    IndexExpr = RHSExp;

    // Use custom logic if this should be the pseudo-object subscript
    // expression.
    if (!LangOpts.isSubscriptPointerArithmetic())
      return BuildObjCSubscriptExpression(RLoc, BaseExpr, IndexExpr, nullptr,
                                          nullptr);

    ResultType = PTy->getPointeeType();
  } else if (const PointerType *PTy = RHSTy->getAs<PointerType>()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = PTy->getPointeeType();
  } else if (const ObjCObjectPointerType *PTy =
               RHSTy->getAs<ObjCObjectPointerType>()) {
     // Handle the uncommon case of "123[Ptr]".
    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = PTy->getPointeeType();
    if (!LangOpts.isSubscriptPointerArithmetic()) {
      Diag(LLoc, diag::err_subscript_nonfragile_interface)
        << ResultType << BaseExpr->getSourceRange();
      return ExprError();
    }
  } else if (const VectorType *VTy = LHSTy->getAs<VectorType>()) {
    BaseExpr = LHSExp;    // vectors: V[123]
    IndexExpr = RHSExp;
    VK = LHSExp->getValueKind();
    if (VK != VK_RValue)
      OK = OK_VectorComponent;

    // FIXME: need to deal with const...
    ResultType = VTy->getElementType();
  } else if (LHSTy->isArrayType()) {
    // If we see an array that wasn't promoted by
    // DefaultFunctionArrayLvalueConversion, it must be an array that
    // wasn't promoted because of the C90 rule that doesn't
    // allow promoting non-lvalue arrays.  Warn, then
    // force the promotion here.
    Diag(LHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        LHSExp->getSourceRange();
    LHSExp = ImpCastExprToType(LHSExp, Context.getArrayDecayedType(LHSTy),
                               CK_ArrayToPointerDecay).get();
    LHSTy = LHSExp->getType();

    BaseExpr = LHSExp;
    IndexExpr = RHSExp;
    ResultType = LHSTy->getAs<PointerType>()->getPointeeType();
  } else if (RHSTy->isArrayType()) {
    // Same as previous, except for 123[f().a] case
    Diag(RHSExp->getLocStart(), diag::ext_subscript_non_lvalue) <<
        RHSExp->getSourceRange();
    RHSExp = ImpCastExprToType(RHSExp, Context.getArrayDecayedType(RHSTy),
                               CK_ArrayToPointerDecay).get();
    RHSTy = RHSExp->getType();

    BaseExpr = RHSExp;
    IndexExpr = LHSExp;
    ResultType = RHSTy->getAs<PointerType>()->getPointeeType();
  } else {
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_value)
       << LHSExp->getSourceRange() << RHSExp->getSourceRange());
  }
  // C99 6.5.2.1p1
  if (!IndexExpr->getType()->isIntegerType() && !IndexExpr->isTypeDependent())
    return ExprError(Diag(LLoc, diag::err_typecheck_subscript_not_integer)
                     << IndexExpr->getSourceRange());

  if ((IndexExpr->getType()->isSpecificBuiltinType(BuiltinType::Char_S) ||
       IndexExpr->getType()->isSpecificBuiltinType(BuiltinType::Char_U))
         && !IndexExpr->isTypeDependent())
    Diag(LLoc, diag::warn_subscript_is_char) << IndexExpr->getSourceRange();

  // C99 6.5.2.1p1: "shall have type "pointer to *object* type". Similarly,
  // C++ [expr.sub]p1: The type "T" shall be a completely-defined object
  // type. Note that Functions are not objects, and that (in C99 parlance)
  // incomplete types are not object types.
  if (ResultType->isFunctionType()) {
    Diag(BaseExpr->getLocStart(), diag::err_subscript_function_type)
      << ResultType << BaseExpr->getSourceRange();
    return ExprError();
  }

  if (ResultType->isVoidType() && !getLangOpts().CPlusPlus) {
    // GNU extension: subscripting on pointer to void
    Diag(LLoc, diag::ext_gnu_subscript_void_type)
      << BaseExpr->getSourceRange();

    // C forbids expressions of unqualified void type from being l-values.
    // See IsCForbiddenLValueType.
    if (!ResultType.hasQualifiers()) VK = VK_RValue;
  } else if (!ResultType->isDependentType() &&
      RequireCompleteType(LLoc, ResultType,
                          diag::err_subscript_incomplete_type, BaseExpr))
    return ExprError();

  assert(VK == VK_RValue || LangOpts.CPlusPlus ||
         !ResultType.isCForbiddenLValueType());

  return new (Context)
      ArraySubscriptExpr(LHSExp, RHSExp, ResultType, VK, OK, RLoc);
}

ExprResult Sema::BuildCXXDefaultArgExpr(SourceLocation CallLoc,
                                        FunctionDecl *FD,
                                        ParmVarDecl *Param) {
  if (Param->hasUnparsedDefaultArg()) {
    Diag(CallLoc,
         diag::err_use_of_default_argument_to_function_declared_later) <<
      FD << cast<CXXRecordDecl>(FD->getDeclContext())->getDeclName();
    Diag(UnparsedDefaultArgLocs[Param],
         diag::note_default_argument_declared_here);
    return ExprError();
  }
  
  if (Param->hasUninstantiatedDefaultArg()) {
    Expr *UninstExpr = Param->getUninstantiatedDefaultArg();

    EnterExpressionEvaluationContext EvalContext(*this, PotentiallyEvaluated,
                                                 Param);

    // Instantiate the expression.
    MultiLevelTemplateArgumentList MutiLevelArgList
      = getTemplateInstantiationArgs(FD, nullptr, /*RelativeToPrimary=*/true);

    InstantiatingTemplate Inst(*this, CallLoc, Param,
                               MutiLevelArgList.getInnermost());
    if (Inst.isInvalid())
      return ExprError();

    ExprResult Result;
    {
      // C++ [dcl.fct.default]p5:
      //   The names in the [default argument] expression are bound, and
      //   the semantic constraints are checked, at the point where the
      //   default argument expression appears.
      ContextRAII SavedContext(*this, FD);
      LocalInstantiationScope Local(*this);
      Result = SubstExpr(UninstExpr, MutiLevelArgList);
    }
    if (Result.isInvalid())
      return ExprError();

    // Check the expression as an initializer for the parameter.
    InitializedEntity Entity
      = InitializedEntity::InitializeParameter(Context, Param);
    InitializationKind Kind
      = InitializationKind::CreateCopy(Param->getLocation(),
             /*FIXME:EqualLoc*/UninstExpr->getLocStart());
    Expr *ResultE = Result.getAs<Expr>();

    InitializationSequence InitSeq(*this, Entity, Kind, ResultE);
    Result = InitSeq.Perform(*this, Entity, Kind, ResultE);
    if (Result.isInvalid())
      return ExprError();

    Result = ActOnFinishFullExpr(Result.getAs<Expr>(),
                                 Param->getOuterLocStart());
    if (Result.isInvalid())
      return ExprError();

    // Remember the instantiated default argument.
    Param->setDefaultArg(Result.getAs<Expr>());
    if (ASTMutationListener *L = getASTMutationListener()) {
      L->DefaultArgumentInstantiated(Param);
    }
  }

  // If the default expression creates temporaries, we need to
  // push them to the current stack of expression temporaries so they'll
  // be properly destroyed.
  // FIXME: We should really be rebuilding the default argument with new
  // bound temporaries; see the comment in PR5810.
  // We don't need to do that with block decls, though, because
  // blocks in default argument expression can never capture anything.
  if (isa<ExprWithCleanups>(Param->getInit())) {
    // Set the "needs cleanups" bit regardless of whether there are
    // any explicit objects.
    ExprNeedsCleanups = true;

    // Append all the objects to the cleanup list.  Right now, this
    // should always be a no-op, because blocks in default argument
    // expressions should never be able to capture anything.
    assert(!cast<ExprWithCleanups>(Param->getInit())->getNumObjects() &&
           "default argument expression has capturing blocks?");
  }

  // We already type-checked the argument, so we know it works. 
  // Just mark all of the declarations in this potentially-evaluated expression
  // as being "referenced".
  MarkDeclarationsReferencedInExpr(Param->getDefaultArg(),
                                   /*SkipLocalVariables=*/true);
  return CXXDefaultArgExpr::Create(Context, CallLoc, Param);
}


Sema::VariadicCallType
Sema::getVariadicCallType(FunctionDecl *FDecl, const FunctionProtoType *Proto,
                          Expr *Fn) {
  if (Proto && Proto->isVariadic()) {
    if (dyn_cast_or_null<CXXConstructorDecl>(FDecl))
      return VariadicConstructor;
    else if (Fn && Fn->getType()->isBlockPointerType())
      return VariadicBlock;
    else if (FDecl) {
      if (CXXMethodDecl *Method = dyn_cast_or_null<CXXMethodDecl>(FDecl))
        if (Method->isInstance())
          return VariadicMethod;
    } else if (Fn && Fn->getType() == Context.BoundMemberTy)
      return VariadicMethod;
    return VariadicFunction;
  }
  return VariadicDoesNotApply;
}

namespace {
class FunctionCallCCC : public FunctionCallFilterCCC {
public:
  FunctionCallCCC(Sema &SemaRef, const IdentifierInfo *FuncName,
                  unsigned NumArgs, MemberExpr *ME)
      : FunctionCallFilterCCC(SemaRef, NumArgs, false, ME),
        FunctionName(FuncName) {}

  bool ValidateCandidate(const TypoCorrection &candidate) override {
    if (!candidate.getCorrectionSpecifier() ||
        candidate.getCorrectionAsIdentifierInfo() != FunctionName) {
      return false;
    }

    return FunctionCallFilterCCC::ValidateCandidate(candidate);
  }

private:
  const IdentifierInfo *const FunctionName;
};
}

static TypoCorrection TryTypoCorrectionForCall(Sema &S, Expr *Fn,
                                               FunctionDecl *FDecl,
                                               ArrayRef<Expr *> Args) {
  MemberExpr *ME = dyn_cast<MemberExpr>(Fn);
  DeclarationName FuncName = FDecl->getDeclName();
  SourceLocation NameLoc = ME ? ME->getMemberLoc() : Fn->getLocStart();

  if (TypoCorrection Corrected = S.CorrectTypo(
          DeclarationNameInfo(FuncName, NameLoc), Sema::LookupOrdinaryName,
          S.getScopeForContext(S.CurContext), nullptr,
          llvm::make_unique<FunctionCallCCC>(S, FuncName.getAsIdentifierInfo(),
                                             Args.size(), ME),
          Sema::CTK_ErrorRecovery)) {
    if (NamedDecl *ND = Corrected.getFoundDecl()) {
      if (Corrected.isOverloaded()) {
        OverloadCandidateSet OCS(NameLoc, OverloadCandidateSet::CSK_Normal);
        OverloadCandidateSet::iterator Best;
        for (NamedDecl *CD : Corrected) {
          if (FunctionDecl *FD = dyn_cast<FunctionDecl>(CD))
            S.AddOverloadCandidate(FD, DeclAccessPair::make(FD, AS_none), Args,
                                   OCS);
        }
        switch (OCS.BestViableFunction(S, NameLoc, Best)) {
        case OR_Success:
          ND = Best->FoundDecl;
          Corrected.setCorrectionDecl(ND);
          break;
        default:
          break;
        }
      }
      ND = ND->getUnderlyingDecl();
      if (isa<ValueDecl>(ND) || isa<FunctionTemplateDecl>(ND))
        return Corrected;
    }
  }
  return TypoCorrection();
}

/// ConvertArgumentsForCall - Converts the arguments specified in
/// Args/NumArgs to the parameter types of the function FDecl with
/// function prototype Proto. Call is the call expression itself, and
/// Fn is the function expression. For a C++ member function, this
/// routine does not attempt to convert the object argument. Returns
/// true if the call is ill-formed.
bool
Sema::ConvertArgumentsForCall(CallExpr *Call, Expr *Fn,
                              FunctionDecl *FDecl,
                              const FunctionProtoType *Proto,
                              ArrayRef<Expr *> Args,
                              SourceLocation RParenLoc,
                              bool IsExecConfig) {
  // Bail out early if calling a builtin with custom typechecking.
  if (FDecl)
    if (unsigned ID = FDecl->getBuiltinID())
      if (Context.BuiltinInfo.hasCustomTypechecking(ID))
        return false;

  // C99 6.5.2.2p7 - the arguments are implicitly converted, as if by
  // assignment, to the types of the corresponding parameter, ...
  unsigned NumParams = Proto->getNumParams();
  bool Invalid = false;
  unsigned MinArgs = FDecl ? FDecl->getMinRequiredArguments() : NumParams;
  unsigned FnKind = Fn->getType()->isBlockPointerType()
                       ? 1 /* block */
                       : (IsExecConfig ? 3 /* kernel function (exec config) */
                                       : 0 /* function */);

  // If too few arguments are available (and we don't have default
  // arguments for the remaining parameters), don't make the call.
  if (Args.size() < NumParams) {
    if (Args.size() < MinArgs) {
      TypoCorrection TC;
      if (FDecl && (TC = TryTypoCorrectionForCall(*this, Fn, FDecl, Args))) {
        unsigned diag_id =
            MinArgs == NumParams && !Proto->isVariadic()
                ? diag::err_typecheck_call_too_few_args_suggest
                : diag::err_typecheck_call_too_few_args_at_least_suggest;
        diagnoseTypo(TC, PDiag(diag_id) << FnKind << MinArgs
                                        << static_cast<unsigned>(Args.size())
                                        << TC.getCorrectionRange());
      } else if (MinArgs == 1 && FDecl && FDecl->getParamDecl(0)->getDeclName())
        Diag(RParenLoc,
             MinArgs == NumParams && !Proto->isVariadic()
                 ? diag::err_typecheck_call_too_few_args_one
                 : diag::err_typecheck_call_too_few_args_at_least_one)
            << FnKind << FDecl->getParamDecl(0) << Fn->getSourceRange();
      else
        Diag(RParenLoc, MinArgs == NumParams && !Proto->isVariadic()
                            ? diag::err_typecheck_call_too_few_args
                            : diag::err_typecheck_call_too_few_args_at_least)
            << FnKind << MinArgs << static_cast<unsigned>(Args.size())
            << Fn->getSourceRange();

      // Emit the location of the prototype.
      if (!TC && FDecl && !FDecl->getBuiltinID() && !IsExecConfig)
        Diag(FDecl->getLocStart(), diag::note_callee_decl)
          << FDecl;

      return true;
    }
    Call->setNumArgs(Context, NumParams);
  }

  // If too many are passed and not variadic, error on the extras and drop
  // them.
  if (Args.size() > NumParams) {
    if (!Proto->isVariadic()) {
      TypoCorrection TC;
      if (FDecl && (TC = TryTypoCorrectionForCall(*this, Fn, FDecl, Args))) {
        unsigned diag_id =
            MinArgs == NumParams && !Proto->isVariadic()
                ? diag::err_typecheck_call_too_many_args_suggest
                : diag::err_typecheck_call_too_many_args_at_most_suggest;
        diagnoseTypo(TC, PDiag(diag_id) << FnKind << NumParams
                                        << static_cast<unsigned>(Args.size())
                                        << TC.getCorrectionRange());
      } else if (NumParams == 1 && FDecl &&
                 FDecl->getParamDecl(0)->getDeclName())
        Diag(Args[NumParams]->getLocStart(),
             MinArgs == NumParams
                 ? diag::err_typecheck_call_too_many_args_one
                 : diag::err_typecheck_call_too_many_args_at_most_one)
            << FnKind << FDecl->getParamDecl(0)
            << static_cast<unsigned>(Args.size()) << Fn->getSourceRange()
            << SourceRange(Args[NumParams]->getLocStart(),
                           Args.back()->getLocEnd());
      else
        Diag(Args[NumParams]->getLocStart(),
             MinArgs == NumParams
                 ? diag::err_typecheck_call_too_many_args
                 : diag::err_typecheck_call_too_many_args_at_most)
            << FnKind << NumParams << static_cast<unsigned>(Args.size())
            << Fn->getSourceRange()
            << SourceRange(Args[NumParams]->getLocStart(),
                           Args.back()->getLocEnd());

      // Emit the location of the prototype.
      if (!TC && FDecl && !FDecl->getBuiltinID() && !IsExecConfig)
        Diag(FDecl->getLocStart(), diag::note_callee_decl)
          << FDecl;
      
      // This deletes the extra arguments.
      Call->setNumArgs(Context, NumParams);
      return true;
    }
  }
  SmallVector<Expr *, 8> AllArgs;
  VariadicCallType CallType = getVariadicCallType(FDecl, Proto, Fn);
  
  Invalid = GatherArgumentsForCall(Call->getLocStart(), FDecl,
                                   Proto, 0, Args, AllArgs, CallType);
  if (Invalid)
    return true;
  unsigned TotalNumArgs = AllArgs.size();
  for (unsigned i = 0; i < TotalNumArgs; ++i)
    Call->setArg(i, AllArgs[i]);

  return false;
}

bool Sema::GatherArgumentsForCall(SourceLocation CallLoc, FunctionDecl *FDecl,
                                  const FunctionProtoType *Proto,
                                  unsigned FirstParam, ArrayRef<Expr *> Args,
                                  SmallVectorImpl<Expr *> &AllArgs,
                                  VariadicCallType CallType, bool AllowExplicit,
                                  bool IsListInitialization) {
  unsigned NumParams = Proto->getNumParams();
  bool Invalid = false;
  size_t ArgIx = 0;
  // Continue to check argument types (even if we have too few/many args).
  for (unsigned i = FirstParam; i < NumParams; i++) {
    QualType ProtoArgType = Proto->getParamType(i);

    Expr *Arg;
    ParmVarDecl *Param = FDecl ? FDecl->getParamDecl(i) : nullptr;
    if (ArgIx < Args.size()) {
      Arg = Args[ArgIx++];

      if (RequireCompleteType(Arg->getLocStart(),
                              ProtoArgType,
                              diag::err_call_incomplete_argument, Arg))
        return true;

      // Strip the unbridged-cast placeholder expression off, if applicable.
      bool CFAudited = false;
      if (Arg->getType() == Context.ARCUnbridgedCastTy &&
          FDecl && FDecl->hasAttr<CFAuditedTransferAttr>() &&
          (!Param || !Param->hasAttr<CFConsumedAttr>()))
        Arg = stripARCUnbridgedCast(Arg);
      else if (getLangOpts().ObjCAutoRefCount &&
               FDecl && FDecl->hasAttr<CFAuditedTransferAttr>() &&
               (!Param || !Param->hasAttr<CFConsumedAttr>()))
        CFAudited = true;

      InitializedEntity Entity =
          Param ? InitializedEntity::InitializeParameter(Context, Param,
                                                         ProtoArgType)
                : InitializedEntity::InitializeParameter(
                      Context, ProtoArgType, Proto->isParamConsumed(i));

      // Remember that parameter belongs to a CF audited API.
      if (CFAudited)
        Entity.setParameterCFAudited();

      ExprResult ArgE = PerformCopyInitialization(
          Entity, SourceLocation(), Arg, IsListInitialization, AllowExplicit);
      if (ArgE.isInvalid())
        return true;

      Arg = ArgE.getAs<Expr>();
    } else {
      assert(Param && "can't use default arguments without a known callee");

      ExprResult ArgExpr =
        BuildCXXDefaultArgExpr(CallLoc, FDecl, Param);
      if (ArgExpr.isInvalid())
        return true;

      Arg = ArgExpr.getAs<Expr>();
    }

    // Check for array bounds violations for each argument to the call. This
    // check only triggers warnings when the argument isn't a more complex Expr
    // with its own checking, such as a BinaryOperator.
    CheckArrayAccess(Arg);

    // Check for violations of C99 static array rules (C99 6.7.5.3p7).
    CheckStaticArrayArgument(CallLoc, Param, Arg);

    AllArgs.push_back(Arg);
  }

  // If this is a variadic call, handle args passed through "...".
  if (CallType != VariadicDoesNotApply) {
    // Assume that extern "C" functions with variadic arguments that
    // return __unknown_anytype aren't *really* variadic.
    if (Proto->getReturnType() == Context.UnknownAnyTy && FDecl &&
        FDecl->isExternC()) {
      for (Expr *A : Args.slice(ArgIx)) {
        QualType paramType; // ignored
        ExprResult arg = checkUnknownAnyArg(CallLoc, A, paramType);
        Invalid |= arg.isInvalid();
        AllArgs.push_back(arg.get());
      }

    // Otherwise do argument promotion, (C99 6.5.2.2p7).
    } else {
      for (Expr *A : Args.slice(ArgIx)) {
        ExprResult Arg = DefaultVariadicArgumentPromotion(A, CallType, FDecl);
        Invalid |= Arg.isInvalid();
        AllArgs.push_back(Arg.get());
      }
    }

    // Check for array bounds violations.
    for (Expr *A : Args.slice(ArgIx))
      CheckArrayAccess(A);
  }
  return Invalid;
}

static void DiagnoseCalleeStaticArrayParam(Sema &S, ParmVarDecl *PVD) {
  TypeLoc TL = PVD->getTypeSourceInfo()->getTypeLoc();
  if (DecayedTypeLoc DTL = TL.getAs<DecayedTypeLoc>())
    TL = DTL.getOriginalLoc();
  if (ArrayTypeLoc ATL = TL.getAs<ArrayTypeLoc>())
    S.Diag(PVD->getLocation(), diag::note_callee_static_array)
      << ATL.getLocalSourceRange();
}

/// CheckStaticArrayArgument - If the given argument corresponds to a static
/// array parameter, check that it is non-null, and that if it is formed by
/// array-to-pointer decay, the underlying array is sufficiently large.
///
/// C99 6.7.5.3p7: If the keyword static also appears within the [ and ] of the
/// array type derivation, then for each call to the function, the value of the
/// corresponding actual argument shall provide access to the first element of
/// an array with at least as many elements as specified by the size expression.
void
Sema::CheckStaticArrayArgument(SourceLocation CallLoc,
                               ParmVarDecl *Param,
                               const Expr *ArgExpr) {
  // Static array parameters are not supported in C++.
  if (!Param || getLangOpts().CPlusPlus)
    return;

  QualType OrigTy = Param->getOriginalType();

  const ArrayType *AT = Context.getAsArrayType(OrigTy);
  if (!AT || AT->getSizeModifier() != ArrayType::Static)
    return;

  if (ArgExpr->isNullPointerConstant(Context,
                                     Expr::NPC_NeverValueDependent)) {
    Diag(CallLoc, diag::warn_null_arg) << ArgExpr->getSourceRange();
    DiagnoseCalleeStaticArrayParam(*this, Param);
    return;
  }

  const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT);
  if (!CAT)
    return;

  const ConstantArrayType *ArgCAT =
    Context.getAsConstantArrayType(ArgExpr->IgnoreParenImpCasts()->getType());
  if (!ArgCAT)
    return;

  if (ArgCAT->getSize().ult(CAT->getSize())) {
    Diag(CallLoc, diag::warn_static_array_too_small)
      << ArgExpr->getSourceRange()
      << (unsigned) ArgCAT->getSize().getZExtValue()
      << (unsigned) CAT->getSize().getZExtValue();
    DiagnoseCalleeStaticArrayParam(*this, Param);
  }
}

/// Given a function expression of unknown-any type, try to rebuild it
/// to have a function type.
static ExprResult rebuildUnknownAnyFunction(Sema &S, Expr *fn);

/// Is the given type a placeholder that we need to lower out
/// immediately during argument processing?
static bool isPlaceholderToRemoveAsArg(QualType type) {
  // Placeholders are never sugared.
  const BuiltinType *placeholder = dyn_cast<BuiltinType>(type);
  if (!placeholder) return false;

  switch (placeholder->getKind()) {
  // Ignore all the non-placeholder types.
#define PLACEHOLDER_TYPE(ID, SINGLETON_ID)
#define BUILTIN_TYPE(ID, SINGLETON_ID) case BuiltinType::ID:
#include "clang/AST/BuiltinTypes.def"
    return false;

  // We cannot lower out overload sets; they might validly be resolved
  // by the call machinery.
  case BuiltinType::Overload:
    return false;

  // Unbridged casts in ARC can be handled in some call positions and
  // should be left in place.
  case BuiltinType::ARCUnbridgedCast:
    return false;

  // Pseudo-objects should be converted as soon as possible.
  case BuiltinType::PseudoObject:
    return true;

  // The debugger mode could theoretically but currently does not try
  // to resolve unknown-typed arguments based on known parameter types.
  case BuiltinType::UnknownAny:
    return true;

  // These are always invalid as call arguments and should be reported.
  case BuiltinType::BoundMember:
  case BuiltinType::BuiltinFn:
  case BuiltinType::OMPArraySection:
    return true;

  }
  llvm_unreachable("bad builtin type kind");
}

/// Check an argument list for placeholders that we won't try to
/// handle later.
static bool checkArgsForPlaceholders(Sema &S, MultiExprArg args) {
  // Apply this processing to all the arguments at once instead of
  // dying at the first failure.
  bool hasInvalid = false;
  for (size_t i = 0, e = args.size(); i != e; i++) {
    if (isPlaceholderToRemoveAsArg(args[i]->getType())) {
      ExprResult result = S.CheckPlaceholderExpr(args[i]);
      if (result.isInvalid()) hasInvalid = true;
      else args[i] = result.get();
    } else if (hasInvalid) {
      (void)S.CorrectDelayedTyposInExpr(args[i]);
    }
  }
  return hasInvalid;
}

/// If a builtin function has a pointer argument with no explicit address
/// space, then it should be able to accept a pointer to any address
/// space as input.  In order to do this, we need to replace the
/// standard builtin declaration with one that uses the same address space
/// as the call.
///
/// \returns nullptr If this builtin is not a candidate for a rewrite i.e.
///                  it does not contain any pointer arguments without
///                  an address space qualifer.  Otherwise the rewritten
///                  FunctionDecl is returned.
/// TODO: Handle pointer return types.
static FunctionDecl *rewriteBuiltinFunctionDecl(Sema *Sema, ASTContext &Context,
                                                const FunctionDecl *FDecl,
                                                MultiExprArg ArgExprs) {

  QualType DeclType = FDecl->getType();
  const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(DeclType);

  if (!Context.BuiltinInfo.hasPtrArgsOrResult(FDecl->getBuiltinID()) ||
      !FT || FT->isVariadic() || ArgExprs.size() != FT->getNumParams())
    return nullptr;

  bool NeedsNewDecl = false;
  unsigned i = 0;
  SmallVector<QualType, 8> OverloadParams;

  for (QualType ParamType : FT->param_types()) {

    // Convert array arguments to pointer to simplify type lookup.
    Expr *Arg = Sema->DefaultFunctionArrayLvalueConversion(ArgExprs[i++]).get();
    QualType ArgType = Arg->getType();
    if (!ParamType->isPointerType() ||
        ParamType.getQualifiers().hasAddressSpace() ||
        !ArgType->isPointerType() ||
        !ArgType->getPointeeType().getQualifiers().hasAddressSpace()) {
      OverloadParams.push_back(ParamType);
      continue;
    }

    NeedsNewDecl = true;
    unsigned AS = ArgType->getPointeeType().getQualifiers().getAddressSpace();

    QualType PointeeType = ParamType->getPointeeType();
    PointeeType = Context.getAddrSpaceQualType(PointeeType, AS);
    OverloadParams.push_back(Context.getPointerType(PointeeType));
  }

  if (!NeedsNewDecl)
    return nullptr;

  FunctionProtoType::ExtProtoInfo EPI;
  QualType OverloadTy = Context.getFunctionType(FT->getReturnType(),
                                                OverloadParams, EPI);
  DeclContext *Parent = Context.getTranslationUnitDecl();
  FunctionDecl *OverloadDecl = FunctionDecl::Create(Context, Parent,
                                                    FDecl->getLocation(),
                                                    FDecl->getLocation(),
                                                    FDecl->getIdentifier(),
                                                    OverloadTy,
                                                    /*TInfo=*/nullptr,
                                                    SC_Extern, false,
                                                    /*hasPrototype=*/true);
  SmallVector<ParmVarDecl*, 16> Params;
  FT = cast<FunctionProtoType>(OverloadTy);
  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
    QualType ParamType = FT->getParamType(i);
    ParmVarDecl *Parm =
        ParmVarDecl::Create(Context, OverloadDecl, SourceLocation(),
                                SourceLocation(), nullptr, ParamType,
                                /*TInfo=*/nullptr, SC_None, nullptr);
    Parm->setScopeInfo(0, i);
    Params.push_back(Parm);
  }
  OverloadDecl->setParams(Params);
  return OverloadDecl;
}

/// ActOnCallExpr - Handle a call to Fn with the specified array of arguments.
/// This provides the location of the left/right parens and a list of comma
/// locations.
ExprResult
Sema::ActOnCallExpr(Scope *S, Expr *Fn, SourceLocation LParenLoc,
                    MultiExprArg ArgExprs, SourceLocation RParenLoc,
                    Expr *ExecConfig, bool IsExecConfig) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Fn);
  if (Result.isInvalid()) return ExprError();
  Fn = Result.get();

  if (checkArgsForPlaceholders(*this, ArgExprs))
    return ExprError();

  if (getLangOpts().CPlusPlus) {
    // If this is a pseudo-destructor expression, build the call immediately.
    if (isa<CXXPseudoDestructorExpr>(Fn)) {
      if (!ArgExprs.empty()) {
        // Pseudo-destructor calls should not have any arguments.
        Diag(Fn->getLocStart(), diag::err_pseudo_dtor_call_with_args)
          << FixItHint::CreateRemoval(
                                    SourceRange(ArgExprs.front()->getLocStart(),
                                                ArgExprs.back()->getLocEnd()));
      }

      return new (Context)
          CallExpr(Context, Fn, None, Context.VoidTy, VK_RValue, RParenLoc);
    }
    if (Fn->getType() == Context.PseudoObjectTy) {
      ExprResult result = CheckPlaceholderExpr(Fn);
      if (result.isInvalid()) return ExprError();
      Fn = result.get();
    }

    // Determine whether this is a dependent call inside a C++ template,
    // in which case we won't do any semantic analysis now.
    // FIXME: Will need to cache the results of name lookup (including ADL) in
    // Fn.
    bool Dependent = false;
    if (Fn->isTypeDependent())
      Dependent = true;
    else if (Expr::hasAnyTypeDependentArguments(ArgExprs))
      Dependent = true;

    if (Dependent) {
      if (ExecConfig) {
        return new (Context) CUDAKernelCallExpr(
            Context, Fn, cast<CallExpr>(ExecConfig), ArgExprs,
            Context.DependentTy, VK_RValue, RParenLoc);
      } else {
        return new (Context) CallExpr(
            Context, Fn, ArgExprs, Context.DependentTy, VK_RValue, RParenLoc);
      }
    }

    // Determine whether this is a call to an object (C++ [over.call.object]).
    if (Fn->getType()->isRecordType())
      return BuildCallToObjectOfClassType(S, Fn, LParenLoc, ArgExprs,
                                          RParenLoc);

    if (Fn->getType() == Context.UnknownAnyTy) {
      ExprResult result = rebuildUnknownAnyFunction(*this, Fn);
      if (result.isInvalid()) return ExprError();
      Fn = result.get();
    }

    if (Fn->getType() == Context.BoundMemberTy) {
      return BuildCallToMemberFunction(S, Fn, LParenLoc, ArgExprs, RParenLoc);
    }
  }

  // Check for overloaded calls.  This can happen even in C due to extensions.
  if (Fn->getType() == Context.OverloadTy) {
    OverloadExpr::FindResult find = OverloadExpr::find(Fn);

    // We aren't supposed to apply this logic for if there's an '&' involved.
    if (!find.HasFormOfMemberPointer) {
      OverloadExpr *ovl = find.Expression;
      if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(ovl))
        return BuildOverloadedCallExpr(S, Fn, ULE, LParenLoc, ArgExprs,
                                       RParenLoc, ExecConfig,
                                       /*AllowTypoCorrection=*/true,
                                       find.IsAddressOfOperand);
      return BuildCallToMemberFunction(S, Fn, LParenLoc, ArgExprs, RParenLoc);
    }
  }

  // If we're directly calling a function, get the appropriate declaration.
  if (Fn->getType() == Context.UnknownAnyTy) {
    ExprResult result = rebuildUnknownAnyFunction(*this, Fn);
    if (result.isInvalid()) return ExprError();
    Fn = result.get();
  }

  Expr *NakedFn = Fn->IgnoreParens();

  bool CallingNDeclIndirectly = false;
  NamedDecl *NDecl = nullptr;
  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(NakedFn)) {
    if (UnOp->getOpcode() == UO_AddrOf) {
      CallingNDeclIndirectly = true;
      NakedFn = UnOp->getSubExpr()->IgnoreParens();
    }
  }

  if (isa<DeclRefExpr>(NakedFn)) {
    NDecl = cast<DeclRefExpr>(NakedFn)->getDecl();

    FunctionDecl *FDecl = dyn_cast<FunctionDecl>(NDecl);
    if (FDecl && FDecl->getBuiltinID()) {
      // Rewrite the function decl for this builtin by replacing parameters
      // with no explicit address space with the address space of the arguments
      // in ArgExprs.
      if ((FDecl = rewriteBuiltinFunctionDecl(this, Context, FDecl, ArgExprs))) {
        NDecl = FDecl;
        Fn = DeclRefExpr::Create(Context, FDecl->getQualifierLoc(),
                           SourceLocation(), FDecl, false,
                           SourceLocation(), FDecl->getType(),
                           Fn->getValueKind(), FDecl);
      }
    }
  } else if (isa<MemberExpr>(NakedFn))
    NDecl = cast<MemberExpr>(NakedFn)->getMemberDecl();

  if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(NDecl)) {
    if (CallingNDeclIndirectly &&
        !checkAddressOfFunctionIsAvailable(FD, /*Complain=*/true,
                                           Fn->getLocStart()))
      return ExprError();

    if (FD->hasAttr<EnableIfAttr>()) {
      if (const EnableIfAttr *Attr = CheckEnableIf(FD, ArgExprs, true)) {
        Diag(Fn->getLocStart(),
             isa<CXXMethodDecl>(FD) ?
                 diag::err_ovl_no_viable_member_function_in_call :
                 diag::err_ovl_no_viable_function_in_call)
          << FD << FD->getSourceRange();
        Diag(FD->getLocation(),
             diag::note_ovl_candidate_disabled_by_enable_if_attr)
            << Attr->getCond()->getSourceRange() << Attr->getMessage();
      }
    }
  }

  return BuildResolvedCallExpr(Fn, NDecl, LParenLoc, ArgExprs, RParenLoc,
                               ExecConfig, IsExecConfig);
}

/// ActOnAsTypeExpr - create a new asType (bitcast) from the arguments.
///
/// __builtin_astype( value, dst type )
///
ExprResult Sema::ActOnAsTypeExpr(Expr *E, ParsedType ParsedDestTy,
                                 SourceLocation BuiltinLoc,
                                 SourceLocation RParenLoc) {
  ExprValueKind VK = VK_RValue;
  ExprObjectKind OK = OK_Ordinary;
  QualType DstTy = GetTypeFromParser(ParsedDestTy);
  QualType SrcTy = E->getType();
  if (Context.getTypeSize(DstTy) != Context.getTypeSize(SrcTy))
    return ExprError(Diag(BuiltinLoc,
                          diag::err_invalid_astype_of_different_size)
                     << DstTy
                     << SrcTy
                     << E->getSourceRange());
  return new (Context) AsTypeExpr(E, DstTy, VK, OK, BuiltinLoc, RParenLoc);
}

/// ActOnConvertVectorExpr - create a new convert-vector expression from the
/// provided arguments.
///
/// __builtin_convertvector( value, dst type )
///
ExprResult Sema::ActOnConvertVectorExpr(Expr *E, ParsedType ParsedDestTy,
                                        SourceLocation BuiltinLoc,
                                        SourceLocation RParenLoc) {
  TypeSourceInfo *TInfo;
  GetTypeFromParser(ParsedDestTy, &TInfo);
  return SemaConvertVectorExpr(E, TInfo, BuiltinLoc, RParenLoc);
}

/// BuildResolvedCallExpr - Build a call to a resolved expression,
/// i.e. an expression not of \p OverloadTy.  The expression should
/// unary-convert to an expression of function-pointer or
/// block-pointer type.
///
/// \param NDecl the declaration being called, if available
ExprResult
Sema::BuildResolvedCallExpr(Expr *Fn, NamedDecl *NDecl,
                            SourceLocation LParenLoc,
                            ArrayRef<Expr *> Args,
                            SourceLocation RParenLoc,
                            Expr *Config, bool IsExecConfig) {
  FunctionDecl *FDecl = dyn_cast_or_null<FunctionDecl>(NDecl);
  unsigned BuiltinID = (FDecl ? FDecl->getBuiltinID() : 0);

  // Functions with 'interrupt' attribute cannot be called directly.
  if (FDecl && FDecl->hasAttr<AnyX86InterruptAttr>()) {
    Diag(Fn->getExprLoc(), diag::err_anyx86_interrupt_called);
    return ExprError();
  }

  // Promote the function operand.
  // We special-case function promotion here because we only allow promoting
  // builtin functions to function pointers in the callee of a call.
  ExprResult Result;
  if (BuiltinID &&
      Fn->getType()->isSpecificBuiltinType(BuiltinType::BuiltinFn)) {
    Result = ImpCastExprToType(Fn, Context.getPointerType(FDecl->getType()),
                               CK_BuiltinFnToFnPtr).get();
  } else {
    Result = CallExprUnaryConversions(Fn);
  }
  if (Result.isInvalid())
    return ExprError();
  Fn = Result.get();

  // Make the call expr early, before semantic checks.  This guarantees cleanup
  // of arguments and function on error.
  CallExpr *TheCall;
  if (Config)
    TheCall = new (Context) CUDAKernelCallExpr(Context, Fn,
                                               cast<CallExpr>(Config), Args,
                                               Context.BoolTy, VK_RValue,
                                               RParenLoc);
  else
    TheCall = new (Context) CallExpr(Context, Fn, Args, Context.BoolTy,
                                     VK_RValue, RParenLoc);

  if (!getLangOpts().CPlusPlus) {
    // C cannot always handle TypoExpr nodes in builtin calls and direct
    // function calls as their argument checking don't necessarily handle
    // dependent types properly, so make sure any TypoExprs have been
    // dealt with.
    ExprResult Result = CorrectDelayedTyposInExpr(TheCall);
    if (!Result.isUsable()) return ExprError();
    TheCall = dyn_cast<CallExpr>(Result.get());
    if (!TheCall) return Result;
    Args = llvm::makeArrayRef(TheCall->getArgs(), TheCall->getNumArgs());
  }

  // Bail out early if calling a builtin with custom typechecking.
  if (BuiltinID && Context.BuiltinInfo.hasCustomTypechecking(BuiltinID))
    return CheckBuiltinFunctionCall(FDecl, BuiltinID, TheCall);

 retry:
  const FunctionType *FuncT;
  if (const PointerType *PT = Fn->getType()->getAs<PointerType>()) {
    // C99 6.5.2.2p1 - "The expression that denotes the called function shall
    // have type pointer to function".
    FuncT = PT->getPointeeType()->getAs<FunctionType>();
    if (!FuncT)
      return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
                         << Fn->getType() << Fn->getSourceRange());
  } else if (const BlockPointerType *BPT =
               Fn->getType()->getAs<BlockPointerType>()) {
    FuncT = BPT->getPointeeType()->castAs<FunctionType>();
  } else {
    // Handle calls to expressions of unknown-any type.
    if (Fn->getType() == Context.UnknownAnyTy) {
      ExprResult rewrite = rebuildUnknownAnyFunction(*this, Fn);
      if (rewrite.isInvalid()) return ExprError();
      Fn = rewrite.get();
      TheCall->setCallee(Fn);
      goto retry;
    }

    return ExprError(Diag(LParenLoc, diag::err_typecheck_call_not_function)
      << Fn->getType() << Fn->getSourceRange());
  }

  if (getLangOpts().CUDA) {
    if (Config) {
      // CUDA: Kernel calls must be to global functions
      if (FDecl && !FDecl->hasAttr<CUDAGlobalAttr>())
        return ExprError(Diag(LParenLoc,diag::err_kern_call_not_global_function)
            << FDecl->getName() << Fn->getSourceRange());

      // CUDA: Kernel function must have 'void' return type
      if (!FuncT->getReturnType()->isVoidType())
        return ExprError(Diag(LParenLoc, diag::err_kern_type_not_void_return)
            << Fn->getType() << Fn->getSourceRange());
    } else {
      // CUDA: Calls to global functions must be configured
      if (FDecl && FDecl->hasAttr<CUDAGlobalAttr>())
        return ExprError(Diag(LParenLoc, diag::err_global_call_not_config)
            << FDecl->getName() << Fn->getSourceRange());
    }
  }

  // Check for a valid return type
  if (CheckCallReturnType(FuncT->getReturnType(), Fn->getLocStart(), TheCall,
                          FDecl))
    return ExprError();

  // We know the result type of the call, set it.
  TheCall->setType(FuncT->getCallResultType(Context));
  TheCall->setValueKind(Expr::getValueKindForType(FuncT->getReturnType()));

  const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FuncT);
  if (Proto) {
    if (ConvertArgumentsForCall(TheCall, Fn, FDecl, Proto, Args, RParenLoc,
                                IsExecConfig))
      return ExprError();
  } else {
    assert(isa<FunctionNoProtoType>(FuncT) && "Unknown FunctionType!");

    if (FDecl) {
      // Check if we have too few/too many template arguments, based
      // on our knowledge of the function definition.
      const FunctionDecl *Def = nullptr;
      if (FDecl->hasBody(Def) && Args.size() != Def->param_size()) {
        Proto = Def->getType()->getAs<FunctionProtoType>();
       if (!Proto || !(Proto->isVariadic() && Args.size() >= Def->param_size()))
          Diag(RParenLoc, diag::warn_call_wrong_number_of_arguments)
          << (Args.size() > Def->param_size()) << FDecl << Fn->getSourceRange();
      }
      
      // If the function we're calling isn't a function prototype, but we have
      // a function prototype from a prior declaratiom, use that prototype.
      if (!FDecl->hasPrototype())
        Proto = FDecl->getType()->getAs<FunctionProtoType>();
    }

    // Promote the arguments (C99 6.5.2.2p6).
    for (unsigned i = 0, e = Args.size(); i != e; i++) {
      Expr *Arg = Args[i];

      if (Proto && i < Proto->getNumParams()) {
        InitializedEntity Entity = InitializedEntity::InitializeParameter(
            Context, Proto->getParamType(i), Proto->isParamConsumed(i));
        ExprResult ArgE =
            PerformCopyInitialization(Entity, SourceLocation(), Arg);
        if (ArgE.isInvalid())
          return true;
        
        Arg = ArgE.getAs<Expr>();

      } else {
        ExprResult ArgE = DefaultArgumentPromotion(Arg);

        if (ArgE.isInvalid())
          return true;

        Arg = ArgE.getAs<Expr>();
      }
      
      if (RequireCompleteType(Arg->getLocStart(),
                              Arg->getType(),
                              diag::err_call_incomplete_argument, Arg))
        return ExprError();

      TheCall->setArg(i, Arg);
    }
  }

  if (CXXMethodDecl *Method = dyn_cast_or_null<CXXMethodDecl>(FDecl))
    if (!Method->isStatic())
      return ExprError(Diag(LParenLoc, diag::err_member_call_without_object)
        << Fn->getSourceRange());

  // Check for sentinels
  if (NDecl)
    DiagnoseSentinelCalls(NDecl, LParenLoc, Args);

  // Do special checking on direct calls to functions.
  if (FDecl) {
    if (CheckFunctionCall(FDecl, TheCall, Proto))
      return ExprError();

    if (BuiltinID)
      return CheckBuiltinFunctionCall(FDecl, BuiltinID, TheCall);
  } else if (NDecl) {
    if (CheckPointerCall(NDecl, TheCall, Proto))
      return ExprError();
  } else {
    if (CheckOtherCall(TheCall, Proto))
      return ExprError();
  }

  return MaybeBindToTemporary(TheCall);
}

ExprResult
Sema::ActOnCompoundLiteral(SourceLocation LParenLoc, ParsedType Ty,
                           SourceLocation RParenLoc, Expr *InitExpr) {
  assert(Ty && "ActOnCompoundLiteral(): missing type");
  assert(InitExpr && "ActOnCompoundLiteral(): missing expression");

  TypeSourceInfo *TInfo;
  QualType literalType = GetTypeFromParser(Ty, &TInfo);
  if (!TInfo)
    TInfo = Context.getTrivialTypeSourceInfo(literalType);

  return BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc, InitExpr);
}

ExprResult
Sema::BuildCompoundLiteralExpr(SourceLocation LParenLoc, TypeSourceInfo *TInfo,
                               SourceLocation RParenLoc, Expr *LiteralExpr) {
  QualType literalType = TInfo->getType();

  if (literalType->isArrayType()) {
    if (RequireCompleteType(LParenLoc, Context.getBaseElementType(literalType),
          diag::err_illegal_decl_array_incomplete_type,
          SourceRange(LParenLoc,
                      LiteralExpr->getSourceRange().getEnd())))
      return ExprError();
    if (literalType->isVariableArrayType())
      return ExprError(Diag(LParenLoc, diag::err_variable_object_no_init)
        << SourceRange(LParenLoc, LiteralExpr->getSourceRange().getEnd()));
  } else if (!literalType->isDependentType() &&
             RequireCompleteType(LParenLoc, literalType,
               diag::err_typecheck_decl_incomplete_type,
               SourceRange(LParenLoc, LiteralExpr->getSourceRange().getEnd())))
    return ExprError();

  InitializedEntity Entity
    = InitializedEntity::InitializeCompoundLiteralInit(TInfo);
  InitializationKind Kind
    = InitializationKind::CreateCStyleCast(LParenLoc, 
                                           SourceRange(LParenLoc, RParenLoc),
                                           /*InitList=*/true);
  InitializationSequence InitSeq(*this, Entity, Kind, LiteralExpr);
  ExprResult Result = InitSeq.Perform(*this, Entity, Kind, LiteralExpr,
                                      &literalType);
  if (Result.isInvalid())
    return ExprError();
  LiteralExpr = Result.get();

  bool isFileScope = getCurFunctionOrMethodDecl() == nullptr;
  if (isFileScope &&
      !LiteralExpr->isTypeDependent() &&
      !LiteralExpr->isValueDependent() &&
      !literalType->isDependentType()) { // 6.5.2.5p3
    if (CheckForConstantInitializer(LiteralExpr, literalType))
      return ExprError();
  }

  // In C, compound literals are l-values for some reason.
  ExprValueKind VK = getLangOpts().CPlusPlus ? VK_RValue : VK_LValue;

  return MaybeBindToTemporary(
           new (Context) CompoundLiteralExpr(LParenLoc, TInfo, literalType,
                                             VK, LiteralExpr, isFileScope));
}

ExprResult
Sema::ActOnInitList(SourceLocation LBraceLoc, MultiExprArg InitArgList,
                    SourceLocation RBraceLoc) {
  // Immediately handle non-overload placeholders.  Overloads can be
  // resolved contextually, but everything else here can't.
  for (unsigned I = 0, E = InitArgList.size(); I != E; ++I) {
    if (InitArgList[I]->getType()->isNonOverloadPlaceholderType()) {
      ExprResult result = CheckPlaceholderExpr(InitArgList[I]);

      // Ignore failures; dropping the entire initializer list because
      // of one failure would be terrible for indexing/etc.
      if (result.isInvalid()) continue;

      InitArgList[I] = result.get();
    }
  }

  // Semantic analysis for initializers is done by ActOnDeclarator() and
  // CheckInitializer() - it requires knowledge of the object being intialized.

  InitListExpr *E = new (Context) InitListExpr(Context, LBraceLoc, InitArgList,
                                               RBraceLoc);
  E->setType(Context.VoidTy); // FIXME: just a place holder for now.
  return E;
}

/// Do an explicit extend of the given block pointer if we're in ARC.
void Sema::maybeExtendBlockObject(ExprResult &E) {
  assert(E.get()->getType()->isBlockPointerType());
  assert(E.get()->isRValue());

  // Only do this in an r-value context.
  if (!getLangOpts().ObjCAutoRefCount) return;

  E = ImplicitCastExpr::Create(Context, E.get()->getType(),
                               CK_ARCExtendBlockObject, E.get(),
                               /*base path*/ nullptr, VK_RValue);
  ExprNeedsCleanups = true;
}

/// Prepare a conversion of the given expression to an ObjC object
/// pointer type.
CastKind Sema::PrepareCastToObjCObjectPointer(ExprResult &E) {
  QualType type = E.get()->getType();
  if (type->isObjCObjectPointerType()) {
    return CK_BitCast;
  } else if (type->isBlockPointerType()) {
    maybeExtendBlockObject(E);
    return CK_BlockPointerToObjCPointerCast;
  } else {
    assert(type->isPointerType());
    return CK_CPointerToObjCPointerCast;
  }
}

/// Prepares for a scalar cast, performing all the necessary stages
/// except the final cast and returning the kind required.
CastKind Sema::PrepareScalarCast(ExprResult &Src, QualType DestTy) {
  // Both Src and Dest are scalar types, i.e. arithmetic or pointer.
  // Also, callers should have filtered out the invalid cases with
  // pointers.  Everything else should be possible.

  QualType SrcTy = Src.get()->getType();
  if (Context.hasSameUnqualifiedType(SrcTy, DestTy))
    return CK_NoOp;

  switch (Type::ScalarTypeKind SrcKind = SrcTy->getScalarTypeKind()) {
  case Type::STK_MemberPointer:
    llvm_unreachable("member pointer type in C");

  case Type::STK_CPointer:
  case Type::STK_BlockPointer:
  case Type::STK_ObjCObjectPointer:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_CPointer: {
      unsigned SrcAS = SrcTy->getPointeeType().getAddressSpace();
      unsigned DestAS = DestTy->getPointeeType().getAddressSpace();
      if (SrcAS != DestAS)
        return CK_AddressSpaceConversion;
      return CK_BitCast;
    }
    case Type::STK_BlockPointer:
      return (SrcKind == Type::STK_BlockPointer
                ? CK_BitCast : CK_AnyPointerToBlockPointerCast);
    case Type::STK_ObjCObjectPointer:
      if (SrcKind == Type::STK_ObjCObjectPointer)
        return CK_BitCast;
      if (SrcKind == Type::STK_CPointer)
        return CK_CPointerToObjCPointerCast;
      maybeExtendBlockObject(Src);
      return CK_BlockPointerToObjCPointerCast;
    case Type::STK_Bool:
      return CK_PointerToBoolean;
    case Type::STK_Integral:
      return CK_PointerToIntegral;
    case Type::STK_Floating:
    case Type::STK_FloatingComplex:
    case Type::STK_IntegralComplex:
    case Type::STK_MemberPointer:
      llvm_unreachable("illegal cast from pointer");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_Bool: // casting from bool is like casting from an integer
  case Type::STK_Integral:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
      if (Src.get()->isNullPointerConstant(Context,
                                           Expr::NPC_ValueDependentIsNull))
        return CK_NullToPointer;
      return CK_IntegralToPointer;
    case Type::STK_Bool:
      return CK_IntegralToBoolean;
    case Type::STK_Integral:
      return CK_IntegralCast;
    case Type::STK_Floating:
      return CK_IntegralToFloating;
    case Type::STK_IntegralComplex:
      Src = ImpCastExprToType(Src.get(),
                      DestTy->castAs<ComplexType>()->getElementType(),
                      CK_IntegralCast);
      return CK_IntegralRealToComplex;
    case Type::STK_FloatingComplex:
      Src = ImpCastExprToType(Src.get(),
                      DestTy->castAs<ComplexType>()->getElementType(),
                      CK_IntegralToFloating);
      return CK_FloatingRealToComplex;
    case Type::STK_MemberPointer:
      llvm_unreachable("member pointer type in C");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_Floating:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_Floating:
      return CK_FloatingCast;
    case Type::STK_Bool:
      return CK_FloatingToBoolean;
    case Type::STK_Integral:
      return CK_FloatingToIntegral;
    case Type::STK_FloatingComplex:
      Src = ImpCastExprToType(Src.get(),
                              DestTy->castAs<ComplexType>()->getElementType(),
                              CK_FloatingCast);
      return CK_FloatingRealToComplex;
    case Type::STK_IntegralComplex:
      Src = ImpCastExprToType(Src.get(),
                              DestTy->castAs<ComplexType>()->getElementType(),
                              CK_FloatingToIntegral);
      return CK_IntegralRealToComplex;
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
      llvm_unreachable("valid float->pointer cast?");
    case Type::STK_MemberPointer:
      llvm_unreachable("member pointer type in C");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_FloatingComplex:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_FloatingComplex:
      return CK_FloatingComplexCast;
    case Type::STK_IntegralComplex:
      return CK_FloatingComplexToIntegralComplex;
    case Type::STK_Floating: {
      QualType ET = SrcTy->castAs<ComplexType>()->getElementType();
      if (Context.hasSameType(ET, DestTy))
        return CK_FloatingComplexToReal;
      Src = ImpCastExprToType(Src.get(), ET, CK_FloatingComplexToReal);
      return CK_FloatingCast;
    }
    case Type::STK_Bool:
      return CK_FloatingComplexToBoolean;
    case Type::STK_Integral:
      Src = ImpCastExprToType(Src.get(),
                              SrcTy->castAs<ComplexType>()->getElementType(),
                              CK_FloatingComplexToReal);
      return CK_FloatingToIntegral;
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
      llvm_unreachable("valid complex float->pointer cast?");
    case Type::STK_MemberPointer:
      llvm_unreachable("member pointer type in C");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_IntegralComplex:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_FloatingComplex:
      return CK_IntegralComplexToFloatingComplex;
    case Type::STK_IntegralComplex:
      return CK_IntegralComplexCast;
    case Type::STK_Integral: {
      QualType ET = SrcTy->castAs<ComplexType>()->getElementType();
      if (Context.hasSameType(ET, DestTy))
        return CK_IntegralComplexToReal;
      Src = ImpCastExprToType(Src.get(), ET, CK_IntegralComplexToReal);
      return CK_IntegralCast;
    }
    case Type::STK_Bool:
      return CK_IntegralComplexToBoolean;
    case Type::STK_Floating:
      Src = ImpCastExprToType(Src.get(),
                              SrcTy->castAs<ComplexType>()->getElementType(),
                              CK_IntegralComplexToReal);
      return CK_IntegralToFloating;
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
      llvm_unreachable("valid complex int->pointer cast?");
    case Type::STK_MemberPointer:
      llvm_unreachable("member pointer type in C");
    }
    llvm_unreachable("Should have returned before this");
  }

  llvm_unreachable("Unhandled scalar cast");
}

static bool breakDownVectorType(QualType type, uint64_t &len,
                                QualType &eltType) {
  // Vectors are simple.
  if (const VectorType *vecType = type->getAs<VectorType>()) {
    len = vecType->getNumElements();
    eltType = vecType->getElementType();
    assert(eltType->isScalarType());
    return true;
  }
  
  // We allow lax conversion to and from non-vector types, but only if
  // they're real types (i.e. non-complex, non-pointer scalar types).
  if (!type->isRealType()) return false;
  
  len = 1;
  eltType = type;
  return true;
}

/// Are the two types lax-compatible vector types?  That is, given
/// that one of them is a vector, do they have equal storage sizes,
/// where the storage size is the number of elements times the element
/// size?
///
/// This will also return false if either of the types is neither a
/// vector nor a real type.
bool Sema::areLaxCompatibleVectorTypes(QualType srcTy, QualType destTy) {
  assert(destTy->isVectorType() || srcTy->isVectorType());
  
  // Disallow lax conversions between scalars and ExtVectors (these
  // conversions are allowed for other vector types because common headers
  // depend on them).  Most scalar OP ExtVector cases are handled by the
  // splat path anyway, which does what we want (convert, not bitcast).
  // What this rules out for ExtVectors is crazy things like char4*float.
  if (srcTy->isScalarType() && destTy->isExtVectorType()) return false;
  if (destTy->isScalarType() && srcTy->isExtVectorType()) return false;

  uint64_t srcLen, destLen;
  QualType srcEltTy, destEltTy;
  if (!breakDownVectorType(srcTy, srcLen, srcEltTy)) return false;
  if (!breakDownVectorType(destTy, destLen, destEltTy)) return false;
  
  // ASTContext::getTypeSize will return the size rounded up to a
  // power of 2, so instead of using that, we need to use the raw
  // element size multiplied by the element count.
  uint64_t srcEltSize = Context.getTypeSize(srcEltTy);
  uint64_t destEltSize = Context.getTypeSize(destEltTy);
  
  return (srcLen * srcEltSize == destLen * destEltSize);
}

/// Is this a legal conversion between two types, one of which is
/// known to be a vector type?
bool Sema::isLaxVectorConversion(QualType srcTy, QualType destTy) {
  assert(destTy->isVectorType() || srcTy->isVectorType());
  
  if (!Context.getLangOpts().LaxVectorConversions)
    return false;
  return areLaxCompatibleVectorTypes(srcTy, destTy);
}

bool Sema::CheckVectorCast(SourceRange R, QualType VectorTy, QualType Ty,
                           CastKind &Kind) {
  assert(VectorTy->isVectorType() && "Not a vector type!");

  if (Ty->isVectorType() || Ty->isIntegralType(Context)) {
    if (!areLaxCompatibleVectorTypes(Ty, VectorTy))
      return Diag(R.getBegin(),
                  Ty->isVectorType() ?
                  diag::err_invalid_conversion_between_vectors :
                  diag::err_invalid_conversion_between_vector_and_integer)
        << VectorTy << Ty << R;
  } else
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar)
      << VectorTy << Ty << R;

  Kind = CK_BitCast;
  return false;
}

ExprResult Sema::prepareVectorSplat(QualType VectorTy, Expr *SplattedExpr) {
  QualType DestElemTy = VectorTy->castAs<VectorType>()->getElementType();

  if (DestElemTy == SplattedExpr->getType())
    return SplattedExpr;

  assert(DestElemTy->isFloatingType() ||
         DestElemTy->isIntegralOrEnumerationType());

  CastKind CK;
  if (VectorTy->isExtVectorType() && SplattedExpr->getType()->isBooleanType()) {
    // OpenCL requires that we convert `true` boolean expressions to -1, but
    // only when splatting vectors.
    if (DestElemTy->isFloatingType()) {
      // To avoid having to have a CK_BooleanToSignedFloating cast kind, we cast
      // in two steps: boolean to signed integral, then to floating.
      ExprResult CastExprRes = ImpCastExprToType(SplattedExpr, Context.IntTy,
                                                 CK_BooleanToSignedIntegral);
      SplattedExpr = CastExprRes.get();
      CK = CK_IntegralToFloating;
    } else {
      CK = CK_BooleanToSignedIntegral;
    }
  } else {
    ExprResult CastExprRes = SplattedExpr;
    CK = PrepareScalarCast(CastExprRes, DestElemTy);
    if (CastExprRes.isInvalid())
      return ExprError();
    SplattedExpr = CastExprRes.get();
  }
  return ImpCastExprToType(SplattedExpr, DestElemTy, CK);
}

ExprResult Sema::CheckExtVectorCast(SourceRange R, QualType DestTy,
                                    Expr *CastExpr, CastKind &Kind) {
  assert(DestTy->isExtVectorType() && "Not an extended vector type!");

  QualType SrcTy = CastExpr->getType();

  // If SrcTy is a VectorType, the total size must match to explicitly cast to
  // an ExtVectorType.
  // In OpenCL, casts between vectors of different types are not allowed.
  // (See OpenCL 6.2).
  if (SrcTy->isVectorType()) {
    if (!areLaxCompatibleVectorTypes(SrcTy, DestTy)
        || (getLangOpts().OpenCL &&
            (DestTy.getCanonicalType() != SrcTy.getCanonicalType()))) {
      Diag(R.getBegin(),diag::err_invalid_conversion_between_ext_vectors)
        << DestTy << SrcTy << R;
      return ExprError();
    }
    Kind = CK_BitCast;
    return CastExpr;
  }

  // All non-pointer scalars can be cast to ExtVector type.  The appropriate
  // conversion will take place first from scalar to elt type, and then
  // splat from elt type to vector.
  if (SrcTy->isPointerType())
    return Diag(R.getBegin(),
                diag::err_invalid_conversion_between_vector_and_scalar)
      << DestTy << SrcTy << R;

  Kind = CK_VectorSplat;
  return prepareVectorSplat(DestTy, CastExpr);
}

ExprResult
Sema::ActOnCastExpr(Scope *S, SourceLocation LParenLoc,
                    Declarator &D, ParsedType &Ty,
                    SourceLocation RParenLoc, Expr *CastExpr) {
  assert(!D.isInvalidType() && (CastExpr != nullptr) &&
         "ActOnCastExpr(): missing type or expr");

  TypeSourceInfo *castTInfo = GetTypeForDeclaratorCast(D, CastExpr->getType());
  if (D.isInvalidType())
    return ExprError();

  if (getLangOpts().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);
  } else {
    // Make sure any TypoExprs have been dealt with.
    ExprResult Res = CorrectDelayedTyposInExpr(CastExpr);
    if (!Res.isUsable())
      return ExprError();
    CastExpr = Res.get();
  }

  checkUnusedDeclAttributes(D);

  QualType castType = castTInfo->getType();
  Ty = CreateParsedType(castType, castTInfo);

  bool isVectorLiteral = false;

  // Check for an altivec or OpenCL literal,
  // i.e. all the elements are integer constants.
  ParenExpr *PE = dyn_cast<ParenExpr>(CastExpr);
  ParenListExpr *PLE = dyn_cast<ParenListExpr>(CastExpr);
  if ((getLangOpts().AltiVec || getLangOpts().ZVector || getLangOpts().OpenCL)
       && castType->isVectorType() && (PE || PLE)) {
    if (PLE && PLE->getNumExprs() == 0) {
      Diag(PLE->getExprLoc(), diag::err_altivec_empty_initializer);
      return ExprError();
    }
    if (PE || PLE->getNumExprs() == 1) {
      Expr *E = (PE ? PE->getSubExpr() : PLE->getExpr(0));
      if (!E->getType()->isVectorType())
        isVectorLiteral = true;
    }
    else
      isVectorLiteral = true;
  }

  // If this is a vector initializer, '(' type ')' '(' init, ..., init ')'
  // then handle it as such.
  if (isVectorLiteral)
    return BuildVectorLiteral(LParenLoc, RParenLoc, CastExpr, castTInfo);

  // If the Expr being casted is a ParenListExpr, handle it specially.
  // This is not an AltiVec-style cast, so turn the ParenListExpr into a
  // sequence of BinOp comma operators.
  if (isa<ParenListExpr>(CastExpr)) {
    ExprResult Result = MaybeConvertParenListExprToParenExpr(S, CastExpr);
    if (Result.isInvalid()) return ExprError();
    CastExpr = Result.get();
  }

  if (getLangOpts().CPlusPlus && !castType->isVoidType() &&
      !getSourceManager().isInSystemMacro(LParenLoc))
    Diag(LParenLoc, diag::warn_old_style_cast) << CastExpr->getSourceRange();
  
  CheckTollFreeBridgeCast(castType, CastExpr);
  
  CheckObjCBridgeRelatedCast(castType, CastExpr);
  
  return BuildCStyleCastExpr(LParenLoc, castTInfo, RParenLoc, CastExpr);
}

ExprResult Sema::BuildVectorLiteral(SourceLocation LParenLoc,
                                    SourceLocation RParenLoc, Expr *E,
                                    TypeSourceInfo *TInfo) {
  assert((isa<ParenListExpr>(E) || isa<ParenExpr>(E)) &&
         "Expected paren or paren list expression");

  Expr **exprs;
  unsigned numExprs;
  Expr *subExpr;
  SourceLocation LiteralLParenLoc, LiteralRParenLoc;
  if (ParenListExpr *PE = dyn_cast<ParenListExpr>(E)) {
    LiteralLParenLoc = PE->getLParenLoc();
    LiteralRParenLoc = PE->getRParenLoc();
    exprs = PE->getExprs();
    numExprs = PE->getNumExprs();
  } else { // isa<ParenExpr> by assertion at function entrance
    LiteralLParenLoc = cast<ParenExpr>(E)->getLParen();
    LiteralRParenLoc = cast<ParenExpr>(E)->getRParen();
    subExpr = cast<ParenExpr>(E)->getSubExpr();
    exprs = &subExpr;
    numExprs = 1;
  }

  QualType Ty = TInfo->getType();
  assert(Ty->isVectorType() && "Expected vector type");

  SmallVector<Expr *, 8> initExprs;
  const VectorType *VTy = Ty->getAs<VectorType>();
  unsigned numElems = Ty->getAs<VectorType>()->getNumElements();
  
  // '(...)' form of vector initialization in AltiVec: the number of
  // initializers must be one or must match the size of the vector.
  // If a single value is specified in the initializer then it will be
  // replicated to all the components of the vector
  if (VTy->getVectorKind() == VectorType::AltiVecVector) {
    // The number of initializers must be one or must match the size of the
    // vector. If a single value is specified in the initializer then it will
    // be replicated to all the components of the vector
    if (numExprs == 1) {
      QualType ElemTy = Ty->getAs<VectorType>()->getElementType();
      ExprResult Literal = DefaultLvalueConversion(exprs[0]);
      if (Literal.isInvalid())
        return ExprError();
      Literal = ImpCastExprToType(Literal.get(), ElemTy,
                                  PrepareScalarCast(Literal, ElemTy));
      return BuildCStyleCastExpr(LParenLoc, TInfo, RParenLoc, Literal.get());
    }
    else if (numExprs < numElems) {
      Diag(E->getExprLoc(),
           diag::err_incorrect_number_of_vector_initializers);
      return ExprError();
    }
    else
      initExprs.append(exprs, exprs + numExprs);
  }
  else {
    // For OpenCL, when the number of initializers is a single value,
    // it will be replicated to all components of the vector.
    if (getLangOpts().OpenCL &&
        VTy->getVectorKind() == VectorType::GenericVector &&
        numExprs == 1) {
        QualType ElemTy = Ty->getAs<VectorType>()->getElementType();
        ExprResult Literal = DefaultLvalueConversion(exprs[0]);
        if (Literal.isInvalid())
          return ExprError();
        Literal = ImpCastExprToType(Literal.get(), ElemTy,
                                    PrepareScalarCast(Literal, ElemTy));
        return BuildCStyleCastExpr(LParenLoc, TInfo, RParenLoc, Literal.get());
    }
    
    initExprs.append(exprs, exprs + numExprs);
  }
  // FIXME: This means that pretty-printing the final AST will produce curly
  // braces instead of the original commas.
  InitListExpr *initE = new (Context) InitListExpr(Context, LiteralLParenLoc,
                                                   initExprs, LiteralRParenLoc);
  initE->setType(Ty);
  return BuildCompoundLiteralExpr(LParenLoc, TInfo, RParenLoc, initE);
}

/// This is not an AltiVec-style cast or or C++ direct-initialization, so turn
/// the ParenListExpr into a sequence of comma binary operators.
ExprResult
Sema::MaybeConvertParenListExprToParenExpr(Scope *S, Expr *OrigExpr) {
  ParenListExpr *E = dyn_cast<ParenListExpr>(OrigExpr);
  if (!E)
    return OrigExpr;

  ExprResult Result(E->getExpr(0));

  for (unsigned i = 1, e = E->getNumExprs(); i != e && !Result.isInvalid(); ++i)
    Result = ActOnBinOp(S, E->getExprLoc(), tok::comma, Result.get(),
                        E->getExpr(i));

  if (Result.isInvalid()) return ExprError();

  return ActOnParenExpr(E->getLParenLoc(), E->getRParenLoc(), Result.get());
}

ExprResult Sema::ActOnParenListExpr(SourceLocation L,
                                    SourceLocation R,
                                    MultiExprArg Val) {
  Expr *expr = new (Context) ParenListExpr(Context, L, Val, R);
  return expr;
}

/// \brief Emit a specialized diagnostic when one expression is a null pointer
/// constant and the other is not a pointer.  Returns true if a diagnostic is
/// emitted.
bool Sema::DiagnoseConditionalForNull(Expr *LHSExpr, Expr *RHSExpr,
                                      SourceLocation QuestionLoc) {
  Expr *NullExpr = LHSExpr;
  Expr *NonPointerExpr = RHSExpr;
  Expr::NullPointerConstantKind NullKind =
      NullExpr->isNullPointerConstant(Context,
                                      Expr::NPC_ValueDependentIsNotNull);

  if (NullKind == Expr::NPCK_NotNull) {
    NullExpr = RHSExpr;
    NonPointerExpr = LHSExpr;
    NullKind =
        NullExpr->isNullPointerConstant(Context,
                                        Expr::NPC_ValueDependentIsNotNull);
  }

  if (NullKind == Expr::NPCK_NotNull)
    return false;

  if (NullKind == Expr::NPCK_ZeroExpression)
    return false;

  if (NullKind == Expr::NPCK_ZeroLiteral) {
    // In this case, check to make sure that we got here from a "NULL"
    // string in the source code.
    NullExpr = NullExpr->IgnoreParenImpCasts();
    SourceLocation loc = NullExpr->getExprLoc();
    if (!findMacroSpelling(loc, "NULL"))
      return false;
  }

  int DiagType = (NullKind == Expr::NPCK_CXX11_nullptr);
  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands_null)
      << NonPointerExpr->getType() << DiagType
      << NonPointerExpr->getSourceRange();
  return true;
}

/// \brief Return false if the condition expression is valid, true otherwise.
static bool checkCondition(Sema &S, Expr *Cond, SourceLocation QuestionLoc) {
  QualType CondTy = Cond->getType();

  // OpenCL v1.1 s6.3.i says the condition cannot be a floating point type.
  if (S.getLangOpts().OpenCL && CondTy->isFloatingType()) {
    S.Diag(QuestionLoc, diag::err_typecheck_cond_expect_nonfloat)
      << CondTy << Cond->getSourceRange();
    return true;
  }

  // C99 6.5.15p2
  if (CondTy->isScalarType()) return false;

  S.Diag(QuestionLoc, diag::err_typecheck_cond_expect_scalar)
    << CondTy << Cond->getSourceRange();
  return true;
}

/// \brief Handle when one or both operands are void type.
static QualType checkConditionalVoidType(Sema &S, ExprResult &LHS,
                                         ExprResult &RHS) {
    Expr *LHSExpr = LHS.get();
    Expr *RHSExpr = RHS.get();

    if (!LHSExpr->getType()->isVoidType())
      S.Diag(RHSExpr->getLocStart(), diag::ext_typecheck_cond_one_void)
        << RHSExpr->getSourceRange();
    if (!RHSExpr->getType()->isVoidType())
      S.Diag(LHSExpr->getLocStart(), diag::ext_typecheck_cond_one_void)
        << LHSExpr->getSourceRange();
    LHS = S.ImpCastExprToType(LHS.get(), S.Context.VoidTy, CK_ToVoid);
    RHS = S.ImpCastExprToType(RHS.get(), S.Context.VoidTy, CK_ToVoid);
    return S.Context.VoidTy;
}

/// \brief Return false if the NullExpr can be promoted to PointerTy,
/// true otherwise.
static bool checkConditionalNullPointer(Sema &S, ExprResult &NullExpr,
                                        QualType PointerTy) {
  if ((!PointerTy->isAnyPointerType() && !PointerTy->isBlockPointerType()) ||
      !NullExpr.get()->isNullPointerConstant(S.Context,
                                            Expr::NPC_ValueDependentIsNull))
    return true;

  NullExpr = S.ImpCastExprToType(NullExpr.get(), PointerTy, CK_NullToPointer);
  return false;
}

/// \brief Checks compatibility between two pointers and return the resulting
/// type.
static QualType checkConditionalPointerCompatibility(Sema &S, ExprResult &LHS,
                                                     ExprResult &RHS,
                                                     SourceLocation Loc) {
  QualType LHSTy = LHS.get()->getType();
  QualType RHSTy = RHS.get()->getType();

  if (S.Context.hasSameType(LHSTy, RHSTy)) {
    // Two identical pointers types are always compatible.
    return LHSTy;
  }

  QualType lhptee, rhptee;

  // Get the pointee types.
  bool IsBlockPointer = false;
  if (const BlockPointerType *LHSBTy = LHSTy->getAs<BlockPointerType>()) {
    lhptee = LHSBTy->getPointeeType();
    rhptee = RHSTy->castAs<BlockPointerType>()->getPointeeType();
    IsBlockPointer = true;
  } else {
    lhptee = LHSTy->castAs<PointerType>()->getPointeeType();
    rhptee = RHSTy->castAs<PointerType>()->getPointeeType();
  }

  // C99 6.5.15p6: If both operands are pointers to compatible types or to
  // differently qualified versions of compatible types, the result type is
  // a pointer to an appropriately qualified version of the composite
  // type.

  // Only CVR-qualifiers exist in the standard, and the differently-qualified
  // clause doesn't make sense for our extensions. E.g. address space 2 should
  // be incompatible with address space 3: they may live on different devices or
  // anything.
  Qualifiers lhQual = lhptee.getQualifiers();
  Qualifiers rhQual = rhptee.getQualifiers();

  unsigned MergedCVRQual = lhQual.getCVRQualifiers() | rhQual.getCVRQualifiers();
  lhQual.removeCVRQualifiers();
  rhQual.removeCVRQualifiers();

  lhptee = S.Context.getQualifiedType(lhptee.getUnqualifiedType(), lhQual);
  rhptee = S.Context.getQualifiedType(rhptee.getUnqualifiedType(), rhQual);

  QualType CompositeTy = S.Context.mergeTypes(lhptee, rhptee);

  if (CompositeTy.isNull()) {
    S.Diag(Loc, diag::ext_typecheck_cond_incompatible_pointers)
      << LHSTy << RHSTy << LHS.get()->getSourceRange()
      << RHS.get()->getSourceRange();
    // In this situation, we assume void* type. No especially good
    // reason, but this is what gcc does, and we do have to pick
    // to get a consistent AST.
    QualType incompatTy = S.Context.getPointerType(S.Context.VoidTy);
    LHS = S.ImpCastExprToType(LHS.get(), incompatTy, CK_BitCast);
    RHS = S.ImpCastExprToType(RHS.get(), incompatTy, CK_BitCast);
    return incompatTy;
  }

  // The pointer types are compatible.
  QualType ResultTy = CompositeTy.withCVRQualifiers(MergedCVRQual);
  if (IsBlockPointer)
    ResultTy = S.Context.getBlockPointerType(ResultTy);
  else
    ResultTy = S.Context.getPointerType(ResultTy);

  LHS = S.ImpCastExprToType(LHS.get(), ResultTy, CK_BitCast);
  RHS = S.ImpCastExprToType(RHS.get(), ResultTy, CK_BitCast);
  return ResultTy;
}

/// \brief Return the resulting type when the operands are both block pointers.
static QualType checkConditionalBlockPointerCompatibility(Sema &S,
                                                          ExprResult &LHS,
                                                          ExprResult &RHS,
                                                          SourceLocation Loc) {
  QualType LHSTy = LHS.get()->getType();
  QualType RHSTy = RHS.get()->getType();

  if (!LHSTy->isBlockPointerType() || !RHSTy->isBlockPointerType()) {
    if (LHSTy->isVoidPointerType() || RHSTy->isVoidPointerType()) {
      QualType destType = S.Context.getPointerType(S.Context.VoidTy);
      LHS = S.ImpCastExprToType(LHS.get(), destType, CK_BitCast);
      RHS = S.ImpCastExprToType(RHS.get(), destType, CK_BitCast);
      return destType;
    }
    S.Diag(Loc, diag::err_typecheck_cond_incompatible_operands)
      << LHSTy << RHSTy << LHS.get()->getSourceRange()
      << RHS.get()->getSourceRange();
    return QualType();
  }

  // We have 2 block pointer types.
  return checkConditionalPointerCompatibility(S, LHS, RHS, Loc);
}

/// \brief Return the resulting type when the operands are both pointers.
static QualType
checkConditionalObjectPointersCompatibility(Sema &S, ExprResult &LHS,
                                            ExprResult &RHS,
                                            SourceLocation Loc) {
  // get the pointer types
  QualType LHSTy = LHS.get()->getType();
  QualType RHSTy = RHS.get()->getType();

  // get the "pointed to" types
  QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
  QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();

  // ignore qualifiers on void (C99 6.5.15p3, clause 6)
  if (lhptee->isVoidType() && rhptee->isIncompleteOrObjectType()) {
    // Figure out necessary qualifiers (C99 6.5.15p6)
    QualType destPointee
      = S.Context.getQualifiedType(lhptee, rhptee.getQualifiers());
    QualType destType = S.Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    LHS = S.ImpCastExprToType(LHS.get(), destType, CK_NoOp);
    // Promote to void*.
    RHS = S.ImpCastExprToType(RHS.get(), destType, CK_BitCast);
    return destType;
  }
  if (rhptee->isVoidType() && lhptee->isIncompleteOrObjectType()) {
    QualType destPointee
      = S.Context.getQualifiedType(rhptee, lhptee.getQualifiers());
    QualType destType = S.Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    RHS = S.ImpCastExprToType(RHS.get(), destType, CK_NoOp);
    // Promote to void*.
    LHS = S.ImpCastExprToType(LHS.get(), destType, CK_BitCast);
    return destType;
  }

  return checkConditionalPointerCompatibility(S, LHS, RHS, Loc);
}

/// \brief Return false if the first expression is not an integer and the second
/// expression is not a pointer, true otherwise.
static bool checkPointerIntegerMismatch(Sema &S, ExprResult &Int,
                                        Expr* PointerExpr, SourceLocation Loc,
                                        bool IsIntFirstExpr) {
  if (!PointerExpr->getType()->isPointerType() ||
      !Int.get()->getType()->isIntegerType())
    return false;

  Expr *Expr1 = IsIntFirstExpr ? Int.get() : PointerExpr;
  Expr *Expr2 = IsIntFirstExpr ? PointerExpr : Int.get();

  S.Diag(Loc, diag::ext_typecheck_cond_pointer_integer_mismatch)
    << Expr1->getType() << Expr2->getType()
    << Expr1->getSourceRange() << Expr2->getSourceRange();
  Int = S.ImpCastExprToType(Int.get(), PointerExpr->getType(),
                            CK_IntegralToPointer);
  return true;
}

/// \brief Simple conversion between integer and floating point types.
///
/// Used when handling the OpenCL conditional operator where the
/// condition is a vector while the other operands are scalar.
///
/// OpenCL v1.1 s6.3.i and s6.11.6 together require that the scalar
/// types are either integer or floating type. Between the two
/// operands, the type with the higher rank is defined as the "result
/// type". The other operand needs to be promoted to the same type. No
/// other type promotion is allowed. We cannot use
/// UsualArithmeticConversions() for this purpose, since it always
/// promotes promotable types.
static QualType OpenCLArithmeticConversions(Sema &S, ExprResult &LHS,
                                            ExprResult &RHS,
                                            SourceLocation QuestionLoc) {
  LHS = S.DefaultFunctionArrayLvalueConversion(LHS.get());
  if (LHS.isInvalid())
    return QualType();
  RHS = S.DefaultFunctionArrayLvalueConversion(RHS.get());
  if (RHS.isInvalid())
    return QualType();

  // For conversion purposes, we ignore any qualifiers.
  // For example, "const float" and "float" are equivalent.
  QualType LHSType =
    S.Context.getCanonicalType(LHS.get()->getType()).getUnqualifiedType();
  QualType RHSType =
    S.Context.getCanonicalType(RHS.get()->getType()).getUnqualifiedType();

  if (!LHSType->isIntegerType() && !LHSType->isRealFloatingType()) {
    S.Diag(QuestionLoc, diag::err_typecheck_cond_expect_int_float)
      << LHSType << LHS.get()->getSourceRange();
    return QualType();
  }

  if (!RHSType->isIntegerType() && !RHSType->isRealFloatingType()) {
    S.Diag(QuestionLoc, diag::err_typecheck_cond_expect_int_float)
      << RHSType << RHS.get()->getSourceRange();
    return QualType();
  }

  // If both types are identical, no conversion is needed.
  if (LHSType == RHSType)
    return LHSType;

  // Now handle "real" floating types (i.e. float, double, long double).
  if (LHSType->isRealFloatingType() || RHSType->isRealFloatingType())
    return handleFloatConversion(S, LHS, RHS, LHSType, RHSType,
                                 /*IsCompAssign = */ false);

  // Finally, we have two differing integer types.
  return handleIntegerConversion<doIntegralCast, doIntegralCast>
  (S, LHS, RHS, LHSType, RHSType, /*IsCompAssign = */ false);
}

/// \brief Convert scalar operands to a vector that matches the
///        condition in length.
///
/// Used when handling the OpenCL conditional operator where the
/// condition is a vector while the other operands are scalar.
///
/// We first compute the "result type" for the scalar operands
/// according to OpenCL v1.1 s6.3.i. Both operands are then converted
/// into a vector of that type where the length matches the condition
/// vector type. s6.11.6 requires that the element types of the result
/// and the condition must have the same number of bits.
static QualType
OpenCLConvertScalarsToVectors(Sema &S, ExprResult &LHS, ExprResult &RHS,
                              QualType CondTy, SourceLocation QuestionLoc) {
  QualType ResTy = OpenCLArithmeticConversions(S, LHS, RHS, QuestionLoc);
  if (ResTy.isNull()) return QualType();

  const VectorType *CV = CondTy->getAs<VectorType>();
  assert(CV);

  // Determine the vector result type
  unsigned NumElements = CV->getNumElements();
  QualType VectorTy = S.Context.getExtVectorType(ResTy, NumElements);

  // Ensure that all types have the same number of bits
  if (S.Context.getTypeSize(CV->getElementType())
      != S.Context.getTypeSize(ResTy)) {
    // Since VectorTy is created internally, it does not pretty print
    // with an OpenCL name. Instead, we just print a description.
    std::string EleTyName = ResTy.getUnqualifiedType().getAsString();
    SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << "(vector of " << NumElements << " '" << EleTyName << "' values)";
    S.Diag(QuestionLoc, diag::err_conditional_vector_element_size)
      << CondTy << OS.str();
    return QualType();
  }

  // Convert operands to the vector result type
  LHS = S.ImpCastExprToType(LHS.get(), VectorTy, CK_VectorSplat);
  RHS = S.ImpCastExprToType(RHS.get(), VectorTy, CK_VectorSplat);

  return VectorTy;
}

/// \brief Return false if this is a valid OpenCL condition vector
static bool checkOpenCLConditionVector(Sema &S, Expr *Cond,
                                       SourceLocation QuestionLoc) {
  // OpenCL v1.1 s6.11.6 says the elements of the vector must be of
  // integral type.
  const VectorType *CondTy = Cond->getType()->getAs<VectorType>();
  assert(CondTy);
  QualType EleTy = CondTy->getElementType();
  if (EleTy->isIntegerType()) return false;

  S.Diag(QuestionLoc, diag::err_typecheck_cond_expect_nonfloat)
    << Cond->getType() << Cond->getSourceRange();
  return true;
}

/// \brief Return false if the vector condition type and the vector
///        result type are compatible.
///
/// OpenCL v1.1 s6.11.6 requires that both vector types have the same
/// number of elements, and their element types have the same number
/// of bits.
static bool checkVectorResult(Sema &S, QualType CondTy, QualType VecResTy,
                              SourceLocation QuestionLoc) {
  const VectorType *CV = CondTy->getAs<VectorType>();
  const VectorType *RV = VecResTy->getAs<VectorType>();
  assert(CV && RV);

  if (CV->getNumElements() != RV->getNumElements()) {
    S.Diag(QuestionLoc, diag::err_conditional_vector_size)
      << CondTy << VecResTy;
    return true;
  }

  QualType CVE = CV->getElementType();
  QualType RVE = RV->getElementType();

  if (S.Context.getTypeSize(CVE) != S.Context.getTypeSize(RVE)) {
    S.Diag(QuestionLoc, diag::err_conditional_vector_element_size)
      << CondTy << VecResTy;
    return true;
  }

  return false;
}

/// \brief Return the resulting type for the conditional operator in
///        OpenCL (aka "ternary selection operator", OpenCL v1.1
///        s6.3.i) when the condition is a vector type.
static QualType
OpenCLCheckVectorConditional(Sema &S, ExprResult &Cond,
                             ExprResult &LHS, ExprResult &RHS,
                             SourceLocation QuestionLoc) {
  Cond = S.DefaultFunctionArrayLvalueConversion(Cond.get()); 
  if (Cond.isInvalid())
    return QualType();
  QualType CondTy = Cond.get()->getType();

  if (checkOpenCLConditionVector(S, Cond.get(), QuestionLoc))
    return QualType();

  // If either operand is a vector then find the vector type of the
  // result as specified in OpenCL v1.1 s6.3.i.
  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    QualType VecResTy = S.CheckVectorOperands(LHS, RHS, QuestionLoc,
                                              /*isCompAssign*/false,
                                              /*AllowBothBool*/true,
                                              /*AllowBoolConversions*/false);
    if (VecResTy.isNull()) return QualType();
    // The result type must match the condition type as specified in
    // OpenCL v1.1 s6.11.6.
    if (checkVectorResult(S, CondTy, VecResTy, QuestionLoc))
      return QualType();
    return VecResTy;
  }

  // Both operands are scalar.
  return OpenCLConvertScalarsToVectors(S, LHS, RHS, CondTy, QuestionLoc);
}

/// Note that LHS is not null here, even if this is the gnu "x ?: y" extension.
/// In that case, LHS = cond.
/// C99 6.5.15
QualType Sema::CheckConditionalOperands(ExprResult &Cond, ExprResult &LHS,
                                        ExprResult &RHS, ExprValueKind &VK,
                                        ExprObjectKind &OK,
                                        SourceLocation QuestionLoc) {

  ExprResult LHSResult = CheckPlaceholderExpr(LHS.get());
  if (!LHSResult.isUsable()) return QualType();
  LHS = LHSResult;

  ExprResult RHSResult = CheckPlaceholderExpr(RHS.get());
  if (!RHSResult.isUsable()) return QualType();
  RHS = RHSResult;

  // C++ is sufficiently different to merit its own checker.
  if (getLangOpts().CPlusPlus)
    return CXXCheckConditionalOperands(Cond, LHS, RHS, VK, OK, QuestionLoc);

  VK = VK_RValue;
  OK = OK_Ordinary;

  // The OpenCL operator with a vector condition is sufficiently
  // different to merit its own checker.
  if (getLangOpts().OpenCL && Cond.get()->getType()->isVectorType())
    return OpenCLCheckVectorConditional(*this, Cond, LHS, RHS, QuestionLoc);

  // First, check the condition.
  Cond = UsualUnaryConversions(Cond.get());
  if (Cond.isInvalid())
    return QualType();
  if (checkCondition(*this, Cond.get(), QuestionLoc))
    return QualType();

  // Now check the two expressions.
  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType())
    return CheckVectorOperands(LHS, RHS, QuestionLoc, /*isCompAssign*/false,
                               /*AllowBothBool*/true,
                               /*AllowBoolConversions*/false);

  QualType ResTy = UsualArithmeticConversions(LHS, RHS);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  QualType LHSTy = LHS.get()->getType();
  QualType RHSTy = RHS.get()->getType();

  // If both operands have arithmetic type, do the usual arithmetic conversions
  // to find a common type: C99 6.5.15p3,5.
  if (LHSTy->isArithmeticType() && RHSTy->isArithmeticType()) {
    LHS = ImpCastExprToType(LHS.get(), ResTy, PrepareScalarCast(LHS, ResTy));
    RHS = ImpCastExprToType(RHS.get(), ResTy, PrepareScalarCast(RHS, ResTy));

    return ResTy;
  }

  // If both operands are the same structure or union type, the result is that
  // type.
  if (const RecordType *LHSRT = LHSTy->getAs<RecordType>()) {    // C99 6.5.15p3
    if (const RecordType *RHSRT = RHSTy->getAs<RecordType>())
      if (LHSRT->getDecl() == RHSRT->getDecl())
        // "If both the operands have structure or union type, the result has
        // that type."  This implies that CV qualifiers are dropped.
        return LHSTy.getUnqualifiedType();
    // FIXME: Type of conditional expression must be complete in C mode.
  }

  // C99 6.5.15p5: "If both operands have void type, the result has void type."
  // The following || allows only one side to be void (a GCC-ism).
  if (LHSTy->isVoidType() || RHSTy->isVoidType()) {
    return checkConditionalVoidType(*this, LHS, RHS);
  }

  // C99 6.5.15p6 - "if one operand is a null pointer constant, the result has
  // the type of the other operand."
  if (!checkConditionalNullPointer(*this, RHS, LHSTy)) return LHSTy;
  if (!checkConditionalNullPointer(*this, LHS, RHSTy)) return RHSTy;

  // All objective-c pointer type analysis is done here.
  QualType compositeType = FindCompositeObjCPointerType(LHS, RHS,
                                                        QuestionLoc);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();
  if (!compositeType.isNull())
    return compositeType;


  // Handle block pointer types.
  if (LHSTy->isBlockPointerType() || RHSTy->isBlockPointerType())
    return checkConditionalBlockPointerCompatibility(*this, LHS, RHS,
                                                     QuestionLoc);

  // Check constraints for C object pointers types (C99 6.5.15p3,6).
  if (LHSTy->isPointerType() && RHSTy->isPointerType())
    return checkConditionalObjectPointersCompatibility(*this, LHS, RHS,
                                                       QuestionLoc);

  // GCC compatibility: soften pointer/integer mismatch.  Note that
  // null pointers have been filtered out by this point.
  if (checkPointerIntegerMismatch(*this, LHS, RHS.get(), QuestionLoc,
      /*isIntFirstExpr=*/true))
    return RHSTy;
  if (checkPointerIntegerMismatch(*this, RHS, LHS.get(), QuestionLoc,
      /*isIntFirstExpr=*/false))
    return LHSTy;

  // Emit a better diagnostic if one of the expressions is a null pointer
  // constant and the other is not a pointer type. In this case, the user most
  // likely forgot to take the address of the other expression.
  if (DiagnoseConditionalForNull(LHS.get(), RHS.get(), QuestionLoc))
    return QualType();

  // Otherwise, the operands are not compatible.
  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHSTy << RHSTy << LHS.get()->getSourceRange()
    << RHS.get()->getSourceRange();
  return QualType();
}

/// FindCompositeObjCPointerType - Helper method to find composite type of
/// two objective-c pointer types of the two input expressions.
QualType Sema::FindCompositeObjCPointerType(ExprResult &LHS, ExprResult &RHS,
                                            SourceLocation QuestionLoc) {
  QualType LHSTy = LHS.get()->getType();
  QualType RHSTy = RHS.get()->getType();

  // Handle things like Class and struct objc_class*.  Here we case the result
  // to the pseudo-builtin, because that will be implicitly cast back to the
  // redefinition type if an attempt is made to access its fields.
  if (LHSTy->isObjCClassType() &&
      (Context.hasSameType(RHSTy, Context.getObjCClassRedefinitionType()))) {
    RHS = ImpCastExprToType(RHS.get(), LHSTy, CK_CPointerToObjCPointerCast);
    return LHSTy;
  }
  if (RHSTy->isObjCClassType() &&
      (Context.hasSameType(LHSTy, Context.getObjCClassRedefinitionType()))) {
    LHS = ImpCastExprToType(LHS.get(), RHSTy, CK_CPointerToObjCPointerCast);
    return RHSTy;
  }
  // And the same for struct objc_object* / id
  if (LHSTy->isObjCIdType() &&
      (Context.hasSameType(RHSTy, Context.getObjCIdRedefinitionType()))) {
    RHS = ImpCastExprToType(RHS.get(), LHSTy, CK_CPointerToObjCPointerCast);
    return LHSTy;
  }
  if (RHSTy->isObjCIdType() &&
      (Context.hasSameType(LHSTy, Context.getObjCIdRedefinitionType()))) {
    LHS = ImpCastExprToType(LHS.get(), RHSTy, CK_CPointerToObjCPointerCast);
    return RHSTy;
  }
  // And the same for struct objc_selector* / SEL
  if (Context.isObjCSelType(LHSTy) &&
      (Context.hasSameType(RHSTy, Context.getObjCSelRedefinitionType()))) {
    RHS = ImpCastExprToType(RHS.get(), LHSTy, CK_BitCast);
    return LHSTy;
  }
  if (Context.isObjCSelType(RHSTy) &&
      (Context.hasSameType(LHSTy, Context.getObjCSelRedefinitionType()))) {
    LHS = ImpCastExprToType(LHS.get(), RHSTy, CK_BitCast);
    return RHSTy;
  }
  // Check constraints for Objective-C object pointers types.
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isObjCObjectPointerType()) {

    if (Context.getCanonicalType(LHSTy) == Context.getCanonicalType(RHSTy)) {
      // Two identical object pointer types are always compatible.
      return LHSTy;
    }
    const ObjCObjectPointerType *LHSOPT = LHSTy->castAs<ObjCObjectPointerType>();
    const ObjCObjectPointerType *RHSOPT = RHSTy->castAs<ObjCObjectPointerType>();
    QualType compositeType = LHSTy;

    // If both operands are interfaces and either operand can be
    // assigned to the other, use that type as the composite
    // type. This allows
    //   xxx ? (A*) a : (B*) b
    // where B is a subclass of A.
    //
    // Additionally, as for assignment, if either type is 'id'
    // allow silent coercion. Finally, if the types are
    // incompatible then make sure to use 'id' as the composite
    // type so the result is acceptable for sending messages to.

    // FIXME: Consider unifying with 'areComparableObjCPointerTypes'.
    // It could return the composite type.
    if (!(compositeType =
          Context.areCommonBaseCompatible(LHSOPT, RHSOPT)).isNull()) {
      // Nothing more to do.
    } else if (Context.canAssignObjCInterfaces(LHSOPT, RHSOPT)) {
      compositeType = RHSOPT->isObjCBuiltinType() ? RHSTy : LHSTy;
    } else if (Context.canAssignObjCInterfaces(RHSOPT, LHSOPT)) {
      compositeType = LHSOPT->isObjCBuiltinType() ? LHSTy : RHSTy;
    } else if ((LHSTy->isObjCQualifiedIdType() ||
                RHSTy->isObjCQualifiedIdType()) &&
               Context.ObjCQualifiedIdTypesAreCompatible(LHSTy, RHSTy, true)) {
      // Need to handle "id<xx>" explicitly.
      // GCC allows qualified id and any Objective-C type to devolve to
      // id. Currently localizing to here until clear this should be
      // part of ObjCQualifiedIdTypesAreCompatible.
      compositeType = Context.getObjCIdType();
    } else if (LHSTy->isObjCIdType() || RHSTy->isObjCIdType()) {
      compositeType = Context.getObjCIdType();
    } else {
      Diag(QuestionLoc, diag::ext_typecheck_cond_incompatible_operands)
      << LHSTy << RHSTy
      << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      QualType incompatTy = Context.getObjCIdType();
      LHS = ImpCastExprToType(LHS.get(), incompatTy, CK_BitCast);
      RHS = ImpCastExprToType(RHS.get(), incompatTy, CK_BitCast);
      return incompatTy;
    }
    // The object pointer types are compatible.
    LHS = ImpCastExprToType(LHS.get(), compositeType, CK_BitCast);
    RHS = ImpCastExprToType(RHS.get(), compositeType, CK_BitCast);
    return compositeType;
  }
  // Check Objective-C object pointer types and 'void *'
  if (LHSTy->isVoidPointerType() && RHSTy->isObjCObjectPointerType()) {
    if (getLangOpts().ObjCAutoRefCount) {
      // ARC forbids the implicit conversion of object pointers to 'void *',
      // so these types are not compatible.
      Diag(QuestionLoc, diag::err_cond_voidptr_arc) << LHSTy << RHSTy
          << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      LHS = RHS = true;
      return QualType();
    }
    QualType lhptee = LHSTy->getAs<PointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType destPointee
    = Context.getQualifiedType(lhptee, rhptee.getQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    LHS = ImpCastExprToType(LHS.get(), destType, CK_NoOp);
    // Promote to void*.
    RHS = ImpCastExprToType(RHS.get(), destType, CK_BitCast);
    return destType;
  }
  if (LHSTy->isObjCObjectPointerType() && RHSTy->isVoidPointerType()) {
    if (getLangOpts().ObjCAutoRefCount) {
      // ARC forbids the implicit conversion of object pointers to 'void *',
      // so these types are not compatible.
      Diag(QuestionLoc, diag::err_cond_voidptr_arc) << LHSTy << RHSTy
          << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      LHS = RHS = true;
      return QualType();
    }
    QualType lhptee = LHSTy->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType rhptee = RHSTy->getAs<PointerType>()->getPointeeType();
    QualType destPointee
    = Context.getQualifiedType(rhptee, lhptee.getQualifiers());
    QualType destType = Context.getPointerType(destPointee);
    // Add qualifiers if necessary.
    RHS = ImpCastExprToType(RHS.get(), destType, CK_NoOp);
    // Promote to void*.
    LHS = ImpCastExprToType(LHS.get(), destType, CK_BitCast);
    return destType;
  }
  return QualType();
}

/// SuggestParentheses - Emit a note with a fixit hint that wraps
/// ParenRange in parentheses.
static void SuggestParentheses(Sema &Self, SourceLocation Loc,
                               const PartialDiagnostic &Note,
                               SourceRange ParenRange) {
  SourceLocation EndLoc = Self.getLocForEndOfToken(ParenRange.getEnd());
  if (ParenRange.getBegin().isFileID() && ParenRange.getEnd().isFileID() &&
      EndLoc.isValid()) {
    Self.Diag(Loc, Note)
      << FixItHint::CreateInsertion(ParenRange.getBegin(), "(")
      << FixItHint::CreateInsertion(EndLoc, ")");
  } else {
    // We can't display the parentheses, so just show the bare note.
    Self.Diag(Loc, Note) << ParenRange;
  }
}

static bool IsArithmeticOp(BinaryOperatorKind Opc) {
  return BinaryOperator::isAdditiveOp(Opc) ||
         BinaryOperator::isMultiplicativeOp(Opc) ||
         BinaryOperator::isShiftOp(Opc);
}

/// IsArithmeticBinaryExpr - Returns true if E is an arithmetic binary
/// expression, either using a built-in or overloaded operator,
/// and sets *OpCode to the opcode and *RHSExprs to the right-hand side
/// expression.
static bool IsArithmeticBinaryExpr(Expr *E, BinaryOperatorKind *Opcode,
                                   Expr **RHSExprs) {
  // Don't strip parenthesis: we should not warn if E is in parenthesis.
  E = E->IgnoreImpCasts();
  E = E->IgnoreConversionOperator();
  E = E->IgnoreImpCasts();

  // Built-in binary operator.
  if (BinaryOperator *OP = dyn_cast<BinaryOperator>(E)) {
    if (IsArithmeticOp(OP->getOpcode())) {
      *Opcode = OP->getOpcode();
      *RHSExprs = OP->getRHS();
      return true;
    }
  }

  // Overloaded operator.
  if (CXXOperatorCallExpr *Call = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (Call->getNumArgs() != 2)
      return false;

    // Make sure this is really a binary operator that is safe to pass into
    // BinaryOperator::getOverloadedOpcode(), e.g. it's not a subscript op.
    OverloadedOperatorKind OO = Call->getOperator();
    if (OO < OO_Plus || OO > OO_Arrow ||
        OO == OO_PlusPlus || OO == OO_MinusMinus)
      return false;

    BinaryOperatorKind OpKind = BinaryOperator::getOverloadedOpcode(OO);
    if (IsArithmeticOp(OpKind)) {
      *Opcode = OpKind;
      *RHSExprs = Call->getArg(1);
      return true;
    }
  }

  return false;
}

/// ExprLooksBoolean - Returns true if E looks boolean, i.e. it has boolean type
/// or is a logical expression such as (x==y) which has int type, but is
/// commonly interpreted as boolean.
static bool ExprLooksBoolean(Expr *E) {
  E = E->IgnoreParenImpCasts();

  if (E->getType()->isBooleanType())
    return true;
  if (BinaryOperator *OP = dyn_cast<BinaryOperator>(E))
    return OP->isComparisonOp() || OP->isLogicalOp();
  if (UnaryOperator *OP = dyn_cast<UnaryOperator>(E))
    return OP->getOpcode() == UO_LNot;
  if (E->getType()->isPointerType())
    return true;

  return false;
}

/// DiagnoseConditionalPrecedence - Emit a warning when a conditional operator
/// and binary operator are mixed in a way that suggests the programmer assumed
/// the conditional operator has higher precedence, for example:
/// "int x = a + someBinaryCondition ? 1 : 2".
static void DiagnoseConditionalPrecedence(Sema &Self,
                                          SourceLocation OpLoc,
                                          Expr *Condition,
                                          Expr *LHSExpr,
                                          Expr *RHSExpr) {
  BinaryOperatorKind CondOpcode;
  Expr *CondRHS;

  if (!IsArithmeticBinaryExpr(Condition, &CondOpcode, &CondRHS))
    return;
  if (!ExprLooksBoolean(CondRHS))
    return;

  // The condition is an arithmetic binary expression, with a right-
  // hand side that looks boolean, so warn.

  Self.Diag(OpLoc, diag::warn_precedence_conditional)
      << Condition->getSourceRange()
      << BinaryOperator::getOpcodeStr(CondOpcode);

  SuggestParentheses(Self, OpLoc,
    Self.PDiag(diag::note_precedence_silence)
      << BinaryOperator::getOpcodeStr(CondOpcode),
    SourceRange(Condition->getLocStart(), Condition->getLocEnd()));

  SuggestParentheses(Self, OpLoc,
    Self.PDiag(diag::note_precedence_conditional_first),
    SourceRange(CondRHS->getLocStart(), RHSExpr->getLocEnd()));
}

/// ActOnConditionalOp - Parse a ?: operation.  Note that 'LHS' may be null
/// in the case of a the GNU conditional expr extension.
ExprResult Sema::ActOnConditionalOp(SourceLocation QuestionLoc,
                                    SourceLocation ColonLoc,
                                    Expr *CondExpr, Expr *LHSExpr,
                                    Expr *RHSExpr) {
  if (!getLangOpts().CPlusPlus) {
    // C cannot handle TypoExpr nodes in the condition because it
    // doesn't handle dependent types properly, so make sure any TypoExprs have
    // been dealt with before checking the operands.
    ExprResult CondResult = CorrectDelayedTyposInExpr(CondExpr);
    if (!CondResult.isUsable()) return ExprError();
    CondExpr = CondResult.get();
  }

  // If this is the gnu "x ?: y" extension, analyze the types as though the LHS
  // was the condition.
  OpaqueValueExpr *opaqueValue = nullptr;
  Expr *commonExpr = nullptr;
  if (!LHSExpr) {
    commonExpr = CondExpr;
    // Lower out placeholder types first.  This is important so that we don't
    // try to capture a placeholder. This happens in few cases in C++; such
    // as Objective-C++'s dictionary subscripting syntax.
    if (commonExpr->hasPlaceholderType()) {
      ExprResult result = CheckPlaceholderExpr(commonExpr);
      if (!result.isUsable()) return ExprError();
      commonExpr = result.get();
    }
    // We usually want to apply unary conversions *before* saving, except
    // in the special case of a C++ l-value conditional.
    if (!(getLangOpts().CPlusPlus
          && !commonExpr->isTypeDependent()
          && commonExpr->getValueKind() == RHSExpr->getValueKind()
          && commonExpr->isGLValue()
          && commonExpr->isOrdinaryOrBitFieldObject()
          && RHSExpr->isOrdinaryOrBitFieldObject()
          && Context.hasSameType(commonExpr->getType(), RHSExpr->getType()))) {
      ExprResult commonRes = UsualUnaryConversions(commonExpr);
      if (commonRes.isInvalid())
        return ExprError();
      commonExpr = commonRes.get();
    }

    opaqueValue = new (Context) OpaqueValueExpr(commonExpr->getExprLoc(),
                                                commonExpr->getType(),
                                                commonExpr->getValueKind(),
                                                commonExpr->getObjectKind(),
                                                commonExpr);
    LHSExpr = CondExpr = opaqueValue;
  }

  ExprValueKind VK = VK_RValue;
  ExprObjectKind OK = OK_Ordinary;
  ExprResult Cond = CondExpr, LHS = LHSExpr, RHS = RHSExpr;
  QualType result = CheckConditionalOperands(Cond, LHS, RHS, 
                                             VK, OK, QuestionLoc);
  if (result.isNull() || Cond.isInvalid() || LHS.isInvalid() ||
      RHS.isInvalid())
    return ExprError();

  DiagnoseConditionalPrecedence(*this, QuestionLoc, Cond.get(), LHS.get(),
                                RHS.get());

  CheckBoolLikeConversion(Cond.get(), QuestionLoc);

  if (!commonExpr)
    return new (Context)
        ConditionalOperator(Cond.get(), QuestionLoc, LHS.get(), ColonLoc,
                            RHS.get(), result, VK, OK);

  return new (Context) BinaryConditionalOperator(
      commonExpr, opaqueValue, Cond.get(), LHS.get(), RHS.get(), QuestionLoc,
      ColonLoc, result, VK, OK);
}

// checkPointerTypesForAssignment - This is a very tricky routine (despite
// being closely modeled after the C99 spec:-). The odd characteristic of this
// routine is it effectively iqnores the qualifiers on the top level pointee.
// This circumvents the usual type rules specified in 6.2.7p1 & 6.7.5.[1-3].
// FIXME: add a couple examples in this comment.
static Sema::AssignConvertType
checkPointerTypesForAssignment(Sema &S, QualType LHSType, QualType RHSType) {
  assert(LHSType.isCanonical() && "LHS not canonicalized!");
  assert(RHSType.isCanonical() && "RHS not canonicalized!");

  // get the "pointed to" type (ignoring qualifiers at the top level)
  const Type *lhptee, *rhptee;
  Qualifiers lhq, rhq;
  std::tie(lhptee, lhq) =
      cast<PointerType>(LHSType)->getPointeeType().split().asPair();
  std::tie(rhptee, rhq) =
      cast<PointerType>(RHSType)->getPointeeType().split().asPair();

  Sema::AssignConvertType ConvTy = Sema::Compatible;

  // C99 6.5.16.1p1: This following citation is common to constraints
  // 3 & 4 (below). ...and the type *pointed to* by the left has all the
  // qualifiers of the type *pointed to* by the right;

  // As a special case, 'non-__weak A *' -> 'non-__weak const *' is okay.
  if (lhq.getObjCLifetime() != rhq.getObjCLifetime() &&
      lhq.compatiblyIncludesObjCLifetime(rhq)) {
    // Ignore lifetime for further calculation.
    lhq.removeObjCLifetime();
    rhq.removeObjCLifetime();
  }

  if (!lhq.compatiblyIncludes(rhq)) {
    // Treat address-space mismatches as fatal.  TODO: address subspaces
    if (!lhq.isAddressSpaceSupersetOf(rhq))
      ConvTy = Sema::IncompatiblePointerDiscardsQualifiers;

    // It's okay to add or remove GC or lifetime qualifiers when converting to
    // and from void*.
    else if (lhq.withoutObjCGCAttr().withoutObjCLifetime()
                        .compatiblyIncludes(
                                rhq.withoutObjCGCAttr().withoutObjCLifetime())
             && (lhptee->isVoidType() || rhptee->isVoidType()))
      ; // keep old

    // Treat lifetime mismatches as fatal.
    else if (lhq.getObjCLifetime() != rhq.getObjCLifetime())
      ConvTy = Sema::IncompatiblePointerDiscardsQualifiers;
    
    // For GCC compatibility, other qualifier mismatches are treated
    // as still compatible in C.
    else ConvTy = Sema::CompatiblePointerDiscardsQualifiers;
  }

  // C99 6.5.16.1p1 (constraint 4): If one operand is a pointer to an object or
  // incomplete type and the other is a pointer to a qualified or unqualified
  // version of void...
  if (lhptee->isVoidType()) {
    if (rhptee->isIncompleteOrObjectType())
      return ConvTy;

    // As an extension, we allow cast to/from void* to function pointer.
    assert(rhptee->isFunctionType());
    return Sema::FunctionVoidPointer;
  }

  if (rhptee->isVoidType()) {
    if (lhptee->isIncompleteOrObjectType())
      return ConvTy;

    // As an extension, we allow cast to/from void* to function pointer.
    assert(lhptee->isFunctionType());
    return Sema::FunctionVoidPointer;
  }

  // C99 6.5.16.1p1 (constraint 3): both operands are pointers to qualified or
  // unqualified versions of compatible types, ...
  QualType ltrans = QualType(lhptee, 0), rtrans = QualType(rhptee, 0);
  if (!S.Context.typesAreCompatible(ltrans, rtrans)) {
    // Check if the pointee types are compatible ignoring the sign.
    // We explicitly check for char so that we catch "char" vs
    // "unsigned char" on systems where "char" is unsigned.
    if (lhptee->isCharType())
      ltrans = S.Context.UnsignedCharTy;
    else if (lhptee->hasSignedIntegerRepresentation())
      ltrans = S.Context.getCorrespondingUnsignedType(ltrans);

    if (rhptee->isCharType())
      rtrans = S.Context.UnsignedCharTy;
    else if (rhptee->hasSignedIntegerRepresentation())
      rtrans = S.Context.getCorrespondingUnsignedType(rtrans);

    if (ltrans == rtrans) {
      // Types are compatible ignoring the sign. Qualifier incompatibility
      // takes priority over sign incompatibility because the sign
      // warning can be disabled.
      if (ConvTy != Sema::Compatible)
        return ConvTy;

      return Sema::IncompatiblePointerSign;
    }

    // If we are a multi-level pointer, it's possible that our issue is simply
    // one of qualification - e.g. char ** -> const char ** is not allowed. If
    // the eventual target type is the same and the pointers have the same
    // level of indirection, this must be the issue.
    if (isa<PointerType>(lhptee) && isa<PointerType>(rhptee)) {
      do {
        lhptee = cast<PointerType>(lhptee)->getPointeeType().getTypePtr();
        rhptee = cast<PointerType>(rhptee)->getPointeeType().getTypePtr();
      } while (isa<PointerType>(lhptee) && isa<PointerType>(rhptee));

      if (lhptee == rhptee)
        return Sema::IncompatibleNestedPointerQualifiers;
    }

    // General pointer incompatibility takes priority over qualifiers.
    return Sema::IncompatiblePointer;
  }
  if (!S.getLangOpts().CPlusPlus &&
      S.IsNoReturnConversion(ltrans, rtrans, ltrans))
    return Sema::IncompatiblePointer;
  return ConvTy;
}

/// checkBlockPointerTypesForAssignment - This routine determines whether two
/// block pointer types are compatible or whether a block and normal pointer
/// are compatible. It is more restrict than comparing two function pointer
// types.
static Sema::AssignConvertType
checkBlockPointerTypesForAssignment(Sema &S, QualType LHSType,
                                    QualType RHSType) {
  assert(LHSType.isCanonical() && "LHS not canonicalized!");
  assert(RHSType.isCanonical() && "RHS not canonicalized!");

  QualType lhptee, rhptee;

  // get the "pointed to" type (ignoring qualifiers at the top level)
  lhptee = cast<BlockPointerType>(LHSType)->getPointeeType();
  rhptee = cast<BlockPointerType>(RHSType)->getPointeeType();

  // In C++, the types have to match exactly.
  if (S.getLangOpts().CPlusPlus)
    return Sema::IncompatibleBlockPointer;

  Sema::AssignConvertType ConvTy = Sema::Compatible;

  // For blocks we enforce that qualifiers are identical.
  if (lhptee.getLocalQualifiers() != rhptee.getLocalQualifiers())
    ConvTy = Sema::CompatiblePointerDiscardsQualifiers;

  if (!S.Context.typesAreBlockPointerCompatible(LHSType, RHSType))
    return Sema::IncompatibleBlockPointer;

  return ConvTy;
}

/// checkObjCPointerTypesForAssignment - Compares two objective-c pointer types
/// for assignment compatibility.
static Sema::AssignConvertType
checkObjCPointerTypesForAssignment(Sema &S, QualType LHSType,
                                   QualType RHSType) {
  assert(LHSType.isCanonical() && "LHS was not canonicalized!");
  assert(RHSType.isCanonical() && "RHS was not canonicalized!");

  if (LHSType->isObjCBuiltinType()) {
    // Class is not compatible with ObjC object pointers.
    if (LHSType->isObjCClassType() && !RHSType->isObjCBuiltinType() &&
        !RHSType->isObjCQualifiedClassType())
      return Sema::IncompatiblePointer;
    return Sema::Compatible;
  }
  if (RHSType->isObjCBuiltinType()) {
    if (RHSType->isObjCClassType() && !LHSType->isObjCBuiltinType() &&
        !LHSType->isObjCQualifiedClassType())
      return Sema::IncompatiblePointer;
    return Sema::Compatible;
  }
  QualType lhptee = LHSType->getAs<ObjCObjectPointerType>()->getPointeeType();
  QualType rhptee = RHSType->getAs<ObjCObjectPointerType>()->getPointeeType();

  if (!lhptee.isAtLeastAsQualifiedAs(rhptee) &&
      // make an exception for id<P>
      !LHSType->isObjCQualifiedIdType())
    return Sema::CompatiblePointerDiscardsQualifiers;

  if (S.Context.typesAreCompatible(LHSType, RHSType))
    return Sema::Compatible;
  if (LHSType->isObjCQualifiedIdType() || RHSType->isObjCQualifiedIdType())
    return Sema::IncompatibleObjCQualifiedId;
  return Sema::IncompatiblePointer;
}

Sema::AssignConvertType
Sema::CheckAssignmentConstraints(SourceLocation Loc,
                                 QualType LHSType, QualType RHSType) {
  // Fake up an opaque expression.  We don't actually care about what
  // cast operations are required, so if CheckAssignmentConstraints
  // adds casts to this they'll be wasted, but fortunately that doesn't
  // usually happen on valid code.
  OpaqueValueExpr RHSExpr(Loc, RHSType, VK_RValue);
  ExprResult RHSPtr = &RHSExpr;
  CastKind K = CK_Invalid;

  return CheckAssignmentConstraints(LHSType, RHSPtr, K, /*ConvertRHS=*/false);
}

/// CheckAssignmentConstraints (C99 6.5.16) - This routine currently
/// has code to accommodate several GCC extensions when type checking
/// pointers. Here are some objectionable examples that GCC considers warnings:
///
///  int a, *pint;
///  short *pshort;
///  struct foo *pfoo;
///
///  pint = pshort; // warning: assignment from incompatible pointer type
///  a = pint; // warning: assignment makes integer from pointer without a cast
///  pint = a; // warning: assignment makes pointer from integer without a cast
///  pint = pfoo; // warning: assignment from incompatible pointer type
///
/// As a result, the code for dealing with pointers is more complex than the
/// C99 spec dictates.
///
/// Sets 'Kind' for any result kind except Incompatible.
Sema::AssignConvertType
Sema::CheckAssignmentConstraints(QualType LHSType, ExprResult &RHS,
                                 CastKind &Kind, bool ConvertRHS) {
  QualType RHSType = RHS.get()->getType();
  QualType OrigLHSType = LHSType;

  // Get canonical types.  We're not formatting these types, just comparing
  // them.
  LHSType = Context.getCanonicalType(LHSType).getUnqualifiedType();
  RHSType = Context.getCanonicalType(RHSType).getUnqualifiedType();

  // Common case: no conversion required.
  if (LHSType == RHSType) {
    Kind = CK_NoOp;
    return Compatible;
  }

  // If we have an atomic type, try a non-atomic assignment, then just add an
  // atomic qualification step.
  if (const AtomicType *AtomicTy = dyn_cast<AtomicType>(LHSType)) {
    Sema::AssignConvertType result =
      CheckAssignmentConstraints(AtomicTy->getValueType(), RHS, Kind);
    if (result != Compatible)
      return result;
    if (Kind != CK_NoOp && ConvertRHS)
      RHS = ImpCastExprToType(RHS.get(), AtomicTy->getValueType(), Kind);
    Kind = CK_NonAtomicToAtomic;
    return Compatible;
  }

  // If the left-hand side is a reference type, then we are in a
  // (rare!) case where we've allowed the use of references in C,
  // e.g., as a parameter type in a built-in function. In this case,
  // just make sure that the type referenced is compatible with the
  // right-hand side type. The caller is responsible for adjusting
  // LHSType so that the resulting expression does not have reference
  // type.
  if (const ReferenceType *LHSTypeRef = LHSType->getAs<ReferenceType>()) {
    if (Context.typesAreCompatible(LHSTypeRef->getPointeeType(), RHSType)) {
      Kind = CK_LValueBitCast;
      return Compatible;
    }
    return Incompatible;
  }

  // Allow scalar to ExtVector assignments, and assignments of an ExtVector type
  // to the same ExtVector type.
  if (LHSType->isExtVectorType()) {
    if (RHSType->isExtVectorType())
      return Incompatible;
    if (RHSType->isArithmeticType()) {
      // CK_VectorSplat does T -> vector T, so first cast to the element type.
      if (ConvertRHS)
        RHS = prepareVectorSplat(LHSType, RHS.get());
      Kind = CK_VectorSplat;
      return Compatible;
    }
  }

  // Conversions to or from vector type.
  if (LHSType->isVectorType() || RHSType->isVectorType()) {
    if (LHSType->isVectorType() && RHSType->isVectorType()) {
      // Allow assignments of an AltiVec vector type to an equivalent GCC
      // vector type and vice versa
      if (Context.areCompatibleVectorTypes(LHSType, RHSType)) {
        Kind = CK_BitCast;
        return Compatible;
      }

      // If we are allowing lax vector conversions, and LHS and RHS are both
      // vectors, the total size only needs to be the same. This is a bitcast;
      // no bits are changed but the result type is different.
      if (isLaxVectorConversion(RHSType, LHSType)) {
        Kind = CK_BitCast;
        return IncompatibleVectors;
      }
    }
    return Incompatible;
  }

  // Arithmetic conversions.
  if (LHSType->isArithmeticType() && RHSType->isArithmeticType() &&
      !(getLangOpts().CPlusPlus && LHSType->isEnumeralType())) {
    if (ConvertRHS)
      Kind = PrepareScalarCast(RHS, LHSType);
    return Compatible;
  }

  // Conversions to normal pointers.
  if (const PointerType *LHSPointer = dyn_cast<PointerType>(LHSType)) {
    // U* -> T*
    if (isa<PointerType>(RHSType)) {
      unsigned AddrSpaceL = LHSPointer->getPointeeType().getAddressSpace();
      unsigned AddrSpaceR = RHSType->getPointeeType().getAddressSpace();
      Kind = AddrSpaceL != AddrSpaceR ? CK_AddressSpaceConversion : CK_BitCast;
      return checkPointerTypesForAssignment(*this, LHSType, RHSType);
    }

    // int -> T*
    if (RHSType->isIntegerType()) {
      Kind = CK_IntegralToPointer; // FIXME: null?
      return IntToPointer;
    }

    // C pointers are not compatible with ObjC object pointers,
    // with two exceptions:
    if (isa<ObjCObjectPointerType>(RHSType)) {
      //  - conversions to void*
      if (LHSPointer->getPointeeType()->isVoidType()) {
        Kind = CK_BitCast;
        return Compatible;
      }

      //  - conversions from 'Class' to the redefinition type
      if (RHSType->isObjCClassType() &&
          Context.hasSameType(LHSType, 
                              Context.getObjCClassRedefinitionType())) {
        Kind = CK_BitCast;
        return Compatible;
      }

      Kind = CK_BitCast;
      return IncompatiblePointer;
    }

    // U^ -> void*
    if (RHSType->getAs<BlockPointerType>()) {
      if (LHSPointer->getPointeeType()->isVoidType()) {
        Kind = CK_BitCast;
        return Compatible;
      }
    }

    return Incompatible;
  }

  // Conversions to block pointers.
  if (isa<BlockPointerType>(LHSType)) {
    // U^ -> T^
    if (RHSType->isBlockPointerType()) {
      Kind = CK_BitCast;
      return checkBlockPointerTypesForAssignment(*this, LHSType, RHSType);
    }

    // int or null -> T^
    if (RHSType->isIntegerType()) {
      Kind = CK_IntegralToPointer; // FIXME: null
      return IntToBlockPointer;
    }

    // id -> T^
    if (getLangOpts().ObjC1 && RHSType->isObjCIdType()) {
      Kind = CK_AnyPointerToBlockPointerCast;
      return Compatible;
    }

    // void* -> T^
    if (const PointerType *RHSPT = RHSType->getAs<PointerType>())
      if (RHSPT->getPointeeType()->isVoidType()) {
        Kind = CK_AnyPointerToBlockPointerCast;
        return Compatible;
      }

    return Incompatible;
  }

  // Conversions to Objective-C pointers.
  if (isa<ObjCObjectPointerType>(LHSType)) {
    // A* -> B*
    if (RHSType->isObjCObjectPointerType()) {
      Kind = CK_BitCast;
      Sema::AssignConvertType result = 
        checkObjCPointerTypesForAssignment(*this, LHSType, RHSType);
      if (getLangOpts().ObjCAutoRefCount &&
          result == Compatible && 
          !CheckObjCARCUnavailableWeakConversion(OrigLHSType, RHSType))
        result = IncompatibleObjCWeakRef;
      return result;
    }

    // int or null -> A*
    if (RHSType->isIntegerType()) {
      Kind = CK_IntegralToPointer; // FIXME: null
      return IntToPointer;
    }

    // In general, C pointers are not compatible with ObjC object pointers,
    // with two exceptions:
    if (isa<PointerType>(RHSType)) {
      Kind = CK_CPointerToObjCPointerCast;

      //  - conversions from 'void*'
      if (RHSType->isVoidPointerType()) {
        return Compatible;
      }

      //  - conversions to 'Class' from its redefinition type
      if (LHSType->isObjCClassType() &&
          Context.hasSameType(RHSType, 
                              Context.getObjCClassRedefinitionType())) {
        return Compatible;
      }

      return IncompatiblePointer;
    }

    // Only under strict condition T^ is compatible with an Objective-C pointer.
    if (RHSType->isBlockPointerType() && 
        LHSType->isBlockCompatibleObjCPointerType(Context)) {
      if (ConvertRHS)
        maybeExtendBlockObject(RHS);
      Kind = CK_BlockPointerToObjCPointerCast;
      return Compatible;
    }

    return Incompatible;
  }

  // Conversions from pointers that are not covered by the above.
  if (isa<PointerType>(RHSType)) {
    // T* -> _Bool
    if (LHSType == Context.BoolTy) {
      Kind = CK_PointerToBoolean;
      return Compatible;
    }

    // T* -> int
    if (LHSType->isIntegerType()) {
      Kind = CK_PointerToIntegral;
      return PointerToInt;
    }

    return Incompatible;
  }

  // Conversions from Objective-C pointers that are not covered by the above.
  if (isa<ObjCObjectPointerType>(RHSType)) {
    // T* -> _Bool
    if (LHSType == Context.BoolTy) {
      Kind = CK_PointerToBoolean;
      return Compatible;
    }

    // T* -> int
    if (LHSType->isIntegerType()) {
      Kind = CK_PointerToIntegral;
      return PointerToInt;
    }

    return Incompatible;
  }

  // struct A -> struct B
  if (isa<TagType>(LHSType) && isa<TagType>(RHSType)) {
    if (Context.typesAreCompatible(LHSType, RHSType)) {
      Kind = CK_NoOp;
      return Compatible;
    }
  }

  return Incompatible;
}

/// \brief Constructs a transparent union from an expression that is
/// used to initialize the transparent union.
static void ConstructTransparentUnion(Sema &S, ASTContext &C,
                                      ExprResult &EResult, QualType UnionType,
                                      FieldDecl *Field) {
  // Build an initializer list that designates the appropriate member
  // of the transparent union.
  Expr *E = EResult.get();
  InitListExpr *Initializer = new (C) InitListExpr(C, SourceLocation(),
                                                   E, SourceLocation());
  Initializer->setType(UnionType);
  Initializer->setInitializedFieldInUnion(Field);

  // Build a compound literal constructing a value of the transparent
  // union type from this initializer list.
  TypeSourceInfo *unionTInfo = C.getTrivialTypeSourceInfo(UnionType);
  EResult = new (C) CompoundLiteralExpr(SourceLocation(), unionTInfo, UnionType,
                                        VK_RValue, Initializer, false);
}

Sema::AssignConvertType
Sema::CheckTransparentUnionArgumentConstraints(QualType ArgType,
                                               ExprResult &RHS) {
  QualType RHSType = RHS.get()->getType();

  // If the ArgType is a Union type, we want to handle a potential
  // transparent_union GCC extension.
  const RecordType *UT = ArgType->getAsUnionType();
  if (!UT || !UT->getDecl()->hasAttr<TransparentUnionAttr>())
    return Incompatible;

  // The field to initialize within the transparent union.
  RecordDecl *UD = UT->getDecl();
  FieldDecl *InitField = nullptr;
  // It's compatible if the expression matches any of the fields.
  for (auto *it : UD->fields()) {
    if (it->getType()->isPointerType()) {
      // If the transparent union contains a pointer type, we allow:
      // 1) void pointer
      // 2) null pointer constant
      if (RHSType->isPointerType())
        if (RHSType->castAs<PointerType>()->getPointeeType()->isVoidType()) {
          RHS = ImpCastExprToType(RHS.get(), it->getType(), CK_BitCast);
          InitField = it;
          break;
        }

      if (RHS.get()->isNullPointerConstant(Context,
                                           Expr::NPC_ValueDependentIsNull)) {
        RHS = ImpCastExprToType(RHS.get(), it->getType(),
                                CK_NullToPointer);
        InitField = it;
        break;
      }
    }

    CastKind Kind = CK_Invalid;
    if (CheckAssignmentConstraints(it->getType(), RHS, Kind)
          == Compatible) {
      RHS = ImpCastExprToType(RHS.get(), it->getType(), Kind);
      InitField = it;
      break;
    }
  }

  if (!InitField)
    return Incompatible;

  ConstructTransparentUnion(*this, Context, RHS, ArgType, InitField);
  return Compatible;
}

Sema::AssignConvertType
Sema::CheckSingleAssignmentConstraints(QualType LHSType, ExprResult &CallerRHS,
                                       bool Diagnose,
                                       bool DiagnoseCFAudited,
                                       bool ConvertRHS) {
  // If ConvertRHS is false, we want to leave the caller's RHS untouched. Sadly,
  // we can't avoid *all* modifications at the moment, so we need some somewhere
  // to put the updated value.
  ExprResult LocalRHS = CallerRHS;
  ExprResult &RHS = ConvertRHS ? CallerRHS : LocalRHS;

  if (getLangOpts().CPlusPlus) {
    if (!LHSType->isRecordType() && !LHSType->isAtomicType()) {
      // C++ 5.17p3: If the left operand is not of class type, the
      // expression is implicitly converted (C++ 4) to the
      // cv-unqualified type of the left operand.
      ExprResult Res;
      if (Diagnose) {
        Res = PerformImplicitConversion(RHS.get(), LHSType.getUnqualifiedType(),
                                        AA_Assigning);
      } else {
        ImplicitConversionSequence ICS =
            TryImplicitConversion(RHS.get(), LHSType.getUnqualifiedType(),
                                  /*SuppressUserConversions=*/false,
                                  /*AllowExplicit=*/false,
                                  /*InOverloadResolution=*/false,
                                  /*CStyle=*/false,
                                  /*AllowObjCWritebackConversion=*/false);
        if (ICS.isFailure())
          return Incompatible;
        Res = PerformImplicitConversion(RHS.get(), LHSType.getUnqualifiedType(),
                                        ICS, AA_Assigning);
      }
      if (Res.isInvalid())
        return Incompatible;
      Sema::AssignConvertType result = Compatible;
      if (getLangOpts().ObjCAutoRefCount &&
          !CheckObjCARCUnavailableWeakConversion(LHSType,
                                                 RHS.get()->getType()))
        result = IncompatibleObjCWeakRef;
      RHS = Res;
      return result;
    }

    // FIXME: Currently, we fall through and treat C++ classes like C
    // structures.
    // FIXME: We also fall through for atomics; not sure what should
    // happen there, though.
  } else if (RHS.get()->getType() == Context.OverloadTy) {
    // As a set of extensions to C, we support overloading on functions. These
    // functions need to be resolved here.
    DeclAccessPair DAP;
    if (FunctionDecl *FD = ResolveAddressOfOverloadedFunction(
            RHS.get(), LHSType, /*Complain=*/false, DAP))
      RHS = FixOverloadedFunctionReference(RHS.get(), DAP, FD);
    else
      return Incompatible;
  }

  // C99 6.5.16.1p1: the left operand is a pointer and the right is
  // a null pointer constant.
  if ((LHSType->isPointerType() || LHSType->isObjCObjectPointerType() ||
       LHSType->isBlockPointerType()) &&
      RHS.get()->isNullPointerConstant(Context,
                                       Expr::NPC_ValueDependentIsNull)) {
    if (Diagnose || ConvertRHS) {
      CastKind Kind;
      CXXCastPath Path;
      CheckPointerConversion(RHS.get(), LHSType, Kind, Path,
                             /*IgnoreBaseAccess=*/false, Diagnose);
      if (ConvertRHS)
        RHS = ImpCastExprToType(RHS.get(), LHSType, Kind, VK_RValue, &Path);
    }
    return Compatible;
  }

  // This check seems unnatural, however it is necessary to ensure the proper
  // conversion of functions/arrays. If the conversion were done for all
  // DeclExpr's (created by ActOnIdExpression), it would mess up the unary
  // expressions that suppress this implicit conversion (&, sizeof).
  //
  // Suppress this for references: C++ 8.5.3p5.
  if (!LHSType->isReferenceType()) {
    // FIXME: We potentially allocate here even if ConvertRHS is false.
    RHS = DefaultFunctionArrayLvalueConversion(RHS.get(), Diagnose);
    if (RHS.isInvalid())
      return Incompatible;
  }

  Expr *PRE = RHS.get()->IgnoreParenCasts();
  if (Diagnose && isa<ObjCProtocolExpr>(PRE)) {
    ObjCProtocolDecl *PDecl = cast<ObjCProtocolExpr>(PRE)->getProtocol();
    if (PDecl && !PDecl->hasDefinition()) {
      Diag(PRE->getExprLoc(), diag::warn_atprotocol_protocol) << PDecl->getName();
      Diag(PDecl->getLocation(), diag::note_entity_declared_at) << PDecl;
    }
  }
  
  CastKind Kind = CK_Invalid;
  Sema::AssignConvertType result =
    CheckAssignmentConstraints(LHSType, RHS, Kind, ConvertRHS);

  // C99 6.5.16.1p2: The value of the right operand is converted to the
  // type of the assignment expression.
  // CheckAssignmentConstraints allows the left-hand side to be a reference,
  // so that we can use references in built-in functions even in C.
  // The getNonReferenceType() call makes sure that the resulting expression
  // does not have reference type.
  if (result != Incompatible && RHS.get()->getType() != LHSType) {
    QualType Ty = LHSType.getNonLValueExprType(Context);
    Expr *E = RHS.get();
    if (getLangOpts().ObjCAutoRefCount)
      CheckObjCARCConversion(SourceRange(), Ty, E, CCK_ImplicitConversion,
                             Diagnose, DiagnoseCFAudited);
    if (getLangOpts().ObjC1 &&
        (CheckObjCBridgeRelatedConversions(E->getLocStart(), LHSType,
                                           E->getType(), E, Diagnose) ||
         ConversionToObjCStringLiteralCheck(LHSType, E, Diagnose))) {
      RHS = E;
      return Compatible;
    }
    
    if (ConvertRHS)
      RHS = ImpCastExprToType(E, Ty, Kind);
  }
  return result;
}

QualType Sema::InvalidOperands(SourceLocation Loc, ExprResult &LHS,
                               ExprResult &RHS) {
  Diag(Loc, diag::err_typecheck_invalid_operands)
    << LHS.get()->getType() << RHS.get()->getType()
    << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
  return QualType();
}

/// Try to convert a value of non-vector type to a vector type by converting
/// the type to the element type of the vector and then performing a splat.
/// If the language is OpenCL, we only use conversions that promote scalar
/// rank; for C, Obj-C, and C++ we allow any real scalar conversion except
/// for float->int.
///
/// \param scalar - if non-null, actually perform the conversions
/// \return true if the operation fails (but without diagnosing the failure)
static bool tryVectorConvertAndSplat(Sema &S, ExprResult *scalar,
                                     QualType scalarTy,
                                     QualType vectorEltTy,
                                     QualType vectorTy) {
  // The conversion to apply to the scalar before splatting it,
  // if necessary.
  CastKind scalarCast = CK_Invalid;
  
  if (vectorEltTy->isIntegralType(S.Context)) {
    if (!scalarTy->isIntegralType(S.Context))
      return true;
    if (S.getLangOpts().OpenCL &&
        S.Context.getIntegerTypeOrder(vectorEltTy, scalarTy) < 0)
      return true;
    scalarCast = CK_IntegralCast;
  } else if (vectorEltTy->isRealFloatingType()) {
    if (scalarTy->isRealFloatingType()) {
      if (S.getLangOpts().OpenCL &&
          S.Context.getFloatingTypeOrder(vectorEltTy, scalarTy) < 0)
        return true;
      scalarCast = CK_FloatingCast;
    }
    else if (scalarTy->isIntegralType(S.Context))
      scalarCast = CK_IntegralToFloating;
    else
      return true;
  } else {
    return true;
  }

  // Adjust scalar if desired.
  if (scalar) {
    if (scalarCast != CK_Invalid)
      *scalar = S.ImpCastExprToType(scalar->get(), vectorEltTy, scalarCast);
    *scalar = S.ImpCastExprToType(scalar->get(), vectorTy, CK_VectorSplat);
  }
  return false;
}

QualType Sema::CheckVectorOperands(ExprResult &LHS, ExprResult &RHS,
                                   SourceLocation Loc, bool IsCompAssign,
                                   bool AllowBothBool,
                                   bool AllowBoolConversions) {
  if (!IsCompAssign) {
    LHS = DefaultFunctionArrayLvalueConversion(LHS.get());
    if (LHS.isInvalid())
      return QualType();
  }
  RHS = DefaultFunctionArrayLvalueConversion(RHS.get());
  if (RHS.isInvalid())
    return QualType();

  // For conversion purposes, we ignore any qualifiers.
  // For example, "const float" and "float" are equivalent.
  QualType LHSType = LHS.get()->getType().getUnqualifiedType();
  QualType RHSType = RHS.get()->getType().getUnqualifiedType();

  const VectorType *LHSVecType = LHSType->getAs<VectorType>();
  const VectorType *RHSVecType = RHSType->getAs<VectorType>();
  assert(LHSVecType || RHSVecType);

  // AltiVec-style "vector bool op vector bool" combinations are allowed
  // for some operators but not others.
  if (!AllowBothBool &&
      LHSVecType && LHSVecType->getVectorKind() == VectorType::AltiVecBool &&
      RHSVecType && RHSVecType->getVectorKind() == VectorType::AltiVecBool)
    return InvalidOperands(Loc, LHS, RHS);

  // If the vector types are identical, return.
  if (Context.hasSameType(LHSType, RHSType))
    return LHSType;

  // If we have compatible AltiVec and GCC vector types, use the AltiVec type.
  if (LHSVecType && RHSVecType &&
      Context.areCompatibleVectorTypes(LHSType, RHSType)) {
    if (isa<ExtVectorType>(LHSVecType)) {
      RHS = ImpCastExprToType(RHS.get(), LHSType, CK_BitCast);
      return LHSType;
    }

    if (!IsCompAssign)
      LHS = ImpCastExprToType(LHS.get(), RHSType, CK_BitCast);
    return RHSType;
  }

  // AllowBoolConversions says that bool and non-bool AltiVec vectors
  // can be mixed, with the result being the non-bool type.  The non-bool
  // operand must have integer element type.
  if (AllowBoolConversions && LHSVecType && RHSVecType &&
      LHSVecType->getNumElements() == RHSVecType->getNumElements() &&
      (Context.getTypeSize(LHSVecType->getElementType()) ==
       Context.getTypeSize(RHSVecType->getElementType()))) {
    if (LHSVecType->getVectorKind() == VectorType::AltiVecVector &&
        LHSVecType->getElementType()->isIntegerType() &&
        RHSVecType->getVectorKind() == VectorType::AltiVecBool) {
      RHS = ImpCastExprToType(RHS.get(), LHSType, CK_BitCast);
      return LHSType;
    }
    if (!IsCompAssign &&
        LHSVecType->getVectorKind() == VectorType::AltiVecBool &&
        RHSVecType->getVectorKind() == VectorType::AltiVecVector &&
        RHSVecType->getElementType()->isIntegerType()) {
      LHS = ImpCastExprToType(LHS.get(), RHSType, CK_BitCast);
      return RHSType;
    }
  }

  // If there's an ext-vector type and a scalar, try to convert the scalar to
  // the vector element type and splat.
  if (!RHSVecType && isa<ExtVectorType>(LHSVecType)) {
    if (!tryVectorConvertAndSplat(*this, &RHS, RHSType,
                                  LHSVecType->getElementType(), LHSType))
      return LHSType;
  }
  if (!LHSVecType && isa<ExtVectorType>(RHSVecType)) {
    if (!tryVectorConvertAndSplat(*this, (IsCompAssign ? nullptr : &LHS),
                                  LHSType, RHSVecType->getElementType(),
                                  RHSType))
      return RHSType;
  }

  // If we're allowing lax vector conversions, only the total (data) size
  // needs to be the same.
  // FIXME: Should we really be allowing this?
  // FIXME: We really just pick the LHS type arbitrarily?
  if (isLaxVectorConversion(RHSType, LHSType)) {
    QualType resultType = LHSType;
    RHS = ImpCastExprToType(RHS.get(), resultType, CK_BitCast);
    return resultType;
  }

  // Okay, the expression is invalid.

  // If there's a non-vector, non-real operand, diagnose that.
  if ((!RHSVecType && !RHSType->isRealType()) ||
      (!LHSVecType && !LHSType->isRealType())) {
    Diag(Loc, diag::err_typecheck_vector_not_convertable_non_scalar)
      << LHSType << RHSType
      << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
    return QualType();
  }

  // OpenCL V1.1 6.2.6.p1:
  // If the operands are of more than one vector type, then an error shall
  // occur. Implicit conversions between vector types are not permitted, per
  // section 6.2.1.
  if (getLangOpts().OpenCL &&
      RHSVecType && isa<ExtVectorType>(RHSVecType) &&
      LHSVecType && isa<ExtVectorType>(LHSVecType)) {
    Diag(Loc, diag::err_opencl_implicit_vector_conversion) << LHSType
                                                           << RHSType;
    return QualType();
  }

  // Otherwise, use the generic diagnostic.
  Diag(Loc, diag::err_typecheck_vector_not_convertable)
    << LHSType << RHSType
    << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
  return QualType();
}

// checkArithmeticNull - Detect when a NULL constant is used improperly in an
// expression.  These are mainly cases where the null pointer is used as an
// integer instead of a pointer.
static void checkArithmeticNull(Sema &S, ExprResult &LHS, ExprResult &RHS,
                                SourceLocation Loc, bool IsCompare) {
  // The canonical way to check for a GNU null is with isNullPointerConstant,
  // but we use a bit of a hack here for speed; this is a relatively
  // hot path, and isNullPointerConstant is slow.
  bool LHSNull = isa<GNUNullExpr>(LHS.get()->IgnoreParenImpCasts());
  bool RHSNull = isa<GNUNullExpr>(RHS.get()->IgnoreParenImpCasts());

  QualType NonNullType = LHSNull ? RHS.get()->getType() : LHS.get()->getType();

  // Avoid analyzing cases where the result will either be invalid (and
  // diagnosed as such) or entirely valid and not something to warn about.
  if ((!LHSNull && !RHSNull) || NonNullType->isBlockPointerType() ||
      NonNullType->isMemberPointerType() || NonNullType->isFunctionType())
    return;

  // Comparison operations would not make sense with a null pointer no matter
  // what the other expression is.
  if (!IsCompare) {
    S.Diag(Loc, diag::warn_null_in_arithmetic_operation)
        << (LHSNull ? LHS.get()->getSourceRange() : SourceRange())
        << (RHSNull ? RHS.get()->getSourceRange() : SourceRange());
    return;
  }

  // The rest of the operations only make sense with a null pointer
  // if the other expression is a pointer.
  if (LHSNull == RHSNull || NonNullType->isAnyPointerType() ||
      NonNullType->canDecayToPointerType())
    return;

  S.Diag(Loc, diag::warn_null_in_comparison_operation)
      << LHSNull /* LHS is NULL */ << NonNullType
      << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
}

static void DiagnoseBadDivideOrRemainderValues(Sema& S, ExprResult &LHS,
                                               ExprResult &RHS,
                                               SourceLocation Loc, bool IsDiv) {
  // Check for division/remainder by zero.
  llvm::APSInt RHSValue;
  if (!RHS.get()->isValueDependent() &&
      RHS.get()->EvaluateAsInt(RHSValue, S.Context) && RHSValue == 0)
    S.DiagRuntimeBehavior(Loc, RHS.get(),
                          S.PDiag(diag::warn_remainder_division_by_zero)
                            << IsDiv << RHS.get()->getSourceRange());
}

QualType Sema::CheckMultiplyDivideOperands(ExprResult &LHS, ExprResult &RHS,
                                           SourceLocation Loc,
                                           bool IsCompAssign, bool IsDiv) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType())
    return CheckVectorOperands(LHS, RHS, Loc, IsCompAssign,
                               /*AllowBothBool*/getLangOpts().AltiVec,
                               /*AllowBoolConversions*/false);

  QualType compType = UsualArithmeticConversions(LHS, RHS, IsCompAssign);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();


  if (compType.isNull() || !compType->isArithmeticType())
    return InvalidOperands(Loc, LHS, RHS);
  if (IsDiv)
    DiagnoseBadDivideOrRemainderValues(*this, LHS, RHS, Loc, IsDiv);
  return compType;
}

QualType Sema::CheckRemainderOperands(
  ExprResult &LHS, ExprResult &RHS, SourceLocation Loc, bool IsCompAssign) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    if (LHS.get()->getType()->hasIntegerRepresentation() && 
        RHS.get()->getType()->hasIntegerRepresentation())
      return CheckVectorOperands(LHS, RHS, Loc, IsCompAssign,
                                 /*AllowBothBool*/getLangOpts().AltiVec,
                                 /*AllowBoolConversions*/false);
    return InvalidOperands(Loc, LHS, RHS);
  }

  QualType compType = UsualArithmeticConversions(LHS, RHS, IsCompAssign);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  if (compType.isNull() || !compType->isIntegerType())
    return InvalidOperands(Loc, LHS, RHS);
  DiagnoseBadDivideOrRemainderValues(*this, LHS, RHS, Loc, false /* IsDiv */);
  return compType;
}

/// \brief Diagnose invalid arithmetic on two void pointers.
static void diagnoseArithmeticOnTwoVoidPointers(Sema &S, SourceLocation Loc,
                                                Expr *LHSExpr, Expr *RHSExpr) {
  S.Diag(Loc, S.getLangOpts().CPlusPlus
                ? diag::err_typecheck_pointer_arith_void_type
                : diag::ext_gnu_void_ptr)
    << 1 /* two pointers */ << LHSExpr->getSourceRange()
                            << RHSExpr->getSourceRange();
}

/// \brief Diagnose invalid arithmetic on a void pointer.
static void diagnoseArithmeticOnVoidPointer(Sema &S, SourceLocation Loc,
                                            Expr *Pointer) {
  S.Diag(Loc, S.getLangOpts().CPlusPlus
                ? diag::err_typecheck_pointer_arith_void_type
                : diag::ext_gnu_void_ptr)
    << 0 /* one pointer */ << Pointer->getSourceRange();
}

/// \brief Diagnose invalid arithmetic on two function pointers.
static void diagnoseArithmeticOnTwoFunctionPointers(Sema &S, SourceLocation Loc,
                                                    Expr *LHS, Expr *RHS) {
  assert(LHS->getType()->isAnyPointerType());
  assert(RHS->getType()->isAnyPointerType());
  S.Diag(Loc, S.getLangOpts().CPlusPlus
                ? diag::err_typecheck_pointer_arith_function_type
                : diag::ext_gnu_ptr_func_arith)
    << 1 /* two pointers */ << LHS->getType()->getPointeeType()
    // We only show the second type if it differs from the first.
    << (unsigned)!S.Context.hasSameUnqualifiedType(LHS->getType(),
                                                   RHS->getType())
    << RHS->getType()->getPointeeType()
    << LHS->getSourceRange() << RHS->getSourceRange();
}

/// \brief Diagnose invalid arithmetic on a function pointer.
static void diagnoseArithmeticOnFunctionPointer(Sema &S, SourceLocation Loc,
                                                Expr *Pointer) {
  assert(Pointer->getType()->isAnyPointerType());
  S.Diag(Loc, S.getLangOpts().CPlusPlus
                ? diag::err_typecheck_pointer_arith_function_type
                : diag::ext_gnu_ptr_func_arith)
    << 0 /* one pointer */ << Pointer->getType()->getPointeeType()
    << 0 /* one pointer, so only one type */
    << Pointer->getSourceRange();
}

/// \brief Emit error if Operand is incomplete pointer type
///
/// \returns True if pointer has incomplete type
static bool checkArithmeticIncompletePointerType(Sema &S, SourceLocation Loc,
                                                 Expr *Operand) {
  QualType ResType = Operand->getType();
  if (const AtomicType *ResAtomicType = ResType->getAs<AtomicType>())
    ResType = ResAtomicType->getValueType();

  assert(ResType->isAnyPointerType() && !ResType->isDependentType());
  QualType PointeeTy = ResType->getPointeeType();
  return S.RequireCompleteType(Loc, PointeeTy,
                               diag::err_typecheck_arithmetic_incomplete_type,
                               PointeeTy, Operand->getSourceRange());
}

/// \brief Check the validity of an arithmetic pointer operand.
///
/// If the operand has pointer type, this code will check for pointer types
/// which are invalid in arithmetic operations. These will be diagnosed
/// appropriately, including whether or not the use is supported as an
/// extension.
///
/// \returns True when the operand is valid to use (even if as an extension).
static bool checkArithmeticOpPointerOperand(Sema &S, SourceLocation Loc,
                                            Expr *Operand) {
  QualType ResType = Operand->getType();
  if (const AtomicType *ResAtomicType = ResType->getAs<AtomicType>())
    ResType = ResAtomicType->getValueType();

  if (!ResType->isAnyPointerType()) return true;

  QualType PointeeTy = ResType->getPointeeType();
  if (PointeeTy->isVoidType()) {
    diagnoseArithmeticOnVoidPointer(S, Loc, Operand);
    return !S.getLangOpts().CPlusPlus;
  }
  if (PointeeTy->isFunctionType()) {
    diagnoseArithmeticOnFunctionPointer(S, Loc, Operand);
    return !S.getLangOpts().CPlusPlus;
  }

  if (checkArithmeticIncompletePointerType(S, Loc, Operand)) return false;

  return true;
}

/// \brief Check the validity of a binary arithmetic operation w.r.t. pointer
/// operands.
///
/// This routine will diagnose any invalid arithmetic on pointer operands much
/// like \see checkArithmeticOpPointerOperand. However, it has special logic
/// for emitting a single diagnostic even for operations where both LHS and RHS
/// are (potentially problematic) pointers.
///
/// \returns True when the operand is valid to use (even if as an extension).
static bool checkArithmeticBinOpPointerOperands(Sema &S, SourceLocation Loc,
                                                Expr *LHSExpr, Expr *RHSExpr) {
  bool isLHSPointer = LHSExpr->getType()->isAnyPointerType();
  bool isRHSPointer = RHSExpr->getType()->isAnyPointerType();
  if (!isLHSPointer && !isRHSPointer) return true;

  QualType LHSPointeeTy, RHSPointeeTy;
  if (isLHSPointer) LHSPointeeTy = LHSExpr->getType()->getPointeeType();
  if (isRHSPointer) RHSPointeeTy = RHSExpr->getType()->getPointeeType();

  // if both are pointers check if operation is valid wrt address spaces
  if (S.getLangOpts().OpenCL && isLHSPointer && isRHSPointer) {
    const PointerType *lhsPtr = LHSExpr->getType()->getAs<PointerType>();
    const PointerType *rhsPtr = RHSExpr->getType()->getAs<PointerType>();
    if (!lhsPtr->isAddressSpaceOverlapping(*rhsPtr)) {
      S.Diag(Loc,
             diag::err_typecheck_op_on_nonoverlapping_address_space_pointers)
          << LHSExpr->getType() << RHSExpr->getType() << 1 /*arithmetic op*/
          << LHSExpr->getSourceRange() << RHSExpr->getSourceRange();
      return false;
    }
  }

  // Check for arithmetic on pointers to incomplete types.
  bool isLHSVoidPtr = isLHSPointer && LHSPointeeTy->isVoidType();
  bool isRHSVoidPtr = isRHSPointer && RHSPointeeTy->isVoidType();
  if (isLHSVoidPtr || isRHSVoidPtr) {
    if (!isRHSVoidPtr) diagnoseArithmeticOnVoidPointer(S, Loc, LHSExpr);
    else if (!isLHSVoidPtr) diagnoseArithmeticOnVoidPointer(S, Loc, RHSExpr);
    else diagnoseArithmeticOnTwoVoidPointers(S, Loc, LHSExpr, RHSExpr);

    return !S.getLangOpts().CPlusPlus;
  }

  bool isLHSFuncPtr = isLHSPointer && LHSPointeeTy->isFunctionType();
  bool isRHSFuncPtr = isRHSPointer && RHSPointeeTy->isFunctionType();
  if (isLHSFuncPtr || isRHSFuncPtr) {
    if (!isRHSFuncPtr) diagnoseArithmeticOnFunctionPointer(S, Loc, LHSExpr);
    else if (!isLHSFuncPtr) diagnoseArithmeticOnFunctionPointer(S, Loc,
                                                                RHSExpr);
    else diagnoseArithmeticOnTwoFunctionPointers(S, Loc, LHSExpr, RHSExpr);

    return !S.getLangOpts().CPlusPlus;
  }

  if (isLHSPointer && checkArithmeticIncompletePointerType(S, Loc, LHSExpr))
    return false;
  if (isRHSPointer && checkArithmeticIncompletePointerType(S, Loc, RHSExpr))
    return false;

  return true;
}

/// diagnoseStringPlusInt - Emit a warning when adding an integer to a string
/// literal.
static void diagnoseStringPlusInt(Sema &Self, SourceLocation OpLoc,
                                  Expr *LHSExpr, Expr *RHSExpr) {
  StringLiteral* StrExpr = dyn_cast<StringLiteral>(LHSExpr->IgnoreImpCasts());
  Expr* IndexExpr = RHSExpr;
  if (!StrExpr) {
    StrExpr = dyn_cast<StringLiteral>(RHSExpr->IgnoreImpCasts());
    IndexExpr = LHSExpr;
  }

  bool IsStringPlusInt = StrExpr &&
      IndexExpr->getType()->isIntegralOrUnscopedEnumerationType();
  if (!IsStringPlusInt || IndexExpr->isValueDependent())
    return;

  llvm::APSInt index;
  if (IndexExpr->EvaluateAsInt(index, Self.getASTContext())) {
    unsigned StrLenWithNull = StrExpr->getLength() + 1;
    if (index.isNonNegative() &&
        index <= llvm::APSInt(llvm::APInt(index.getBitWidth(), StrLenWithNull),
                              index.isUnsigned()))
      return;
  }

  SourceRange DiagRange(LHSExpr->getLocStart(), RHSExpr->getLocEnd());
  Self.Diag(OpLoc, diag::warn_string_plus_int)
      << DiagRange << IndexExpr->IgnoreImpCasts()->getType();

  // Only print a fixit for "str" + int, not for int + "str".
  if (IndexExpr == RHSExpr) {
    SourceLocation EndLoc = Self.getLocForEndOfToken(RHSExpr->getLocEnd());
    Self.Diag(OpLoc, diag::note_string_plus_scalar_silence)
        << FixItHint::CreateInsertion(LHSExpr->getLocStart(), "&")
        << FixItHint::CreateReplacement(SourceRange(OpLoc), "[")
        << FixItHint::CreateInsertion(EndLoc, "]");
  } else
    Self.Diag(OpLoc, diag::note_string_plus_scalar_silence);
}

/// \brief Emit a warning when adding a char literal to a string.
static void diagnoseStringPlusChar(Sema &Self, SourceLocation OpLoc,
                                   Expr *LHSExpr, Expr *RHSExpr) {
  const Expr *StringRefExpr = LHSExpr;
  const CharacterLiteral *CharExpr =
      dyn_cast<CharacterLiteral>(RHSExpr->IgnoreImpCasts());

  if (!CharExpr) {
    CharExpr = dyn_cast<CharacterLiteral>(LHSExpr->IgnoreImpCasts());
    StringRefExpr = RHSExpr;
  }

  if (!CharExpr || !StringRefExpr)
    return;

  const QualType StringType = StringRefExpr->getType();

  // Return if not a PointerType.
  if (!StringType->isAnyPointerType())
    return;

  // Return if not a CharacterType.
  if (!StringType->getPointeeType()->isAnyCharacterType())
    return;

  ASTContext &Ctx = Self.getASTContext();
  SourceRange DiagRange(LHSExpr->getLocStart(), RHSExpr->getLocEnd());

  const QualType CharType = CharExpr->getType();
  if (!CharType->isAnyCharacterType() &&
      CharType->isIntegerType() &&
      llvm::isUIntN(Ctx.getCharWidth(), CharExpr->getValue())) {
    Self.Diag(OpLoc, diag::warn_string_plus_char)
        << DiagRange << Ctx.CharTy;
  } else {
    Self.Diag(OpLoc, diag::warn_string_plus_char)
        << DiagRange << CharExpr->getType();
  }

  // Only print a fixit for str + char, not for char + str.
  if (isa<CharacterLiteral>(RHSExpr->IgnoreImpCasts())) {
    SourceLocation EndLoc = Self.getLocForEndOfToken(RHSExpr->getLocEnd());
    Self.Diag(OpLoc, diag::note_string_plus_scalar_silence)
        << FixItHint::CreateInsertion(LHSExpr->getLocStart(), "&")
        << FixItHint::CreateReplacement(SourceRange(OpLoc), "[")
        << FixItHint::CreateInsertion(EndLoc, "]");
  } else {
    Self.Diag(OpLoc, diag::note_string_plus_scalar_silence);
  }
}

/// \brief Emit error when two pointers are incompatible.
static void diagnosePointerIncompatibility(Sema &S, SourceLocation Loc,
                                           Expr *LHSExpr, Expr *RHSExpr) {
  assert(LHSExpr->getType()->isAnyPointerType());
  assert(RHSExpr->getType()->isAnyPointerType());
  S.Diag(Loc, diag::err_typecheck_sub_ptr_compatible)
    << LHSExpr->getType() << RHSExpr->getType() << LHSExpr->getSourceRange()
    << RHSExpr->getSourceRange();
}

// C99 6.5.6
QualType Sema::CheckAdditionOperands(ExprResult &LHS, ExprResult &RHS,
                                     SourceLocation Loc, BinaryOperatorKind Opc,
                                     QualType* CompLHSTy) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    QualType compType = CheckVectorOperands(
        LHS, RHS, Loc, CompLHSTy,
        /*AllowBothBool*/getLangOpts().AltiVec,
        /*AllowBoolConversions*/getLangOpts().ZVector);
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  QualType compType = UsualArithmeticConversions(LHS, RHS, CompLHSTy);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  // Diagnose "string literal" '+' int and string '+' "char literal".
  if (Opc == BO_Add) {
    diagnoseStringPlusInt(*this, Loc, LHS.get(), RHS.get());
    diagnoseStringPlusChar(*this, Loc, LHS.get(), RHS.get());
  }

  // handle the common case first (both operands are arithmetic).
  if (!compType.isNull() && compType->isArithmeticType()) {
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  // Type-checking.  Ultimately the pointer's going to be in PExp;
  // note that we bias towards the LHS being the pointer.
  Expr *PExp = LHS.get(), *IExp = RHS.get();

  bool isObjCPointer;
  if (PExp->getType()->isPointerType()) {
    isObjCPointer = false;
  } else if (PExp->getType()->isObjCObjectPointerType()) {
    isObjCPointer = true;
  } else {
    std::swap(PExp, IExp);
    if (PExp->getType()->isPointerType()) {
      isObjCPointer = false;
    } else if (PExp->getType()->isObjCObjectPointerType()) {
      isObjCPointer = true;
    } else {
      return InvalidOperands(Loc, LHS, RHS);
    }
  }
  assert(PExp->getType()->isAnyPointerType());

  if (!IExp->getType()->isIntegerType())
    return InvalidOperands(Loc, LHS, RHS);

  if (!checkArithmeticOpPointerOperand(*this, Loc, PExp))
    return QualType();

  if (isObjCPointer && checkArithmeticOnObjCPointer(*this, Loc, PExp))
    return QualType();

  // Check array bounds for pointer arithemtic
  CheckArrayAccess(PExp, IExp);

  if (CompLHSTy) {
    QualType LHSTy = Context.isPromotableBitField(LHS.get());
    if (LHSTy.isNull()) {
      LHSTy = LHS.get()->getType();
      if (LHSTy->isPromotableIntegerType())
        LHSTy = Context.getPromotedIntegerType(LHSTy);
    }
    *CompLHSTy = LHSTy;
  }

  return PExp->getType();
}

// C99 6.5.6
QualType Sema::CheckSubtractionOperands(ExprResult &LHS, ExprResult &RHS,
                                        SourceLocation Loc,
                                        QualType* CompLHSTy) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    QualType compType = CheckVectorOperands(
        LHS, RHS, Loc, CompLHSTy,
        /*AllowBothBool*/getLangOpts().AltiVec,
        /*AllowBoolConversions*/getLangOpts().ZVector);
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  QualType compType = UsualArithmeticConversions(LHS, RHS, CompLHSTy);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  // Enforce type constraints: C99 6.5.6p3.

  // Handle the common case first (both operands are arithmetic).
  if (!compType.isNull() && compType->isArithmeticType()) {
    if (CompLHSTy) *CompLHSTy = compType;
    return compType;
  }

  // Either ptr - int   or   ptr - ptr.
  if (LHS.get()->getType()->isAnyPointerType()) {
    QualType lpointee = LHS.get()->getType()->getPointeeType();

    // Diagnose bad cases where we step over interface counts.
    if (LHS.get()->getType()->isObjCObjectPointerType() &&
        checkArithmeticOnObjCPointer(*this, Loc, LHS.get()))
      return QualType();

    // The result type of a pointer-int computation is the pointer type.
    if (RHS.get()->getType()->isIntegerType()) {
      if (!checkArithmeticOpPointerOperand(*this, Loc, LHS.get()))
        return QualType();

      // Check array bounds for pointer arithemtic
      CheckArrayAccess(LHS.get(), RHS.get(), /*ArraySubscriptExpr*/nullptr,
                       /*AllowOnePastEnd*/true, /*IndexNegated*/true);

      if (CompLHSTy) *CompLHSTy = LHS.get()->getType();
      return LHS.get()->getType();
    }

    // Handle pointer-pointer subtractions.
    if (const PointerType *RHSPTy
          = RHS.get()->getType()->getAs<PointerType>()) {
      QualType rpointee = RHSPTy->getPointeeType();

      if (getLangOpts().CPlusPlus) {
        // Pointee types must be the same: C++ [expr.add]
        if (!Context.hasSameUnqualifiedType(lpointee, rpointee)) {
          diagnosePointerIncompatibility(*this, Loc, LHS.get(), RHS.get());
        }
      } else {
        // Pointee types must be compatible C99 6.5.6p3
        if (!Context.typesAreCompatible(
                Context.getCanonicalType(lpointee).getUnqualifiedType(),
                Context.getCanonicalType(rpointee).getUnqualifiedType())) {
          diagnosePointerIncompatibility(*this, Loc, LHS.get(), RHS.get());
          return QualType();
        }
      }

      if (!checkArithmeticBinOpPointerOperands(*this, Loc,
                                               LHS.get(), RHS.get()))
        return QualType();

      // The pointee type may have zero size.  As an extension, a structure or
      // union may have zero size or an array may have zero length.  In this
      // case subtraction does not make sense.
      if (!rpointee->isVoidType() && !rpointee->isFunctionType()) {
        CharUnits ElementSize = Context.getTypeSizeInChars(rpointee);
        if (ElementSize.isZero()) {
          Diag(Loc,diag::warn_sub_ptr_zero_size_types)
            << rpointee.getUnqualifiedType()
            << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
        }
      }

      if (CompLHSTy) *CompLHSTy = LHS.get()->getType();
      return Context.getPointerDiffType();
    }
  }

  return InvalidOperands(Loc, LHS, RHS);
}

static bool isScopedEnumerationType(QualType T) {
  if (const EnumType *ET = T->getAs<EnumType>())
    return ET->getDecl()->isScoped();
  return false;
}

static void DiagnoseBadShiftValues(Sema& S, ExprResult &LHS, ExprResult &RHS,
                                   SourceLocation Loc, BinaryOperatorKind Opc,
                                   QualType LHSType) {
  // OpenCL 6.3j: shift values are effectively % word size of LHS (more defined),
  // so skip remaining warnings as we don't want to modify values within Sema.
  if (S.getLangOpts().OpenCL)
    return;

  llvm::APSInt Right;
  // Check right/shifter operand
  if (RHS.get()->isValueDependent() ||
      !RHS.get()->EvaluateAsInt(Right, S.Context))
    return;

  if (Right.isNegative()) {
    S.DiagRuntimeBehavior(Loc, RHS.get(),
                          S.PDiag(diag::warn_shift_negative)
                            << RHS.get()->getSourceRange());
    return;
  }
  llvm::APInt LeftBits(Right.getBitWidth(),
                       S.Context.getTypeSize(LHS.get()->getType()));
  if (Right.uge(LeftBits)) {
    S.DiagRuntimeBehavior(Loc, RHS.get(),
                          S.PDiag(diag::warn_shift_gt_typewidth)
                            << RHS.get()->getSourceRange());
    return;
  }
  if (Opc != BO_Shl)
    return;

  // When left shifting an ICE which is signed, we can check for overflow which
  // according to C++ has undefined behavior ([expr.shift] 5.8/2). Unsigned
  // integers have defined behavior modulo one more than the maximum value
  // representable in the result type, so never warn for those.
  llvm::APSInt Left;
  if (LHS.get()->isValueDependent() ||
      LHSType->hasUnsignedIntegerRepresentation() ||
      !LHS.get()->EvaluateAsInt(Left, S.Context))
    return;

  // If LHS does not have a signed type and non-negative value
  // then, the behavior is undefined. Warn about it.
  if (Left.isNegative()) {
    S.DiagRuntimeBehavior(Loc, LHS.get(),
                          S.PDiag(diag::warn_shift_lhs_negative)
                            << LHS.get()->getSourceRange());
    return;
  }

  llvm::APInt ResultBits =
      static_cast<llvm::APInt&>(Right) + Left.getMinSignedBits();
  if (LeftBits.uge(ResultBits))
    return;
  llvm::APSInt Result = Left.extend(ResultBits.getLimitedValue());
  Result = Result.shl(Right);

  // Print the bit representation of the signed integer as an unsigned
  // hexadecimal number.
  SmallString<40> HexResult;
  Result.toString(HexResult, 16, /*Signed =*/false, /*Literal =*/true);

  // If we are only missing a sign bit, this is less likely to result in actual
  // bugs -- if the result is cast back to an unsigned type, it will have the
  // expected value. Thus we place this behind a different warning that can be
  // turned off separately if needed.
  if (LeftBits == ResultBits - 1) {
    S.Diag(Loc, diag::warn_shift_result_sets_sign_bit)
        << HexResult << LHSType
        << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
    return;
  }

  S.Diag(Loc, diag::warn_shift_result_gt_typewidth)
    << HexResult.str() << Result.getMinSignedBits() << LHSType
    << Left.getBitWidth() << LHS.get()->getSourceRange()
    << RHS.get()->getSourceRange();
}

/// \brief Return the resulting type when an OpenCL vector is shifted
///        by a scalar or vector shift amount.
static QualType checkOpenCLVectorShift(Sema &S,
                                       ExprResult &LHS, ExprResult &RHS,
                                       SourceLocation Loc, bool IsCompAssign) {
  // OpenCL v1.1 s6.3.j says RHS can be a vector only if LHS is a vector.
  if (!LHS.get()->getType()->isVectorType()) {
    S.Diag(Loc, diag::err_shift_rhs_only_vector)
      << RHS.get()->getType() << LHS.get()->getType()
      << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
    return QualType();
  }

  if (!IsCompAssign) {
    LHS = S.UsualUnaryConversions(LHS.get());
    if (LHS.isInvalid()) return QualType();
  }

  RHS = S.UsualUnaryConversions(RHS.get());
  if (RHS.isInvalid()) return QualType();

  QualType LHSType = LHS.get()->getType();
  const VectorType *LHSVecTy = LHSType->castAs<VectorType>();
  QualType LHSEleType = LHSVecTy->getElementType();

  // Note that RHS might not be a vector.
  QualType RHSType = RHS.get()->getType();
  const VectorType *RHSVecTy = RHSType->getAs<VectorType>();
  QualType RHSEleType = RHSVecTy ? RHSVecTy->getElementType() : RHSType;

  // OpenCL v1.1 s6.3.j says that the operands need to be integers.
  if (!LHSEleType->isIntegerType()) {
    S.Diag(Loc, diag::err_typecheck_expect_int)
      << LHS.get()->getType() << LHS.get()->getSourceRange();
    return QualType();
  }

  if (!RHSEleType->isIntegerType()) {
    S.Diag(Loc, diag::err_typecheck_expect_int)
      << RHS.get()->getType() << RHS.get()->getSourceRange();
    return QualType();
  }

  if (RHSVecTy) {
    // OpenCL v1.1 s6.3.j says that for vector types, the operators
    // are applied component-wise. So if RHS is a vector, then ensure
    // that the number of elements is the same as LHS...
    if (RHSVecTy->getNumElements() != LHSVecTy->getNumElements()) {
      S.Diag(Loc, diag::err_typecheck_vector_lengths_not_equal)
        << LHS.get()->getType() << RHS.get()->getType()
        << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      return QualType();
    }
  } else {
    // ...else expand RHS to match the number of elements in LHS.
    QualType VecTy =
      S.Context.getExtVectorType(RHSEleType, LHSVecTy->getNumElements());
    RHS = S.ImpCastExprToType(RHS.get(), VecTy, CK_VectorSplat);
  }

  return LHSType;
}

// C99 6.5.7
QualType Sema::CheckShiftOperands(ExprResult &LHS, ExprResult &RHS,
                                  SourceLocation Loc, BinaryOperatorKind Opc,
                                  bool IsCompAssign) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  // Vector shifts promote their scalar inputs to vector type.
  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    if (LangOpts.OpenCL)
      return checkOpenCLVectorShift(*this, LHS, RHS, Loc, IsCompAssign);
    if (LangOpts.ZVector) {
      // The shift operators for the z vector extensions work basically
      // like OpenCL shifts, except that neither the LHS nor the RHS is
      // allowed to be a "vector bool".
      if (auto LHSVecType = LHS.get()->getType()->getAs<VectorType>())
        if (LHSVecType->getVectorKind() == VectorType::AltiVecBool)
          return InvalidOperands(Loc, LHS, RHS);
      if (auto RHSVecType = RHS.get()->getType()->getAs<VectorType>())
        if (RHSVecType->getVectorKind() == VectorType::AltiVecBool)
          return InvalidOperands(Loc, LHS, RHS);
      return checkOpenCLVectorShift(*this, LHS, RHS, Loc, IsCompAssign);
    }
    return CheckVectorOperands(LHS, RHS, Loc, IsCompAssign,
                               /*AllowBothBool*/true,
                               /*AllowBoolConversions*/false);
  }

  // Shifts don't perform usual arithmetic conversions, they just do integer
  // promotions on each operand. C99 6.5.7p3

  // For the LHS, do usual unary conversions, but then reset them away
  // if this is a compound assignment.
  ExprResult OldLHS = LHS;
  LHS = UsualUnaryConversions(LHS.get());
  if (LHS.isInvalid())
    return QualType();
  QualType LHSType = LHS.get()->getType();
  if (IsCompAssign) LHS = OldLHS;

  // The RHS is simpler.
  RHS = UsualUnaryConversions(RHS.get());
  if (RHS.isInvalid())
    return QualType();
  QualType RHSType = RHS.get()->getType();

  // C99 6.5.7p2: Each of the operands shall have integer type.
  if (!LHSType->hasIntegerRepresentation() ||
      !RHSType->hasIntegerRepresentation())
    return InvalidOperands(Loc, LHS, RHS);

  // C++0x: Don't allow scoped enums. FIXME: Use something better than
  // hasIntegerRepresentation() above instead of this.
  if (isScopedEnumerationType(LHSType) ||
      isScopedEnumerationType(RHSType)) {
    return InvalidOperands(Loc, LHS, RHS);
  }
  // Sanity-check shift operands
  DiagnoseBadShiftValues(*this, LHS, RHS, Loc, Opc, LHSType);

  // "The type of the result is that of the promoted left operand."
  return LHSType;
}

static bool IsWithinTemplateSpecialization(Decl *D) {
  if (DeclContext *DC = D->getDeclContext()) {
    if (isa<ClassTemplateSpecializationDecl>(DC))
      return true;
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
      return FD->isFunctionTemplateSpecialization();
  }
  return false;
}

/// If two different enums are compared, raise a warning.
static void checkEnumComparison(Sema &S, SourceLocation Loc, Expr *LHS,
                                Expr *RHS) {
  QualType LHSStrippedType = LHS->IgnoreParenImpCasts()->getType();
  QualType RHSStrippedType = RHS->IgnoreParenImpCasts()->getType();

  const EnumType *LHSEnumType = LHSStrippedType->getAs<EnumType>();
  if (!LHSEnumType)
    return;
  const EnumType *RHSEnumType = RHSStrippedType->getAs<EnumType>();
  if (!RHSEnumType)
    return;

  // Ignore anonymous enums.
  if (!LHSEnumType->getDecl()->getIdentifier())
    return;
  if (!RHSEnumType->getDecl()->getIdentifier())
    return;

  if (S.Context.hasSameUnqualifiedType(LHSStrippedType, RHSStrippedType))
    return;

  S.Diag(Loc, diag::warn_comparison_of_mixed_enum_types)
      << LHSStrippedType << RHSStrippedType
      << LHS->getSourceRange() << RHS->getSourceRange();
}

/// \brief Diagnose bad pointer comparisons.
static void diagnoseDistinctPointerComparison(Sema &S, SourceLocation Loc,
                                              ExprResult &LHS, ExprResult &RHS,
                                              bool IsError) {
  S.Diag(Loc, IsError ? diag::err_typecheck_comparison_of_distinct_pointers
                      : diag::ext_typecheck_comparison_of_distinct_pointers)
    << LHS.get()->getType() << RHS.get()->getType()
    << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
}

/// \brief Returns false if the pointers are converted to a composite type,
/// true otherwise.
static bool convertPointersToCompositeType(Sema &S, SourceLocation Loc,
                                           ExprResult &LHS, ExprResult &RHS) {
  // C++ [expr.rel]p2:
  //   [...] Pointer conversions (4.10) and qualification
  //   conversions (4.4) are performed on pointer operands (or on
  //   a pointer operand and a null pointer constant) to bring
  //   them to their composite pointer type. [...]
  //
  // C++ [expr.eq]p1 uses the same notion for (in)equality
  // comparisons of pointers.

  // C++ [expr.eq]p2:
  //   In addition, pointers to members can be compared, or a pointer to
  //   member and a null pointer constant. Pointer to member conversions
  //   (4.11) and qualification conversions (4.4) are performed to bring
  //   them to a common type. If one operand is a null pointer constant,
  //   the common type is the type of the other operand. Otherwise, the
  //   common type is a pointer to member type similar (4.4) to the type
  //   of one of the operands, with a cv-qualification signature (4.4)
  //   that is the union of the cv-qualification signatures of the operand
  //   types.

  QualType LHSType = LHS.get()->getType();
  QualType RHSType = RHS.get()->getType();
  assert((LHSType->isPointerType() && RHSType->isPointerType()) ||
         (LHSType->isMemberPointerType() && RHSType->isMemberPointerType()));

  bool NonStandardCompositeType = false;
  bool *BoolPtr = S.isSFINAEContext() ? nullptr : &NonStandardCompositeType;
  QualType T = S.FindCompositePointerType(Loc, LHS, RHS, BoolPtr);
  if (T.isNull()) {
    diagnoseDistinctPointerComparison(S, Loc, LHS, RHS, /*isError*/true);
    return true;
  }

  if (NonStandardCompositeType)
    S.Diag(Loc, diag::ext_typecheck_comparison_of_distinct_pointers_nonstandard)
      << LHSType << RHSType << T << LHS.get()->getSourceRange()
      << RHS.get()->getSourceRange();

  LHS = S.ImpCastExprToType(LHS.get(), T, CK_BitCast);
  RHS = S.ImpCastExprToType(RHS.get(), T, CK_BitCast);
  return false;
}

static void diagnoseFunctionPointerToVoidComparison(Sema &S, SourceLocation Loc,
                                                    ExprResult &LHS,
                                                    ExprResult &RHS,
                                                    bool IsError) {
  S.Diag(Loc, IsError ? diag::err_typecheck_comparison_of_fptr_to_void
                      : diag::ext_typecheck_comparison_of_fptr_to_void)
    << LHS.get()->getType() << RHS.get()->getType()
    << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
}

static bool isObjCObjectLiteral(ExprResult &E) {
  switch (E.get()->IgnoreParenImpCasts()->getStmtClass()) {
  case Stmt::ObjCArrayLiteralClass:
  case Stmt::ObjCDictionaryLiteralClass:
  case Stmt::ObjCStringLiteralClass:
  case Stmt::ObjCBoxedExprClass:
    return true;
  default:
    // Note that ObjCBoolLiteral is NOT an object literal!
    return false;
  }
}

static bool hasIsEqualMethod(Sema &S, const Expr *LHS, const Expr *RHS) {
  const ObjCObjectPointerType *Type =
    LHS->getType()->getAs<ObjCObjectPointerType>();

  // If this is not actually an Objective-C object, bail out.
  if (!Type)
    return false;

  // Get the LHS object's interface type.
  QualType InterfaceType = Type->getPointeeType();

  // If the RHS isn't an Objective-C object, bail out.
  if (!RHS->getType()->isObjCObjectPointerType())
    return false;

  // Try to find the -isEqual: method.
  Selector IsEqualSel = S.NSAPIObj->getIsEqualSelector();
  ObjCMethodDecl *Method = S.LookupMethodInObjectType(IsEqualSel,
                                                      InterfaceType,
                                                      /*instance=*/true);
  if (!Method) {
    if (Type->isObjCIdType()) {
      // For 'id', just check the global pool.
      Method = S.LookupInstanceMethodInGlobalPool(IsEqualSel, SourceRange(),
                                                  /*receiverId=*/true);
    } else {
      // Check protocols.
      Method = S.LookupMethodInQualifiedType(IsEqualSel, Type,
                                             /*instance=*/true);
    }
  }

  if (!Method)
    return false;

  QualType T = Method->parameters()[0]->getType();
  if (!T->isObjCObjectPointerType())
    return false;

  QualType R = Method->getReturnType();
  if (!R->isScalarType())
    return false;

  return true;
}

Sema::ObjCLiteralKind Sema::CheckLiteralKind(Expr *FromE) {
  FromE = FromE->IgnoreParenImpCasts();
  switch (FromE->getStmtClass()) {
    default:
      break;
    case Stmt::ObjCStringLiteralClass:
      // "string literal"
      return LK_String;
    case Stmt::ObjCArrayLiteralClass:
      // "array literal"
      return LK_Array;
    case Stmt::ObjCDictionaryLiteralClass:
      // "dictionary literal"
      return LK_Dictionary;
    case Stmt::BlockExprClass:
      return LK_Block;
    case Stmt::ObjCBoxedExprClass: {
      Expr *Inner = cast<ObjCBoxedExpr>(FromE)->getSubExpr()->IgnoreParens();
      switch (Inner->getStmtClass()) {
        case Stmt::IntegerLiteralClass:
        case Stmt::FloatingLiteralClass:
        case Stmt::CharacterLiteralClass:
        case Stmt::ObjCBoolLiteralExprClass:
        case Stmt::CXXBoolLiteralExprClass:
          // "numeric literal"
          return LK_Numeric;
        case Stmt::ImplicitCastExprClass: {
          CastKind CK = cast<CastExpr>(Inner)->getCastKind();
          // Boolean literals can be represented by implicit casts.
          if (CK == CK_IntegralToBoolean || CK == CK_IntegralCast)
            return LK_Numeric;
          break;
        }
        default:
          break;
      }
      return LK_Boxed;
    }
  }
  return LK_None;
}

static void diagnoseObjCLiteralComparison(Sema &S, SourceLocation Loc,
                                          ExprResult &LHS, ExprResult &RHS,
                                          BinaryOperator::Opcode Opc){
  Expr *Literal;
  Expr *Other;
  if (isObjCObjectLiteral(LHS)) {
    Literal = LHS.get();
    Other = RHS.get();
  } else {
    Literal = RHS.get();
    Other = LHS.get();
  }

  // Don't warn on comparisons against nil.
  Other = Other->IgnoreParenCasts();
  if (Other->isNullPointerConstant(S.getASTContext(),
                                   Expr::NPC_ValueDependentIsNotNull))
    return;

  // This should be kept in sync with warn_objc_literal_comparison.
  // LK_String should always be after the other literals, since it has its own
  // warning flag.
  Sema::ObjCLiteralKind LiteralKind = S.CheckLiteralKind(Literal);
  assert(LiteralKind != Sema::LK_Block);
  if (LiteralKind == Sema::LK_None) {
    llvm_unreachable("Unknown Objective-C object literal kind");
  }

  if (LiteralKind == Sema::LK_String)
    S.Diag(Loc, diag::warn_objc_string_literal_comparison)
      << Literal->getSourceRange();
  else
    S.Diag(Loc, diag::warn_objc_literal_comparison)
      << LiteralKind << Literal->getSourceRange();

  if (BinaryOperator::isEqualityOp(Opc) &&
      hasIsEqualMethod(S, LHS.get(), RHS.get())) {
    SourceLocation Start = LHS.get()->getLocStart();
    SourceLocation End = S.getLocForEndOfToken(RHS.get()->getLocEnd());
    CharSourceRange OpRange =
      CharSourceRange::getCharRange(Loc, S.getLocForEndOfToken(Loc));

    S.Diag(Loc, diag::note_objc_literal_comparison_isequal)
      << FixItHint::CreateInsertion(Start, Opc == BO_EQ ? "[" : "![")
      << FixItHint::CreateReplacement(OpRange, " isEqual:")
      << FixItHint::CreateInsertion(End, "]");
  }
}

static void diagnoseLogicalNotOnLHSofComparison(Sema &S, ExprResult &LHS,
                                                ExprResult &RHS,
                                                SourceLocation Loc,
                                                BinaryOperatorKind Opc) {
  // Check that left hand side is !something.
  UnaryOperator *UO = dyn_cast<UnaryOperator>(LHS.get()->IgnoreImpCasts());
  if (!UO || UO->getOpcode() != UO_LNot) return;

  // Only check if the right hand side is non-bool arithmetic type.
  if (RHS.get()->isKnownToHaveBooleanValue()) return;

  // Make sure that the something in !something is not bool.
  Expr *SubExpr = UO->getSubExpr()->IgnoreImpCasts();
  if (SubExpr->isKnownToHaveBooleanValue()) return;

  // Emit warning.
  S.Diag(UO->getOperatorLoc(), diag::warn_logical_not_on_lhs_of_comparison)
      << Loc;

  // First note suggest !(x < y)
  SourceLocation FirstOpen = SubExpr->getLocStart();
  SourceLocation FirstClose = RHS.get()->getLocEnd();
  FirstClose = S.getLocForEndOfToken(FirstClose);
  if (FirstClose.isInvalid())
    FirstOpen = SourceLocation();
  S.Diag(UO->getOperatorLoc(), diag::note_logical_not_fix)
      << FixItHint::CreateInsertion(FirstOpen, "(")
      << FixItHint::CreateInsertion(FirstClose, ")");

  // Second note suggests (!x) < y
  SourceLocation SecondOpen = LHS.get()->getLocStart();
  SourceLocation SecondClose = LHS.get()->getLocEnd();
  SecondClose = S.getLocForEndOfToken(SecondClose);
  if (SecondClose.isInvalid())
    SecondOpen = SourceLocation();
  S.Diag(UO->getOperatorLoc(), diag::note_logical_not_silence_with_parens)
      << FixItHint::CreateInsertion(SecondOpen, "(")
      << FixItHint::CreateInsertion(SecondClose, ")");
}

// Get the decl for a simple expression: a reference to a variable,
// an implicit C++ field reference, or an implicit ObjC ivar reference.
static ValueDecl *getCompareDecl(Expr *E) {
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(E))
    return DR->getDecl();
  if (ObjCIvarRefExpr* Ivar = dyn_cast<ObjCIvarRefExpr>(E)) {
    if (Ivar->isFreeIvar())
      return Ivar->getDecl();
  }
  if (MemberExpr* Mem = dyn_cast<MemberExpr>(E)) {
    if (Mem->isImplicitAccess())
      return Mem->getMemberDecl();
  }
  return nullptr;
}

// C99 6.5.8, C++ [expr.rel]
QualType Sema::CheckCompareOperands(ExprResult &LHS, ExprResult &RHS,
                                    SourceLocation Loc, BinaryOperatorKind Opc,
                                    bool IsRelational) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/true);

  // Handle vector comparisons separately.
  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType())
    return CheckVectorCompareOperands(LHS, RHS, Loc, IsRelational);

  QualType LHSType = LHS.get()->getType();
  QualType RHSType = RHS.get()->getType();

  Expr *LHSStripped = LHS.get()->IgnoreParenImpCasts();
  Expr *RHSStripped = RHS.get()->IgnoreParenImpCasts();

  checkEnumComparison(*this, Loc, LHS.get(), RHS.get());
  diagnoseLogicalNotOnLHSofComparison(*this, LHS, RHS, Loc, Opc);

  if (!LHSType->hasFloatingRepresentation() &&
      !(LHSType->isBlockPointerType() && IsRelational) &&
      !LHS.get()->getLocStart().isMacroID() &&
      !RHS.get()->getLocStart().isMacroID() &&
      ActiveTemplateInstantiations.empty()) {
    // For non-floating point types, check for self-comparisons of the form
    // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
    // often indicate logic errors in the program.
    //
    // NOTE: Don't warn about comparison expressions resulting from macro
    // expansion. Also don't warn about comparisons which are only self
    // comparisons within a template specialization. The warnings should catch
    // obvious cases in the definition of the template anyways. The idea is to
    // warn when the typed comparison operator will always evaluate to the same
    // result.
    ValueDecl *DL = getCompareDecl(LHSStripped);
    ValueDecl *DR = getCompareDecl(RHSStripped);
    if (DL && DR && DL == DR && !IsWithinTemplateSpecialization(DL)) {
      DiagRuntimeBehavior(Loc, nullptr, PDiag(diag::warn_comparison_always)
                          << 0 // self-
                          << (Opc == BO_EQ
                              || Opc == BO_LE
                              || Opc == BO_GE));
    } else if (DL && DR && LHSType->isArrayType() && RHSType->isArrayType() &&
               !DL->getType()->isReferenceType() &&
               !DR->getType()->isReferenceType()) {
        // what is it always going to eval to?
        char always_evals_to;
        switch(Opc) {
        case BO_EQ: // e.g. array1 == array2
          always_evals_to = 0; // false
          break;
        case BO_NE: // e.g. array1 != array2
          always_evals_to = 1; // true
          break;
        default:
          // best we can say is 'a constant'
          always_evals_to = 2; // e.g. array1 <= array2
          break;
        }
        DiagRuntimeBehavior(Loc, nullptr, PDiag(diag::warn_comparison_always)
                            << 1 // array
                            << always_evals_to);
    }

    if (isa<CastExpr>(LHSStripped))
      LHSStripped = LHSStripped->IgnoreParenCasts();
    if (isa<CastExpr>(RHSStripped))
      RHSStripped = RHSStripped->IgnoreParenCasts();

    // Warn about comparisons against a string constant (unless the other
    // operand is null), the user probably wants strcmp.
    Expr *literalString = nullptr;
    Expr *literalStringStripped = nullptr;
    if ((isa<StringLiteral>(LHSStripped) || isa<ObjCEncodeExpr>(LHSStripped)) &&
        !RHSStripped->isNullPointerConstant(Context,
                                            Expr::NPC_ValueDependentIsNull)) {
      literalString = LHS.get();
      literalStringStripped = LHSStripped;
    } else if ((isa<StringLiteral>(RHSStripped) ||
                isa<ObjCEncodeExpr>(RHSStripped)) &&
               !LHSStripped->isNullPointerConstant(Context,
                                            Expr::NPC_ValueDependentIsNull)) {
      literalString = RHS.get();
      literalStringStripped = RHSStripped;
    }

    if (literalString) {
      DiagRuntimeBehavior(Loc, nullptr,
        PDiag(diag::warn_stringcompare)
          << isa<ObjCEncodeExpr>(literalStringStripped)
          << literalString->getSourceRange());
    }
  }

  // C99 6.5.8p3 / C99 6.5.9p4
  UsualArithmeticConversions(LHS, RHS);
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  LHSType = LHS.get()->getType();
  RHSType = RHS.get()->getType();

  // The result of comparisons is 'bool' in C++, 'int' in C.
  QualType ResultTy = Context.getLogicalOperationType();

  if (IsRelational) {
    if (LHSType->isRealType() && RHSType->isRealType())
      return ResultTy;
  } else {
    // Check for comparisons of floating point operands using != and ==.
    if (LHSType->hasFloatingRepresentation())
      CheckFloatComparison(Loc, LHS.get(), RHS.get());

    if (LHSType->isArithmeticType() && RHSType->isArithmeticType())
      return ResultTy;
  }

  const Expr::NullPointerConstantKind LHSNullKind =
      LHS.get()->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull);
  const Expr::NullPointerConstantKind RHSNullKind =
      RHS.get()->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull);
  bool LHSIsNull = LHSNullKind != Expr::NPCK_NotNull;
  bool RHSIsNull = RHSNullKind != Expr::NPCK_NotNull;

  if (!IsRelational && LHSIsNull != RHSIsNull) {
    bool IsEquality = Opc == BO_EQ;
    if (RHSIsNull)
      DiagnoseAlwaysNonNullPointer(LHS.get(), RHSNullKind, IsEquality,
                                   RHS.get()->getSourceRange());
    else
      DiagnoseAlwaysNonNullPointer(RHS.get(), LHSNullKind, IsEquality,
                                   LHS.get()->getSourceRange());
  }

  // All of the following pointer-related warnings are GCC extensions, except
  // when handling null pointer constants. 
  if (LHSType->isPointerType() && RHSType->isPointerType()) { // C99 6.5.8p2
    QualType LCanPointeeTy =
      LHSType->castAs<PointerType>()->getPointeeType().getCanonicalType();
    QualType RCanPointeeTy =
      RHSType->castAs<PointerType>()->getPointeeType().getCanonicalType();

    if (getLangOpts().CPlusPlus) {
      if (LCanPointeeTy == RCanPointeeTy)
        return ResultTy;
      if (!IsRelational &&
          (LCanPointeeTy->isVoidType() || RCanPointeeTy->isVoidType())) {
        // Valid unless comparison between non-null pointer and function pointer
        // This is a gcc extension compatibility comparison.
        // In a SFINAE context, we treat this as a hard error to maintain
        // conformance with the C++ standard.
        if ((LCanPointeeTy->isFunctionType() || RCanPointeeTy->isFunctionType())
            && !LHSIsNull && !RHSIsNull) {
          diagnoseFunctionPointerToVoidComparison(
              *this, Loc, LHS, RHS, /*isError*/ (bool)isSFINAEContext());
          
          if (isSFINAEContext())
            return QualType();
          
          RHS = ImpCastExprToType(RHS.get(), LHSType, CK_BitCast);
          return ResultTy;
        }
      }

      if (convertPointersToCompositeType(*this, Loc, LHS, RHS))
        return QualType();
      else
        return ResultTy;
    }
    // C99 6.5.9p2 and C99 6.5.8p2
    if (Context.typesAreCompatible(LCanPointeeTy.getUnqualifiedType(),
                                   RCanPointeeTy.getUnqualifiedType())) {
      // Valid unless a relational comparison of function pointers
      if (IsRelational && LCanPointeeTy->isFunctionType()) {
        Diag(Loc, diag::ext_typecheck_ordered_comparison_of_function_pointers)
          << LHSType << RHSType << LHS.get()->getSourceRange()
          << RHS.get()->getSourceRange();
      }
    } else if (!IsRelational &&
               (LCanPointeeTy->isVoidType() || RCanPointeeTy->isVoidType())) {
      // Valid unless comparison between non-null pointer and function pointer
      if ((LCanPointeeTy->isFunctionType() || RCanPointeeTy->isFunctionType())
          && !LHSIsNull && !RHSIsNull)
        diagnoseFunctionPointerToVoidComparison(*this, Loc, LHS, RHS,
                                                /*isError*/false);
    } else {
      // Invalid
      diagnoseDistinctPointerComparison(*this, Loc, LHS, RHS, /*isError*/false);
    }
    if (LCanPointeeTy != RCanPointeeTy) {
      // Treat NULL constant as a special case in OpenCL.
      if (getLangOpts().OpenCL && !LHSIsNull && !RHSIsNull) {
        const PointerType *LHSPtr = LHSType->getAs<PointerType>();
        if (!LHSPtr->isAddressSpaceOverlapping(*RHSType->getAs<PointerType>())) {
          Diag(Loc,
               diag::err_typecheck_op_on_nonoverlapping_address_space_pointers)
              << LHSType << RHSType << 0 /* comparison */
              << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
        }
      }
      unsigned AddrSpaceL = LCanPointeeTy.getAddressSpace();
      unsigned AddrSpaceR = RCanPointeeTy.getAddressSpace();
      CastKind Kind = AddrSpaceL != AddrSpaceR ? CK_AddressSpaceConversion
                                               : CK_BitCast;
      if (LHSIsNull && !RHSIsNull)
        LHS = ImpCastExprToType(LHS.get(), RHSType, Kind);
      else
        RHS = ImpCastExprToType(RHS.get(), LHSType, Kind);
    }
    return ResultTy;
  }

  if (getLangOpts().CPlusPlus) {
    // Comparison of nullptr_t with itself.
    if (LHSType->isNullPtrType() && RHSType->isNullPtrType())
      return ResultTy;
    
    // Comparison of pointers with null pointer constants and equality
    // comparisons of member pointers to null pointer constants.
    if (RHSIsNull &&
        ((LHSType->isAnyPointerType() || LHSType->isNullPtrType()) ||
         (!IsRelational && 
          (LHSType->isMemberPointerType() || LHSType->isBlockPointerType())))) {
      RHS = ImpCastExprToType(RHS.get(), LHSType, 
                        LHSType->isMemberPointerType()
                          ? CK_NullToMemberPointer
                          : CK_NullToPointer);
      return ResultTy;
    }
    if (LHSIsNull &&
        ((RHSType->isAnyPointerType() || RHSType->isNullPtrType()) ||
         (!IsRelational && 
          (RHSType->isMemberPointerType() || RHSType->isBlockPointerType())))) {
      LHS = ImpCastExprToType(LHS.get(), RHSType, 
                        RHSType->isMemberPointerType()
                          ? CK_NullToMemberPointer
                          : CK_NullToPointer);
      return ResultTy;
    }

    // Comparison of member pointers.
    if (!IsRelational &&
        LHSType->isMemberPointerType() && RHSType->isMemberPointerType()) {
      if (convertPointersToCompositeType(*this, Loc, LHS, RHS))
        return QualType();
      else
        return ResultTy;
    }

    // Handle scoped enumeration types specifically, since they don't promote
    // to integers.
    if (LHS.get()->getType()->isEnumeralType() &&
        Context.hasSameUnqualifiedType(LHS.get()->getType(),
                                       RHS.get()->getType()))
      return ResultTy;
  }

  // Handle block pointer types.
  if (!IsRelational && LHSType->isBlockPointerType() &&
      RHSType->isBlockPointerType()) {
    QualType lpointee = LHSType->castAs<BlockPointerType>()->getPointeeType();
    QualType rpointee = RHSType->castAs<BlockPointerType>()->getPointeeType();

    if (!LHSIsNull && !RHSIsNull &&
        !Context.typesAreCompatible(lpointee, rpointee)) {
      Diag(Loc, diag::err_typecheck_comparison_of_distinct_blocks)
        << LHSType << RHSType << LHS.get()->getSourceRange()
        << RHS.get()->getSourceRange();
    }
    RHS = ImpCastExprToType(RHS.get(), LHSType, CK_BitCast);
    return ResultTy;
  }

  // Allow block pointers to be compared with null pointer constants.
  if (!IsRelational
      && ((LHSType->isBlockPointerType() && RHSType->isPointerType())
          || (LHSType->isPointerType() && RHSType->isBlockPointerType()))) {
    if (!LHSIsNull && !RHSIsNull) {
      if (!((RHSType->isPointerType() && RHSType->castAs<PointerType>()
             ->getPointeeType()->isVoidType())
            || (LHSType->isPointerType() && LHSType->castAs<PointerType>()
                ->getPointeeType()->isVoidType())))
        Diag(Loc, diag::err_typecheck_comparison_of_distinct_blocks)
          << LHSType << RHSType << LHS.get()->getSourceRange()
          << RHS.get()->getSourceRange();
    }
    if (LHSIsNull && !RHSIsNull)
      LHS = ImpCastExprToType(LHS.get(), RHSType,
                              RHSType->isPointerType() ? CK_BitCast
                                : CK_AnyPointerToBlockPointerCast);
    else
      RHS = ImpCastExprToType(RHS.get(), LHSType,
                              LHSType->isPointerType() ? CK_BitCast
                                : CK_AnyPointerToBlockPointerCast);
    return ResultTy;
  }

  if (LHSType->isObjCObjectPointerType() ||
      RHSType->isObjCObjectPointerType()) {
    const PointerType *LPT = LHSType->getAs<PointerType>();
    const PointerType *RPT = RHSType->getAs<PointerType>();
    if (LPT || RPT) {
      bool LPtrToVoid = LPT ? LPT->getPointeeType()->isVoidType() : false;
      bool RPtrToVoid = RPT ? RPT->getPointeeType()->isVoidType() : false;

      if (!LPtrToVoid && !RPtrToVoid &&
          !Context.typesAreCompatible(LHSType, RHSType)) {
        diagnoseDistinctPointerComparison(*this, Loc, LHS, RHS,
                                          /*isError*/false);
      }
      if (LHSIsNull && !RHSIsNull) {
        Expr *E = LHS.get();
        if (getLangOpts().ObjCAutoRefCount)
          CheckObjCARCConversion(SourceRange(), RHSType, E, CCK_ImplicitConversion);
        LHS = ImpCastExprToType(E, RHSType,
                                RPT ? CK_BitCast :CK_CPointerToObjCPointerCast);
      }
      else {
        Expr *E = RHS.get();
        if (getLangOpts().ObjCAutoRefCount)
          CheckObjCARCConversion(SourceRange(), LHSType, E,
                                 CCK_ImplicitConversion, /*Diagnose=*/true,
                                 /*DiagnoseCFAudited=*/false, Opc);
        RHS = ImpCastExprToType(E, LHSType,
                                LPT ? CK_BitCast :CK_CPointerToObjCPointerCast);
      }
      return ResultTy;
    }
    if (LHSType->isObjCObjectPointerType() &&
        RHSType->isObjCObjectPointerType()) {
      if (!Context.areComparableObjCPointerTypes(LHSType, RHSType))
        diagnoseDistinctPointerComparison(*this, Loc, LHS, RHS,
                                          /*isError*/false);
      if (isObjCObjectLiteral(LHS) || isObjCObjectLiteral(RHS))
        diagnoseObjCLiteralComparison(*this, Loc, LHS, RHS, Opc);

      if (LHSIsNull && !RHSIsNull)
        LHS = ImpCastExprToType(LHS.get(), RHSType, CK_BitCast);
      else
        RHS = ImpCastExprToType(RHS.get(), LHSType, CK_BitCast);
      return ResultTy;
    }
  }
  if ((LHSType->isAnyPointerType() && RHSType->isIntegerType()) ||
      (LHSType->isIntegerType() && RHSType->isAnyPointerType())) {
    unsigned DiagID = 0;
    bool isError = false;
    if (LangOpts.DebuggerSupport) {
      // Under a debugger, allow the comparison of pointers to integers,
      // since users tend to want to compare addresses.
    } else if ((LHSIsNull && LHSType->isIntegerType()) ||
        (RHSIsNull && RHSType->isIntegerType())) {
      if (IsRelational && !getLangOpts().CPlusPlus)
        DiagID = diag::ext_typecheck_ordered_comparison_of_pointer_and_zero;
    } else if (IsRelational && !getLangOpts().CPlusPlus)
      DiagID = diag::ext_typecheck_ordered_comparison_of_pointer_integer;
    else if (getLangOpts().CPlusPlus) {
      DiagID = diag::err_typecheck_comparison_of_pointer_integer;
      isError = true;
    } else
      DiagID = diag::ext_typecheck_comparison_of_pointer_integer;

    if (DiagID) {
      Diag(Loc, DiagID)
        << LHSType << RHSType << LHS.get()->getSourceRange()
        << RHS.get()->getSourceRange();
      if (isError)
        return QualType();
    }
    
    if (LHSType->isIntegerType())
      LHS = ImpCastExprToType(LHS.get(), RHSType,
                        LHSIsNull ? CK_NullToPointer : CK_IntegralToPointer);
    else
      RHS = ImpCastExprToType(RHS.get(), LHSType,
                        RHSIsNull ? CK_NullToPointer : CK_IntegralToPointer);
    return ResultTy;
  }
  
  // Handle block pointers.
  if (!IsRelational && RHSIsNull
      && LHSType->isBlockPointerType() && RHSType->isIntegerType()) {
    RHS = ImpCastExprToType(RHS.get(), LHSType, CK_NullToPointer);
    return ResultTy;
  }
  if (!IsRelational && LHSIsNull
      && LHSType->isIntegerType() && RHSType->isBlockPointerType()) {
    LHS = ImpCastExprToType(LHS.get(), RHSType, CK_NullToPointer);
    return ResultTy;
  }

  return InvalidOperands(Loc, LHS, RHS);
}


// Return a signed type that is of identical size and number of elements.
// For floating point vectors, return an integer type of identical size 
// and number of elements.
QualType Sema::GetSignedVectorType(QualType V) {
  const VectorType *VTy = V->getAs<VectorType>();
  unsigned TypeSize = Context.getTypeSize(VTy->getElementType());
  if (TypeSize == Context.getTypeSize(Context.CharTy))
    return Context.getExtVectorType(Context.CharTy, VTy->getNumElements());
  else if (TypeSize == Context.getTypeSize(Context.ShortTy))
    return Context.getExtVectorType(Context.ShortTy, VTy->getNumElements());
  else if (TypeSize == Context.getTypeSize(Context.IntTy))
    return Context.getExtVectorType(Context.IntTy, VTy->getNumElements());
  else if (TypeSize == Context.getTypeSize(Context.LongTy))
    return Context.getExtVectorType(Context.LongTy, VTy->getNumElements());
  assert(TypeSize == Context.getTypeSize(Context.LongLongTy) &&
         "Unhandled vector element size in vector compare");
  return Context.getExtVectorType(Context.LongLongTy, VTy->getNumElements());
}

/// CheckVectorCompareOperands - vector comparisons are a clang extension that
/// operates on extended vector types.  Instead of producing an IntTy result,
/// like a scalar comparison, a vector comparison produces a vector of integer
/// types.
QualType Sema::CheckVectorCompareOperands(ExprResult &LHS, ExprResult &RHS,
                                          SourceLocation Loc,
                                          bool IsRelational) {
  // Check to make sure we're operating on vectors of the same type and width,
  // Allowing one side to be a scalar of element type.
  QualType vType = CheckVectorOperands(LHS, RHS, Loc, /*isCompAssign*/false,
                              /*AllowBothBool*/true,
                              /*AllowBoolConversions*/getLangOpts().ZVector);
  if (vType.isNull())
    return vType;

  QualType LHSType = LHS.get()->getType();

  // If AltiVec, the comparison results in a numeric type, i.e.
  // bool for C++, int for C
  if (getLangOpts().AltiVec &&
      vType->getAs<VectorType>()->getVectorKind() == VectorType::AltiVecVector)
    return Context.getLogicalOperationType();

  // For non-floating point types, check for self-comparisons of the form
  // x == x, x != x, x < x, etc.  These always evaluate to a constant, and
  // often indicate logic errors in the program.
  if (!LHSType->hasFloatingRepresentation() &&
      ActiveTemplateInstantiations.empty()) {
    if (DeclRefExpr* DRL
          = dyn_cast<DeclRefExpr>(LHS.get()->IgnoreParenImpCasts()))
      if (DeclRefExpr* DRR
            = dyn_cast<DeclRefExpr>(RHS.get()->IgnoreParenImpCasts()))
        if (DRL->getDecl() == DRR->getDecl())
          DiagRuntimeBehavior(Loc, nullptr,
                              PDiag(diag::warn_comparison_always)
                                << 0 // self-
                                << 2 // "a constant"
                              );
  }

  // Check for comparisons of floating point operands using != and ==.
  if (!IsRelational && LHSType->hasFloatingRepresentation()) {
    assert (RHS.get()->getType()->hasFloatingRepresentation());
    CheckFloatComparison(Loc, LHS.get(), RHS.get());
  }
  
  // Return a signed type for the vector.
  return GetSignedVectorType(LHSType);
}

QualType Sema::CheckVectorLogicalOperands(ExprResult &LHS, ExprResult &RHS,
                                          SourceLocation Loc) {
  // Ensure that either both operands are of the same vector type, or
  // one operand is of a vector type and the other is of its element type.
  QualType vType = CheckVectorOperands(LHS, RHS, Loc, false,
                                       /*AllowBothBool*/true,
                                       /*AllowBoolConversions*/false);
  if (vType.isNull())
    return InvalidOperands(Loc, LHS, RHS);
  if (getLangOpts().OpenCL && getLangOpts().OpenCLVersion < 120 &&
      vType->hasFloatingRepresentation())
    return InvalidOperands(Loc, LHS, RHS);
  
  return GetSignedVectorType(LHS.get()->getType());
}

inline QualType Sema::CheckBitwiseOperands(
  ExprResult &LHS, ExprResult &RHS, SourceLocation Loc, bool IsCompAssign) {
  checkArithmeticNull(*this, LHS, RHS, Loc, /*isCompare=*/false);

  if (LHS.get()->getType()->isVectorType() ||
      RHS.get()->getType()->isVectorType()) {
    if (LHS.get()->getType()->hasIntegerRepresentation() &&
        RHS.get()->getType()->hasIntegerRepresentation())
      return CheckVectorOperands(LHS, RHS, Loc, IsCompAssign,
                        /*AllowBothBool*/true,
                        /*AllowBoolConversions*/getLangOpts().ZVector);
    return InvalidOperands(Loc, LHS, RHS);
  }

  ExprResult LHSResult = LHS, RHSResult = RHS;
  QualType compType = UsualArithmeticConversions(LHSResult, RHSResult,
                                                 IsCompAssign);
  if (LHSResult.isInvalid() || RHSResult.isInvalid())
    return QualType();
  LHS = LHSResult.get();
  RHS = RHSResult.get();

  if (!compType.isNull() && compType->isIntegralOrUnscopedEnumerationType())
    return compType;
  return InvalidOperands(Loc, LHS, RHS);
}

// C99 6.5.[13,14]
inline QualType Sema::CheckLogicalOperands(ExprResult &LHS, ExprResult &RHS,
                                           SourceLocation Loc,
                                           BinaryOperatorKind Opc) {
  // Check vector operands differently.
  if (LHS.get()->getType()->isVectorType() || RHS.get()->getType()->isVectorType())
    return CheckVectorLogicalOperands(LHS, RHS, Loc);
  
  // Diagnose cases where the user write a logical and/or but probably meant a
  // bitwise one.  We do this when the LHS is a non-bool integer and the RHS
  // is a constant.
  if (LHS.get()->getType()->isIntegerType() &&
      !LHS.get()->getType()->isBooleanType() &&
      RHS.get()->getType()->isIntegerType() && !RHS.get()->isValueDependent() &&
      // Don't warn in macros or template instantiations.
      !Loc.isMacroID() && ActiveTemplateInstantiations.empty()) {
    // If the RHS can be constant folded, and if it constant folds to something
    // that isn't 0 or 1 (which indicate a potential logical operation that
    // happened to fold to true/false) then warn.
    // Parens on the RHS are ignored.
    llvm::APSInt Result;
    if (RHS.get()->EvaluateAsInt(Result, Context))
      if ((getLangOpts().Bool && !RHS.get()->getType()->isBooleanType() &&
           !RHS.get()->getExprLoc().isMacroID()) ||
          (Result != 0 && Result != 1)) {
        Diag(Loc, diag::warn_logical_instead_of_bitwise)
          << RHS.get()->getSourceRange()
          << (Opc == BO_LAnd ? "&&" : "||");
        // Suggest replacing the logical operator with the bitwise version
        Diag(Loc, diag::note_logical_instead_of_bitwise_change_operator)
            << (Opc == BO_LAnd ? "&" : "|")
            << FixItHint::CreateReplacement(SourceRange(
                                                 Loc, getLocForEndOfToken(Loc)),
                                            Opc == BO_LAnd ? "&" : "|");
        if (Opc == BO_LAnd)
          // Suggest replacing "Foo() && kNonZero" with "Foo()"
          Diag(Loc, diag::note_logical_instead_of_bitwise_remove_constant)
              << FixItHint::CreateRemoval(
                  SourceRange(getLocForEndOfToken(LHS.get()->getLocEnd()),
                              RHS.get()->getLocEnd()));
      }
  }

  if (!Context.getLangOpts().CPlusPlus) {
    // OpenCL v1.1 s6.3.g: The logical operators and (&&), or (||) do
    // not operate on the built-in scalar and vector float types.
    if (Context.getLangOpts().OpenCL &&
        Context.getLangOpts().OpenCLVersion < 120) {
      if (LHS.get()->getType()->isFloatingType() ||
          RHS.get()->getType()->isFloatingType())
        return InvalidOperands(Loc, LHS, RHS);
    }

    LHS = UsualUnaryConversions(LHS.get());
    if (LHS.isInvalid())
      return QualType();

    RHS = UsualUnaryConversions(RHS.get());
    if (RHS.isInvalid())
      return QualType();

    if (!LHS.get()->getType()->isScalarType() ||
        !RHS.get()->getType()->isScalarType())
      return InvalidOperands(Loc, LHS, RHS);

    return Context.IntTy;
  }

  // The following is safe because we only use this method for
  // non-overloadable operands.

  // C++ [expr.log.and]p1
  // C++ [expr.log.or]p1
  // The operands are both contextually converted to type bool.
  ExprResult LHSRes = PerformContextuallyConvertToBool(LHS.get());
  if (LHSRes.isInvalid())
    return InvalidOperands(Loc, LHS, RHS);
  LHS = LHSRes;

  ExprResult RHSRes = PerformContextuallyConvertToBool(RHS.get());
  if (RHSRes.isInvalid())
    return InvalidOperands(Loc, LHS, RHS);
  RHS = RHSRes;

  // C++ [expr.log.and]p2
  // C++ [expr.log.or]p2
  // The result is a bool.
  return Context.BoolTy;
}

static bool IsReadonlyMessage(Expr *E, Sema &S) {
  const MemberExpr *ME = dyn_cast<MemberExpr>(E);
  if (!ME) return false;
  if (!isa<FieldDecl>(ME->getMemberDecl())) return false;
  ObjCMessageExpr *Base =
    dyn_cast<ObjCMessageExpr>(ME->getBase()->IgnoreParenImpCasts());
  if (!Base) return false;
  return Base->getMethodDecl() != nullptr;
}

/// Is the given expression (which must be 'const') a reference to a
/// variable which was originally non-const, but which has become
/// 'const' due to being captured within a block?
enum NonConstCaptureKind { NCCK_None, NCCK_Block, NCCK_Lambda };
static NonConstCaptureKind isReferenceToNonConstCapture(Sema &S, Expr *E) {
  assert(E->isLValue() && E->getType().isConstQualified());
  E = E->IgnoreParens();

  // Must be a reference to a declaration from an enclosing scope.
  DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE) return NCCK_None;
  if (!DRE->refersToEnclosingVariableOrCapture()) return NCCK_None;

  // The declaration must be a variable which is not declared 'const'.
  VarDecl *var = dyn_cast<VarDecl>(DRE->getDecl());
  if (!var) return NCCK_None;
  if (var->getType().isConstQualified()) return NCCK_None;
  assert(var->hasLocalStorage() && "capture added 'const' to non-local?");

  // Decide whether the first capture was for a block or a lambda.
  DeclContext *DC = S.CurContext, *Prev = nullptr;
  while (DC != var->getDeclContext()) {
    Prev = DC;
    DC = DC->getParent();
  }
  // Unless we have an init-capture, we've gone one step too far.
  if (!var->isInitCapture())
    DC = Prev;
  return (isa<BlockDecl>(DC) ? NCCK_Block : NCCK_Lambda);
}

static bool IsTypeModifiable(QualType Ty, bool IsDereference) {
  Ty = Ty.getNonReferenceType();
  if (IsDereference && Ty->isPointerType())
    Ty = Ty->getPointeeType();
  return !Ty.isConstQualified();
}

/// Emit the "read-only variable not assignable" error and print notes to give
/// more information about why the variable is not assignable, such as pointing
/// to the declaration of a const variable, showing that a method is const, or
/// that the function is returning a const reference.
static void DiagnoseConstAssignment(Sema &S, const Expr *E,
                                    SourceLocation Loc) {
  // Update err_typecheck_assign_const and note_typecheck_assign_const
  // when this enum is changed.
  enum {
    ConstFunction,
    ConstVariable,
    ConstMember,
    ConstMethod,
    ConstUnknown,  // Keep as last element
  };

  SourceRange ExprRange = E->getSourceRange();

  // Only emit one error on the first const found.  All other consts will emit
  // a note to the error.
  bool DiagnosticEmitted = false;

  // Track if the current expression is the result of a derefence, and if the
  // next checked expression is the result of a derefence.
  bool IsDereference = false;
  bool NextIsDereference = false;

  // Loop to process MemberExpr chains.
  while (true) {
    IsDereference = NextIsDereference;
    NextIsDereference = false;

    E = E->IgnoreParenImpCasts();
    if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
      NextIsDereference = ME->isArrow();
      const ValueDecl *VD = ME->getMemberDecl();
      if (const FieldDecl *Field = dyn_cast<FieldDecl>(VD)) {
        // Mutable fields can be modified even if the class is const.
        if (Field->isMutable()) {
          assert(DiagnosticEmitted && "Expected diagnostic not emitted.");
          break;
        }

        if (!IsTypeModifiable(Field->getType(), IsDereference)) {
          if (!DiagnosticEmitted) {
            S.Diag(Loc, diag::err_typecheck_assign_const)
                << ExprRange << ConstMember << false /*static*/ << Field
                << Field->getType();
            DiagnosticEmitted = true;
          }
          S.Diag(VD->getLocation(), diag::note_typecheck_assign_const)
              << ConstMember << false /*static*/ << Field << Field->getType()
              << Field->getSourceRange();
        }
        E = ME->getBase();
        continue;
      } else if (const VarDecl *VDecl = dyn_cast<VarDecl>(VD)) {
        if (VDecl->getType().isConstQualified()) {
          if (!DiagnosticEmitted) {
            S.Diag(Loc, diag::err_typecheck_assign_const)
                << ExprRange << ConstMember << true /*static*/ << VDecl
                << VDecl->getType();
            DiagnosticEmitted = true;
          }
          S.Diag(VD->getLocation(), diag::note_typecheck_assign_const)
              << ConstMember << true /*static*/ << VDecl << VDecl->getType()
              << VDecl->getSourceRange();
        }
        // Static fields do not inherit constness from parents.
        break;
      }
      break;
    } // End MemberExpr
    break;
  }

  if (const CallExpr *CE = dyn_cast<CallExpr>(E)) {
    // Function calls
    const FunctionDecl *FD = CE->getDirectCallee();
    if (FD && !IsTypeModifiable(FD->getReturnType(), IsDereference)) {
      if (!DiagnosticEmitted) {
        S.Diag(Loc, diag::err_typecheck_assign_const) << ExprRange
                                                      << ConstFunction << FD;
        DiagnosticEmitted = true;
      }
      S.Diag(FD->getReturnTypeSourceRange().getBegin(),
             diag::note_typecheck_assign_const)
          << ConstFunction << FD << FD->getReturnType()
          << FD->getReturnTypeSourceRange();
    }
  } else if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    // Point to variable declaration.
    if (const ValueDecl *VD = DRE->getDecl()) {
      if (!IsTypeModifiable(VD->getType(), IsDereference)) {
        if (!DiagnosticEmitted) {
          S.Diag(Loc, diag::err_typecheck_assign_const)
              << ExprRange << ConstVariable << VD << VD->getType();
          DiagnosticEmitted = true;
        }
        S.Diag(VD->getLocation(), diag::note_typecheck_assign_const)
            << ConstVariable << VD << VD->getType() << VD->getSourceRange();
      }
    }
  } else if (isa<CXXThisExpr>(E)) {
    if (const DeclContext *DC = S.getFunctionLevelDeclContext()) {
      if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DC)) {
        if (MD->isConst()) {
          if (!DiagnosticEmitted) {
            S.Diag(Loc, diag::err_typecheck_assign_const) << ExprRange
                                                          << ConstMethod << MD;
            DiagnosticEmitted = true;
          }
          S.Diag(MD->getLocation(), diag::note_typecheck_assign_const)
              << ConstMethod << MD << MD->getSourceRange();
        }
      }
    }
  }

  if (DiagnosticEmitted)
    return;

  // Can't determine a more specific message, so display the generic error.
  S.Diag(Loc, diag::err_typecheck_assign_const) << ExprRange << ConstUnknown;
}

/// CheckForModifiableLvalue - Verify that E is a modifiable lvalue.  If not,
/// emit an error and return true.  If so, return false.
static bool CheckForModifiableLvalue(Expr *E, SourceLocation Loc, Sema &S) {
  assert(!E->hasPlaceholderType(BuiltinType::PseudoObject));
  SourceLocation OrigLoc = Loc;
  Expr::isModifiableLvalueResult IsLV = E->isModifiableLvalue(S.Context,
                                                              &Loc);
  if (IsLV == Expr::MLV_ClassTemporary && IsReadonlyMessage(E, S))
    IsLV = Expr::MLV_InvalidMessageExpression;
  if (IsLV == Expr::MLV_Valid)
    return false;

  unsigned DiagID = 0;
  bool NeedType = false;
  switch (IsLV) { // C99 6.5.16p2
  case Expr::MLV_ConstQualified:
    // Use a specialized diagnostic when we're assigning to an object
    // from an enclosing function or block.
    if (NonConstCaptureKind NCCK = isReferenceToNonConstCapture(S, E)) {
      if (NCCK == NCCK_Block)
        DiagID = diag::err_block_decl_ref_not_modifiable_lvalue;
      else
        DiagID = diag::err_lambda_decl_ref_not_modifiable_lvalue;
      break;
    }

    // In ARC, use some specialized diagnostics for occasions where we
    // infer 'const'.  These are always pseudo-strong variables.
    if (S.getLangOpts().ObjCAutoRefCount) {
      DeclRefExpr *declRef = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts());
      if (declRef && isa<VarDecl>(declRef->getDecl())) {
        VarDecl *var = cast<VarDecl>(declRef->getDecl());

        // Use the normal diagnostic if it's pseudo-__strong but the
        // user actually wrote 'const'.
        if (var->isARCPseudoStrong() &&
            (!var->getTypeSourceInfo() ||
             !var->getTypeSourceInfo()->getType().isConstQualified())) {
          // There are two pseudo-strong cases:
          //  - self
          ObjCMethodDecl *method = S.getCurMethodDecl();
          if (method && var == method->getSelfDecl())
            DiagID = method->isClassMethod()
              ? diag::err_typecheck_arc_assign_self_class_method
              : diag::err_typecheck_arc_assign_self;

          //  - fast enumeration variables
          else
            DiagID = diag::err_typecheck_arr_assign_enumeration;

          SourceRange Assign;
          if (Loc != OrigLoc)
            Assign = SourceRange(OrigLoc, OrigLoc);
          S.Diag(Loc, DiagID) << E->getSourceRange() << Assign;
          // We need to preserve the AST regardless, so migration tool
          // can do its job.
          return false;
        }
      }
    }

    // If none of the special cases above are triggered, then this is a
    // simple const assignment.
    if (DiagID == 0) {
      DiagnoseConstAssignment(S, E, Loc);
      return true;
    }

    break;
  case Expr::MLV_ConstAddrSpace:
    DiagnoseConstAssignment(S, E, Loc);
    return true;
  case Expr::MLV_ArrayType:
  case Expr::MLV_ArrayTemporary:
    DiagID = diag::err_typecheck_array_not_modifiable_lvalue;
    NeedType = true;
    break;
  case Expr::MLV_NotObjectType:
    DiagID = diag::err_typecheck_non_object_not_modifiable_lvalue;
    NeedType = true;
    break;
  case Expr::MLV_LValueCast:
    DiagID = diag::err_typecheck_lvalue_casts_not_supported;
    break;
  case Expr::MLV_Valid:
    llvm_unreachable("did not take early return for MLV_Valid");
  case Expr::MLV_InvalidExpression:
  case Expr::MLV_MemberFunction:
  case Expr::MLV_ClassTemporary:
    DiagID = diag::err_typecheck_expression_not_modifiable_lvalue;
    break;
  case Expr::MLV_IncompleteType:
  case Expr::MLV_IncompleteVoidType:
    return S.RequireCompleteType(Loc, E->getType(),
             diag::err_typecheck_incomplete_type_not_modifiable_lvalue, E);
  case Expr::MLV_DuplicateVectorComponents:
    DiagID = diag::err_typecheck_duplicate_vector_components_not_mlvalue;
    break;
  case Expr::MLV_NoSetterProperty:
    llvm_unreachable("readonly properties should be processed differently");
  case Expr::MLV_InvalidMessageExpression:
    DiagID = diag::error_readonly_message_assignment;
    break;
  case Expr::MLV_SubObjCPropertySetting:
    DiagID = diag::error_no_subobject_property_setting;
    break;
  }

  SourceRange Assign;
  if (Loc != OrigLoc)
    Assign = SourceRange(OrigLoc, OrigLoc);
  if (NeedType)
    S.Diag(Loc, DiagID) << E->getType() << E->getSourceRange() << Assign;
  else
    S.Diag(Loc, DiagID) << E->getSourceRange() << Assign;
  return true;
}

static void CheckIdentityFieldAssignment(Expr *LHSExpr, Expr *RHSExpr,
                                         SourceLocation Loc,
                                         Sema &Sema) {
  // C / C++ fields
  MemberExpr *ML = dyn_cast<MemberExpr>(LHSExpr);
  MemberExpr *MR = dyn_cast<MemberExpr>(RHSExpr);
  if (ML && MR && ML->getMemberDecl() == MR->getMemberDecl()) {
    if (isa<CXXThisExpr>(ML->getBase()) && isa<CXXThisExpr>(MR->getBase()))
      Sema.Diag(Loc, diag::warn_identity_field_assign) << 0;
  }

  // Objective-C instance variables
  ObjCIvarRefExpr *OL = dyn_cast<ObjCIvarRefExpr>(LHSExpr);
  ObjCIvarRefExpr *OR = dyn_cast<ObjCIvarRefExpr>(RHSExpr);
  if (OL && OR && OL->getDecl() == OR->getDecl()) {
    DeclRefExpr *RL = dyn_cast<DeclRefExpr>(OL->getBase()->IgnoreImpCasts());
    DeclRefExpr *RR = dyn_cast<DeclRefExpr>(OR->getBase()->IgnoreImpCasts());
    if (RL && RR && RL->getDecl() == RR->getDecl())
      Sema.Diag(Loc, diag::warn_identity_field_assign) << 1;
  }
}

// C99 6.5.16.1
QualType Sema::CheckAssignmentOperands(Expr *LHSExpr, ExprResult &RHS,
                                       SourceLocation Loc,
                                       QualType CompoundType) {
  assert(!LHSExpr->hasPlaceholderType(BuiltinType::PseudoObject));

  // Verify that LHS is a modifiable lvalue, and emit error if not.
  if (CheckForModifiableLvalue(LHSExpr, Loc, *this))
    return QualType();

  QualType LHSType = LHSExpr->getType();
  QualType RHSType = CompoundType.isNull() ? RHS.get()->getType() :
                                             CompoundType;
  AssignConvertType ConvTy;
  if (CompoundType.isNull()) {
    Expr *RHSCheck = RHS.get();

    CheckIdentityFieldAssignment(LHSExpr, RHSCheck, Loc, *this);

    QualType LHSTy(LHSType);
    ConvTy = CheckSingleAssignmentConstraints(LHSTy, RHS);
    if (RHS.isInvalid())
      return QualType();
    // Special case of NSObject attributes on c-style pointer types.
    if (ConvTy == IncompatiblePointer &&
        ((Context.isObjCNSObjectType(LHSType) &&
          RHSType->isObjCObjectPointerType()) ||
         (Context.isObjCNSObjectType(RHSType) &&
          LHSType->isObjCObjectPointerType())))
      ConvTy = Compatible;

    if (ConvTy == Compatible &&
        LHSType->isObjCObjectType())
        Diag(Loc, diag::err_objc_object_assignment)
          << LHSType;

    // If the RHS is a unary plus or minus, check to see if they = and + are
    // right next to each other.  If so, the user may have typo'd "x =+ 4"
    // instead of "x += 4".
    if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(RHSCheck))
      RHSCheck = ICE->getSubExpr();
    if (UnaryOperator *UO = dyn_cast<UnaryOperator>(RHSCheck)) {
      if ((UO->getOpcode() == UO_Plus ||
           UO->getOpcode() == UO_Minus) &&
          Loc.isFileID() && UO->getOperatorLoc().isFileID() &&
          // Only if the two operators are exactly adjacent.
          Loc.getLocWithOffset(1) == UO->getOperatorLoc() &&
          // And there is a space or other character before the subexpr of the
          // unary +/-.  We don't want to warn on "x=-1".
          Loc.getLocWithOffset(2) != UO->getSubExpr()->getLocStart() &&
          UO->getSubExpr()->getLocStart().isFileID()) {
        Diag(Loc, diag::warn_not_compound_assign)
          << (UO->getOpcode() == UO_Plus ? "+" : "-")
          << SourceRange(UO->getOperatorLoc(), UO->getOperatorLoc());
      }
    }

    if (ConvTy == Compatible) {
      if (LHSType.getObjCLifetime() == Qualifiers::OCL_Strong) {
        // Warn about retain cycles where a block captures the LHS, but
        // not if the LHS is a simple variable into which the block is
        // being stored...unless that variable can be captured by reference!
        const Expr *InnerLHS = LHSExpr->IgnoreParenCasts();
        const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(InnerLHS);
        if (!DRE || DRE->getDecl()->hasAttr<BlocksAttr>())
          checkRetainCycles(LHSExpr, RHS.get());

        // It is safe to assign a weak reference into a strong variable.
        // Although this code can still have problems:
        //   id x = self.weakProp;
        //   id y = self.weakProp;
        // we do not warn to warn spuriously when 'x' and 'y' are on separate
        // paths through the function. This should be revisited if
        // -Wrepeated-use-of-weak is made flow-sensitive.
        if (!Diags.isIgnored(diag::warn_arc_repeated_use_of_weak,
                             RHS.get()->getLocStart()))
          getCurFunction()->markSafeWeakUse(RHS.get());

      } else if (getLangOpts().ObjCAutoRefCount) {
        checkUnsafeExprAssigns(Loc, LHSExpr, RHS.get());
      }
    }
  } else {
    // Compound assignment "x += y"
    ConvTy = CheckAssignmentConstraints(Loc, LHSType, RHSType);
  }

  if (DiagnoseAssignmentResult(ConvTy, Loc, LHSType, RHSType,
                               RHS.get(), AA_Assigning))
    return QualType();

  CheckForNullPointerDereference(*this, LHSExpr);

  // C99 6.5.16p3: The type of an assignment expression is the type of the
  // left operand unless the left operand has qualified type, in which case
  // it is the unqualified version of the type of the left operand.
  // C99 6.5.16.1p2: In simple assignment, the value of the right operand
  // is converted to the type of the assignment expression (above).
  // C++ 5.17p1: the type of the assignment expression is that of its left
  // operand.
  return (getLangOpts().CPlusPlus
          ? LHSType : LHSType.getUnqualifiedType());
}

// C99 6.5.17
static QualType CheckCommaOperands(Sema &S, ExprResult &LHS, ExprResult &RHS,
                                   SourceLocation Loc) {
  LHS = S.CheckPlaceholderExpr(LHS.get());
  RHS = S.CheckPlaceholderExpr(RHS.get());
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();

  // C's comma performs lvalue conversion (C99 6.3.2.1) on both its
  // operands, but not unary promotions.
  // C++'s comma does not do any conversions at all (C++ [expr.comma]p1).

  // So we treat the LHS as a ignored value, and in C++ we allow the
  // containing site to determine what should be done with the RHS.
  LHS = S.IgnoredValueConversions(LHS.get());
  if (LHS.isInvalid())
    return QualType();

  S.DiagnoseUnusedExprResult(LHS.get());

  if (!S.getLangOpts().CPlusPlus) {
    RHS = S.DefaultFunctionArrayLvalueConversion(RHS.get());
    if (RHS.isInvalid())
      return QualType();
    if (!RHS.get()->getType()->isVoidType())
      S.RequireCompleteType(Loc, RHS.get()->getType(),
                            diag::err_incomplete_type);
  }

  return RHS.get()->getType();
}

/// CheckIncrementDecrementOperand - unlike most "Check" methods, this routine
/// doesn't need to call UsualUnaryConversions or UsualArithmeticConversions.
static QualType CheckIncrementDecrementOperand(Sema &S, Expr *Op,
                                               ExprValueKind &VK,
                                               ExprObjectKind &OK,
                                               SourceLocation OpLoc,
                                               bool IsInc, bool IsPrefix) {
  if (Op->isTypeDependent())
    return S.Context.DependentTy;

  QualType ResType = Op->getType();
  // Atomic types can be used for increment / decrement where the non-atomic
  // versions can, so ignore the _Atomic() specifier for the purpose of
  // checking.
  if (const AtomicType *ResAtomicType = ResType->getAs<AtomicType>())
    ResType = ResAtomicType->getValueType();

  assert(!ResType.isNull() && "no type for increment/decrement expression");

  if (S.getLangOpts().CPlusPlus && ResType->isBooleanType()) {
    // Decrement of bool is not allowed.
    if (!IsInc) {
      S.Diag(OpLoc, diag::err_decrement_bool) << Op->getSourceRange();
      return QualType();
    }
    // Increment of bool sets it to true, but is deprecated.
    S.Diag(OpLoc, S.getLangOpts().CPlusPlus1z ? diag::ext_increment_bool
                                              : diag::warn_increment_bool)
      << Op->getSourceRange();
  } else if (S.getLangOpts().CPlusPlus && ResType->isEnumeralType()) {
    // Error on enum increments and decrements in C++ mode
    S.Diag(OpLoc, diag::err_increment_decrement_enum) << IsInc << ResType;
    return QualType();
  } else if (ResType->isRealType()) {
    // OK!
  } else if (ResType->isPointerType()) {
    // C99 6.5.2.4p2, 6.5.6p2
    if (!checkArithmeticOpPointerOperand(S, OpLoc, Op))
      return QualType();
  } else if (ResType->isObjCObjectPointerType()) {
    // On modern runtimes, ObjC pointer arithmetic is forbidden.
    // Otherwise, we just need a complete type.
    if (checkArithmeticIncompletePointerType(S, OpLoc, Op) ||
        checkArithmeticOnObjCPointer(S, OpLoc, Op))
      return QualType();    
  } else if (ResType->isAnyComplexType()) {
    // C99 does not support ++/-- on complex types, we allow as an extension.
    S.Diag(OpLoc, diag::ext_integer_increment_complex)
      << ResType << Op->getSourceRange();
  } else if (ResType->isPlaceholderType()) {
    ExprResult PR = S.CheckPlaceholderExpr(Op);
    if (PR.isInvalid()) return QualType();
    return CheckIncrementDecrementOperand(S, PR.get(), VK, OK, OpLoc,
                                          IsInc, IsPrefix);
  } else if (S.getLangOpts().AltiVec && ResType->isVectorType()) {
    // OK! ( C/C++ Language Extensions for CBEA(Version 2.6) 10.3 )
  } else if (S.getLangOpts().ZVector && ResType->isVectorType() &&
             (ResType->getAs<VectorType>()->getVectorKind() !=
              VectorType::AltiVecBool)) {
    // The z vector extensions allow ++ and -- for non-bool vectors.
  } else if(S.getLangOpts().OpenCL && ResType->isVectorType() &&
            ResType->getAs<VectorType>()->getElementType()->isIntegerType()) {
    // OpenCL V1.2 6.3 says dec/inc ops operate on integer vector types.
  } else {
    S.Diag(OpLoc, diag::err_typecheck_illegal_increment_decrement)
      << ResType << int(IsInc) << Op->getSourceRange();
    return QualType();
  }
  // At this point, we know we have a real, complex or pointer type.
  // Now make sure the operand is a modifiable lvalue.
  if (CheckForModifiableLvalue(Op, OpLoc, S))
    return QualType();
  // In C++, a prefix increment is the same type as the operand. Otherwise
  // (in C or with postfix), the increment is the unqualified type of the
  // operand.
  if (IsPrefix && S.getLangOpts().CPlusPlus) {
    VK = VK_LValue;
    OK = Op->getObjectKind();
    return ResType;
  } else {
    VK = VK_RValue;
    return ResType.getUnqualifiedType();
  }
}
  

/// getPrimaryDecl - Helper function for CheckAddressOfOperand().
/// This routine allows us to typecheck complex/recursive expressions
/// where the declaration is needed for type checking. We only need to
/// handle cases when the expression references a function designator
/// or is an lvalue. Here are some examples:
///  - &(x) => x
///  - &*****f => f for f a function designator.
///  - &s.xx => s
///  - &s.zz[1].yy -> s, if zz is an array
///  - *(x + 1) -> x, if x is an array
///  - &"123"[2] -> 0
///  - & __real__ x -> x
static ValueDecl *getPrimaryDecl(Expr *E) {
  switch (E->getStmtClass()) {
  case Stmt::DeclRefExprClass:
    return cast<DeclRefExpr>(E)->getDecl();
  case Stmt::MemberExprClass:
    // If this is an arrow operator, the address is an offset from
    // the base's value, so the object the base refers to is
    // irrelevant.
    if (cast<MemberExpr>(E)->isArrow())
      return nullptr;
    // Otherwise, the expression refers to a part of the base
    return getPrimaryDecl(cast<MemberExpr>(E)->getBase());
  case Stmt::ArraySubscriptExprClass: {
    // FIXME: This code shouldn't be necessary!  We should catch the implicit
    // promotion of register arrays earlier.
    Expr* Base = cast<ArraySubscriptExpr>(E)->getBase();
    if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(Base)) {
      if (ICE->getSubExpr()->getType()->isArrayType())
        return getPrimaryDecl(ICE->getSubExpr());
    }
    return nullptr;
  }
  case Stmt::UnaryOperatorClass: {
    UnaryOperator *UO = cast<UnaryOperator>(E);

    switch(UO->getOpcode()) {
    case UO_Real:
    case UO_Imag:
    case UO_Extension:
      return getPrimaryDecl(UO->getSubExpr());
    default:
      return nullptr;
    }
  }
  case Stmt::ParenExprClass:
    return getPrimaryDecl(cast<ParenExpr>(E)->getSubExpr());
  case Stmt::ImplicitCastExprClass:
    // If the result of an implicit cast is an l-value, we care about
    // the sub-expression; otherwise, the result here doesn't matter.
    return getPrimaryDecl(cast<ImplicitCastExpr>(E)->getSubExpr());
  default:
    return nullptr;
  }
}

namespace {
  enum {
    AO_Bit_Field = 0,
    AO_Vector_Element = 1,
    AO_Property_Expansion = 2,
    AO_Register_Variable = 3,
    AO_No_Error = 4
  };
}
/// \brief Diagnose invalid operand for address of operations.
///
/// \param Type The type of operand which cannot have its address taken.
static void diagnoseAddressOfInvalidType(Sema &S, SourceLocation Loc,
                                         Expr *E, unsigned Type) {
  S.Diag(Loc, diag::err_typecheck_address_of) << Type << E->getSourceRange();
}

/// CheckAddressOfOperand - The operand of & must be either a function
/// designator or an lvalue designating an object. If it is an lvalue, the
/// object cannot be declared with storage class register or be a bit field.
/// Note: The usual conversions are *not* applied to the operand of the &
/// operator (C99 6.3.2.1p[2-4]), and its result is never an lvalue.
/// In C++, the operand might be an overloaded function name, in which case
/// we allow the '&' but retain the overloaded-function type.
QualType Sema::CheckAddressOfOperand(ExprResult &OrigOp, SourceLocation OpLoc) {
  if (const BuiltinType *PTy = OrigOp.get()->getType()->getAsPlaceholderType()){
    if (PTy->getKind() == BuiltinType::Overload) {
      Expr *E = OrigOp.get()->IgnoreParens();
      if (!isa<OverloadExpr>(E)) {
        assert(cast<UnaryOperator>(E)->getOpcode() == UO_AddrOf);
        Diag(OpLoc, diag::err_typecheck_invalid_lvalue_addrof_addrof_function)
          << OrigOp.get()->getSourceRange();
        return QualType();
      }

      OverloadExpr *Ovl = cast<OverloadExpr>(E);
      if (isa<UnresolvedMemberExpr>(Ovl))
        if (!ResolveSingleFunctionTemplateSpecialization(Ovl)) {
          Diag(OpLoc, diag::err_invalid_form_pointer_member_function)
            << OrigOp.get()->getSourceRange();
          return QualType();
        }

      return Context.OverloadTy;
    }

    if (PTy->getKind() == BuiltinType::UnknownAny)
      return Context.UnknownAnyTy;

    if (PTy->getKind() == BuiltinType::BoundMember) {
      Diag(OpLoc, diag::err_invalid_form_pointer_member_function)
        << OrigOp.get()->getSourceRange();
      return QualType();
    }

    OrigOp = CheckPlaceholderExpr(OrigOp.get());
    if (OrigOp.isInvalid()) return QualType();
  }

  if (OrigOp.get()->isTypeDependent())
    return Context.DependentTy;

  assert(!OrigOp.get()->getType()->isPlaceholderType());

  // Make sure to ignore parentheses in subsequent checks
  Expr *op = OrigOp.get()->IgnoreParens();

  // OpenCL v1.0 s6.8.a.3: Pointers to functions are not allowed.
  if (LangOpts.OpenCL && op->getType()->isFunctionType()) {
    Diag(op->getExprLoc(), diag::err_opencl_taking_function_address);
    return QualType();
  }

  if (getLangOpts().C99) {
    // Implement C99-only parts of addressof rules.
    if (UnaryOperator* uOp = dyn_cast<UnaryOperator>(op)) {
      if (uOp->getOpcode() == UO_Deref)
        // Per C99 6.5.3.2, the address of a deref always returns a valid result
        // (assuming the deref expression is valid).
        return uOp->getSubExpr()->getType();
    }
    // Technically, there should be a check for array subscript
    // expressions here, but the result of one is always an lvalue anyway.
  }
  ValueDecl *dcl = getPrimaryDecl(op);

  if (auto *FD = dyn_cast_or_null<FunctionDecl>(dcl))
    if (!checkAddressOfFunctionIsAvailable(FD, /*Complain=*/true,
                                           op->getLocStart()))
      return QualType();

  Expr::LValueClassification lval = op->ClassifyLValue(Context);
  unsigned AddressOfError = AO_No_Error;

  if (lval == Expr::LV_ClassTemporary || lval == Expr::LV_ArrayTemporary) { 
    bool sfinae = (bool)isSFINAEContext();
    Diag(OpLoc, isSFINAEContext() ? diag::err_typecheck_addrof_temporary
                                  : diag::ext_typecheck_addrof_temporary)
      << op->getType() << op->getSourceRange();
    if (sfinae)
      return QualType();
    // Materialize the temporary as an lvalue so that we can take its address.
    OrigOp = op = new (Context)
        MaterializeTemporaryExpr(op->getType(), OrigOp.get(), true);
  } else if (isa<ObjCSelectorExpr>(op)) {
    return Context.getPointerType(op->getType());
  } else if (lval == Expr::LV_MemberFunction) {
    // If it's an instance method, make a member pointer.
    // The expression must have exactly the form &A::foo.

    // If the underlying expression isn't a decl ref, give up.
    if (!isa<DeclRefExpr>(op)) {
      Diag(OpLoc, diag::err_invalid_form_pointer_member_function)
        << OrigOp.get()->getSourceRange();
      return QualType();
    }
    DeclRefExpr *DRE = cast<DeclRefExpr>(op);
    CXXMethodDecl *MD = cast<CXXMethodDecl>(DRE->getDecl());

    // The id-expression was parenthesized.
    if (OrigOp.get() != DRE) {
      Diag(OpLoc, diag::err_parens_pointer_member_function)
        << OrigOp.get()->getSourceRange();

    // The method was named without a qualifier.
    } else if (!DRE->getQualifier()) {
      if (MD->getParent()->getName().empty())
        Diag(OpLoc, diag::err_unqualified_pointer_member_function)
          << op->getSourceRange();
      else {
        SmallString<32> Str;
        StringRef Qual = (MD->getParent()->getName() + "::").toStringRef(Str);
        Diag(OpLoc, diag::err_unqualified_pointer_member_function)
          << op->getSourceRange()
          << FixItHint::CreateInsertion(op->getSourceRange().getBegin(), Qual);
      }
    }

    // Taking the address of a dtor is illegal per C++ [class.dtor]p2.
    if (isa<CXXDestructorDecl>(MD))
      Diag(OpLoc, diag::err_typecheck_addrof_dtor) << op->getSourceRange();

    QualType MPTy = Context.getMemberPointerType(
        op->getType(), Context.getTypeDeclType(MD->getParent()).getTypePtr());
    // Under the MS ABI, lock down the inheritance model now.
    if (Context.getTargetInfo().getCXXABI().isMicrosoft())
      (void)isCompleteType(OpLoc, MPTy);
    return MPTy;
  } else if (lval != Expr::LV_Valid && lval != Expr::LV_IncompleteVoidType) {
    // C99 6.5.3.2p1
    // The operand must be either an l-value or a function designator
    if (!op->getType()->isFunctionType()) {
      // Use a special diagnostic for loads from property references.
      if (isa<PseudoObjectExpr>(op)) {
        AddressOfError = AO_Property_Expansion;
      } else {
        Diag(OpLoc, diag::err_typecheck_invalid_lvalue_addrof)
          << op->getType() << op->getSourceRange();
        return QualType();
      }
    }
  } else if (op->getObjectKind() == OK_BitField) { // C99 6.5.3.2p1
    // The operand cannot be a bit-field
    AddressOfError = AO_Bit_Field;
  } else if (op->getObjectKind() == OK_VectorComponent) {
    // The operand cannot be an element of a vector
    AddressOfError = AO_Vector_Element;
  } else if (dcl) { // C99 6.5.3.2p1
    // We have an lvalue with a decl. Make sure the decl is not declared
    // with the register storage-class specifier.
    if (const VarDecl *vd = dyn_cast<VarDecl>(dcl)) {
      // in C++ it is not error to take address of a register
      // variable (c++03 7.1.1P3)
      if (vd->getStorageClass() == SC_Register &&
          !getLangOpts().CPlusPlus) {
        AddressOfError = AO_Register_Variable;
      }
    } else if (isa<MSPropertyDecl>(dcl)) {
      AddressOfError = AO_Property_Expansion;
    } else if (isa<FunctionTemplateDecl>(dcl)) {
      return Context.OverloadTy;
    } else if (isa<FieldDecl>(dcl) || isa<IndirectFieldDecl>(dcl)) {
      // Okay: we can take the address of a field.
      // Could be a pointer to member, though, if there is an explicit
      // scope qualifier for the class.
      if (isa<DeclRefExpr>(op) && cast<DeclRefExpr>(op)->getQualifier()) {
        DeclContext *Ctx = dcl->getDeclContext();
        if (Ctx && Ctx->isRecord()) {
          if (dcl->getType()->isReferenceType()) {
            Diag(OpLoc,
                 diag::err_cannot_form_pointer_to_member_of_reference_type)
              << dcl->getDeclName() << dcl->getType();
            return QualType();
          }

          while (cast<RecordDecl>(Ctx)->isAnonymousStructOrUnion())
            Ctx = Ctx->getParent();

          QualType MPTy = Context.getMemberPointerType(
              op->getType(),
              Context.getTypeDeclType(cast<RecordDecl>(Ctx)).getTypePtr());
          // Under the MS ABI, lock down the inheritance model now.
          if (Context.getTargetInfo().getCXXABI().isMicrosoft())
            (void)isCompleteType(OpLoc, MPTy);
          return MPTy;
        }
      }
    } else if (!isa<FunctionDecl>(dcl) && !isa<NonTypeTemplateParmDecl>(dcl))
      llvm_unreachable("Unknown/unexpected decl type");
  }

  if (AddressOfError != AO_No_Error) {
    diagnoseAddressOfInvalidType(*this, OpLoc, op, AddressOfError);
    return QualType();
  }

  if (lval == Expr::LV_IncompleteVoidType) {
    // Taking the address of a void variable is technically illegal, but we
    // allow it in cases which are otherwise valid.
    // Example: "extern void x; void* y = &x;".
    Diag(OpLoc, diag::ext_typecheck_addrof_void) << op->getSourceRange();
  }

  // If the operand has type "type", the result has type "pointer to type".
  if (op->getType()->isObjCObjectType())
    return Context.getObjCObjectPointerType(op->getType());
  return Context.getPointerType(op->getType());
}

static void RecordModifiableNonNullParam(Sema &S, const Expr *Exp) {
  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Exp);
  if (!DRE)
    return;
  const Decl *D = DRE->getDecl();
  if (!D)
    return;
  const ParmVarDecl *Param = dyn_cast<ParmVarDecl>(D);
  if (!Param)
    return;
  if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(Param->getDeclContext()))
    if (!FD->hasAttr<NonNullAttr>() && !Param->hasAttr<NonNullAttr>())
      return;
  if (FunctionScopeInfo *FD = S.getCurFunction())
    if (!FD->ModifiedNonNullParams.count(Param))
      FD->ModifiedNonNullParams.insert(Param);
}

/// CheckIndirectionOperand - Type check unary indirection (prefix '*').
static QualType CheckIndirectionOperand(Sema &S, Expr *Op, ExprValueKind &VK,
                                        SourceLocation OpLoc) {
  if (Op->isTypeDependent())
    return S.Context.DependentTy;

  ExprResult ConvResult = S.UsualUnaryConversions(Op);
  if (ConvResult.isInvalid())
    return QualType();
  Op = ConvResult.get();
  QualType OpTy = Op->getType();
  QualType Result;

  if (isa<CXXReinterpretCastExpr>(Op)) {
    QualType OpOrigType = Op->IgnoreParenCasts()->getType();
    S.CheckCompatibleReinterpretCast(OpOrigType, OpTy, /*IsDereference*/true,
                                     Op->getSourceRange());
  }

  if (const PointerType *PT = OpTy->getAs<PointerType>())
    Result = PT->getPointeeType();
  else if (const ObjCObjectPointerType *OPT =
             OpTy->getAs<ObjCObjectPointerType>())
    Result = OPT->getPointeeType();
  else {
    ExprResult PR = S.CheckPlaceholderExpr(Op);
    if (PR.isInvalid()) return QualType();
    if (PR.get() != Op)
      return CheckIndirectionOperand(S, PR.get(), VK, OpLoc);
  }

  if (Result.isNull()) {
    S.Diag(OpLoc, diag::err_typecheck_indirection_requires_pointer)
      << OpTy << Op->getSourceRange();
    return QualType();
  }

  // Note that per both C89 and C99, indirection is always legal, even if Result
  // is an incomplete type or void.  It would be possible to warn about
  // dereferencing a void pointer, but it's completely well-defined, and such a
  // warning is unlikely to catch any mistakes. In C++, indirection is not valid
  // for pointers to 'void' but is fine for any other pointer type:
  //
  // C++ [expr.unary.op]p1:
  //   [...] the expression to which [the unary * operator] is applied shall
  //   be a pointer to an object type, or a pointer to a function type
  if (S.getLangOpts().CPlusPlus && Result->isVoidType())
    S.Diag(OpLoc, diag::ext_typecheck_indirection_through_void_pointer)
      << OpTy << Op->getSourceRange();

  // Dereferences are usually l-values...
  VK = VK_LValue;

  // ...except that certain expressions are never l-values in C.
  if (!S.getLangOpts().CPlusPlus && Result.isCForbiddenLValueType())
    VK = VK_RValue;
  
  return Result;
}

BinaryOperatorKind Sema::ConvertTokenKindToBinaryOpcode(tok::TokenKind Kind) {
  BinaryOperatorKind Opc;
  switch (Kind) {
  default: llvm_unreachable("Unknown binop!");
  case tok::periodstar:           Opc = BO_PtrMemD; break;
  case tok::arrowstar:            Opc = BO_PtrMemI; break;
  case tok::star:                 Opc = BO_Mul; break;
  case tok::slash:                Opc = BO_Div; break;
  case tok::percent:              Opc = BO_Rem; break;
  case tok::plus:                 Opc = BO_Add; break;
  case tok::minus:                Opc = BO_Sub; break;
  case tok::lessless:             Opc = BO_Shl; break;
  case tok::greatergreater:       Opc = BO_Shr; break;
  case tok::lessequal:            Opc = BO_LE; break;
  case tok::less:                 Opc = BO_LT; break;
  case tok::greaterequal:         Opc = BO_GE; break;
  case tok::greater:              Opc = BO_GT; break;
  case tok::exclaimequal:         Opc = BO_NE; break;
  case tok::equalequal:           Opc = BO_EQ; break;
  case tok::amp:                  Opc = BO_And; break;
  case tok::caret:                Opc = BO_Xor; break;
  case tok::pipe:                 Opc = BO_Or; break;
  case tok::ampamp:               Opc = BO_LAnd; break;
  case tok::pipepipe:             Opc = BO_LOr; break;
  case tok::equal:                Opc = BO_Assign; break;
  case tok::starequal:            Opc = BO_MulAssign; break;
  case tok::slashequal:           Opc = BO_DivAssign; break;
  case tok::percentequal:         Opc = BO_RemAssign; break;
  case tok::plusequal:            Opc = BO_AddAssign; break;
  case tok::minusequal:           Opc = BO_SubAssign; break;
  case tok::lesslessequal:        Opc = BO_ShlAssign; break;
  case tok::greatergreaterequal:  Opc = BO_ShrAssign; break;
  case tok::ampequal:             Opc = BO_AndAssign; break;
  case tok::caretequal:           Opc = BO_XorAssign; break;
  case tok::pipeequal:            Opc = BO_OrAssign; break;
  case tok::comma:                Opc = BO_Comma; break;
  }
  return Opc;
}

static inline UnaryOperatorKind ConvertTokenKindToUnaryOpcode(
  tok::TokenKind Kind) {
  UnaryOperatorKind Opc;
  switch (Kind) {
  default: llvm_unreachable("Unknown unary op!");
  case tok::plusplus:     Opc = UO_PreInc; break;
  case tok::minusminus:   Opc = UO_PreDec; break;
  case tok::amp:          Opc = UO_AddrOf; break;
  case tok::star:         Opc = UO_Deref; break;
  case tok::plus:         Opc = UO_Plus; break;
  case tok::minus:        Opc = UO_Minus; break;
  case tok::tilde:        Opc = UO_Not; break;
  case tok::exclaim:      Opc = UO_LNot; break;
  case tok::kw___real:    Opc = UO_Real; break;
  case tok::kw___imag:    Opc = UO_Imag; break;
  case tok::kw___extension__: Opc = UO_Extension; break;
  }
  return Opc;
}

/// DiagnoseSelfAssignment - Emits a warning if a value is assigned to itself.
/// This warning is only emitted for builtin assignment operations. It is also
/// suppressed in the event of macro expansions.
static void DiagnoseSelfAssignment(Sema &S, Expr *LHSExpr, Expr *RHSExpr,
                                   SourceLocation OpLoc) {
  if (!S.ActiveTemplateInstantiations.empty())
    return;
  if (OpLoc.isInvalid() || OpLoc.isMacroID())
    return;
  LHSExpr = LHSExpr->IgnoreParenImpCasts();
  RHSExpr = RHSExpr->IgnoreParenImpCasts();
  const DeclRefExpr *LHSDeclRef = dyn_cast<DeclRefExpr>(LHSExpr);
  const DeclRefExpr *RHSDeclRef = dyn_cast<DeclRefExpr>(RHSExpr);
  if (!LHSDeclRef || !RHSDeclRef ||
      LHSDeclRef->getLocation().isMacroID() ||
      RHSDeclRef->getLocation().isMacroID())
    return;
  const ValueDecl *LHSDecl =
    cast<ValueDecl>(LHSDeclRef->getDecl()->getCanonicalDecl());
  const ValueDecl *RHSDecl =
    cast<ValueDecl>(RHSDeclRef->getDecl()->getCanonicalDecl());
  if (LHSDecl != RHSDecl)
    return;
  if (LHSDecl->getType().isVolatileQualified())
    return;
  if (const ReferenceType *RefTy = LHSDecl->getType()->getAs<ReferenceType>())
    if (RefTy->getPointeeType().isVolatileQualified())
      return;

  S.Diag(OpLoc, diag::warn_self_assignment)
      << LHSDeclRef->getType()
      << LHSExpr->getSourceRange() << RHSExpr->getSourceRange();
}

/// Check if a bitwise-& is performed on an Objective-C pointer.  This
/// is usually indicative of introspection within the Objective-C pointer.
static void checkObjCPointerIntrospection(Sema &S, ExprResult &L, ExprResult &R,
                                          SourceLocation OpLoc) {
  if (!S.getLangOpts().ObjC1)
    return;

  const Expr *ObjCPointerExpr = nullptr, *OtherExpr = nullptr;
  const Expr *LHS = L.get();
  const Expr *RHS = R.get();

  if (LHS->IgnoreParenCasts()->getType()->isObjCObjectPointerType()) {
    ObjCPointerExpr = LHS;
    OtherExpr = RHS;
  }
  else if (RHS->IgnoreParenCasts()->getType()->isObjCObjectPointerType()) {
    ObjCPointerExpr = RHS;
    OtherExpr = LHS;
  }

  // This warning is deliberately made very specific to reduce false
  // positives with logic that uses '&' for hashing.  This logic mainly
  // looks for code trying to introspect into tagged pointers, which
  // code should generally never do.
  if (ObjCPointerExpr && isa<IntegerLiteral>(OtherExpr->IgnoreParenCasts())) {
    unsigned Diag = diag::warn_objc_pointer_masking;
    // Determine if we are introspecting the result of performSelectorXXX.
    const Expr *Ex = ObjCPointerExpr->IgnoreParenCasts();
    // Special case messages to -performSelector and friends, which
    // can return non-pointer values boxed in a pointer value.
    // Some clients may wish to silence warnings in this subcase.
    if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(Ex)) {
      Selector S = ME->getSelector();
      StringRef SelArg0 = S.getNameForSlot(0);
      if (SelArg0.startswith("performSelector"))
        Diag = diag::warn_objc_pointer_masking_performSelector;
    }
    
    S.Diag(OpLoc, Diag)
      << ObjCPointerExpr->getSourceRange();
  }
}

static NamedDecl *getDeclFromExpr(Expr *E) {
  if (!E)
    return nullptr;
  if (auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl();
  if (auto *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (auto *IRE = dyn_cast<ObjCIvarRefExpr>(E))
    return IRE->getDecl();
  return nullptr;
}

/// CreateBuiltinBinOp - Creates a new built-in binary operation with
/// operator @p Opc at location @c TokLoc. This routine only supports
/// built-in operations; ActOnBinOp handles overloaded operators.
ExprResult Sema::CreateBuiltinBinOp(SourceLocation OpLoc,
                                    BinaryOperatorKind Opc,
                                    Expr *LHSExpr, Expr *RHSExpr) {
  if (getLangOpts().CPlusPlus11 && isa<InitListExpr>(RHSExpr)) {
    // The syntax only allows initializer lists on the RHS of assignment,
    // so we don't need to worry about accepting invalid code for
    // non-assignment operators.
    // C++11 5.17p9:
    //   The meaning of x = {v} [...] is that of x = T(v) [...]. The meaning
    //   of x = {} is x = T().
    InitializationKind Kind =
        InitializationKind::CreateDirectList(RHSExpr->getLocStart());
    InitializedEntity Entity =
        InitializedEntity::InitializeTemporary(LHSExpr->getType());
    InitializationSequence InitSeq(*this, Entity, Kind, RHSExpr);
    ExprResult Init = InitSeq.Perform(*this, Entity, Kind, RHSExpr);
    if (Init.isInvalid())
      return Init;
    RHSExpr = Init.get();
  }

  ExprResult LHS = LHSExpr, RHS = RHSExpr;
  QualType ResultTy;     // Result type of the binary operator.
  // The following two variables are used for compound assignment operators
  QualType CompLHSTy;    // Type of LHS after promotions for computation
  QualType CompResultTy; // Type of computation result
  ExprValueKind VK = VK_RValue;
  ExprObjectKind OK = OK_Ordinary;

  if (!getLangOpts().CPlusPlus) {
    // C cannot handle TypoExpr nodes on either side of a binop because it
    // doesn't handle dependent types properly, so make sure any TypoExprs have
    // been dealt with before checking the operands.
    LHS = CorrectDelayedTyposInExpr(LHSExpr);
    RHS = CorrectDelayedTyposInExpr(RHSExpr, [Opc, LHS](Expr *E) {
      if (Opc != BO_Assign)
        return ExprResult(E);
      // Avoid correcting the RHS to the same Expr as the LHS.
      Decl *D = getDeclFromExpr(E);
      return (D && D == getDeclFromExpr(LHS.get())) ? ExprError() : E;
    });
    if (!LHS.isUsable() || !RHS.isUsable())
      return ExprError();
  }

  if (getLangOpts().OpenCL) {
    // OpenCLC v2.0 s6.13.11.1 allows atomic variables to be initialized by
    // the ATOMIC_VAR_INIT macro.
    if (LHSExpr->getType()->isAtomicType() ||
        RHSExpr->getType()->isAtomicType()) {
      SourceRange SR(LHSExpr->getLocStart(), RHSExpr->getLocEnd());
      if (BO_Assign == Opc)
        Diag(OpLoc, diag::err_atomic_init_constant) << SR;
      else
        ResultTy = InvalidOperands(OpLoc, LHS, RHS);
      return ExprError();
    }
  }

  switch (Opc) {
  case BO_Assign:
    ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, QualType());
    if (getLangOpts().CPlusPlus &&
        LHS.get()->getObjectKind() != OK_ObjCProperty) {
      VK = LHS.get()->getValueKind();
      OK = LHS.get()->getObjectKind();
    }
    if (!ResultTy.isNull()) {
      DiagnoseSelfAssignment(*this, LHS.get(), RHS.get(), OpLoc);
      DiagnoseSelfMove(LHS.get(), RHS.get(), OpLoc);
    }
    RecordModifiableNonNullParam(*this, LHS.get());
    break;
  case BO_PtrMemD:
  case BO_PtrMemI:
    ResultTy = CheckPointerToMemberOperands(LHS, RHS, VK, OpLoc,
                                            Opc == BO_PtrMemI);
    break;
  case BO_Mul:
  case BO_Div:
    ResultTy = CheckMultiplyDivideOperands(LHS, RHS, OpLoc, false,
                                           Opc == BO_Div);
    break;
  case BO_Rem:
    ResultTy = CheckRemainderOperands(LHS, RHS, OpLoc);
    break;
  case BO_Add:
    ResultTy = CheckAdditionOperands(LHS, RHS, OpLoc, Opc);
    break;
  case BO_Sub:
    ResultTy = CheckSubtractionOperands(LHS, RHS, OpLoc);
    break;
  case BO_Shl:
  case BO_Shr:
    ResultTy = CheckShiftOperands(LHS, RHS, OpLoc, Opc);
    break;
  case BO_LE:
  case BO_LT:
  case BO_GE:
  case BO_GT:
    ResultTy = CheckCompareOperands(LHS, RHS, OpLoc, Opc, true);
    break;
  case BO_EQ:
  case BO_NE:
    ResultTy = CheckCompareOperands(LHS, RHS, OpLoc, Opc, false);
    break;
  case BO_And:
    checkObjCPointerIntrospection(*this, LHS, RHS, OpLoc);
  case BO_Xor:
  case BO_Or:
    ResultTy = CheckBitwiseOperands(LHS, RHS, OpLoc);
    break;
  case BO_LAnd:
  case BO_LOr:
    ResultTy = CheckLogicalOperands(LHS, RHS, OpLoc, Opc);
    break;
  case BO_MulAssign:
  case BO_DivAssign:
    CompResultTy = CheckMultiplyDivideOperands(LHS, RHS, OpLoc, true,
                                               Opc == BO_DivAssign);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_RemAssign:
    CompResultTy = CheckRemainderOperands(LHS, RHS, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_AddAssign:
    CompResultTy = CheckAdditionOperands(LHS, RHS, OpLoc, Opc, &CompLHSTy);
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_SubAssign:
    CompResultTy = CheckSubtractionOperands(LHS, RHS, OpLoc, &CompLHSTy);
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_ShlAssign:
  case BO_ShrAssign:
    CompResultTy = CheckShiftOperands(LHS, RHS, OpLoc, Opc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_AndAssign:
  case BO_OrAssign: // fallthrough
    DiagnoseSelfAssignment(*this, LHS.get(), RHS.get(), OpLoc);
  case BO_XorAssign:
    CompResultTy = CheckBitwiseOperands(LHS, RHS, OpLoc, true);
    CompLHSTy = CompResultTy;
    if (!CompResultTy.isNull() && !LHS.isInvalid() && !RHS.isInvalid())
      ResultTy = CheckAssignmentOperands(LHS.get(), RHS, OpLoc, CompResultTy);
    break;
  case BO_Comma:
    ResultTy = CheckCommaOperands(*this, LHS, RHS, OpLoc);
    if (getLangOpts().CPlusPlus && !RHS.isInvalid()) {
      VK = RHS.get()->getValueKind();
      OK = RHS.get()->getObjectKind();
    }
    break;
  }
  if (ResultTy.isNull() || LHS.isInvalid() || RHS.isInvalid())
    return ExprError();

  // Check for array bounds violations for both sides of the BinaryOperator
  CheckArrayAccess(LHS.get());
  CheckArrayAccess(RHS.get());

  if (const ObjCIsaExpr *OISA = dyn_cast<ObjCIsaExpr>(LHS.get()->IgnoreParenCasts())) {
    NamedDecl *ObjectSetClass = LookupSingleName(TUScope,
                                                 &Context.Idents.get("object_setClass"),
                                                 SourceLocation(), LookupOrdinaryName);
    if (ObjectSetClass && isa<ObjCIsaExpr>(LHS.get())) {
      SourceLocation RHSLocEnd = getLocForEndOfToken(RHS.get()->getLocEnd());
      Diag(LHS.get()->getExprLoc(), diag::warn_objc_isa_assign) <<
      FixItHint::CreateInsertion(LHS.get()->getLocStart(), "object_setClass(") <<
      FixItHint::CreateReplacement(SourceRange(OISA->getOpLoc(), OpLoc), ",") <<
      FixItHint::CreateInsertion(RHSLocEnd, ")");
    }
    else
      Diag(LHS.get()->getExprLoc(), diag::warn_objc_isa_assign);
  }
  else if (const ObjCIvarRefExpr *OIRE =
           dyn_cast<ObjCIvarRefExpr>(LHS.get()->IgnoreParenCasts()))
    DiagnoseDirectIsaAccess(*this, OIRE, OpLoc, RHS.get());
  
  if (CompResultTy.isNull())
    return new (Context) BinaryOperator(LHS.get(), RHS.get(), Opc, ResultTy, VK,
                                        OK, OpLoc, FPFeatures.fp_contract);
  if (getLangOpts().CPlusPlus && LHS.get()->getObjectKind() !=
      OK_ObjCProperty) {
    VK = VK_LValue;
    OK = LHS.get()->getObjectKind();
  }
  return new (Context) CompoundAssignOperator(
      LHS.get(), RHS.get(), Opc, ResultTy, VK, OK, CompLHSTy, CompResultTy,
      OpLoc, FPFeatures.fp_contract);
}

/// DiagnoseBitwisePrecedence - Emit a warning when bitwise and comparison
/// operators are mixed in a way that suggests that the programmer forgot that
/// comparison operators have higher precedence. The most typical example of
/// such code is "flags & 0x0020 != 0", which is equivalent to "flags & 1".
static void DiagnoseBitwisePrecedence(Sema &Self, BinaryOperatorKind Opc,
                                      SourceLocation OpLoc, Expr *LHSExpr,
                                      Expr *RHSExpr) {
  BinaryOperator *LHSBO = dyn_cast<BinaryOperator>(LHSExpr);
  BinaryOperator *RHSBO = dyn_cast<BinaryOperator>(RHSExpr);

  // Check that one of the sides is a comparison operator and the other isn't.
  bool isLeftComp = LHSBO && LHSBO->isComparisonOp();
  bool isRightComp = RHSBO && RHSBO->isComparisonOp();
  if (isLeftComp == isRightComp)
    return;

  // Bitwise operations are sometimes used as eager logical ops.
  // Don't diagnose this.
  bool isLeftBitwise = LHSBO && LHSBO->isBitwiseOp();
  bool isRightBitwise = RHSBO && RHSBO->isBitwiseOp();
  if (isLeftBitwise || isRightBitwise)
    return;

  SourceRange DiagRange = isLeftComp ? SourceRange(LHSExpr->getLocStart(),
                                                   OpLoc)
                                     : SourceRange(OpLoc, RHSExpr->getLocEnd());
  StringRef OpStr = isLeftComp ? LHSBO->getOpcodeStr() : RHSBO->getOpcodeStr();
  SourceRange ParensRange = isLeftComp ?
      SourceRange(LHSBO->getRHS()->getLocStart(), RHSExpr->getLocEnd())
    : SourceRange(LHSExpr->getLocStart(), RHSBO->getLHS()->getLocEnd());

  Self.Diag(OpLoc, diag::warn_precedence_bitwise_rel)
    << DiagRange << BinaryOperator::getOpcodeStr(Opc) << OpStr;
  SuggestParentheses(Self, OpLoc,
    Self.PDiag(diag::note_precedence_silence) << OpStr,
    (isLeftComp ? LHSExpr : RHSExpr)->getSourceRange());
  SuggestParentheses(Self, OpLoc,
    Self.PDiag(diag::note_precedence_bitwise_first)
      << BinaryOperator::getOpcodeStr(Opc),
    ParensRange);
}

/// \brief It accepts a '&&' expr that is inside a '||' one.
/// Emit a diagnostic together with a fixit hint that wraps the '&&' expression
/// in parentheses.
static void
EmitDiagnosticForLogicalAndInLogicalOr(Sema &Self, SourceLocation OpLoc,
                                       BinaryOperator *Bop) {
  assert(Bop->getOpcode() == BO_LAnd);
  Self.Diag(Bop->getOperatorLoc(), diag::warn_logical_and_in_logical_or)
      << Bop->getSourceRange() << OpLoc;
  SuggestParentheses(Self, Bop->getOperatorLoc(),
    Self.PDiag(diag::note_precedence_silence)
      << Bop->getOpcodeStr(),
    Bop->getSourceRange());
}

/// \brief Returns true if the given expression can be evaluated as a constant
/// 'true'.
static bool EvaluatesAsTrue(Sema &S, Expr *E) {
  bool Res;
  return !E->isValueDependent() &&
         E->EvaluateAsBooleanCondition(Res, S.getASTContext()) && Res;
}

/// \brief Returns true if the given expression can be evaluated as a constant
/// 'false'.
static bool EvaluatesAsFalse(Sema &S, Expr *E) {
  bool Res;
  return !E->isValueDependent() &&
         E->EvaluateAsBooleanCondition(Res, S.getASTContext()) && !Res;
}

/// \brief Look for '&&' in the left hand of a '||' expr.
static void DiagnoseLogicalAndInLogicalOrLHS(Sema &S, SourceLocation OpLoc,
                                             Expr *LHSExpr, Expr *RHSExpr) {
  if (BinaryOperator *Bop = dyn_cast<BinaryOperator>(LHSExpr)) {
    if (Bop->getOpcode() == BO_LAnd) {
      // If it's "a && b || 0" don't warn since the precedence doesn't matter.
      if (EvaluatesAsFalse(S, RHSExpr))
        return;
      // If it's "1 && a || b" don't warn since the precedence doesn't matter.
      if (!EvaluatesAsTrue(S, Bop->getLHS()))
        return EmitDiagnosticForLogicalAndInLogicalOr(S, OpLoc, Bop);
    } else if (Bop->getOpcode() == BO_LOr) {
      if (BinaryOperator *RBop = dyn_cast<BinaryOperator>(Bop->getRHS())) {
        // If it's "a || b && 1 || c" we didn't warn earlier for
        // "a || b && 1", but warn now.
        if (RBop->getOpcode() == BO_LAnd && EvaluatesAsTrue(S, RBop->getRHS()))
          return EmitDiagnosticForLogicalAndInLogicalOr(S, OpLoc, RBop);
      }
    }
  }
}

/// \brief Look for '&&' in the right hand of a '||' expr.
static void DiagnoseLogicalAndInLogicalOrRHS(Sema &S, SourceLocation OpLoc,
                                             Expr *LHSExpr, Expr *RHSExpr) {
  if (BinaryOperator *Bop = dyn_cast<BinaryOperator>(RHSExpr)) {
    if (Bop->getOpcode() == BO_LAnd) {
      // If it's "0 || a && b" don't warn since the precedence doesn't matter.
      if (EvaluatesAsFalse(S, LHSExpr))
        return;
      // If it's "a || b && 1" don't warn since the precedence doesn't matter.
      if (!EvaluatesAsTrue(S, Bop->getRHS()))
        return EmitDiagnosticForLogicalAndInLogicalOr(S, OpLoc, Bop);
    }
  }
}

/// \brief Look for bitwise op in the left or right hand of a bitwise op with
/// lower precedence and emit a diagnostic together with a fixit hint that wraps
/// the '&' expression in parentheses.
static void DiagnoseBitwiseOpInBitwiseOp(Sema &S, BinaryOperatorKind Opc,
                                         SourceLocation OpLoc, Expr *SubExpr) {
  if (BinaryOperator *Bop = dyn_cast<BinaryOperator>(SubExpr)) {
    if (Bop->isBitwiseOp() && Bop->getOpcode() < Opc) {
      S.Diag(Bop->getOperatorLoc(), diag::warn_bitwise_op_in_bitwise_op)
        << Bop->getOpcodeStr() << BinaryOperator::getOpcodeStr(Opc)
        << Bop->getSourceRange() << OpLoc;
      SuggestParentheses(S, Bop->getOperatorLoc(),
        S.PDiag(diag::note_precedence_silence)
          << Bop->getOpcodeStr(),
        Bop->getSourceRange());
    }
  }
}

static void DiagnoseAdditionInShift(Sema &S, SourceLocation OpLoc,
                                    Expr *SubExpr, StringRef Shift) {
  if (BinaryOperator *Bop = dyn_cast<BinaryOperator>(SubExpr)) {
    if (Bop->getOpcode() == BO_Add || Bop->getOpcode() == BO_Sub) {
      StringRef Op = Bop->getOpcodeStr();
      S.Diag(Bop->getOperatorLoc(), diag::warn_addition_in_bitshift)
          << Bop->getSourceRange() << OpLoc << Shift << Op;
      SuggestParentheses(S, Bop->getOperatorLoc(),
          S.PDiag(diag::note_precedence_silence) << Op,
          Bop->getSourceRange());
    }
  }
}

static void DiagnoseShiftCompare(Sema &S, SourceLocation OpLoc,
                                 Expr *LHSExpr, Expr *RHSExpr) {
  CXXOperatorCallExpr *OCE = dyn_cast<CXXOperatorCallExpr>(LHSExpr);
  if (!OCE)
    return;

  FunctionDecl *FD = OCE->getDirectCallee();
  if (!FD || !FD->isOverloadedOperator())
    return;

  OverloadedOperatorKind Kind = FD->getOverloadedOperator();
  if (Kind != OO_LessLess && Kind != OO_GreaterGreater)
    return;

  S.Diag(OpLoc, diag::warn_overloaded_shift_in_comparison)
      << LHSExpr->getSourceRange() << RHSExpr->getSourceRange()
      << (Kind == OO_LessLess);
  SuggestParentheses(S, OCE->getOperatorLoc(),
                     S.PDiag(diag::note_precedence_silence)
                         << (Kind == OO_LessLess ? "<<" : ">>"),
                     OCE->getSourceRange());
  SuggestParentheses(S, OpLoc,
                     S.PDiag(diag::note_evaluate_comparison_first),
                     SourceRange(OCE->getArg(1)->getLocStart(),
                                 RHSExpr->getLocEnd()));
}

/// DiagnoseBinOpPrecedence - Emit warnings for expressions with tricky
/// precedence.
static void DiagnoseBinOpPrecedence(Sema &Self, BinaryOperatorKind Opc,
                                    SourceLocation OpLoc, Expr *LHSExpr,
                                    Expr *RHSExpr){
  // Diagnose "arg1 'bitwise' arg2 'eq' arg3".
  if (BinaryOperator::isBitwiseOp(Opc))
    DiagnoseBitwisePrecedence(Self, Opc, OpLoc, LHSExpr, RHSExpr);

  // Diagnose "arg1 & arg2 | arg3"
  if ((Opc == BO_Or || Opc == BO_Xor) &&
      !OpLoc.isMacroID()/* Don't warn in macros. */) {
    DiagnoseBitwiseOpInBitwiseOp(Self, Opc, OpLoc, LHSExpr);
    DiagnoseBitwiseOpInBitwiseOp(Self, Opc, OpLoc, RHSExpr);
  }

  // Warn about arg1 || arg2 && arg3, as GCC 4.3+ does.
  // We don't warn for 'assert(a || b && "bad")' since this is safe.
  if (Opc == BO_LOr && !OpLoc.isMacroID()/* Don't warn in macros. */) {
    DiagnoseLogicalAndInLogicalOrLHS(Self, OpLoc, LHSExpr, RHSExpr);
    DiagnoseLogicalAndInLogicalOrRHS(Self, OpLoc, LHSExpr, RHSExpr);
  }

  if ((Opc == BO_Shl && LHSExpr->getType()->isIntegralType(Self.getASTContext()))
      || Opc == BO_Shr) {
    StringRef Shift = BinaryOperator::getOpcodeStr(Opc);
    DiagnoseAdditionInShift(Self, OpLoc, LHSExpr, Shift);
    DiagnoseAdditionInShift(Self, OpLoc, RHSExpr, Shift);
  }

  // Warn on overloaded shift operators and comparisons, such as:
  // cout << 5 == 4;
  if (BinaryOperator::isComparisonOp(Opc))
    DiagnoseShiftCompare(Self, OpLoc, LHSExpr, RHSExpr);
}

// Binary Operators.  'Tok' is the token for the operator.
ExprResult Sema::ActOnBinOp(Scope *S, SourceLocation TokLoc,
                            tok::TokenKind Kind,
                            Expr *LHSExpr, Expr *RHSExpr) {
  BinaryOperatorKind Opc = ConvertTokenKindToBinaryOpcode(Kind);
  assert(LHSExpr && "ActOnBinOp(): missing left expression");
  assert(RHSExpr && "ActOnBinOp(): missing right expression");

  // Emit warnings for tricky precedence issues, e.g. "bitfield & 0x4 == 0"
  DiagnoseBinOpPrecedence(*this, Opc, TokLoc, LHSExpr, RHSExpr);

  return BuildBinOp(S, TokLoc, Opc, LHSExpr, RHSExpr);
}

/// Build an overloaded binary operator expression in the given scope.
static ExprResult BuildOverloadedBinOp(Sema &S, Scope *Sc, SourceLocation OpLoc,
                                       BinaryOperatorKind Opc,
                                       Expr *LHS, Expr *RHS) {
  // Find all of the overloaded operators visible from this
  // point. We perform both an operator-name lookup from the local
  // scope and an argument-dependent lookup based on the types of
  // the arguments.
  UnresolvedSet<16> Functions;
  OverloadedOperatorKind OverOp
    = BinaryOperator::getOverloadedOperator(Opc);
  if (Sc && OverOp != OO_None && OverOp != OO_Equal)
    S.LookupOverloadedOperatorName(OverOp, Sc, LHS->getType(),
                                   RHS->getType(), Functions);

  // Build the (potentially-overloaded, potentially-dependent)
  // binary operation.
  return S.CreateOverloadedBinOp(OpLoc, Opc, Functions, LHS, RHS);
}

ExprResult Sema::BuildBinOp(Scope *S, SourceLocation OpLoc,
                            BinaryOperatorKind Opc,
                            Expr *LHSExpr, Expr *RHSExpr) {
  // We want to end up calling one of checkPseudoObjectAssignment
  // (if the LHS is a pseudo-object), BuildOverloadedBinOp (if
  // both expressions are overloadable or either is type-dependent),
  // or CreateBuiltinBinOp (in any other case).  We also want to get
  // any placeholder types out of the way.

  // Handle pseudo-objects in the LHS.
  if (const BuiltinType *pty = LHSExpr->getType()->getAsPlaceholderType()) {
    // Assignments with a pseudo-object l-value need special analysis.
    if (pty->getKind() == BuiltinType::PseudoObject &&
        BinaryOperator::isAssignmentOp(Opc))
      return checkPseudoObjectAssignment(S, OpLoc, Opc, LHSExpr, RHSExpr);

    // Don't resolve overloads if the other type is overloadable.
    if (pty->getKind() == BuiltinType::Overload) {
      // We can't actually test that if we still have a placeholder,
      // though.  Fortunately, none of the exceptions we see in that
      // code below are valid when the LHS is an overload set.  Note
      // that an overload set can be dependently-typed, but it never
      // instantiates to having an overloadable type.
      ExprResult resolvedRHS = CheckPlaceholderExpr(RHSExpr);
      if (resolvedRHS.isInvalid()) return ExprError();
      RHSExpr = resolvedRHS.get();

      if (RHSExpr->isTypeDependent() ||
          RHSExpr->getType()->isOverloadableType())
        return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);
    }
        
    ExprResult LHS = CheckPlaceholderExpr(LHSExpr);
    if (LHS.isInvalid()) return ExprError();
    LHSExpr = LHS.get();
  }

  // Handle pseudo-objects in the RHS.
  if (const BuiltinType *pty = RHSExpr->getType()->getAsPlaceholderType()) {
    // An overload in the RHS can potentially be resolved by the type
    // being assigned to.
    if (Opc == BO_Assign && pty->getKind() == BuiltinType::Overload) {
      if (LHSExpr->isTypeDependent() || RHSExpr->isTypeDependent())
        return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);

      if (LHSExpr->getType()->isOverloadableType())
        return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);

      return CreateBuiltinBinOp(OpLoc, Opc, LHSExpr, RHSExpr);
    }

    // Don't resolve overloads if the other type is overloadable.
    if (pty->getKind() == BuiltinType::Overload &&
        LHSExpr->getType()->isOverloadableType())
      return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);

    ExprResult resolvedRHS = CheckPlaceholderExpr(RHSExpr);
    if (!resolvedRHS.isUsable()) return ExprError();
    RHSExpr = resolvedRHS.get();
  }

  if (getLangOpts().CPlusPlus) {
    // If either expression is type-dependent, always build an
    // overloaded op.
    if (LHSExpr->isTypeDependent() || RHSExpr->isTypeDependent())
      return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);

    // Otherwise, build an overloaded op if either expression has an
    // overloadable type.
    if (LHSExpr->getType()->isOverloadableType() ||
        RHSExpr->getType()->isOverloadableType())
      return BuildOverloadedBinOp(*this, S, OpLoc, Opc, LHSExpr, RHSExpr);
  }

  // Build a built-in binary operation.
  return CreateBuiltinBinOp(OpLoc, Opc, LHSExpr, RHSExpr);
}

ExprResult Sema::CreateBuiltinUnaryOp(SourceLocation OpLoc,
                                      UnaryOperatorKind Opc,
                                      Expr *InputExpr) {
  ExprResult Input = InputExpr;
  ExprValueKind VK = VK_RValue;
  ExprObjectKind OK = OK_Ordinary;
  QualType resultType;
  if (getLangOpts().OpenCL) {
    // The only legal unary operation for atomics is '&'.
    if (Opc != UO_AddrOf && InputExpr->getType()->isAtomicType()) {
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
                       << InputExpr->getType()
                       << Input.get()->getSourceRange());
    }
  }
  switch (Opc) {
  case UO_PreInc:
  case UO_PreDec:
  case UO_PostInc:
  case UO_PostDec:
    resultType = CheckIncrementDecrementOperand(*this, Input.get(), VK, OK,
                                                OpLoc,
                                                Opc == UO_PreInc ||
                                                Opc == UO_PostInc,
                                                Opc == UO_PreInc ||
                                                Opc == UO_PreDec);
    break;
  case UO_AddrOf:
    resultType = CheckAddressOfOperand(Input, OpLoc);
    RecordModifiableNonNullParam(*this, InputExpr);
    break;
  case UO_Deref: {
    Input = DefaultFunctionArrayLvalueConversion(Input.get());
    if (Input.isInvalid()) return ExprError();
    resultType = CheckIndirectionOperand(*this, Input.get(), VK, OpLoc);
    break;
  }
  case UO_Plus:
  case UO_Minus:
    Input = UsualUnaryConversions(Input.get());
    if (Input.isInvalid()) return ExprError();
    resultType = Input.get()->getType();
    if (resultType->isDependentType())
      break;
    if (resultType->isArithmeticType()) // C99 6.5.3.3p1
      break;
    else if (resultType->isVectorType() &&
             // The z vector extensions don't allow + or - with bool vectors.
             (!Context.getLangOpts().ZVector ||
              resultType->getAs<VectorType>()->getVectorKind() !=
              VectorType::AltiVecBool))
      break;
    else if (getLangOpts().CPlusPlus && // C++ [expr.unary.op]p6
             Opc == UO_Plus &&
             resultType->isPointerType())
      break;

    return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
      << resultType << Input.get()->getSourceRange());

  case UO_Not: // bitwise complement
    Input = UsualUnaryConversions(Input.get());
    if (Input.isInvalid())
      return ExprError();
    resultType = Input.get()->getType();
    if (resultType->isDependentType())
      break;
    // C99 6.5.3.3p1. We allow complex int and float as a GCC extension.
    if (resultType->isComplexType() || resultType->isComplexIntegerType())
      // C99 does not support '~' for complex conjugation.
      Diag(OpLoc, diag::ext_integer_complement_complex)
          << resultType << Input.get()->getSourceRange();
    else if (resultType->hasIntegerRepresentation())
      break;
    else if (resultType->isExtVectorType()) {
      if (Context.getLangOpts().OpenCL) {
        // OpenCL v1.1 s6.3.f: The bitwise operator not (~) does not operate
        // on vector float types.
        QualType T = resultType->getAs<ExtVectorType>()->getElementType();
        if (!T->isIntegerType())
          return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
                           << resultType << Input.get()->getSourceRange());
      }
      break;
    } else {
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
                       << resultType << Input.get()->getSourceRange());
    }
    break;

  case UO_LNot: // logical negation
    // Unlike +/-/~, integer promotions aren't done here (C99 6.5.3.3p5).
    Input = DefaultFunctionArrayLvalueConversion(Input.get());
    if (Input.isInvalid()) return ExprError();
    resultType = Input.get()->getType();

    // Though we still have to promote half FP to float...
    if (resultType->isHalfType() && !Context.getLangOpts().NativeHalfType) {
      Input = ImpCastExprToType(Input.get(), Context.FloatTy, CK_FloatingCast).get();
      resultType = Context.FloatTy;
    }

    if (resultType->isDependentType())
      break;
    if (resultType->isScalarType() && !isScopedEnumerationType(resultType)) {
      // C99 6.5.3.3p1: ok, fallthrough;
      if (Context.getLangOpts().CPlusPlus) {
        // C++03 [expr.unary.op]p8, C++0x [expr.unary.op]p9:
        // operand contextually converted to bool.
        Input = ImpCastExprToType(Input.get(), Context.BoolTy,
                                  ScalarTypeToBooleanCastKind(resultType));
      } else if (Context.getLangOpts().OpenCL &&
                 Context.getLangOpts().OpenCLVersion < 120) {
        // OpenCL v1.1 6.3.h: The logical operator not (!) does not
        // operate on scalar float types.
        if (!resultType->isIntegerType())
          return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
                           << resultType << Input.get()->getSourceRange());
      }
    } else if (resultType->isExtVectorType()) {
      if (Context.getLangOpts().OpenCL &&
          Context.getLangOpts().OpenCLVersion < 120) {
        // OpenCL v1.1 6.3.h: The logical operator not (!) does not
        // operate on vector float types.
        QualType T = resultType->getAs<ExtVectorType>()->getElementType();
        if (!T->isIntegerType())
          return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
                           << resultType << Input.get()->getSourceRange());
      }
      // Vector logical not returns the signed variant of the operand type.
      resultType = GetSignedVectorType(resultType);
      break;
    } else {
      return ExprError(Diag(OpLoc, diag::err_typecheck_unary_expr)
        << resultType << Input.get()->getSourceRange());
    }
    
    // LNot always has type int. C99 6.5.3.3p5.
    // In C++, it's bool. C++ 5.3.1p8
    resultType = Context.getLogicalOperationType();
    break;
  case UO_Real:
  case UO_Imag:
    resultType = CheckRealImagOperand(*this, Input, OpLoc, Opc == UO_Real);
    // _Real maps ordinary l-values into ordinary l-values. _Imag maps ordinary
    // complex l-values to ordinary l-values and all other values to r-values.
    if (Input.isInvalid()) return ExprError();
    if (Opc == UO_Real || Input.get()->getType()->isAnyComplexType()) {
      if (Input.get()->getValueKind() != VK_RValue &&
          Input.get()->getObjectKind() == OK_Ordinary)
        VK = Input.get()->getValueKind();
    } else if (!getLangOpts().CPlusPlus) {
      // In C, a volatile scalar is read by __imag. In C++, it is not.
      Input = DefaultLvalueConversion(Input.get());
    }
    break;
  case UO_Extension:
  case UO_Coawait:
    resultType = Input.get()->getType();
    VK = Input.get()->getValueKind();
    OK = Input.get()->getObjectKind();
    break;
  }
  if (resultType.isNull() || Input.isInvalid())
    return ExprError();

  // Check for array bounds violations in the operand of the UnaryOperator,
  // except for the '*' and '&' operators that have to be handled specially
  // by CheckArrayAccess (as there are special cases like &array[arraysize]
  // that are explicitly defined as valid by the standard).
  if (Opc != UO_AddrOf && Opc != UO_Deref)
    CheckArrayAccess(Input.get());

  return new (Context)
      UnaryOperator(Input.get(), Opc, resultType, VK, OK, OpLoc);
}

/// \brief Determine whether the given expression is a qualified member
/// access expression, of a form that could be turned into a pointer to member
/// with the address-of operator.
static bool isQualifiedMemberAccess(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (!DRE->getQualifier())
      return false;
    
    ValueDecl *VD = DRE->getDecl();
    if (!VD->isCXXClassMember())
      return false;
    
    if (isa<FieldDecl>(VD) || isa<IndirectFieldDecl>(VD))
      return true;
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(VD))
      return Method->isInstance();
      
    return false;
  }
  
  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(E)) {
    if (!ULE->getQualifier())
      return false;
    
    for (NamedDecl *D : ULE->decls()) {
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
        if (Method->isInstance())
          return true;
      } else {
        // Overload set does not contain methods.
        break;
      }
    }
    
    return false;
  }
  
  return false;
}

ExprResult Sema::BuildUnaryOp(Scope *S, SourceLocation OpLoc,
                              UnaryOperatorKind Opc, Expr *Input) {
  // First things first: handle placeholders so that the
  // overloaded-operator check considers the right type.
  if (const BuiltinType *pty = Input->getType()->getAsPlaceholderType()) {
    // Increment and decrement of pseudo-object references.
    if (pty->getKind() == BuiltinType::PseudoObject &&
        UnaryOperator::isIncrementDecrementOp(Opc))
      return checkPseudoObjectIncDec(S, OpLoc, Opc, Input);

    // extension is always a builtin operator.
    if (Opc == UO_Extension)
      return CreateBuiltinUnaryOp(OpLoc, Opc, Input);

    // & gets special logic for several kinds of placeholder.
    // The builtin code knows what to do.
    if (Opc == UO_AddrOf &&
        (pty->getKind() == BuiltinType::Overload ||
         pty->getKind() == BuiltinType::UnknownAny ||
         pty->getKind() == BuiltinType::BoundMember))
      return CreateBuiltinUnaryOp(OpLoc, Opc, Input);

    // Anything else needs to be handled now.
    ExprResult Result = CheckPlaceholderExpr(Input);
    if (Result.isInvalid()) return ExprError();
    Input = Result.get();
  }

  if (getLangOpts().CPlusPlus && Input->getType()->isOverloadableType() &&
      UnaryOperator::getOverloadedOperator(Opc) != OO_None &&
      !(Opc == UO_AddrOf && isQualifiedMemberAccess(Input))) {
    // Find all of the overloaded operators visible from this
    // point. We perform both an operator-name lookup from the local
    // scope and an argument-dependent lookup based on the types of
    // the arguments.
    UnresolvedSet<16> Functions;
    OverloadedOperatorKind OverOp = UnaryOperator::getOverloadedOperator(Opc);
    if (S && OverOp != OO_None)
      LookupOverloadedOperatorName(OverOp, S, Input->getType(), QualType(),
                                   Functions);

    return CreateOverloadedUnaryOp(OpLoc, Opc, Functions, Input);
  }

  return CreateBuiltinUnaryOp(OpLoc, Opc, Input);
}

// Unary Operators.  'Tok' is the token for the operator.
ExprResult Sema::ActOnUnaryOp(Scope *S, SourceLocation OpLoc,
                              tok::TokenKind Op, Expr *Input) {
  return BuildUnaryOp(S, OpLoc, ConvertTokenKindToUnaryOpcode(Op), Input);
}

/// ActOnAddrLabel - Parse the GNU address of label extension: "&&foo".
ExprResult Sema::ActOnAddrLabel(SourceLocation OpLoc, SourceLocation LabLoc,
                                LabelDecl *TheDecl) {
  TheDecl->markUsed(Context);
  // Create the AST node.  The address of a label always has type 'void*'.
  return new (Context) AddrLabelExpr(OpLoc, LabLoc, TheDecl,
                                     Context.getPointerType(Context.VoidTy));
}

/// Given the last statement in a statement-expression, check whether
/// the result is a producing expression (like a call to an
/// ns_returns_retained function) and, if so, rebuild it to hoist the
/// release out of the full-expression.  Otherwise, return null.
/// Cannot fail.
static Expr *maybeRebuildARCConsumingStmt(Stmt *Statement) {
  // Should always be wrapped with one of these.
  ExprWithCleanups *cleanups = dyn_cast<ExprWithCleanups>(Statement);
  if (!cleanups) return nullptr;

  ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(cleanups->getSubExpr());
  if (!cast || cast->getCastKind() != CK_ARCConsumeObject)
    return nullptr;

  // Splice out the cast.  This shouldn't modify any interesting
  // features of the statement.
  Expr *producer = cast->getSubExpr();
  assert(producer->getType() == cast->getType());
  assert(producer->getValueKind() == cast->getValueKind());
  cleanups->setSubExpr(producer);
  return cleanups;
}

void Sema::ActOnStartStmtExpr() {
  PushExpressionEvaluationContext(ExprEvalContexts.back().Context);
}

void Sema::ActOnStmtExprError() {
  // Note that function is also called by TreeTransform when leaving a
  // StmtExpr scope without rebuilding anything.

  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

ExprResult
Sema::ActOnStmtExpr(SourceLocation LPLoc, Stmt *SubStmt,
                    SourceLocation RPLoc) { // "({..})"
  assert(SubStmt && isa<CompoundStmt>(SubStmt) && "Invalid action invocation!");
  CompoundStmt *Compound = cast<CompoundStmt>(SubStmt);

  if (hasAnyUnrecoverableErrorsInThisFunction())
    DiscardCleanupsInEvaluationContext();
  assert(!ExprNeedsCleanups && "cleanups within StmtExpr not correctly bound!");
  PopExpressionEvaluationContext();

  // FIXME: there are a variety of strange constraints to enforce here, for
  // example, it is not possible to goto into a stmt expression apparently.
  // More semantic analysis is needed.

  // If there are sub-stmts in the compound stmt, take the type of the last one
  // as the type of the stmtexpr.
  QualType Ty = Context.VoidTy;
  bool StmtExprMayBindToTemp = false;
  if (!Compound->body_empty()) {
    Stmt *LastStmt = Compound->body_back();
    LabelStmt *LastLabelStmt = nullptr;
    // If LastStmt is a label, skip down through into the body.
    while (LabelStmt *Label = dyn_cast<LabelStmt>(LastStmt)) {
      LastLabelStmt = Label;
      LastStmt = Label->getSubStmt();
    }

    if (Expr *LastE = dyn_cast<Expr>(LastStmt)) {
      // Do function/array conversion on the last expression, but not
      // lvalue-to-rvalue.  However, initialize an unqualified type.
      ExprResult LastExpr = DefaultFunctionArrayConversion(LastE);
      if (LastExpr.isInvalid())
        return ExprError();
      Ty = LastExpr.get()->getType().getUnqualifiedType();

      if (!Ty->isDependentType() && !LastExpr.get()->isTypeDependent()) {
        // In ARC, if the final expression ends in a consume, splice
        // the consume out and bind it later.  In the alternate case
        // (when dealing with a retainable type), the result
        // initialization will create a produce.  In both cases the
        // result will be +1, and we'll need to balance that out with
        // a bind.
        if (Expr *rebuiltLastStmt
              = maybeRebuildARCConsumingStmt(LastExpr.get())) {
          LastExpr = rebuiltLastStmt;
        } else {
          LastExpr = PerformCopyInitialization(
                            InitializedEntity::InitializeResult(LPLoc, 
                                                                Ty,
                                                                false),
                                                   SourceLocation(),
                                               LastExpr);
        }

        if (LastExpr.isInvalid())
          return ExprError();
        if (LastExpr.get() != nullptr) {
          if (!LastLabelStmt)
            Compound->setLastStmt(LastExpr.get());
          else
            LastLabelStmt->setSubStmt(LastExpr.get());
          StmtExprMayBindToTemp = true;
        }
      }
    }
  }

  // FIXME: Check that expression type is complete/non-abstract; statement
  // expressions are not lvalues.
  Expr *ResStmtExpr = new (Context) StmtExpr(Compound, Ty, LPLoc, RPLoc);
  if (StmtExprMayBindToTemp)
    return MaybeBindToTemporary(ResStmtExpr);
  return ResStmtExpr;
}

ExprResult Sema::BuildBuiltinOffsetOf(SourceLocation BuiltinLoc,
                                      TypeSourceInfo *TInfo,
                                      ArrayRef<OffsetOfComponent> Components,
                                      SourceLocation RParenLoc) {
  QualType ArgTy = TInfo->getType();
  bool Dependent = ArgTy->isDependentType();
  SourceRange TypeRange = TInfo->getTypeLoc().getLocalSourceRange();
  
  // We must have at least one component that refers to the type, and the first
  // one is known to be a field designator.  Verify that the ArgTy represents
  // a struct/union/class.
  if (!Dependent && !ArgTy->isRecordType())
    return ExprError(Diag(BuiltinLoc, diag::err_offsetof_record_type) 
                       << ArgTy << TypeRange);
  
  // Type must be complete per C99 7.17p3 because a declaring a variable
  // with an incomplete type would be ill-formed.
  if (!Dependent 
      && RequireCompleteType(BuiltinLoc, ArgTy,
                             diag::err_offsetof_incomplete_type, TypeRange))
    return ExprError();
  
  // offsetof with non-identifier designators (e.g. "offsetof(x, a.b[c])") are a
  // GCC extension, diagnose them.
  // FIXME: This diagnostic isn't actually visible because the location is in
  // a system header!
  if (Components.size() != 1)
    Diag(BuiltinLoc, diag::ext_offsetof_extended_field_designator)
      << SourceRange(Components[1].LocStart, Components.back().LocEnd);
  
  bool DidWarnAboutNonPOD = false;
  QualType CurrentType = ArgTy;
  SmallVector<OffsetOfNode, 4> Comps;
  SmallVector<Expr*, 4> Exprs;
  for (const OffsetOfComponent &OC : Components) {
    if (OC.isBrackets) {
      // Offset of an array sub-field.  TODO: Should we allow vector elements?
      if (!CurrentType->isDependentType()) {
        const ArrayType *AT = Context.getAsArrayType(CurrentType);
        if(!AT)
          return ExprError(Diag(OC.LocEnd, diag::err_offsetof_array_type)
                           << CurrentType);
        CurrentType = AT->getElementType();
      } else
        CurrentType = Context.DependentTy;
      
      ExprResult IdxRval = DefaultLvalueConversion(static_cast<Expr*>(OC.U.E));
      if (IdxRval.isInvalid())
        return ExprError();
      Expr *Idx = IdxRval.get();

      // The expression must be an integral expression.
      // FIXME: An integral constant expression?
      if (!Idx->isTypeDependent() && !Idx->isValueDependent() &&
          !Idx->getType()->isIntegerType())
        return ExprError(Diag(Idx->getLocStart(),
                              diag::err_typecheck_subscript_not_integer)
                         << Idx->getSourceRange());

      // Record this array index.
      Comps.push_back(OffsetOfNode(OC.LocStart, Exprs.size(), OC.LocEnd));
      Exprs.push_back(Idx);
      continue;
    }
    
    // Offset of a field.
    if (CurrentType->isDependentType()) {
      // We have the offset of a field, but we can't look into the dependent
      // type. Just record the identifier of the field.
      Comps.push_back(OffsetOfNode(OC.LocStart, OC.U.IdentInfo, OC.LocEnd));
      CurrentType = Context.DependentTy;
      continue;
    }
    
    // We need to have a complete type to look into.
    if (RequireCompleteType(OC.LocStart, CurrentType,
                            diag::err_offsetof_incomplete_type))
      return ExprError();
    
    // Look for the designated field.
    const RecordType *RC = CurrentType->getAs<RecordType>();
    if (!RC) 
      return ExprError(Diag(OC.LocEnd, diag::err_offsetof_record_type)
                       << CurrentType);
    RecordDecl *RD = RC->getDecl();
    
    // C++ [lib.support.types]p5:
    //   The macro offsetof accepts a restricted set of type arguments in this
    //   International Standard. type shall be a POD structure or a POD union
    //   (clause 9).
    // C++11 [support.types]p4:
    //   If type is not a standard-layout class (Clause 9), the results are
    //   undefined.
    if (CXXRecordDecl *CRD = dyn_cast<CXXRecordDecl>(RD)) {
      bool IsSafe = LangOpts.CPlusPlus11? CRD->isStandardLayout() : CRD->isPOD();
      unsigned DiagID =
        LangOpts.CPlusPlus11? diag::ext_offsetof_non_standardlayout_type
                            : diag::ext_offsetof_non_pod_type;

      if (!IsSafe && !DidWarnAboutNonPOD &&
          DiagRuntimeBehavior(BuiltinLoc, nullptr,
                              PDiag(DiagID)
                              << SourceRange(Components[0].LocStart, OC.LocEnd)
                              << CurrentType))
        DidWarnAboutNonPOD = true;
    }
    
    // Look for the field.
    LookupResult R(*this, OC.U.IdentInfo, OC.LocStart, LookupMemberName);
    LookupQualifiedName(R, RD);
    FieldDecl *MemberDecl = R.getAsSingle<FieldDecl>();
    IndirectFieldDecl *IndirectMemberDecl = nullptr;
    if (!MemberDecl) {
      if ((IndirectMemberDecl = R.getAsSingle<IndirectFieldDecl>()))
        MemberDecl = IndirectMemberDecl->getAnonField();
    }

    if (!MemberDecl)
      return ExprError(Diag(BuiltinLoc, diag::err_no_member)
                       << OC.U.IdentInfo << RD << SourceRange(OC.LocStart, 
                                                              OC.LocEnd));
    
    // C99 7.17p3:
    //   (If the specified member is a bit-field, the behavior is undefined.)
    //
    // We diagnose this as an error.
    if (MemberDecl->isBitField()) {
      Diag(OC.LocEnd, diag::err_offsetof_bitfield)
        << MemberDecl->getDeclName()
        << SourceRange(BuiltinLoc, RParenLoc);
      Diag(MemberDecl->getLocation(), diag::note_bitfield_decl);
      return ExprError();
    }

    RecordDecl *Parent = MemberDecl->getParent();
    if (IndirectMemberDecl)
      Parent = cast<RecordDecl>(IndirectMemberDecl->getDeclContext());

    // If the member was found in a base class, introduce OffsetOfNodes for
    // the base class indirections.
    CXXBasePaths Paths;
    if (IsDerivedFrom(OC.LocStart, CurrentType, Context.getTypeDeclType(Parent),
                      Paths)) {
      if (Paths.getDetectedVirtual()) {
        Diag(OC.LocEnd, diag::err_offsetof_field_of_virtual_base)
          << MemberDecl->getDeclName()
          << SourceRange(BuiltinLoc, RParenLoc);
        return ExprError();
      }

      CXXBasePath &Path = Paths.front();
      for (const CXXBasePathElement &B : Path)
        Comps.push_back(OffsetOfNode(B.Base));
    }

    if (IndirectMemberDecl) {
      for (auto *FI : IndirectMemberDecl->chain()) {
        assert(isa<FieldDecl>(FI));
        Comps.push_back(OffsetOfNode(OC.LocStart,
                                     cast<FieldDecl>(FI), OC.LocEnd));
      }
    } else
      Comps.push_back(OffsetOfNode(OC.LocStart, MemberDecl, OC.LocEnd));

    CurrentType = MemberDecl->getType().getNonReferenceType(); 
  }
  
  return OffsetOfExpr::Create(Context, Context.getSizeType(), BuiltinLoc, TInfo,
                              Comps, Exprs, RParenLoc);
}

ExprResult Sema::ActOnBuiltinOffsetOf(Scope *S,
                                      SourceLocation BuiltinLoc,
                                      SourceLocation TypeLoc,
                                      ParsedType ParsedArgTy,
                                      ArrayRef<OffsetOfComponent> Components,
                                      SourceLocation RParenLoc) {
  
  TypeSourceInfo *ArgTInfo;
  QualType ArgTy = GetTypeFromParser(ParsedArgTy, &ArgTInfo);
  if (ArgTy.isNull())
    return ExprError();

  if (!ArgTInfo)
    ArgTInfo = Context.getTrivialTypeSourceInfo(ArgTy, TypeLoc);

  return BuildBuiltinOffsetOf(BuiltinLoc, ArgTInfo, Components, RParenLoc);
}


ExprResult Sema::ActOnChooseExpr(SourceLocation BuiltinLoc,
                                 Expr *CondExpr,
                                 Expr *LHSExpr, Expr *RHSExpr,
                                 SourceLocation RPLoc) {
  assert((CondExpr && LHSExpr && RHSExpr) && "Missing type argument(s)");

  ExprValueKind VK = VK_RValue;
  ExprObjectKind OK = OK_Ordinary;
  QualType resType;
  bool ValueDependent = false;
  bool CondIsTrue = false;
  if (CondExpr->isTypeDependent() || CondExpr->isValueDependent()) {
    resType = Context.DependentTy;
    ValueDependent = true;
  } else {
    // The conditional expression is required to be a constant expression.
    llvm::APSInt condEval(32);
    ExprResult CondICE
      = VerifyIntegerConstantExpression(CondExpr, &condEval,
          diag::err_typecheck_choose_expr_requires_constant, false);
    if (CondICE.isInvalid())
      return ExprError();
    CondExpr = CondICE.get();
    CondIsTrue = condEval.getZExtValue();

    // If the condition is > zero, then the AST type is the same as the LSHExpr.
    Expr *ActiveExpr = CondIsTrue ? LHSExpr : RHSExpr;

    resType = ActiveExpr->getType();
    ValueDependent = ActiveExpr->isValueDependent();
    VK = ActiveExpr->getValueKind();
    OK = ActiveExpr->getObjectKind();
  }

  return new (Context)
      ChooseExpr(BuiltinLoc, CondExpr, LHSExpr, RHSExpr, resType, VK, OK, RPLoc,
                 CondIsTrue, resType->isDependentType(), ValueDependent);
}

//===----------------------------------------------------------------------===//
// Clang Extensions.
//===----------------------------------------------------------------------===//

/// ActOnBlockStart - This callback is invoked when a block literal is started.
void Sema::ActOnBlockStart(SourceLocation CaretLoc, Scope *CurScope) {
  BlockDecl *Block = BlockDecl::Create(Context, CurContext, CaretLoc);

  if (LangOpts.CPlusPlus) {
    Decl *ManglingContextDecl;
    if (MangleNumberingContext *MCtx =
            getCurrentMangleNumberContext(Block->getDeclContext(),
                                          ManglingContextDecl)) {
      unsigned ManglingNumber = MCtx->getManglingNumber(Block);
      Block->setBlockMangling(ManglingNumber, ManglingContextDecl);
    }
  }

  PushBlockScope(CurScope, Block);
  CurContext->addDecl(Block);
  if (CurScope)
    PushDeclContext(CurScope, Block);
  else
    CurContext = Block;

  getCurBlock()->HasImplicitReturnType = true;

  // Enter a new evaluation context to insulate the block from any
  // cleanups from the enclosing full-expression.
  PushExpressionEvaluationContext(PotentiallyEvaluated);  
}

void Sema::ActOnBlockArguments(SourceLocation CaretLoc, Declarator &ParamInfo,
                               Scope *CurScope) {
  assert(ParamInfo.getIdentifier() == nullptr &&
         "block-id should have no identifier!");
  assert(ParamInfo.getContext() == Declarator::BlockLiteralContext);
  BlockScopeInfo *CurBlock = getCurBlock();

  TypeSourceInfo *Sig = GetTypeForDeclarator(ParamInfo, CurScope);
  QualType T = Sig->getType();

  // FIXME: We should allow unexpanded parameter packs here, but that would,
  // in turn, make the block expression contain unexpanded parameter packs.
  if (DiagnoseUnexpandedParameterPack(CaretLoc, Sig, UPPC_Block)) {
    // Drop the parameters.
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.HasTrailingReturn = false;
    EPI.TypeQuals |= DeclSpec::TQ_const;
    T = Context.getFunctionType(Context.DependentTy, None, EPI);
    Sig = Context.getTrivialTypeSourceInfo(T);
  }
  
  // GetTypeForDeclarator always produces a function type for a block
  // literal signature.  Furthermore, it is always a FunctionProtoType
  // unless the function was written with a typedef.
  assert(T->isFunctionType() &&
         "GetTypeForDeclarator made a non-function block signature");

  // Look for an explicit signature in that function type.
  FunctionProtoTypeLoc ExplicitSignature;

  TypeLoc tmp = Sig->getTypeLoc().IgnoreParens();
  if ((ExplicitSignature = tmp.getAs<FunctionProtoTypeLoc>())) {

    // Check whether that explicit signature was synthesized by
    // GetTypeForDeclarator.  If so, don't save that as part of the
    // written signature.
    if (ExplicitSignature.getLocalRangeBegin() ==
        ExplicitSignature.getLocalRangeEnd()) {
      // This would be much cheaper if we stored TypeLocs instead of
      // TypeSourceInfos.
      TypeLoc Result = ExplicitSignature.getReturnLoc();
      unsigned Size = Result.getFullDataSize();
      Sig = Context.CreateTypeSourceInfo(Result.getType(), Size);
      Sig->getTypeLoc().initializeFullCopy(Result, Size);

      ExplicitSignature = FunctionProtoTypeLoc();
    }
  }

  CurBlock->TheDecl->setSignatureAsWritten(Sig);
  CurBlock->FunctionType = T;

  const FunctionType *Fn = T->getAs<FunctionType>();
  QualType RetTy = Fn->getReturnType();
  bool isVariadic =
    (isa<FunctionProtoType>(Fn) && cast<FunctionProtoType>(Fn)->isVariadic());

  CurBlock->TheDecl->setIsVariadic(isVariadic);

  // Context.DependentTy is used as a placeholder for a missing block
  // return type.  TODO:  what should we do with declarators like:
  //   ^ * { ... }
  // If the answer is "apply template argument deduction"....
  if (RetTy != Context.DependentTy) {
    CurBlock->ReturnType = RetTy;
    CurBlock->TheDecl->setBlockMissingReturnType(false);
    CurBlock->HasImplicitReturnType = false;
  }

  // Push block parameters from the declarator if we had them.
  SmallVector<ParmVarDecl*, 8> Params;
  if (ExplicitSignature) {
    for (unsigned I = 0, E = ExplicitSignature.getNumParams(); I != E; ++I) {
      ParmVarDecl *Param = ExplicitSignature.getParam(I);
      if (Param->getIdentifier() == nullptr &&
          !Param->isImplicit() &&
          !Param->isInvalidDecl() &&
          !getLangOpts().CPlusPlus)
        Diag(Param->getLocation(), diag::err_parameter_name_omitted);
      Params.push_back(Param);
    }

  // Fake up parameter variables if we have a typedef, like
  //   ^ fntype { ... }
  } else if (const FunctionProtoType *Fn = T->getAs<FunctionProtoType>()) {
    for (const auto &I : Fn->param_types()) {
      ParmVarDecl *Param = BuildParmVarDeclForTypedef(
          CurBlock->TheDecl, ParamInfo.getLocStart(), I);
      Params.push_back(Param);
    }
  }

  // Set the parameters on the block decl.
  if (!Params.empty()) {
    CurBlock->TheDecl->setParams(Params);
    CheckParmsForFunctionDef(CurBlock->TheDecl->param_begin(),
                             CurBlock->TheDecl->param_end(),
                             /*CheckParameterNames=*/false);
  }
  
  // Finally we can process decl attributes.
  ProcessDeclAttributes(CurScope, CurBlock->TheDecl, ParamInfo);

  // Put the parameter variables in scope.
  for (auto AI : CurBlock->TheDecl->params()) {
    AI->setOwningFunction(CurBlock->TheDecl);

    // If this has an identifier, add it to the scope stack.
    if (AI->getIdentifier()) {
      CheckShadow(CurBlock->TheScope, AI);

      PushOnScopeChains(AI, CurBlock->TheScope);
    }
  }
}

/// ActOnBlockError - If there is an error parsing a block, this callback
/// is invoked to pop the information about the block from the action impl.
void Sema::ActOnBlockError(SourceLocation CaretLoc, Scope *CurScope) {
  // Leave the expression-evaluation context.
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  // Pop off CurBlock, handle nested blocks.
  PopDeclContext();
  PopFunctionScopeInfo();
}

/// ActOnBlockStmtExpr - This is called when the body of a block statement
/// literal was successfully completed.  ^(int x){...}
ExprResult Sema::ActOnBlockStmtExpr(SourceLocation CaretLoc,
                                    Stmt *Body, Scope *CurScope) {
  // If blocks are disabled, emit an error.
  if (!LangOpts.Blocks)
    Diag(CaretLoc, diag::err_blocks_disable);

  // Leave the expression-evaluation context.
  if (hasAnyUnrecoverableErrorsInThisFunction())
    DiscardCleanupsInEvaluationContext();
  assert(!ExprNeedsCleanups && "cleanups within block not correctly bound!");
  PopExpressionEvaluationContext();

  BlockScopeInfo *BSI = cast<BlockScopeInfo>(FunctionScopes.back());

  if (BSI->HasImplicitReturnType)
    deduceClosureReturnType(*BSI);

  PopDeclContext();

  QualType RetTy = Context.VoidTy;
  if (!BSI->ReturnType.isNull())
    RetTy = BSI->ReturnType;

  bool NoReturn = BSI->TheDecl->hasAttr<NoReturnAttr>();
  QualType BlockTy;

  // Set the captured variables on the block.
  // FIXME: Share capture structure between BlockDecl and CapturingScopeInfo!
  SmallVector<BlockDecl::Capture, 4> Captures;
  for (CapturingScopeInfo::Capture &Cap : BSI->Captures) {
    if (Cap.isThisCapture())
      continue;
    BlockDecl::Capture NewCap(Cap.getVariable(), Cap.isBlockCapture(),
                              Cap.isNested(), Cap.getInitExpr());
    Captures.push_back(NewCap);
  }
  BSI->TheDecl->setCaptures(Context, Captures, BSI->CXXThisCaptureIndex != 0);

  // If the user wrote a function type in some form, try to use that.
  if (!BSI->FunctionType.isNull()) {
    const FunctionType *FTy = BSI->FunctionType->getAs<FunctionType>();

    FunctionType::ExtInfo Ext = FTy->getExtInfo();
    if (NoReturn && !Ext.getNoReturn()) Ext = Ext.withNoReturn(true);
    
    // Turn protoless block types into nullary block types.
    if (isa<FunctionNoProtoType>(FTy)) {
      FunctionProtoType::ExtProtoInfo EPI;
      EPI.ExtInfo = Ext;
      BlockTy = Context.getFunctionType(RetTy, None, EPI);

    // Otherwise, if we don't need to change anything about the function type,
    // preserve its sugar structure.
    } else if (FTy->getReturnType() == RetTy &&
               (!NoReturn || FTy->getNoReturnAttr())) {
      BlockTy = BSI->FunctionType;

    // Otherwise, make the minimal modifications to the function type.
    } else {
      const FunctionProtoType *FPT = cast<FunctionProtoType>(FTy);
      FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
      EPI.TypeQuals = 0; // FIXME: silently?
      EPI.ExtInfo = Ext;
      BlockTy = Context.getFunctionType(RetTy, FPT->getParamTypes(), EPI);
    }

  // If we don't have a function type, just build one from nothing.
  } else {
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.ExtInfo = FunctionType::ExtInfo().withNoReturn(NoReturn);
    BlockTy = Context.getFunctionType(RetTy, None, EPI);
  }

  DiagnoseUnusedParameters(BSI->TheDecl->param_begin(),
                           BSI->TheDecl->param_end());
  BlockTy = Context.getBlockPointerType(BlockTy);

  // If needed, diagnose invalid gotos and switches in the block.
  if (getCurFunction()->NeedsScopeChecking() &&
      !PP.isCodeCompletionEnabled())
    DiagnoseInvalidJumps(cast<CompoundStmt>(Body));

  BSI->TheDecl->setBody(cast<CompoundStmt>(Body));

  // Try to apply the named return value optimization. We have to check again
  // if we can do this, though, because blocks keep return statements around
  // to deduce an implicit return type.
  if (getLangOpts().CPlusPlus && RetTy->isRecordType() &&
      !BSI->TheDecl->isDependentContext())
    computeNRVO(Body, BSI);
  
  BlockExpr *Result = new (Context) BlockExpr(BSI->TheDecl, BlockTy);
  AnalysisBasedWarnings::Policy WP = AnalysisWarnings.getDefaultPolicy();
  PopFunctionScopeInfo(&WP, Result->getBlockDecl(), Result);

  // If the block isn't obviously global, i.e. it captures anything at
  // all, then we need to do a few things in the surrounding context:
  if (Result->getBlockDecl()->hasCaptures()) {
    // First, this expression has a new cleanup object.
    ExprCleanupObjects.push_back(Result->getBlockDecl());
    ExprNeedsCleanups = true;

    // It also gets a branch-protected scope if any of the captured
    // variables needs destruction.
    for (const auto &CI : Result->getBlockDecl()->captures()) {
      const VarDecl *var = CI.getVariable();
      if (var->getType().isDestructedType() != QualType::DK_none) {
        getCurFunction()->setHasBranchProtectedScope();
        break;
      }
    }
  }

  return Result;
}

ExprResult Sema::ActOnVAArg(SourceLocation BuiltinLoc, Expr *E, ParsedType Ty,
                            SourceLocation RPLoc) {
  TypeSourceInfo *TInfo;
  GetTypeFromParser(Ty, &TInfo);
  return BuildVAArgExpr(BuiltinLoc, E, TInfo, RPLoc);
}

ExprResult Sema::BuildVAArgExpr(SourceLocation BuiltinLoc,
                                Expr *E, TypeSourceInfo *TInfo,
                                SourceLocation RPLoc) {
  Expr *OrigExpr = E;
  bool IsMS = false;

  // CUDA device code does not support varargs.
  if (getLangOpts().CUDA && getLangOpts().CUDAIsDevice) {
    if (const FunctionDecl *F = dyn_cast<FunctionDecl>(CurContext)) {
      CUDAFunctionTarget T = IdentifyCUDATarget(F);
      if (T == CFT_Global || T == CFT_Device || T == CFT_HostDevice)
        return ExprError(Diag(E->getLocStart(), diag::err_va_arg_in_device));
    }
  }

  // It might be a __builtin_ms_va_list. (But don't ever mark a va_arg()
  // as Microsoft ABI on an actual Microsoft platform, where
  // __builtin_ms_va_list and __builtin_va_list are the same.)
  if (!E->isTypeDependent() && Context.getTargetInfo().hasBuiltinMSVaList() &&
      Context.getTargetInfo().getBuiltinVaListKind() != TargetInfo::CharPtrBuiltinVaList) {
    QualType MSVaListType = Context.getBuiltinMSVaListType();
    if (Context.hasSameType(MSVaListType, E->getType())) {
      if (CheckForModifiableLvalue(E, BuiltinLoc, *this))
        return ExprError();
      IsMS = true;
    }
  }

  // Get the va_list type
  QualType VaListType = Context.getBuiltinVaListType();
  if (!IsMS) {
    if (VaListType->isArrayType()) {
      // Deal with implicit array decay; for example, on x86-64,
      // va_list is an array, but it's supposed to decay to
      // a pointer for va_arg.
      VaListType = Context.getArrayDecayedType(VaListType);
      // Make sure the input expression also decays appropriately.
      ExprResult Result = UsualUnaryConversions(E);
      if (Result.isInvalid())
        return ExprError();
      E = Result.get();
    } else if (VaListType->isRecordType() && getLangOpts().CPlusPlus) {
      // If va_list is a record type and we are compiling in C++ mode,
      // check the argument using reference binding.
      InitializedEntity Entity = InitializedEntity::InitializeParameter(
          Context, Context.getLValueReferenceType(VaListType), false);
      ExprResult Init = PerformCopyInitialization(Entity, SourceLocation(), E);
      if (Init.isInvalid())
        return ExprError();
      E = Init.getAs<Expr>();
    } else {
      // Otherwise, the va_list argument must be an l-value because
      // it is modified by va_arg.
      if (!E->isTypeDependent() &&
          CheckForModifiableLvalue(E, BuiltinLoc, *this))
        return ExprError();
    }
  }

  if (!IsMS && !E->isTypeDependent() &&
      !Context.hasSameType(VaListType, E->getType()))
    return ExprError(Diag(E->getLocStart(),
                         diag::err_first_argument_to_va_arg_not_of_type_va_list)
      << OrigExpr->getType() << E->getSourceRange());

  if (!TInfo->getType()->isDependentType()) {
    if (RequireCompleteType(TInfo->getTypeLoc().getBeginLoc(), TInfo->getType(),
                            diag::err_second_parameter_to_va_arg_incomplete,
                            TInfo->getTypeLoc()))
      return ExprError();

    if (RequireNonAbstractType(TInfo->getTypeLoc().getBeginLoc(),
                               TInfo->getType(),
                               diag::err_second_parameter_to_va_arg_abstract,
                               TInfo->getTypeLoc()))
      return ExprError();

    if (!TInfo->getType().isPODType(Context)) {
      Diag(TInfo->getTypeLoc().getBeginLoc(),
           TInfo->getType()->isObjCLifetimeType()
             ? diag::warn_second_parameter_to_va_arg_ownership_qualified
             : diag::warn_second_parameter_to_va_arg_not_pod)
        << TInfo->getType()
        << TInfo->getTypeLoc().getSourceRange();
    }

    // Check for va_arg where arguments of the given type will be promoted
    // (i.e. this va_arg is guaranteed to have undefined behavior).
    QualType PromoteType;
    if (TInfo->getType()->isPromotableIntegerType()) {
      PromoteType = Context.getPromotedIntegerType(TInfo->getType());
      if (Context.typesAreCompatible(PromoteType, TInfo->getType()))
        PromoteType = QualType();
    }
    if (TInfo->getType()->isSpecificBuiltinType(BuiltinType::Float))
      PromoteType = Context.DoubleTy;
    if (!PromoteType.isNull())
      DiagRuntimeBehavior(TInfo->getTypeLoc().getBeginLoc(), E,
                  PDiag(diag::warn_second_parameter_to_va_arg_never_compatible)
                          << TInfo->getType()
                          << PromoteType
                          << TInfo->getTypeLoc().getSourceRange());
  }

  QualType T = TInfo->getType().getNonLValueExprType(Context);
  return new (Context) VAArgExpr(BuiltinLoc, E, TInfo, RPLoc, T, IsMS);
}

ExprResult Sema::ActOnGNUNullExpr(SourceLocation TokenLoc) {
  // The type of __null will be int or long, depending on the size of
  // pointers on the target.
  QualType Ty;
  unsigned pw = Context.getTargetInfo().getPointerWidth(0);
  if (pw == Context.getTargetInfo().getIntWidth())
    Ty = Context.IntTy;
  else if (pw == Context.getTargetInfo().getLongWidth())
    Ty = Context.LongTy;
  else if (pw == Context.getTargetInfo().getLongLongWidth())
    Ty = Context.LongLongTy;
  else {
    llvm_unreachable("I don't know size of pointer!");
  }

  return new (Context) GNUNullExpr(Ty, TokenLoc);
}

bool Sema::ConversionToObjCStringLiteralCheck(QualType DstType, Expr *&Exp,
                                              bool Diagnose) {
  if (!getLangOpts().ObjC1)
    return false;

  const ObjCObjectPointerType *PT = DstType->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  if (!PT->isObjCIdType()) {
    // Check if the destination is the 'NSString' interface.
    const ObjCInterfaceDecl *ID = PT->getInterfaceDecl();
    if (!ID || !ID->getIdentifier()->isStr("NSString"))
      return false;
  }
  
  // Ignore any parens, implicit casts (should only be
  // array-to-pointer decays), and not-so-opaque values.  The last is
  // important for making this trigger for property assignments.
  Expr *SrcExpr = Exp->IgnoreParenImpCasts();
  if (OpaqueValueExpr *OV = dyn_cast<OpaqueValueExpr>(SrcExpr))
    if (OV->getSourceExpr())
      SrcExpr = OV->getSourceExpr()->IgnoreParenImpCasts();

  StringLiteral *SL = dyn_cast<StringLiteral>(SrcExpr);
  if (!SL || !SL->isAscii())
    return false;
  if (Diagnose)
    Diag(SL->getLocStart(), diag::err_missing_atsign_prefix)
      << FixItHint::CreateInsertion(SL->getLocStart(), "@");
  Exp = BuildObjCStringLiteral(SL->getLocStart(), SL).get();
  return true;
}

static bool maybeDiagnoseAssignmentToFunction(Sema &S, QualType DstType,
                                              const Expr *SrcExpr) {
  if (!DstType->isFunctionPointerType() ||
      !SrcExpr->getType()->isFunctionType())
    return false;

  auto *DRE = dyn_cast<DeclRefExpr>(SrcExpr->IgnoreParenImpCasts());
  if (!DRE)
    return false;

  auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
  if (!FD)
    return false;

  return !S.checkAddressOfFunctionIsAvailable(FD,
                                              /*Complain=*/true,
                                              SrcExpr->getLocStart());
}

bool Sema::DiagnoseAssignmentResult(AssignConvertType ConvTy,
                                    SourceLocation Loc,
                                    QualType DstType, QualType SrcType,
                                    Expr *SrcExpr, AssignmentAction Action,
                                    bool *Complained) {
  if (Complained)
    *Complained = false;

  // Decode the result (notice that AST's are still created for extensions).
  bool CheckInferredResultType = false;
  bool isInvalid = false;
  unsigned DiagKind = 0;
  FixItHint Hint;
  ConversionFixItGenerator ConvHints;
  bool MayHaveConvFixit = false;
  bool MayHaveFunctionDiff = false;
  const ObjCInterfaceDecl *IFace = nullptr;
  const ObjCProtocolDecl *PDecl = nullptr;

  switch (ConvTy) {
  case Compatible:
      DiagnoseAssignmentEnum(DstType, SrcType, SrcExpr);
      return false;

  case PointerToInt:
    DiagKind = diag::ext_typecheck_convert_pointer_int;
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case IntToPointer:
    DiagKind = diag::ext_typecheck_convert_int_pointer;
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    break;
  case IncompatiblePointer:
      DiagKind =
        (Action == AA_Passing_CFAudited ?
          diag::err_arc_typecheck_convert_incompatible_pointer :
          diag::ext_typecheck_convert_incompatible_pointer);
    CheckInferredResultType = DstType->isObjCObjectPointerType() &&
      SrcType->isObjCObjectPointerType();
    if (Hint.isNull() && !CheckInferredResultType) {
      ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    }
    else if (CheckInferredResultType) {
      SrcType = SrcType.getUnqualifiedType();
      DstType = DstType.getUnqualifiedType();
    }
    MayHaveConvFixit = true;
    break;
  case IncompatiblePointerSign:
    DiagKind = diag::ext_typecheck_convert_incompatible_pointer_sign;
    break;
  case FunctionVoidPointer:
    DiagKind = diag::ext_typecheck_convert_pointer_void_func;
    break;
  case IncompatiblePointerDiscardsQualifiers: {
    // Perform array-to-pointer decay if necessary.
    if (SrcType->isArrayType()) SrcType = Context.getArrayDecayedType(SrcType);

    Qualifiers lhq = SrcType->getPointeeType().getQualifiers();
    Qualifiers rhq = DstType->getPointeeType().getQualifiers();
    if (lhq.getAddressSpace() != rhq.getAddressSpace()) {
      DiagKind = diag::err_typecheck_incompatible_address_space;
      break;


    } else if (lhq.getObjCLifetime() != rhq.getObjCLifetime()) {
      DiagKind = diag::err_typecheck_incompatible_ownership;
      break;
    }

    llvm_unreachable("unknown error case for discarding qualifiers!");
    // fallthrough
  }
  case CompatiblePointerDiscardsQualifiers:
    // If the qualifiers lost were because we were applying the
    // (deprecated) C++ conversion from a string literal to a char*
    // (or wchar_t*), then there was no error (C++ 4.2p2).  FIXME:
    // Ideally, this check would be performed in
    // checkPointerTypesForAssignment. However, that would require a
    // bit of refactoring (so that the second argument is an
    // expression, rather than a type), which should be done as part
    // of a larger effort to fix checkPointerTypesForAssignment for
    // C++ semantics.
    if (getLangOpts().CPlusPlus &&
        IsStringLiteralToNonConstPointerConversion(SrcExpr, DstType))
      return false;
    DiagKind = diag::ext_typecheck_convert_discards_qualifiers;
    break;
  case IncompatibleNestedPointerQualifiers:
    DiagKind = diag::ext_nested_pointer_qualifier_mismatch;
    break;
  case IntToBlockPointer:
    DiagKind = diag::err_int_to_block_pointer;
    break;
  case IncompatibleBlockPointer:
    DiagKind = diag::err_typecheck_convert_incompatible_block_pointer;
    break;
  case IncompatibleObjCQualifiedId: {
    if (SrcType->isObjCQualifiedIdType()) {
      const ObjCObjectPointerType *srcOPT =
                SrcType->getAs<ObjCObjectPointerType>();
      for (auto *srcProto : srcOPT->quals()) {
        PDecl = srcProto;
        break;
      }
      if (const ObjCInterfaceType *IFaceT =
            DstType->getAs<ObjCObjectPointerType>()->getInterfaceType())
        IFace = IFaceT->getDecl();
    }
    else if (DstType->isObjCQualifiedIdType()) {
      const ObjCObjectPointerType *dstOPT =
        DstType->getAs<ObjCObjectPointerType>();
      for (auto *dstProto : dstOPT->quals()) {
        PDecl = dstProto;
        break;
      }
      if (const ObjCInterfaceType *IFaceT =
            SrcType->getAs<ObjCObjectPointerType>()->getInterfaceType())
        IFace = IFaceT->getDecl();
    }
    DiagKind = diag::warn_incompatible_qualified_id;
    break;
  }
  case IncompatibleVectors:
    DiagKind = diag::warn_incompatible_vectors;
    break;
  case IncompatibleObjCWeakRef:
    DiagKind = diag::err_arc_weak_unavailable_assign;
    break;
  case Incompatible:
    if (maybeDiagnoseAssignmentToFunction(*this, DstType, SrcExpr)) {
      if (Complained)
        *Complained = true;
      return true;
    }

    DiagKind = diag::err_typecheck_convert_incompatible;
    ConvHints.tryToFixConversion(SrcExpr, SrcType, DstType, *this);
    MayHaveConvFixit = true;
    isInvalid = true;
    MayHaveFunctionDiff = true;
    break;
  }

  QualType FirstType, SecondType;
  switch (Action) {
  case AA_Assigning:
  case AA_Initializing:
    // The destination type comes first.
    FirstType = DstType;
    SecondType = SrcType;
    break;

  case AA_Returning:
  case AA_Passing:
  case AA_Passing_CFAudited:
  case AA_Converting:
  case AA_Sending:
  case AA_Casting:
    // The source type comes first.
    FirstType = SrcType;
    SecondType = DstType;
    break;
  }

  PartialDiagnostic FDiag = PDiag(DiagKind);
  if (Action == AA_Passing_CFAudited)
    FDiag << FirstType << SecondType << AA_Passing << SrcExpr->getSourceRange();
  else
    FDiag << FirstType << SecondType << Action << SrcExpr->getSourceRange();

  // If we can fix the conversion, suggest the FixIts.
  assert(ConvHints.isNull() || Hint.isNull());
  if (!ConvHints.isNull()) {
    for (FixItHint &H : ConvHints.Hints)
      FDiag << H;
  } else {
    FDiag << Hint;
  }
  if (MayHaveConvFixit) { FDiag << (unsigned) (ConvHints.Kind); }

  if (MayHaveFunctionDiff)
    HandleFunctionTypeMismatch(FDiag, SecondType, FirstType);

  Diag(Loc, FDiag);
  if (DiagKind == diag::warn_incompatible_qualified_id &&
      PDecl && IFace && !IFace->hasDefinition())
      Diag(IFace->getLocation(), diag::not_incomplete_class_and_qualified_id)
        << IFace->getName() << PDecl->getName();
    
  if (SecondType == Context.OverloadTy)
    NoteAllOverloadCandidates(OverloadExpr::find(SrcExpr).Expression,
                              FirstType, /*TakingAddress=*/true);

  if (CheckInferredResultType)
    EmitRelatedResultTypeNote(SrcExpr);

  if (Action == AA_Returning && ConvTy == IncompatiblePointer)
    EmitRelatedResultTypeNoteForReturn(DstType);
  
  if (Complained)
    *Complained = true;
  return isInvalid;
}

ExprResult Sema::VerifyIntegerConstantExpression(Expr *E,
                                                 llvm::APSInt *Result) {
  class SimpleICEDiagnoser : public VerifyICEDiagnoser {
  public:
    void diagnoseNotICE(Sema &S, SourceLocation Loc, SourceRange SR) override {
      S.Diag(Loc, diag::err_expr_not_ice) << S.LangOpts.CPlusPlus << SR;
    }
  } Diagnoser;
  
  return VerifyIntegerConstantExpression(E, Result, Diagnoser);
}

ExprResult Sema::VerifyIntegerConstantExpression(Expr *E,
                                                 llvm::APSInt *Result,
                                                 unsigned DiagID,
                                                 bool AllowFold) {
  class IDDiagnoser : public VerifyICEDiagnoser {
    unsigned DiagID;
    
  public:
    IDDiagnoser(unsigned DiagID)
      : VerifyICEDiagnoser(DiagID == 0), DiagID(DiagID) { }
    
    void diagnoseNotICE(Sema &S, SourceLocation Loc, SourceRange SR) override {
      S.Diag(Loc, DiagID) << SR;
    }
  } Diagnoser(DiagID);
  
  return VerifyIntegerConstantExpression(E, Result, Diagnoser, AllowFold);
}

void Sema::VerifyICEDiagnoser::diagnoseFold(Sema &S, SourceLocation Loc,
                                            SourceRange SR) {
  S.Diag(Loc, diag::ext_expr_not_ice) << SR << S.LangOpts.CPlusPlus;
}

ExprResult
Sema::VerifyIntegerConstantExpression(Expr *E, llvm::APSInt *Result,
                                      VerifyICEDiagnoser &Diagnoser,
                                      bool AllowFold) {
  SourceLocation DiagLoc = E->getLocStart();

  if (getLangOpts().CPlusPlus11) {
    // C++11 [expr.const]p5:
    //   If an expression of literal class type is used in a context where an
    //   integral constant expression is required, then that class type shall
    //   have a single non-explicit conversion function to an integral or
    //   unscoped enumeration type
    ExprResult Converted;
    class CXX11ConvertDiagnoser : public ICEConvertDiagnoser {
    public:
      CXX11ConvertDiagnoser(bool Silent)
          : ICEConvertDiagnoser(/*AllowScopedEnumerations*/false,
                                Silent, true) {}

      SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                           QualType T) override {
        return S.Diag(Loc, diag::err_ice_not_integral) << T;
      }

      SemaDiagnosticBuilder diagnoseIncomplete(
          Sema &S, SourceLocation Loc, QualType T) override {
        return S.Diag(Loc, diag::err_ice_incomplete_type) << T;
      }

      SemaDiagnosticBuilder diagnoseExplicitConv(
          Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) override {
        return S.Diag(Loc, diag::err_ice_explicit_conversion) << T << ConvTy;
      }

      SemaDiagnosticBuilder noteExplicitConv(
          Sema &S, CXXConversionDecl *Conv, QualType ConvTy) override {
        return S.Diag(Conv->getLocation(), diag::note_ice_conversion_here)
                 << ConvTy->isEnumeralType() << ConvTy;
      }

      SemaDiagnosticBuilder diagnoseAmbiguous(
          Sema &S, SourceLocation Loc, QualType T) override {
        return S.Diag(Loc, diag::err_ice_ambiguous_conversion) << T;
      }

      SemaDiagnosticBuilder noteAmbiguous(
          Sema &S, CXXConversionDecl *Conv, QualType ConvTy) override {
        return S.Diag(Conv->getLocation(), diag::note_ice_conversion_here)
                 << ConvTy->isEnumeralType() << ConvTy;
      }

      SemaDiagnosticBuilder diagnoseConversion(
          Sema &S, SourceLocation Loc, QualType T, QualType ConvTy) override {
        llvm_unreachable("conversion functions are permitted");
      }
    } ConvertDiagnoser(Diagnoser.Suppress);

    Converted = PerformContextualImplicitConversion(DiagLoc, E,
                                                    ConvertDiagnoser);
    if (Converted.isInvalid())
      return Converted;
    E = Converted.get();
    if (!E->getType()->isIntegralOrUnscopedEnumerationType())
      return ExprError();
  } else if (!E->getType()->isIntegralOrUnscopedEnumerationType()) {
    // An ICE must be of integral or unscoped enumeration type.
    if (!Diagnoser.Suppress)
      Diagnoser.diagnoseNotICE(*this, DiagLoc, E->getSourceRange());
    return ExprError();
  }

  // Circumvent ICE checking in C++11 to avoid evaluating the expression twice
  // in the non-ICE case.
  if (!getLangOpts().CPlusPlus11 && E->isIntegerConstantExpr(Context)) {
    if (Result)
      *Result = E->EvaluateKnownConstInt(Context);
    return E;
  }

  Expr::EvalResult EvalResult;
  SmallVector<PartialDiagnosticAt, 8> Notes;
  EvalResult.Diag = &Notes;

  // Try to evaluate the expression, and produce diagnostics explaining why it's
  // not a constant expression as a side-effect.
  bool Folded = E->EvaluateAsRValue(EvalResult, Context) &&
                EvalResult.Val.isInt() && !EvalResult.HasSideEffects;

  // In C++11, we can rely on diagnostics being produced for any expression
  // which is not a constant expression. If no diagnostics were produced, then
  // this is a constant expression.
  if (Folded && getLangOpts().CPlusPlus11 && Notes.empty()) {
    if (Result)
      *Result = EvalResult.Val.getInt();
    return E;
  }

  // If our only note is the usual "invalid subexpression" note, just point
  // the caret at its location rather than producing an essentially
  // redundant note.
  if (Notes.size() == 1 && Notes[0].second.getDiagID() ==
        diag::note_invalid_subexpr_in_const_expr) {
    DiagLoc = Notes[0].first;
    Notes.clear();
  }

  if (!Folded || !AllowFold) {
    if (!Diagnoser.Suppress) {
      Diagnoser.diagnoseNotICE(*this, DiagLoc, E->getSourceRange());
      for (const PartialDiagnosticAt &Note : Notes)
        Diag(Note.first, Note.second);
    }

    return ExprError();
  }

  Diagnoser.diagnoseFold(*this, DiagLoc, E->getSourceRange());
  for (const PartialDiagnosticAt &Note : Notes)
    Diag(Note.first, Note.second);

  if (Result)
    *Result = EvalResult.Val.getInt();
  return E;
}

namespace {
  // Handle the case where we conclude a expression which we speculatively
  // considered to be unevaluated is actually evaluated.
  class TransformToPE : public TreeTransform<TransformToPE> {
    typedef TreeTransform<TransformToPE> BaseTransform;

  public:
    TransformToPE(Sema &SemaRef) : BaseTransform(SemaRef) { }

    // Make sure we redo semantic analysis
    bool AlwaysRebuild() { return true; }

    // Make sure we handle LabelStmts correctly.
    // FIXME: This does the right thing, but maybe we need a more general
    // fix to TreeTransform?
    StmtResult TransformLabelStmt(LabelStmt *S) {
      S->getDecl()->setStmt(nullptr);
      return BaseTransform::TransformLabelStmt(S);
    }

    // We need to special-case DeclRefExprs referring to FieldDecls which
    // are not part of a member pointer formation; normal TreeTransforming
    // doesn't catch this case because of the way we represent them in the AST.
    // FIXME: This is a bit ugly; is it really the best way to handle this
    // case?
    //
    // Error on DeclRefExprs referring to FieldDecls.
    ExprResult TransformDeclRefExpr(DeclRefExpr *E) {
      if (isa<FieldDecl>(E->getDecl()) &&
          !SemaRef.isUnevaluatedContext())
        return SemaRef.Diag(E->getLocation(),
                            diag::err_invalid_non_static_member_use)
            << E->getDecl() << E->getSourceRange();

      return BaseTransform::TransformDeclRefExpr(E);
    }

    // Exception: filter out member pointer formation
    ExprResult TransformUnaryOperator(UnaryOperator *E) {
      if (E->getOpcode() == UO_AddrOf && E->getType()->isMemberPointerType())
        return E;

      return BaseTransform::TransformUnaryOperator(E);
    }

    ExprResult TransformLambdaExpr(LambdaExpr *E) {
      // Lambdas never need to be transformed.
      return E;
    }
  };
}

ExprResult Sema::TransformToPotentiallyEvaluated(Expr *E) {
  assert(isUnevaluatedContext() &&
         "Should only transform unevaluated expressions");
  ExprEvalContexts.back().Context =
      ExprEvalContexts[ExprEvalContexts.size()-2].Context;
  if (isUnevaluatedContext())
    return E;
  return TransformToPE(*this).TransformExpr(E);
}

void
Sema::PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext,
                                      Decl *LambdaContextDecl,
                                      bool IsDecltype) {
  ExprEvalContexts.emplace_back(NewContext, ExprCleanupObjects.size(),
                                ExprNeedsCleanups, LambdaContextDecl,
                                IsDecltype);
  ExprNeedsCleanups = false;
  if (!MaybeODRUseExprs.empty())
    std::swap(MaybeODRUseExprs, ExprEvalContexts.back().SavedMaybeODRUseExprs);
}

void
Sema::PushExpressionEvaluationContext(ExpressionEvaluationContext NewContext,
                                      ReuseLambdaContextDecl_t,
                                      bool IsDecltype) {
  Decl *ClosureContextDecl = ExprEvalContexts.back().ManglingContextDecl;
  PushExpressionEvaluationContext(NewContext, ClosureContextDecl, IsDecltype);
}

void Sema::PopExpressionEvaluationContext() {
  ExpressionEvaluationContextRecord& Rec = ExprEvalContexts.back();
  unsigned NumTypos = Rec.NumTypos;

  if (!Rec.Lambdas.empty()) {
    if (Rec.isUnevaluated() || Rec.Context == ConstantEvaluated) {
      unsigned D;
      if (Rec.isUnevaluated()) {
        // C++11 [expr.prim.lambda]p2:
        //   A lambda-expression shall not appear in an unevaluated operand
        //   (Clause 5).
        D = diag::err_lambda_unevaluated_operand;
      } else {
        // C++1y [expr.const]p2:
        //   A conditional-expression e is a core constant expression unless the
        //   evaluation of e, following the rules of the abstract machine, would
        //   evaluate [...] a lambda-expression.
        D = diag::err_lambda_in_constant_expression;
      }
      for (const auto *L : Rec.Lambdas)
        Diag(L->getLocStart(), D);
    } else {
      // Mark the capture expressions odr-used. This was deferred
      // during lambda expression creation.
      for (auto *Lambda : Rec.Lambdas) {
        for (auto *C : Lambda->capture_inits())
          MarkDeclarationsReferencedInExpr(C);
      }
    }
  }

  // When are coming out of an unevaluated context, clear out any
  // temporaries that we may have created as part of the evaluation of
  // the expression in that context: they aren't relevant because they
  // will never be constructed.
  if (Rec.isUnevaluated() || Rec.Context == ConstantEvaluated) {
    ExprCleanupObjects.erase(ExprCleanupObjects.begin() + Rec.NumCleanupObjects,
                             ExprCleanupObjects.end());
    ExprNeedsCleanups = Rec.ParentNeedsCleanups;
    CleanupVarDeclMarking();
    std::swap(MaybeODRUseExprs, Rec.SavedMaybeODRUseExprs);
  // Otherwise, merge the contexts together.
  } else {
    ExprNeedsCleanups |= Rec.ParentNeedsCleanups;
    MaybeODRUseExprs.insert(Rec.SavedMaybeODRUseExprs.begin(),
                            Rec.SavedMaybeODRUseExprs.end());
  }

  // Pop the current expression evaluation context off the stack.
  ExprEvalContexts.pop_back();

  if (!ExprEvalContexts.empty())
    ExprEvalContexts.back().NumTypos += NumTypos;
  else
    assert(NumTypos == 0 && "There are outstanding typos after popping the "
                            "last ExpressionEvaluationContextRecord");
}

void Sema::DiscardCleanupsInEvaluationContext() {
  ExprCleanupObjects.erase(
         ExprCleanupObjects.begin() + ExprEvalContexts.back().NumCleanupObjects,
         ExprCleanupObjects.end());
  ExprNeedsCleanups = false;
  MaybeODRUseExprs.clear();
}

ExprResult Sema::HandleExprEvaluationContextForTypeof(Expr *E) {
  if (!E->getType()->isVariablyModifiedType())
    return E;
  return TransformToPotentiallyEvaluated(E);
}

static bool IsPotentiallyEvaluatedContext(Sema &SemaRef) {
  // Do not mark anything as "used" within a dependent context; wait for
  // an instantiation.
  if (SemaRef.CurContext->isDependentContext())
    return false;

  switch (SemaRef.ExprEvalContexts.back().Context) {
    case Sema::Unevaluated:
    case Sema::UnevaluatedAbstract:
      // We are in an expression that is not potentially evaluated; do nothing.
      // (Depending on how you read the standard, we actually do need to do
      // something here for null pointer constants, but the standard's
      // definition of a null pointer constant is completely crazy.)
      return false;

    case Sema::ConstantEvaluated:
    case Sema::PotentiallyEvaluated:
      // We are in a potentially evaluated expression (or a constant-expression
      // in C++03); we need to do implicit template instantiation, implicitly
      // define class members, and mark most declarations as used.
      return true;

    case Sema::PotentiallyEvaluatedIfUsed:
      // Referenced declarations will only be used if the construct in the
      // containing expression is used.
      return false;
  }
  llvm_unreachable("Invalid context");
}

/// \brief Mark a function referenced, and check whether it is odr-used
/// (C++ [basic.def.odr]p2, C99 6.9p3)
void Sema::MarkFunctionReferenced(SourceLocation Loc, FunctionDecl *Func,
                                  bool OdrUse) {
  assert(Func && "No function?");

  Func->setReferenced();

  // C++11 [basic.def.odr]p3:
  //   A function whose name appears as a potentially-evaluated expression is
  //   odr-used if it is the unique lookup result or the selected member of a
  //   set of overloaded functions [...].
  //
  // We (incorrectly) mark overload resolution as an unevaluated context, so we
  // can just check that here. Skip the rest of this function if we've already
  // marked the function as used.
  if (Func->isUsed(/*CheckUsedAttr=*/false) ||
      !IsPotentiallyEvaluatedContext(*this)) {
    // C++11 [temp.inst]p3:
    //   Unless a function template specialization has been explicitly
    //   instantiated or explicitly specialized, the function template
    //   specialization is implicitly instantiated when the specialization is
    //   referenced in a context that requires a function definition to exist.
    //
    // We consider constexpr function templates to be referenced in a context
    // that requires a definition to exist whenever they are referenced.
    //
    // FIXME: This instantiates constexpr functions too frequently. If this is
    // really an unevaluated context (and we're not just in the definition of a
    // function template or overload resolution or other cases which we
    // incorrectly consider to be unevaluated contexts), and we're not in a
    // subexpression which we actually need to evaluate (for instance, a
    // template argument, array bound or an expression in a braced-init-list),
    // we are not permitted to instantiate this constexpr function definition.
    //
    // FIXME: This also implicitly defines special members too frequently. They
    // are only supposed to be implicitly defined if they are odr-used, but they
    // are not odr-used from constant expressions in unevaluated contexts.
    // However, they cannot be referenced if they are deleted, and they are
    // deleted whenever the implicit definition of the special member would
    // fail.
    if (!Func->isConstexpr() || Func->getBody())
      return;
    CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Func);
    if (!Func->isImplicitlyInstantiable() && (!MD || MD->isUserProvided()))
      return;
  }

  // Note that this declaration has been used.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Func)) {
    Constructor = cast<CXXConstructorDecl>(Constructor->getFirstDecl());
    if (Constructor->isDefaulted() && !Constructor->isDeleted()) {
      if (Constructor->isDefaultConstructor()) {
        if (Constructor->isTrivial() && !Constructor->hasAttr<DLLExportAttr>())
          return;
        DefineImplicitDefaultConstructor(Loc, Constructor);
      } else if (Constructor->isCopyConstructor()) {
        DefineImplicitCopyConstructor(Loc, Constructor);
      } else if (Constructor->isMoveConstructor()) {
        DefineImplicitMoveConstructor(Loc, Constructor);
      }
    } else if (Constructor->getInheritedConstructor()) {
      DefineInheritingConstructor(Loc, Constructor);
    }
  } else if (CXXDestructorDecl *Destructor =
                 dyn_cast<CXXDestructorDecl>(Func)) {
    Destructor = cast<CXXDestructorDecl>(Destructor->getFirstDecl());
    if (Destructor->isDefaulted() && !Destructor->isDeleted()) {
      if (Destructor->isTrivial() && !Destructor->hasAttr<DLLExportAttr>())
        return;
      DefineImplicitDestructor(Loc, Destructor);
    }
    if (Destructor->isVirtual() && getLangOpts().AppleKext)
      MarkVTableUsed(Loc, Destructor->getParent());
  } else if (CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(Func)) {
    if (MethodDecl->isOverloadedOperator() &&
        MethodDecl->getOverloadedOperator() == OO_Equal) {
      MethodDecl = cast<CXXMethodDecl>(MethodDecl->getFirstDecl());
      if (MethodDecl->isDefaulted() && !MethodDecl->isDeleted()) {
        if (MethodDecl->isCopyAssignmentOperator())
          DefineImplicitCopyAssignment(Loc, MethodDecl);
        else
          DefineImplicitMoveAssignment(Loc, MethodDecl);
      }
    } else if (isa<CXXConversionDecl>(MethodDecl) &&
               MethodDecl->getParent()->isLambda()) {
      CXXConversionDecl *Conversion =
          cast<CXXConversionDecl>(MethodDecl->getFirstDecl());
      if (Conversion->isLambdaToBlockPointerConversion())
        DefineImplicitLambdaToBlockPointerConversion(Loc, Conversion);
      else
        DefineImplicitLambdaToFunctionPointerConversion(Loc, Conversion);
    } else if (MethodDecl->isVirtual() && getLangOpts().AppleKext)
      MarkVTableUsed(Loc, MethodDecl->getParent());
  }

  // Recursive functions should be marked when used from another function.
  // FIXME: Is this really right?
  if (CurContext == Func) return;

  // Resolve the exception specification for any function which is
  // used: CodeGen will need it.
  const FunctionProtoType *FPT = Func->getType()->getAs<FunctionProtoType>();
  if (FPT && isUnresolvedExceptionSpec(FPT->getExceptionSpecType()))
    ResolveExceptionSpec(Loc, FPT);

  if (!OdrUse) return;

  // Implicit instantiation of function templates and member functions of
  // class templates.
  if (Func->isImplicitlyInstantiable()) {
    bool AlreadyInstantiated = false;
    SourceLocation PointOfInstantiation = Loc;
    if (FunctionTemplateSpecializationInfo *SpecInfo
                              = Func->getTemplateSpecializationInfo()) {
      if (SpecInfo->getPointOfInstantiation().isInvalid())
        SpecInfo->setPointOfInstantiation(Loc);
      else if (SpecInfo->getTemplateSpecializationKind()
                 == TSK_ImplicitInstantiation) {
        AlreadyInstantiated = true;
        PointOfInstantiation = SpecInfo->getPointOfInstantiation();
      }
    } else if (MemberSpecializationInfo *MSInfo
                                = Func->getMemberSpecializationInfo()) {
      if (MSInfo->getPointOfInstantiation().isInvalid())
        MSInfo->setPointOfInstantiation(Loc);
      else if (MSInfo->getTemplateSpecializationKind()
                 == TSK_ImplicitInstantiation) {
        AlreadyInstantiated = true;
        PointOfInstantiation = MSInfo->getPointOfInstantiation();
      }
    }

    if (!AlreadyInstantiated || Func->isConstexpr()) {
      if (isa<CXXRecordDecl>(Func->getDeclContext()) &&
          cast<CXXRecordDecl>(Func->getDeclContext())->isLocalClass() &&
          ActiveTemplateInstantiations.size())
        PendingLocalImplicitInstantiations.push_back(
            std::make_pair(Func, PointOfInstantiation));
      else if (Func->isConstexpr())
        // Do not defer instantiations of constexpr functions, to avoid the
        // expression evaluator needing to call back into Sema if it sees a
        // call to such a function.
        InstantiateFunctionDefinition(PointOfInstantiation, Func);
      else {
        PendingInstantiations.push_back(std::make_pair(Func,
                                                       PointOfInstantiation));
        // Notify the consumer that a function was implicitly instantiated.
        Consumer.HandleCXXImplicitFunctionInstantiation(Func);
      }
    }
  } else {
    // Walk redefinitions, as some of them may be instantiable.
    for (auto i : Func->redecls()) {
      if (!i->isUsed(false) && i->isImplicitlyInstantiable())
        MarkFunctionReferenced(Loc, i);
    }
  }

  // Keep track of used but undefined functions.
  if (!Func->isDefined()) {
    if (mightHaveNonExternalLinkage(Func))
      UndefinedButUsed.insert(std::make_pair(Func->getCanonicalDecl(), Loc));
    else if (Func->getMostRecentDecl()->isInlined() &&
             !LangOpts.GNUInline &&
             !Func->getMostRecentDecl()->hasAttr<GNUInlineAttr>())
      UndefinedButUsed.insert(std::make_pair(Func->getCanonicalDecl(), Loc));
  }

  // Normally the most current decl is marked used while processing the use and
  // any subsequent decls are marked used by decl merging. This fails with
  // template instantiation since marking can happen at the end of the file
  // and, because of the two phase lookup, this function is called with at
  // decl in the middle of a decl chain. We loop to maintain the invariant
  // that once a decl is used, all decls after it are also used.
  for (FunctionDecl *F = Func->getMostRecentDecl();; F = F->getPreviousDecl()) {
    F->markUsed(Context);
    if (F == Func)
      break;
  }
}

static void
diagnoseUncapturableValueReference(Sema &S, SourceLocation loc,
                                   VarDecl *var, DeclContext *DC) {
  DeclContext *VarDC = var->getDeclContext();

  //  If the parameter still belongs to the translation unit, then
  //  we're actually just using one parameter in the declaration of
  //  the next.
  if (isa<ParmVarDecl>(var) &&
      isa<TranslationUnitDecl>(VarDC))
    return;

  // For C code, don't diagnose about capture if we're not actually in code
  // right now; it's impossible to write a non-constant expression outside of
  // function context, so we'll get other (more useful) diagnostics later.
  //
  // For C++, things get a bit more nasty... it would be nice to suppress this
  // diagnostic for certain cases like using a local variable in an array bound
  // for a member of a local class, but the correct predicate is not obvious.
  if (!S.getLangOpts().CPlusPlus && !S.CurContext->isFunctionOrMethod())
    return;

  if (isa<CXXMethodDecl>(VarDC) &&
      cast<CXXRecordDecl>(VarDC->getParent())->isLambda()) {
    S.Diag(loc, diag::err_reference_to_local_var_in_enclosing_lambda)
      << var->getIdentifier();
  } else if (FunctionDecl *fn = dyn_cast<FunctionDecl>(VarDC)) {
    S.Diag(loc, diag::err_reference_to_local_var_in_enclosing_function)
      << var->getIdentifier() << fn->getDeclName();
  } else if (isa<BlockDecl>(VarDC)) {
    S.Diag(loc, diag::err_reference_to_local_var_in_enclosing_block)
      << var->getIdentifier();
  } else {
    // FIXME: Is there any other context where a local variable can be
    // declared?
    S.Diag(loc, diag::err_reference_to_local_var_in_enclosing_context)
      << var->getIdentifier();
  }

  S.Diag(var->getLocation(), diag::note_entity_declared_at)
      << var->getIdentifier();

  // FIXME: Add additional diagnostic info about class etc. which prevents
  // capture.
}

 
static bool isVariableAlreadyCapturedInScopeInfo(CapturingScopeInfo *CSI, VarDecl *Var, 
                                      bool &SubCapturesAreNested,
                                      QualType &CaptureType, 
                                      QualType &DeclRefType) {
   // Check whether we've already captured it.
  if (CSI->CaptureMap.count(Var)) {
    // If we found a capture, any subcaptures are nested.
    SubCapturesAreNested = true;
      
    // Retrieve the capture type for this variable.
    CaptureType = CSI->getCapture(Var).getCaptureType();
      
    // Compute the type of an expression that refers to this variable.
    DeclRefType = CaptureType.getNonReferenceType();

    // Similarly to mutable captures in lambda, all the OpenMP captures by copy
    // are mutable in the sense that user can change their value - they are
    // private instances of the captured declarations.
    const CapturingScopeInfo::Capture &Cap = CSI->getCapture(Var);
    if (Cap.isCopyCapture() &&
        !(isa<LambdaScopeInfo>(CSI) && cast<LambdaScopeInfo>(CSI)->Mutable) &&
        !(isa<CapturedRegionScopeInfo>(CSI) &&
          cast<CapturedRegionScopeInfo>(CSI)->CapRegionKind == CR_OpenMP))
      DeclRefType.addConst();
    return true;
  }
  return false;
}

// Only block literals, captured statements, and lambda expressions can
// capture; other scopes don't work.
static DeclContext *getParentOfCapturingContextOrNull(DeclContext *DC, VarDecl *Var, 
                                 SourceLocation Loc, 
                                 const bool Diagnose, Sema &S) {
  if (isa<BlockDecl>(DC) || isa<CapturedDecl>(DC) || isLambdaCallOperator(DC))
    return getLambdaAwareParentOfDeclContext(DC);
  else if (Var->hasLocalStorage()) {
    if (Diagnose)
       diagnoseUncapturableValueReference(S, Loc, Var, DC);
  }
  return nullptr;
}

// Certain capturing entities (lambdas, blocks etc.) are not allowed to capture 
// certain types of variables (unnamed, variably modified types etc.)
// so check for eligibility.
static bool isVariableCapturable(CapturingScopeInfo *CSI, VarDecl *Var, 
                                 SourceLocation Loc, 
                                 const bool Diagnose, Sema &S) {

  bool IsBlock = isa<BlockScopeInfo>(CSI);
  bool IsLambda = isa<LambdaScopeInfo>(CSI);

  // Lambdas are not allowed to capture unnamed variables
  // (e.g. anonymous unions).
  // FIXME: The C++11 rule don't actually state this explicitly, but I'm
  // assuming that's the intent.
  if (IsLambda && !Var->getDeclName()) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_lambda_capture_anonymous_var);
      S.Diag(Var->getLocation(), diag::note_declared_at);
    }
    return false;
  }

  // Prohibit variably-modified types in blocks; they're difficult to deal with.
  if (Var->getType()->isVariablyModifiedType() && IsBlock) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_ref_vm_type);
      S.Diag(Var->getLocation(), diag::note_previous_decl) 
        << Var->getDeclName();
    }
    return false;
  }
  // Prohibit structs with flexible array members too.
  // We cannot capture what is in the tail end of the struct.
  if (const RecordType *VTTy = Var->getType()->getAs<RecordType>()) {
    if (VTTy->getDecl()->hasFlexibleArrayMember()) {
      if (Diagnose) {
        if (IsBlock)
          S.Diag(Loc, diag::err_ref_flexarray_type);
        else
          S.Diag(Loc, diag::err_lambda_capture_flexarray_type)
            << Var->getDeclName();
        S.Diag(Var->getLocation(), diag::note_previous_decl)
          << Var->getDeclName();
      }
      return false;
    }
  }
  const bool HasBlocksAttr = Var->hasAttr<BlocksAttr>();
  // Lambdas and captured statements are not allowed to capture __block
  // variables; they don't support the expected semantics.
  if (HasBlocksAttr && (IsLambda || isa<CapturedRegionScopeInfo>(CSI))) {
    if (Diagnose) {
      S.Diag(Loc, diag::err_capture_block_variable)
        << Var->getDeclName() << !IsLambda;
      S.Diag(Var->getLocation(), diag::note_previous_decl)
        << Var->getDeclName();
    }
    return false;
  }

  return true;
}

// Returns true if the capture by block was successful.
static bool captureInBlock(BlockScopeInfo *BSI, VarDecl *Var, 
                                 SourceLocation Loc, 
                                 const bool BuildAndDiagnose, 
                                 QualType &CaptureType,
                                 QualType &DeclRefType, 
                                 const bool Nested,
                                 Sema &S) {
  Expr *CopyExpr = nullptr;
  bool ByRef = false;
      
  // Blocks are not allowed to capture arrays.
  if (CaptureType->isArrayType()) {
    if (BuildAndDiagnose) {
      S.Diag(Loc, diag::err_ref_array_type);
      S.Diag(Var->getLocation(), diag::note_previous_decl) 
      << Var->getDeclName();
    }
    return false;
  }

  // Forbid the block-capture of autoreleasing variables.
  if (CaptureType.getObjCLifetime() == Qualifiers::OCL_Autoreleasing) {
    if (BuildAndDiagnose) {
      S.Diag(Loc, diag::err_arc_autoreleasing_capture)
        << /*block*/ 0;
      S.Diag(Var->getLocation(), diag::note_previous_decl)
        << Var->getDeclName();
    }
    return false;
  }
  const bool HasBlocksAttr = Var->hasAttr<BlocksAttr>();
  if (HasBlocksAttr || CaptureType->isReferenceType()) {
    // Block capture by reference does not change the capture or
    // declaration reference types.
    ByRef = true;
  } else {
    // Block capture by copy introduces 'const'.
    CaptureType = CaptureType.getNonReferenceType().withConst();
    DeclRefType = CaptureType;
                
    if (S.getLangOpts().CPlusPlus && BuildAndDiagnose) {
      if (const RecordType *Record = DeclRefType->getAs<RecordType>()) {
        // The capture logic needs the destructor, so make sure we mark it.
        // Usually this is unnecessary because most local variables have
        // their destructors marked at declaration time, but parameters are
        // an exception because it's technically only the call site that
        // actually requires the destructor.
        if (isa<ParmVarDecl>(Var))
          S.FinalizeVarWithDestructor(Var, Record);

        // Enter a new evaluation context to insulate the copy
        // full-expression.
        EnterExpressionEvaluationContext scope(S, S.PotentiallyEvaluated);

        // According to the blocks spec, the capture of a variable from
        // the stack requires a const copy constructor.  This is not true
        // of the copy/move done to move a __block variable to the heap.
        Expr *DeclRef = new (S.Context) DeclRefExpr(Var, Nested,
                                                  DeclRefType.withConst(), 
                                                  VK_LValue, Loc);
            
        ExprResult Result
          = S.PerformCopyInitialization(
              InitializedEntity::InitializeBlock(Var->getLocation(),
                                                  CaptureType, false),
              Loc, DeclRef);
            
        // Build a full-expression copy expression if initialization
        // succeeded and used a non-trivial constructor.  Recover from
        // errors by pretending that the copy isn't necessary.
        if (!Result.isInvalid() &&
            !cast<CXXConstructExpr>(Result.get())->getConstructor()
                ->isTrivial()) {
          Result = S.MaybeCreateExprWithCleanups(Result);
          CopyExpr = Result.get();
        }
      }
    }
  }

  // Actually capture the variable.
  if (BuildAndDiagnose)
    BSI->addCapture(Var, HasBlocksAttr, ByRef, Nested, Loc, 
                    SourceLocation(), CaptureType, CopyExpr);

  return true;

}


/// \brief Capture the given variable in the captured region.
static bool captureInCapturedRegion(CapturedRegionScopeInfo *RSI,
                                    VarDecl *Var, 
                                    SourceLocation Loc, 
                                    const bool BuildAndDiagnose, 
                                    QualType &CaptureType,
                                    QualType &DeclRefType, 
                                    const bool RefersToCapturedVariable,
                                    Sema &S) {
  
  // By default, capture variables by reference.
  bool ByRef = true;
  // Using an LValue reference type is consistent with Lambdas (see below).
  if (S.getLangOpts().OpenMP) {
    ByRef = S.IsOpenMPCapturedByRef(Var, RSI);
    if (S.IsOpenMPCapturedDecl(Var))
      DeclRefType = DeclRefType.getUnqualifiedType();
  }

  if (ByRef)
    CaptureType = S.Context.getLValueReferenceType(DeclRefType);
  else
    CaptureType = DeclRefType;

  Expr *CopyExpr = nullptr;
  if (BuildAndDiagnose) {
    // The current implementation assumes that all variables are captured
    // by references. Since there is no capture by copy, no expression
    // evaluation will be needed.
    RecordDecl *RD = RSI->TheRecordDecl;

    FieldDecl *Field
      = FieldDecl::Create(S.Context, RD, Loc, Loc, nullptr, CaptureType,
                          S.Context.getTrivialTypeSourceInfo(CaptureType, Loc),
                          nullptr, false, ICIS_NoInit);
    Field->setImplicit(true);
    Field->setAccess(AS_private);
    RD->addDecl(Field);
 
    CopyExpr = new (S.Context) DeclRefExpr(Var, RefersToCapturedVariable,
                                            DeclRefType, VK_LValue, Loc);
    Var->setReferenced(true);
    Var->markUsed(S.Context);
  }

  // Actually capture the variable.
  if (BuildAndDiagnose)
    RSI->addCapture(Var, /*isBlock*/false, ByRef, RefersToCapturedVariable, Loc,
                    SourceLocation(), CaptureType, CopyExpr);
  
  
  return true;
}

/// \brief Create a field within the lambda class for the variable
/// being captured.
static void addAsFieldToClosureType(Sema &S, LambdaScopeInfo *LSI, VarDecl *Var,
                                    QualType FieldType, QualType DeclRefType,
                                    SourceLocation Loc,
                                    bool RefersToCapturedVariable) {
  CXXRecordDecl *Lambda = LSI->Lambda;

  // Build the non-static data member.
  FieldDecl *Field
    = FieldDecl::Create(S.Context, Lambda, Loc, Loc, nullptr, FieldType,
                        S.Context.getTrivialTypeSourceInfo(FieldType, Loc),
                        nullptr, false, ICIS_NoInit);
  Field->setImplicit(true);
  Field->setAccess(AS_private);
  Lambda->addDecl(Field);
}

/// \brief Capture the given variable in the lambda.
static bool captureInLambda(LambdaScopeInfo *LSI,
                            VarDecl *Var, 
                            SourceLocation Loc, 
                            const bool BuildAndDiagnose, 
                            QualType &CaptureType,
                            QualType &DeclRefType, 
                            const bool RefersToCapturedVariable,
                            const Sema::TryCaptureKind Kind, 
                            SourceLocation EllipsisLoc,
                            const bool IsTopScope,
                            Sema &S) {

  // Determine whether we are capturing by reference or by value.
  bool ByRef = false;
  if (IsTopScope && Kind != Sema::TryCapture_Implicit) {
    ByRef = (Kind == Sema::TryCapture_ExplicitByRef);
  } else {
    ByRef = (LSI->ImpCaptureStyle == LambdaScopeInfo::ImpCap_LambdaByref);
  }
    
  // Compute the type of the field that will capture this variable.
  if (ByRef) {
    // C++11 [expr.prim.lambda]p15:
    //   An entity is captured by reference if it is implicitly or
    //   explicitly captured but not captured by copy. It is
    //   unspecified whether additional unnamed non-static data
    //   members are declared in the closure type for entities
    //   captured by reference.
    //
    // FIXME: It is not clear whether we want to build an lvalue reference
    // to the DeclRefType or to CaptureType.getNonReferenceType(). GCC appears
    // to do the former, while EDG does the latter. Core issue 1249 will 
    // clarify, but for now we follow GCC because it's a more permissive and
    // easily defensible position.
    CaptureType = S.Context.getLValueReferenceType(DeclRefType);
  } else {
    // C++11 [expr.prim.lambda]p14:
    //   For each entity captured by copy, an unnamed non-static
    //   data member is declared in the closure type. The
    //   declaration order of these members is unspecified. The type
    //   of such a data member is the type of the corresponding
    //   captured entity if the entity is not a reference to an
    //   object, or the referenced type otherwise. [Note: If the
    //   captured entity is a reference to a function, the
    //   corresponding data member is also a reference to a
    //   function. - end note ]
    if (const ReferenceType *RefType = CaptureType->getAs<ReferenceType>()){
      if (!RefType->getPointeeType()->isFunctionType())
        CaptureType = RefType->getPointeeType();
    }

    // Forbid the lambda copy-capture of autoreleasing variables.
    if (CaptureType.getObjCLifetime() == Qualifiers::OCL_Autoreleasing) {
      if (BuildAndDiagnose) {
        S.Diag(Loc, diag::err_arc_autoreleasing_capture) << /*lambda*/ 1;
        S.Diag(Var->getLocation(), diag::note_previous_decl)
          << Var->getDeclName();
      }
      return false;
    }

    // Make sure that by-copy captures are of a complete and non-abstract type.
    if (BuildAndDiagnose) {
      if (!CaptureType->isDependentType() &&
          S.RequireCompleteType(Loc, CaptureType,
                                diag::err_capture_of_incomplete_type,
                                Var->getDeclName()))
        return false;

      if (S.RequireNonAbstractType(Loc, CaptureType,
                                   diag::err_capture_of_abstract_type))
        return false;
    }
  }

  // Capture this variable in the lambda.
  if (BuildAndDiagnose)
    addAsFieldToClosureType(S, LSI, Var, CaptureType, DeclRefType, Loc,
                            RefersToCapturedVariable);
    
  // Compute the type of a reference to this captured variable.
  if (ByRef)
    DeclRefType = CaptureType.getNonReferenceType();
  else {
    // C++ [expr.prim.lambda]p5:
    //   The closure type for a lambda-expression has a public inline 
    //   function call operator [...]. This function call operator is 
    //   declared const (9.3.1) if and only if the lambda-expression’s 
    //   parameter-declaration-clause is not followed by mutable.
    DeclRefType = CaptureType.getNonReferenceType();
    if (!LSI->Mutable && !CaptureType->isReferenceType())
      DeclRefType.addConst();      
  }
    
  // Add the capture.
  if (BuildAndDiagnose)
    LSI->addCapture(Var, /*IsBlock=*/false, ByRef, RefersToCapturedVariable, 
                    Loc, EllipsisLoc, CaptureType, /*CopyExpr=*/nullptr);
      
  return true;
}

bool Sema::tryCaptureVariable(
    VarDecl *Var, SourceLocation ExprLoc, TryCaptureKind Kind,
    SourceLocation EllipsisLoc, bool BuildAndDiagnose, QualType &CaptureType,
    QualType &DeclRefType, const unsigned *const FunctionScopeIndexToStopAt) {
  // An init-capture is notionally from the context surrounding its
  // declaration, but its parent DC is the lambda class.
  DeclContext *VarDC = Var->getDeclContext();
  if (Var->isInitCapture())
    VarDC = VarDC->getParent();
  
  DeclContext *DC = CurContext;
  const unsigned MaxFunctionScopesIndex = FunctionScopeIndexToStopAt 
      ? *FunctionScopeIndexToStopAt : FunctionScopes.size() - 1;  
  // We need to sync up the Declaration Context with the
  // FunctionScopeIndexToStopAt
  if (FunctionScopeIndexToStopAt) {
    unsigned FSIndex = FunctionScopes.size() - 1;
    while (FSIndex != MaxFunctionScopesIndex) {
      DC = getLambdaAwareParentOfDeclContext(DC);
      --FSIndex;
    }
  }

  
  // If the variable is declared in the current context, there is no need to
  // capture it.
  if (VarDC == DC) return true;

  // Capture global variables if it is required to use private copy of this
  // variable.
  bool IsGlobal = !Var->hasLocalStorage();
  if (IsGlobal && !(LangOpts.OpenMP && IsOpenMPCapturedDecl(Var)))
    return true;

  // Walk up the stack to determine whether we can capture the variable,
  // performing the "simple" checks that don't depend on type. We stop when
  // we've either hit the declared scope of the variable or find an existing
  // capture of that variable.  We start from the innermost capturing-entity
  // (the DC) and ensure that all intervening capturing-entities 
  // (blocks/lambdas etc.) between the innermost capturer and the variable`s
  // declcontext can either capture the variable or have already captured
  // the variable.
  CaptureType = Var->getType();
  DeclRefType = CaptureType.getNonReferenceType();
  bool Nested = false;
  bool Explicit = (Kind != TryCapture_Implicit);
  unsigned FunctionScopesIndex = MaxFunctionScopesIndex;
  unsigned OpenMPLevel = 0;
  do {
    // Only block literals, captured statements, and lambda expressions can
    // capture; other scopes don't work.
    DeclContext *ParentDC = getParentOfCapturingContextOrNull(DC, Var, 
                                                              ExprLoc, 
                                                              BuildAndDiagnose,
                                                              *this);
    // We need to check for the parent *first* because, if we *have*
    // private-captured a global variable, we need to recursively capture it in
    // intermediate blocks, lambdas, etc.
    if (!ParentDC) {
      if (IsGlobal) {
        FunctionScopesIndex = MaxFunctionScopesIndex - 1;
        break;
      }
      return true;
    }

    FunctionScopeInfo  *FSI = FunctionScopes[FunctionScopesIndex];
    CapturingScopeInfo *CSI = cast<CapturingScopeInfo>(FSI);


    // Check whether we've already captured it.
    if (isVariableAlreadyCapturedInScopeInfo(CSI, Var, Nested, CaptureType, 
                                             DeclRefType)) 
      break;
    // If we are instantiating a generic lambda call operator body, 
    // we do not want to capture new variables.  What was captured
    // during either a lambdas transformation or initial parsing
    // should be used. 
    if (isGenericLambdaCallOperatorSpecialization(DC)) {
      if (BuildAndDiagnose) {
        LambdaScopeInfo *LSI = cast<LambdaScopeInfo>(CSI);   
        if (LSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_None) {
          Diag(ExprLoc, diag::err_lambda_impcap) << Var->getDeclName();
          Diag(Var->getLocation(), diag::note_previous_decl) 
             << Var->getDeclName();
          Diag(LSI->Lambda->getLocStart(), diag::note_lambda_decl);          
        } else
          diagnoseUncapturableValueReference(*this, ExprLoc, Var, DC);
      }
      return true;
    }
    // Certain capturing entities (lambdas, blocks etc.) are not allowed to capture 
    // certain types of variables (unnamed, variably modified types etc.)
    // so check for eligibility.
    if (!isVariableCapturable(CSI, Var, ExprLoc, BuildAndDiagnose, *this))
       return true;

    // Try to capture variable-length arrays types.
    if (Var->getType()->isVariablyModifiedType()) {
      // We're going to walk down into the type and look for VLA
      // expressions.
      QualType QTy = Var->getType();
      if (ParmVarDecl *PVD = dyn_cast_or_null<ParmVarDecl>(Var))
        QTy = PVD->getOriginalType();
      do {
        const Type *Ty = QTy.getTypePtr();
        switch (Ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
          QTy = QualType();
          break;
        // These types are never variably-modified.
        case Type::Builtin:
        case Type::Complex:
        case Type::Vector:
        case Type::ExtVector:
        case Type::Record:
        case Type::Enum:
        case Type::Elaborated:
        case Type::TemplateSpecialization:
        case Type::ObjCObject:
        case Type::ObjCInterface:
        case Type::ObjCObjectPointer:
        case Type::Pipe:
          llvm_unreachable("type class is never variably-modified!");
        case Type::Adjusted:
          QTy = cast<AdjustedType>(Ty)->getOriginalType();
          break;
        case Type::Decayed:
          QTy = cast<DecayedType>(Ty)->getPointeeType();
          break;
        case Type::Pointer:
          QTy = cast<PointerType>(Ty)->getPointeeType();
          break;
        case Type::BlockPointer:
          QTy = cast<BlockPointerType>(Ty)->getPointeeType();
          break;
        case Type::LValueReference:
        case Type::RValueReference:
          QTy = cast<ReferenceType>(Ty)->getPointeeType();
          break;
        case Type::MemberPointer:
          QTy = cast<MemberPointerType>(Ty)->getPointeeType();
          break;
        case Type::ConstantArray:
        case Type::IncompleteArray:
          // Losing element qualification here is fine.
          QTy = cast<ArrayType>(Ty)->getElementType();
          break;
        case Type::VariableArray: {
          // Losing element qualification here is fine.
          const VariableArrayType *VAT = cast<VariableArrayType>(Ty);

          // Unknown size indication requires no size computation.
          // Otherwise, evaluate and record it.
          if (auto Size = VAT->getSizeExpr()) {
            if (!CSI->isVLATypeCaptured(VAT)) {
              RecordDecl *CapRecord = nullptr;
              if (auto LSI = dyn_cast<LambdaScopeInfo>(CSI)) {
                CapRecord = LSI->Lambda;
              } else if (auto CRSI = dyn_cast<CapturedRegionScopeInfo>(CSI)) {
                CapRecord = CRSI->TheRecordDecl;
              }
              if (CapRecord) {
                auto ExprLoc = Size->getExprLoc();
                auto SizeType = Context.getSizeType();
                // Build the non-static data member.
                auto Field = FieldDecl::Create(
                    Context, CapRecord, ExprLoc, ExprLoc,
                    /*Id*/ nullptr, SizeType, /*TInfo*/ nullptr,
                    /*BW*/ nullptr, /*Mutable*/ false,
                    /*InitStyle*/ ICIS_NoInit);
                Field->setImplicit(true);
                Field->setAccess(AS_private);
                Field->setCapturedVLAType(VAT);
                CapRecord->addDecl(Field);

                CSI->addVLATypeCapture(ExprLoc, SizeType);
              }
            }
          }
          QTy = VAT->getElementType();
          break;
        }
        case Type::FunctionProto:
        case Type::FunctionNoProto:
          QTy = cast<FunctionType>(Ty)->getReturnType();
          break;
        case Type::Paren:
        case Type::TypeOf:
        case Type::UnaryTransform:
        case Type::Attributed:
        case Type::SubstTemplateTypeParm:
        case Type::PackExpansion:
          // Keep walking after single level desugaring.
          QTy = QTy.getSingleStepDesugaredType(getASTContext());
          break;
        case Type::Typedef:
          QTy = cast<TypedefType>(Ty)->desugar();
          break;
        case Type::Decltype:
          QTy = cast<DecltypeType>(Ty)->desugar();
          break;
        case Type::Auto:
          QTy = cast<AutoType>(Ty)->getDeducedType();
          break;
        case Type::TypeOfExpr:
          QTy = cast<TypeOfExprType>(Ty)->getUnderlyingExpr()->getType();
          break;
        case Type::Atomic:
          QTy = cast<AtomicType>(Ty)->getValueType();
          break;
        }
      } while (!QTy.isNull() && QTy->isVariablyModifiedType());
    }

    if (getLangOpts().OpenMP) {
      if (auto *RSI = dyn_cast<CapturedRegionScopeInfo>(CSI)) {
        // OpenMP private variables should not be captured in outer scope, so
        // just break here. Similarly, global variables that are captured in a
        // target region should not be captured outside the scope of the region.
        if (RSI->CapRegionKind == CR_OpenMP) {
          auto isTargetCap = isOpenMPTargetCapturedDecl(Var, OpenMPLevel);
          // When we detect target captures we are looking from inside the
          // target region, therefore we need to propagate the capture from the
          // enclosing region. Therefore, the capture is not initially nested.
          if (isTargetCap)
            FunctionScopesIndex--;

          if (isTargetCap || isOpenMPPrivateDecl(Var, OpenMPLevel)) {
            Nested = !isTargetCap;
            DeclRefType = DeclRefType.getUnqualifiedType();
            CaptureType = Context.getLValueReferenceType(DeclRefType);
            break;
          }
          ++OpenMPLevel;
        }
      }
    }
    if (CSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_None && !Explicit) {
      // No capture-default, and this is not an explicit capture 
      // so cannot capture this variable.  
      if (BuildAndDiagnose) {
        Diag(ExprLoc, diag::err_lambda_impcap) << Var->getDeclName();
        Diag(Var->getLocation(), diag::note_previous_decl) 
          << Var->getDeclName();
        Diag(cast<LambdaScopeInfo>(CSI)->Lambda->getLocStart(),
             diag::note_lambda_decl);
        // FIXME: If we error out because an outer lambda can not implicitly
        // capture a variable that an inner lambda explicitly captures, we
        // should have the inner lambda do the explicit capture - because
        // it makes for cleaner diagnostics later.  This would purely be done
        // so that the diagnostic does not misleadingly claim that a variable 
        // can not be captured by a lambda implicitly even though it is captured 
        // explicitly.  Suggestion:
        //  - create const bool VariableCaptureWasInitiallyExplicit = Explicit 
        //    at the function head
        //  - cache the StartingDeclContext - this must be a lambda 
        //  - captureInLambda in the innermost lambda the variable.
      }
      return true;
    }

    FunctionScopesIndex--;
    DC = ParentDC;
    Explicit = false;
  } while (!VarDC->Equals(DC));

  // Walk back down the scope stack, (e.g. from outer lambda to inner lambda)
  // computing the type of the capture at each step, checking type-specific 
  // requirements, and adding captures if requested. 
  // If the variable had already been captured previously, we start capturing 
  // at the lambda nested within that one.   
  for (unsigned I = ++FunctionScopesIndex, N = MaxFunctionScopesIndex + 1; I != N; 
       ++I) {
    CapturingScopeInfo *CSI = cast<CapturingScopeInfo>(FunctionScopes[I]);
    
    if (BlockScopeInfo *BSI = dyn_cast<BlockScopeInfo>(CSI)) {
      if (!captureInBlock(BSI, Var, ExprLoc, 
                          BuildAndDiagnose, CaptureType, 
                          DeclRefType, Nested, *this))
        return true;
      Nested = true;
    } else if (CapturedRegionScopeInfo *RSI = dyn_cast<CapturedRegionScopeInfo>(CSI)) {
      if (!captureInCapturedRegion(RSI, Var, ExprLoc, 
                                   BuildAndDiagnose, CaptureType, 
                                   DeclRefType, Nested, *this))
        return true;
      Nested = true;
    } else {
      LambdaScopeInfo *LSI = cast<LambdaScopeInfo>(CSI);
      if (!captureInLambda(LSI, Var, ExprLoc, 
                           BuildAndDiagnose, CaptureType, 
                           DeclRefType, Nested, Kind, EllipsisLoc, 
                            /*IsTopScope*/I == N - 1, *this))
        return true;
      Nested = true;
    }
  }
  return false;
}

bool Sema::tryCaptureVariable(VarDecl *Var, SourceLocation Loc,
                              TryCaptureKind Kind, SourceLocation EllipsisLoc) {  
  QualType CaptureType;
  QualType DeclRefType;
  return tryCaptureVariable(Var, Loc, Kind, EllipsisLoc,
                            /*BuildAndDiagnose=*/true, CaptureType,
                            DeclRefType, nullptr);
}

bool Sema::NeedToCaptureVariable(VarDecl *Var, SourceLocation Loc) {
  QualType CaptureType;
  QualType DeclRefType;
  return !tryCaptureVariable(Var, Loc, TryCapture_Implicit, SourceLocation(),
                             /*BuildAndDiagnose=*/false, CaptureType,
                             DeclRefType, nullptr);
}

QualType Sema::getCapturedDeclRefType(VarDecl *Var, SourceLocation Loc) {
  QualType CaptureType;
  QualType DeclRefType;
  
  // Determine whether we can capture this variable.
  if (tryCaptureVariable(Var, Loc, TryCapture_Implicit, SourceLocation(),
                         /*BuildAndDiagnose=*/false, CaptureType, 
                         DeclRefType, nullptr))
    return QualType();

  return DeclRefType;
}



// If either the type of the variable or the initializer is dependent, 
// return false. Otherwise, determine whether the variable is a constant
// expression. Use this if you need to know if a variable that might or
// might not be dependent is truly a constant expression.
static inline bool IsVariableNonDependentAndAConstantExpression(VarDecl *Var, 
    ASTContext &Context) {
 
  if (Var->getType()->isDependentType()) 
    return false;
  const VarDecl *DefVD = nullptr;
  Var->getAnyInitializer(DefVD);
  if (!DefVD) 
    return false;
  EvaluatedStmt *Eval = DefVD->ensureEvaluatedStmt();
  Expr *Init = cast<Expr>(Eval->Value);
  if (Init->isValueDependent()) 
    return false;
  return IsVariableAConstantExpression(Var, Context); 
}


void Sema::UpdateMarkingForLValueToRValue(Expr *E) {
  // Per C++11 [basic.def.odr], a variable is odr-used "unless it is 
  // an object that satisfies the requirements for appearing in a
  // constant expression (5.19) and the lvalue-to-rvalue conversion (4.1)
  // is immediately applied."  This function handles the lvalue-to-rvalue
  // conversion part.
  MaybeODRUseExprs.erase(E->IgnoreParens());
  
  // If we are in a lambda, check if this DeclRefExpr or MemberExpr refers
  // to a variable that is a constant expression, and if so, identify it as
  // a reference to a variable that does not involve an odr-use of that 
  // variable. 
  if (LambdaScopeInfo *LSI = getCurLambda()) {
    Expr *SansParensExpr = E->IgnoreParens();
    VarDecl *Var = nullptr;
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(SansParensExpr)) 
      Var = dyn_cast<VarDecl>(DRE->getFoundDecl());
    else if (MemberExpr *ME = dyn_cast<MemberExpr>(SansParensExpr))
      Var = dyn_cast<VarDecl>(ME->getMemberDecl());
    
    if (Var && IsVariableNonDependentAndAConstantExpression(Var, Context)) 
      LSI->markVariableExprAsNonODRUsed(SansParensExpr);    
  }
}

ExprResult Sema::ActOnConstantExpression(ExprResult Res) {
  Res = CorrectDelayedTyposInExpr(Res);

  if (!Res.isUsable())
    return Res;

  // If a constant-expression is a reference to a variable where we delay
  // deciding whether it is an odr-use, just assume we will apply the
  // lvalue-to-rvalue conversion.  In the one case where this doesn't happen
  // (a non-type template argument), we have special handling anyway.
  UpdateMarkingForLValueToRValue(Res.get());
  return Res;
}

void Sema::CleanupVarDeclMarking() {
  for (Expr *E : MaybeODRUseExprs) {
    VarDecl *Var;
    SourceLocation Loc;
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
      Var = cast<VarDecl>(DRE->getDecl());
      Loc = DRE->getLocation();
    } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
      Var = cast<VarDecl>(ME->getMemberDecl());
      Loc = ME->getMemberLoc();
    } else {
      llvm_unreachable("Unexpected expression");
    }

    MarkVarDeclODRUsed(Var, Loc, *this,
                       /*MaxFunctionScopeIndex Pointer*/ nullptr);
  }

  MaybeODRUseExprs.clear();
}


static void DoMarkVarDeclReferenced(Sema &SemaRef, SourceLocation Loc,
                                    VarDecl *Var, Expr *E) {
  assert((!E || isa<DeclRefExpr>(E) || isa<MemberExpr>(E)) &&
         "Invalid Expr argument to DoMarkVarDeclReferenced");
  Var->setReferenced();

  TemplateSpecializationKind TSK = Var->getTemplateSpecializationKind();
  bool MarkODRUsed = true;

  // If the context is not potentially evaluated, this is not an odr-use and
  // does not trigger instantiation.
  if (!IsPotentiallyEvaluatedContext(SemaRef)) {
    if (SemaRef.isUnevaluatedContext())
      return;

    // If we don't yet know whether this context is going to end up being an
    // evaluated context, and we're referencing a variable from an enclosing
    // scope, add a potential capture.
    //
    // FIXME: Is this necessary? These contexts are only used for default
    // arguments, where local variables can't be used.
    const bool RefersToEnclosingScope =
        (SemaRef.CurContext != Var->getDeclContext() &&
         Var->getDeclContext()->isFunctionOrMethod() && Var->hasLocalStorage());
    if (RefersToEnclosingScope) {
      if (LambdaScopeInfo *const LSI = SemaRef.getCurLambda()) {
        // If a variable could potentially be odr-used, defer marking it so
        // until we finish analyzing the full expression for any
        // lvalue-to-rvalue
        // or discarded value conversions that would obviate odr-use.
        // Add it to the list of potential captures that will be analyzed
        // later (ActOnFinishFullExpr) for eventual capture and odr-use marking
        // unless the variable is a reference that was initialized by a constant
        // expression (this will never need to be captured or odr-used).
        assert(E && "Capture variable should be used in an expression.");
        if (!Var->getType()->isReferenceType() ||
            !IsVariableNonDependentAndAConstantExpression(Var, SemaRef.Context))
          LSI->addPotentialCapture(E->IgnoreParens());
      }
    }

    if (!isTemplateInstantiation(TSK))
      return;

    // Instantiate, but do not mark as odr-used, variable templates.
    MarkODRUsed = false;
  }

  VarTemplateSpecializationDecl *VarSpec =
      dyn_cast<VarTemplateSpecializationDecl>(Var);
  assert(!isa<VarTemplatePartialSpecializationDecl>(Var) &&
         "Can't instantiate a partial template specialization.");

  // Perform implicit instantiation of static data members, static data member
  // templates of class templates, and variable template specializations. Delay
  // instantiations of variable templates, except for those that could be used
  // in a constant expression.
  if (isTemplateInstantiation(TSK)) {
    bool TryInstantiating = TSK == TSK_ImplicitInstantiation;

    if (TryInstantiating && !isa<VarTemplateSpecializationDecl>(Var)) {
      if (Var->getPointOfInstantiation().isInvalid()) {
        // This is a modification of an existing AST node. Notify listeners.
        if (ASTMutationListener *L = SemaRef.getASTMutationListener())
          L->StaticDataMemberInstantiated(Var);
      } else if (!Var->isUsableInConstantExpressions(SemaRef.Context))
        // Don't bother trying to instantiate it again, unless we might need
        // its initializer before we get to the end of the TU.
        TryInstantiating = false;
    }

    if (Var->getPointOfInstantiation().isInvalid())
      Var->setTemplateSpecializationKind(TSK, Loc);

    if (TryInstantiating) {
      SourceLocation PointOfInstantiation = Var->getPointOfInstantiation();
      bool InstantiationDependent = false;
      bool IsNonDependent =
          VarSpec ? !TemplateSpecializationType::anyDependentTemplateArguments(
                        VarSpec->getTemplateArgsInfo(), InstantiationDependent)
                  : true;

      // Do not instantiate specializations that are still type-dependent.
      if (IsNonDependent) {
        if (Var->isUsableInConstantExpressions(SemaRef.Context)) {
          // Do not defer instantiations of variables which could be used in a
          // constant expression.
          SemaRef.InstantiateVariableDefinition(PointOfInstantiation, Var);
        } else {
          SemaRef.PendingInstantiations
              .push_back(std::make_pair(Var, PointOfInstantiation));
        }
      }
    }
  }

  if(!MarkODRUsed) return;

  // Per C++11 [basic.def.odr], a variable is odr-used "unless it satisfies
  // the requirements for appearing in a constant expression (5.19) and, if
  // it is an object, the lvalue-to-rvalue conversion (4.1)
  // is immediately applied."  We check the first part here, and
  // Sema::UpdateMarkingForLValueToRValue deals with the second part.
  // Note that we use the C++11 definition everywhere because nothing in
  // C++03 depends on whether we get the C++03 version correct. The second
  // part does not apply to references, since they are not objects.
  if (E && IsVariableAConstantExpression(Var, SemaRef.Context)) {
    // A reference initialized by a constant expression can never be
    // odr-used, so simply ignore it.
    if (!Var->getType()->isReferenceType())
      SemaRef.MaybeODRUseExprs.insert(E);
  } else
    MarkVarDeclODRUsed(Var, Loc, SemaRef,
                       /*MaxFunctionScopeIndex ptr*/ nullptr);
}

/// \brief Mark a variable referenced, and check whether it is odr-used
/// (C++ [basic.def.odr]p2, C99 6.9p3).  Note that this should not be
/// used directly for normal expressions referring to VarDecl.
void Sema::MarkVariableReferenced(SourceLocation Loc, VarDecl *Var) {
  DoMarkVarDeclReferenced(*this, Loc, Var, nullptr);
}

static void MarkExprReferenced(Sema &SemaRef, SourceLocation Loc,
                               Decl *D, Expr *E, bool OdrUse) {
  if (VarDecl *Var = dyn_cast<VarDecl>(D)) {
    DoMarkVarDeclReferenced(SemaRef, Loc, Var, E);
    return;
  }

  SemaRef.MarkAnyDeclReferenced(Loc, D, OdrUse);

  // If this is a call to a method via a cast, also mark the method in the
  // derived class used in case codegen can devirtualize the call.
  const MemberExpr *ME = dyn_cast<MemberExpr>(E);
  if (!ME)
    return;
  CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(ME->getMemberDecl());
  if (!MD)
    return;
  // Only attempt to devirtualize if this is truly a virtual call.
  bool IsVirtualCall = MD->isVirtual() &&
                          ME->performsVirtualDispatch(SemaRef.getLangOpts());
  if (!IsVirtualCall)
    return;
  const Expr *Base = ME->getBase();
  const CXXRecordDecl *MostDerivedClassDecl = Base->getBestDynamicClassType();
  if (!MostDerivedClassDecl)
    return;
  CXXMethodDecl *DM = MD->getCorrespondingMethodInClass(MostDerivedClassDecl);
  if (!DM || DM->isPure())
    return;
  SemaRef.MarkAnyDeclReferenced(Loc, DM, OdrUse);
} 

/// \brief Perform reference-marking and odr-use handling for a DeclRefExpr.
void Sema::MarkDeclRefReferenced(DeclRefExpr *E) {
  // TODO: update this with DR# once a defect report is filed.
  // C++11 defect. The address of a pure member should not be an ODR use, even
  // if it's a qualified reference.
  bool OdrUse = true;
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(E->getDecl()))
    if (Method->isVirtual())
      OdrUse = false;
  MarkExprReferenced(*this, E->getLocation(), E->getDecl(), E, OdrUse);
}

/// \brief Perform reference-marking and odr-use handling for a MemberExpr.
void Sema::MarkMemberReferenced(MemberExpr *E) {
  // C++11 [basic.def.odr]p2:
  //   A non-overloaded function whose name appears as a potentially-evaluated
  //   expression or a member of a set of candidate functions, if selected by
  //   overload resolution when referred to from a potentially-evaluated
  //   expression, is odr-used, unless it is a pure virtual function and its
  //   name is not explicitly qualified.
  bool OdrUse = true;
  if (E->performsVirtualDispatch(getLangOpts())) {
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(E->getMemberDecl()))
      if (Method->isPure())
        OdrUse = false;
  }
  SourceLocation Loc = E->getMemberLoc().isValid() ?
                            E->getMemberLoc() : E->getLocStart();
  MarkExprReferenced(*this, Loc, E->getMemberDecl(), E, OdrUse);
}

/// \brief Perform marking for a reference to an arbitrary declaration.  It
/// marks the declaration referenced, and performs odr-use checking for
/// functions and variables. This method should not be used when building a
/// normal expression which refers to a variable.
void Sema::MarkAnyDeclReferenced(SourceLocation Loc, Decl *D, bool OdrUse) {
  if (OdrUse) {
    if (auto *VD = dyn_cast<VarDecl>(D)) {
      MarkVariableReferenced(Loc, VD);
      return;
    }
  }
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    MarkFunctionReferenced(Loc, FD, OdrUse);
    return;
  }
  D->setReferenced();
}

namespace {
  // Mark all of the declarations referenced
  // FIXME: Not fully implemented yet! We need to have a better understanding
  // of when we're entering
  class MarkReferencedDecls : public RecursiveASTVisitor<MarkReferencedDecls> {
    Sema &S;
    SourceLocation Loc;

  public:
    typedef RecursiveASTVisitor<MarkReferencedDecls> Inherited;

    MarkReferencedDecls(Sema &S, SourceLocation Loc) : S(S), Loc(Loc) { }

    bool TraverseTemplateArgument(const TemplateArgument &Arg);
    bool TraverseRecordType(RecordType *T);
  };
}

bool MarkReferencedDecls::TraverseTemplateArgument(
    const TemplateArgument &Arg) {
  if (Arg.getKind() == TemplateArgument::Declaration) {
    if (Decl *D = Arg.getAsDecl())
      S.MarkAnyDeclReferenced(Loc, D, true);
  }

  return Inherited::TraverseTemplateArgument(Arg);
}

bool MarkReferencedDecls::TraverseRecordType(RecordType *T) {
  if (ClassTemplateSpecializationDecl *Spec
                  = dyn_cast<ClassTemplateSpecializationDecl>(T->getDecl())) {
    const TemplateArgumentList &Args = Spec->getTemplateArgs();
    return TraverseTemplateArguments(Args.data(), Args.size());
  }

  return true;
}

void Sema::MarkDeclarationsReferencedInType(SourceLocation Loc, QualType T) {
  MarkReferencedDecls Marker(*this, Loc);
  Marker.TraverseType(Context.getCanonicalType(T));
}

namespace {
  /// \brief Helper class that marks all of the declarations referenced by
  /// potentially-evaluated subexpressions as "referenced".
  class EvaluatedExprMarker : public EvaluatedExprVisitor<EvaluatedExprMarker> {
    Sema &S;
    bool SkipLocalVariables;
    
  public:
    typedef EvaluatedExprVisitor<EvaluatedExprMarker> Inherited;
    
    EvaluatedExprMarker(Sema &S, bool SkipLocalVariables) 
      : Inherited(S.Context), S(S), SkipLocalVariables(SkipLocalVariables) { }
    
    void VisitDeclRefExpr(DeclRefExpr *E) {
      // If we were asked not to visit local variables, don't.
      if (SkipLocalVariables) {
        if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl()))
          if (VD->hasLocalStorage())
            return;
      }
      
      S.MarkDeclRefReferenced(E);
    }

    void VisitMemberExpr(MemberExpr *E) {
      S.MarkMemberReferenced(E);
      Inherited::VisitMemberExpr(E);
    }
    
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
      S.MarkFunctionReferenced(E->getLocStart(),
            const_cast<CXXDestructorDecl*>(E->getTemporary()->getDestructor()));
      Visit(E->getSubExpr());
    }
    
    void VisitCXXNewExpr(CXXNewExpr *E) {
      if (E->getOperatorNew())
        S.MarkFunctionReferenced(E->getLocStart(), E->getOperatorNew());
      if (E->getOperatorDelete())
        S.MarkFunctionReferenced(E->getLocStart(), E->getOperatorDelete());
      Inherited::VisitCXXNewExpr(E);
    }

    void VisitCXXDeleteExpr(CXXDeleteExpr *E) {
      if (E->getOperatorDelete())
        S.MarkFunctionReferenced(E->getLocStart(), E->getOperatorDelete());
      QualType Destroyed = S.Context.getBaseElementType(E->getDestroyedType());
      if (const RecordType *DestroyedRec = Destroyed->getAs<RecordType>()) {
        CXXRecordDecl *Record = cast<CXXRecordDecl>(DestroyedRec->getDecl());
        S.MarkFunctionReferenced(E->getLocStart(), 
                                    S.LookupDestructor(Record));
      }
      
      Inherited::VisitCXXDeleteExpr(E);
    }
    
    void VisitCXXConstructExpr(CXXConstructExpr *E) {
      S.MarkFunctionReferenced(E->getLocStart(), E->getConstructor());
      Inherited::VisitCXXConstructExpr(E);
    }
    
    void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) {
      Visit(E->getExpr());
    }

    void VisitImplicitCastExpr(ImplicitCastExpr *E) {
      Inherited::VisitImplicitCastExpr(E);

      if (E->getCastKind() == CK_LValueToRValue)
        S.UpdateMarkingForLValueToRValue(E->getSubExpr());
    }
  };
}

/// \brief Mark any declarations that appear within this expression or any
/// potentially-evaluated subexpressions as "referenced".
///
/// \param SkipLocalVariables If true, don't mark local variables as 
/// 'referenced'.
void Sema::MarkDeclarationsReferencedInExpr(Expr *E, 
                                            bool SkipLocalVariables) {
  EvaluatedExprMarker(*this, SkipLocalVariables).Visit(E);
}

/// \brief Emit a diagnostic that describes an effect on the run-time behavior
/// of the program being compiled.
///
/// This routine emits the given diagnostic when the code currently being
/// type-checked is "potentially evaluated", meaning that there is a
/// possibility that the code will actually be executable. Code in sizeof()
/// expressions, code used only during overload resolution, etc., are not
/// potentially evaluated. This routine will suppress such diagnostics or,
/// in the absolutely nutty case of potentially potentially evaluated
/// expressions (C++ typeid), queue the diagnostic to potentially emit it
/// later.
///
/// This routine should be used for all diagnostics that describe the run-time
/// behavior of a program, such as passing a non-POD value through an ellipsis.
/// Failure to do so will likely result in spurious diagnostics or failures
/// during overload resolution or within sizeof/alignof/typeof/typeid.
bool Sema::DiagRuntimeBehavior(SourceLocation Loc, const Stmt *Statement,
                               const PartialDiagnostic &PD) {
  switch (ExprEvalContexts.back().Context) {
  case Unevaluated:
  case UnevaluatedAbstract:
    // The argument will never be evaluated, so don't complain.
    break;

  case ConstantEvaluated:
    // Relevant diagnostics should be produced by constant evaluation.
    break;

  case PotentiallyEvaluated:
  case PotentiallyEvaluatedIfUsed:
    if (Statement && getCurFunctionOrMethodDecl()) {
      FunctionScopes.back()->PossiblyUnreachableDiags.
        push_back(sema::PossiblyUnreachableDiag(PD, Loc, Statement));
    }
    else
      Diag(Loc, PD);
      
    return true;
  }

  return false;
}

bool Sema::CheckCallReturnType(QualType ReturnType, SourceLocation Loc,
                               CallExpr *CE, FunctionDecl *FD) {
  if (ReturnType->isVoidType() || !ReturnType->isIncompleteType())
    return false;

  // If we're inside a decltype's expression, don't check for a valid return
  // type or construct temporaries until we know whether this is the last call.
  if (ExprEvalContexts.back().IsDecltype) {
    ExprEvalContexts.back().DelayedDecltypeCalls.push_back(CE);
    return false;
  }

  class CallReturnIncompleteDiagnoser : public TypeDiagnoser {
    FunctionDecl *FD;
    CallExpr *CE;
    
  public:
    CallReturnIncompleteDiagnoser(FunctionDecl *FD, CallExpr *CE)
      : FD(FD), CE(CE) { }

    void diagnose(Sema &S, SourceLocation Loc, QualType T) override {
      if (!FD) {
        S.Diag(Loc, diag::err_call_incomplete_return)
          << T << CE->getSourceRange();
        return;
      }
      
      S.Diag(Loc, diag::err_call_function_incomplete_return)
        << CE->getSourceRange() << FD->getDeclName() << T;
      S.Diag(FD->getLocation(), diag::note_entity_declared_at)
          << FD->getDeclName();
    }
  } Diagnoser(FD, CE);
  
  if (RequireCompleteType(Loc, ReturnType, Diagnoser))
    return true;

  return false;
}

// Diagnose the s/=/==/ and s/\|=/!=/ typos. Note that adding parentheses
// will prevent this condition from triggering, which is what we want.
void Sema::DiagnoseAssignmentAsCondition(Expr *E) {
  SourceLocation Loc;

  unsigned diagnostic = diag::warn_condition_is_assignment;
  bool IsOrAssign = false;

  if (BinaryOperator *Op = dyn_cast<BinaryOperator>(E)) {
    if (Op->getOpcode() != BO_Assign && Op->getOpcode() != BO_OrAssign)
      return;

    IsOrAssign = Op->getOpcode() == BO_OrAssign;

    // Greylist some idioms by putting them into a warning subcategory.
    if (ObjCMessageExpr *ME
          = dyn_cast<ObjCMessageExpr>(Op->getRHS()->IgnoreParenCasts())) {
      Selector Sel = ME->getSelector();

      // self = [<foo> init...]
      if (isSelfExpr(Op->getLHS()) && ME->getMethodFamily() == OMF_init)
        diagnostic = diag::warn_condition_is_idiomatic_assignment;

      // <foo> = [<bar> nextObject]
      else if (Sel.isUnarySelector() && Sel.getNameForSlot(0) == "nextObject")
        diagnostic = diag::warn_condition_is_idiomatic_assignment;
    }

    Loc = Op->getOperatorLoc();
  } else if (CXXOperatorCallExpr *Op = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (Op->getOperator() != OO_Equal && Op->getOperator() != OO_PipeEqual)
      return;

    IsOrAssign = Op->getOperator() == OO_PipeEqual;
    Loc = Op->getOperatorLoc();
  } else if (PseudoObjectExpr *POE = dyn_cast<PseudoObjectExpr>(E))
    return DiagnoseAssignmentAsCondition(POE->getSyntacticForm());
  else {
    // Not an assignment.
    return;
  }

  Diag(Loc, diagnostic) << E->getSourceRange();

  SourceLocation Open = E->getLocStart();
  SourceLocation Close = getLocForEndOfToken(E->getSourceRange().getEnd());
  Diag(Loc, diag::note_condition_assign_silence)
        << FixItHint::CreateInsertion(Open, "(")
        << FixItHint::CreateInsertion(Close, ")");

  if (IsOrAssign)
    Diag(Loc, diag::note_condition_or_assign_to_comparison)
      << FixItHint::CreateReplacement(Loc, "!=");
  else
    Diag(Loc, diag::note_condition_assign_to_comparison)
      << FixItHint::CreateReplacement(Loc, "==");
}

/// \brief Redundant parentheses over an equality comparison can indicate
/// that the user intended an assignment used as condition.
void Sema::DiagnoseEqualityWithExtraParens(ParenExpr *ParenE) {
  // Don't warn if the parens came from a macro.
  SourceLocation parenLoc = ParenE->getLocStart();
  if (parenLoc.isInvalid() || parenLoc.isMacroID())
    return;
  // Don't warn for dependent expressions.
  if (ParenE->isTypeDependent())
    return;

  Expr *E = ParenE->IgnoreParens();

  if (BinaryOperator *opE = dyn_cast<BinaryOperator>(E))
    if (opE->getOpcode() == BO_EQ &&
        opE->getLHS()->IgnoreParenImpCasts()->isModifiableLvalue(Context)
                                                           == Expr::MLV_Valid) {
      SourceLocation Loc = opE->getOperatorLoc();
      
      Diag(Loc, diag::warn_equality_with_extra_parens) << E->getSourceRange();
      SourceRange ParenERange = ParenE->getSourceRange();
      Diag(Loc, diag::note_equality_comparison_silence)
        << FixItHint::CreateRemoval(ParenERange.getBegin())
        << FixItHint::CreateRemoval(ParenERange.getEnd());
      Diag(Loc, diag::note_equality_comparison_to_assign)
        << FixItHint::CreateReplacement(Loc, "=");
    }
}

ExprResult Sema::CheckBooleanCondition(Expr *E, SourceLocation Loc) {
  DiagnoseAssignmentAsCondition(E);
  if (ParenExpr *parenE = dyn_cast<ParenExpr>(E))
    DiagnoseEqualityWithExtraParens(parenE);

  ExprResult result = CheckPlaceholderExpr(E);
  if (result.isInvalid()) return ExprError();
  E = result.get();

  if (!E->isTypeDependent()) {
    if (getLangOpts().CPlusPlus)
      return CheckCXXBooleanCondition(E); // C++ 6.4p4

    ExprResult ERes = DefaultFunctionArrayLvalueConversion(E);
    if (ERes.isInvalid())
      return ExprError();
    E = ERes.get();

    QualType T = E->getType();
    if (!T->isScalarType()) { // C99 6.8.4.1p1
      Diag(Loc, diag::err_typecheck_statement_requires_scalar)
        << T << E->getSourceRange();
      return ExprError();
    }
    CheckBoolLikeConversion(E, Loc);
  }

  return E;
}

ExprResult Sema::ActOnBooleanCondition(Scope *S, SourceLocation Loc,
                                       Expr *SubExpr) {
  if (!SubExpr)
    return ExprError();

  return CheckBooleanCondition(SubExpr, Loc);
}

namespace {
  /// A visitor for rebuilding a call to an __unknown_any expression
  /// to have an appropriate type.
  struct RebuildUnknownAnyFunction
    : StmtVisitor<RebuildUnknownAnyFunction, ExprResult> {

    Sema &S;

    RebuildUnknownAnyFunction(Sema &S) : S(S) {}

    ExprResult VisitStmt(Stmt *S) {
      llvm_unreachable("unexpected statement!");
    }

    ExprResult VisitExpr(Expr *E) {
      S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_call)
        << E->getSourceRange();
      return ExprError();
    }

    /// Rebuild an expression which simply semantically wraps another
    /// expression which it shares the type and value kind of.
    template <class T> ExprResult rebuildSugarExpr(T *E) {
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();

      Expr *SubExpr = SubResult.get();
      E->setSubExpr(SubExpr);
      E->setType(SubExpr->getType());
      E->setValueKind(SubExpr->getValueKind());
      assert(E->getObjectKind() == OK_Ordinary);
      return E;
    }

    ExprResult VisitParenExpr(ParenExpr *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryExtension(UnaryOperator *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryAddrOf(UnaryOperator *E) {
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();

      Expr *SubExpr = SubResult.get();
      E->setSubExpr(SubExpr);
      E->setType(S.Context.getPointerType(SubExpr->getType()));
      assert(E->getValueKind() == VK_RValue);
      assert(E->getObjectKind() == OK_Ordinary);
      return E;
    }

    ExprResult resolveDecl(Expr *E, ValueDecl *VD) {
      if (!isa<FunctionDecl>(VD)) return VisitExpr(E);

      E->setType(VD->getType());

      assert(E->getValueKind() == VK_RValue);
      if (S.getLangOpts().CPlusPlus &&
          !(isa<CXXMethodDecl>(VD) &&
            cast<CXXMethodDecl>(VD)->isInstance()))
        E->setValueKind(VK_LValue);

      return E;
    }

    ExprResult VisitMemberExpr(MemberExpr *E) {
      return resolveDecl(E, E->getMemberDecl());
    }

    ExprResult VisitDeclRefExpr(DeclRefExpr *E) {
      return resolveDecl(E, E->getDecl());
    }
  };
}

/// Given a function expression of unknown-any type, try to rebuild it
/// to have a function type.
static ExprResult rebuildUnknownAnyFunction(Sema &S, Expr *FunctionExpr) {
  ExprResult Result = RebuildUnknownAnyFunction(S).Visit(FunctionExpr);
  if (Result.isInvalid()) return ExprError();
  return S.DefaultFunctionArrayConversion(Result.get());
}

namespace {
  /// A visitor for rebuilding an expression of type __unknown_anytype
  /// into one which resolves the type directly on the referring
  /// expression.  Strict preservation of the original source
  /// structure is not a goal.
  struct RebuildUnknownAnyExpr
    : StmtVisitor<RebuildUnknownAnyExpr, ExprResult> {

    Sema &S;

    /// The current destination type.
    QualType DestType;

    RebuildUnknownAnyExpr(Sema &S, QualType CastType)
      : S(S), DestType(CastType) {}

    ExprResult VisitStmt(Stmt *S) {
      llvm_unreachable("unexpected statement!");
    }

    ExprResult VisitExpr(Expr *E) {
      S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_expr)
        << E->getSourceRange();
      return ExprError();
    }

    ExprResult VisitCallExpr(CallExpr *E);
    ExprResult VisitObjCMessageExpr(ObjCMessageExpr *E);

    /// Rebuild an expression which simply semantically wraps another
    /// expression which it shares the type and value kind of.
    template <class T> ExprResult rebuildSugarExpr(T *E) {
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();
      Expr *SubExpr = SubResult.get();
      E->setSubExpr(SubExpr);
      E->setType(SubExpr->getType());
      E->setValueKind(SubExpr->getValueKind());
      assert(E->getObjectKind() == OK_Ordinary);
      return E;
    }

    ExprResult VisitParenExpr(ParenExpr *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryExtension(UnaryOperator *E) {
      return rebuildSugarExpr(E);
    }

    ExprResult VisitUnaryAddrOf(UnaryOperator *E) {
      const PointerType *Ptr = DestType->getAs<PointerType>();
      if (!Ptr) {
        S.Diag(E->getOperatorLoc(), diag::err_unknown_any_addrof)
          << E->getSourceRange();
        return ExprError();
      }
      assert(E->getValueKind() == VK_RValue);
      assert(E->getObjectKind() == OK_Ordinary);
      E->setType(DestType);

      // Build the sub-expression as if it were an object of the pointee type.
      DestType = Ptr->getPointeeType();
      ExprResult SubResult = Visit(E->getSubExpr());
      if (SubResult.isInvalid()) return ExprError();
      E->setSubExpr(SubResult.get());
      return E;
    }

    ExprResult VisitImplicitCastExpr(ImplicitCastExpr *E);

    ExprResult resolveDecl(Expr *E, ValueDecl *VD);

    ExprResult VisitMemberExpr(MemberExpr *E) {
      return resolveDecl(E, E->getMemberDecl());
    }

    ExprResult VisitDeclRefExpr(DeclRefExpr *E) {
      return resolveDecl(E, E->getDecl());
    }
  };
}

/// Rebuilds a call expression which yielded __unknown_anytype.
ExprResult RebuildUnknownAnyExpr::VisitCallExpr(CallExpr *E) {
  Expr *CalleeExpr = E->getCallee();

  enum FnKind {
    FK_MemberFunction,
    FK_FunctionPointer,
    FK_BlockPointer
  };

  FnKind Kind;
  QualType CalleeType = CalleeExpr->getType();
  if (CalleeType == S.Context.BoundMemberTy) {
    assert(isa<CXXMemberCallExpr>(E) || isa<CXXOperatorCallExpr>(E));
    Kind = FK_MemberFunction;
    CalleeType = Expr::findBoundMemberType(CalleeExpr);
  } else if (const PointerType *Ptr = CalleeType->getAs<PointerType>()) {
    CalleeType = Ptr->getPointeeType();
    Kind = FK_FunctionPointer;
  } else {
    CalleeType = CalleeType->castAs<BlockPointerType>()->getPointeeType();
    Kind = FK_BlockPointer;
  }
  const FunctionType *FnType = CalleeType->castAs<FunctionType>();

  // Verify that this is a legal result type of a function.
  if (DestType->isArrayType() || DestType->isFunctionType()) {
    unsigned diagID = diag::err_func_returning_array_function;
    if (Kind == FK_BlockPointer)
      diagID = diag::err_block_returning_array_function;

    S.Diag(E->getExprLoc(), diagID)
      << DestType->isFunctionType() << DestType;
    return ExprError();
  }

  // Otherwise, go ahead and set DestType as the call's result.
  E->setType(DestType.getNonLValueExprType(S.Context));
  E->setValueKind(Expr::getValueKindForType(DestType));
  assert(E->getObjectKind() == OK_Ordinary);

  // Rebuild the function type, replacing the result type with DestType.
  const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FnType);
  if (Proto) {
    // __unknown_anytype(...) is a special case used by the debugger when
    // it has no idea what a function's signature is.
    //
    // We want to build this call essentially under the K&R
    // unprototyped rules, but making a FunctionNoProtoType in C++
    // would foul up all sorts of assumptions.  However, we cannot
    // simply pass all arguments as variadic arguments, nor can we
    // portably just call the function under a non-variadic type; see
    // the comment on IR-gen's TargetInfo::isNoProtoCallVariadic.
    // However, it turns out that in practice it is generally safe to
    // call a function declared as "A foo(B,C,D);" under the prototype
    // "A foo(B,C,D,...);".  The only known exception is with the
    // Windows ABI, where any variadic function is implicitly cdecl
    // regardless of its normal CC.  Therefore we change the parameter
    // types to match the types of the arguments.
    //
    // This is a hack, but it is far superior to moving the
    // corresponding target-specific code from IR-gen to Sema/AST.

    ArrayRef<QualType> ParamTypes = Proto->getParamTypes();
    SmallVector<QualType, 8> ArgTypes;
    if (ParamTypes.empty() && Proto->isVariadic()) { // the special case
      ArgTypes.reserve(E->getNumArgs());
      for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
        Expr *Arg = E->getArg(i);
        QualType ArgType = Arg->getType();
        if (E->isLValue()) {
          ArgType = S.Context.getLValueReferenceType(ArgType);
        } else if (E->isXValue()) {
          ArgType = S.Context.getRValueReferenceType(ArgType);
        }
        ArgTypes.push_back(ArgType);
      }
      ParamTypes = ArgTypes;
    }
    DestType = S.Context.getFunctionType(DestType, ParamTypes,
                                         Proto->getExtProtoInfo());
  } else {
    DestType = S.Context.getFunctionNoProtoType(DestType,
                                                FnType->getExtInfo());
  }

  // Rebuild the appropriate pointer-to-function type.
  switch (Kind) { 
  case FK_MemberFunction:
    // Nothing to do.
    break;

  case FK_FunctionPointer:
    DestType = S.Context.getPointerType(DestType);
    break;

  case FK_BlockPointer:
    DestType = S.Context.getBlockPointerType(DestType);
    break;
  }

  // Finally, we can recurse.
  ExprResult CalleeResult = Visit(CalleeExpr);
  if (!CalleeResult.isUsable()) return ExprError();
  E->setCallee(CalleeResult.get());

  // Bind a temporary if necessary.
  return S.MaybeBindToTemporary(E);
}

ExprResult RebuildUnknownAnyExpr::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  // Verify that this is a legal result type of a call.
  if (DestType->isArrayType() || DestType->isFunctionType()) {
    S.Diag(E->getExprLoc(), diag::err_func_returning_array_function)
      << DestType->isFunctionType() << DestType;
    return ExprError();
  }

  // Rewrite the method result type if available.
  if (ObjCMethodDecl *Method = E->getMethodDecl()) {
    assert(Method->getReturnType() == S.Context.UnknownAnyTy);
    Method->setReturnType(DestType);
  }

  // Change the type of the message.
  E->setType(DestType.getNonReferenceType());
  E->setValueKind(Expr::getValueKindForType(DestType));

  return S.MaybeBindToTemporary(E);
}

ExprResult RebuildUnknownAnyExpr::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  // The only case we should ever see here is a function-to-pointer decay.
  if (E->getCastKind() == CK_FunctionToPointerDecay) {
    assert(E->getValueKind() == VK_RValue);
    assert(E->getObjectKind() == OK_Ordinary);
  
    E->setType(DestType);
  
    // Rebuild the sub-expression as the pointee (function) type.
    DestType = DestType->castAs<PointerType>()->getPointeeType();
  
    ExprResult Result = Visit(E->getSubExpr());
    if (!Result.isUsable()) return ExprError();
  
    E->setSubExpr(Result.get());
    return E;
  } else if (E->getCastKind() == CK_LValueToRValue) {
    assert(E->getValueKind() == VK_RValue);
    assert(E->getObjectKind() == OK_Ordinary);

    assert(isa<BlockPointerType>(E->getType()));

    E->setType(DestType);

    // The sub-expression has to be a lvalue reference, so rebuild it as such.
    DestType = S.Context.getLValueReferenceType(DestType);

    ExprResult Result = Visit(E->getSubExpr());
    if (!Result.isUsable()) return ExprError();

    E->setSubExpr(Result.get());
    return E;
  } else {
    llvm_unreachable("Unhandled cast type!");
  }
}

ExprResult RebuildUnknownAnyExpr::resolveDecl(Expr *E, ValueDecl *VD) {
  ExprValueKind ValueKind = VK_LValue;
  QualType Type = DestType;

  // We know how to make this work for certain kinds of decls:

  //  - functions
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(VD)) {
    if (const PointerType *Ptr = Type->getAs<PointerType>()) {
      DestType = Ptr->getPointeeType();
      ExprResult Result = resolveDecl(E, VD);
      if (Result.isInvalid()) return ExprError();
      return S.ImpCastExprToType(Result.get(), Type,
                                 CK_FunctionToPointerDecay, VK_RValue);
    }

    if (!Type->isFunctionType()) {
      S.Diag(E->getExprLoc(), diag::err_unknown_any_function)
        << VD << E->getSourceRange();
      return ExprError();
    }
    if (const FunctionProtoType *FT = Type->getAs<FunctionProtoType>()) {
      // We must match the FunctionDecl's type to the hack introduced in
      // RebuildUnknownAnyExpr::VisitCallExpr to vararg functions of unknown
      // type. See the lengthy commentary in that routine.
      QualType FDT = FD->getType();
      const FunctionType *FnType = FDT->castAs<FunctionType>();
      const FunctionProtoType *Proto = dyn_cast_or_null<FunctionProtoType>(FnType);
      DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E);
      if (DRE && Proto && Proto->getParamTypes().empty() && Proto->isVariadic()) {
        SourceLocation Loc = FD->getLocation();
        FunctionDecl *NewFD = FunctionDecl::Create(FD->getASTContext(),
                                      FD->getDeclContext(),
                                      Loc, Loc, FD->getNameInfo().getName(),
                                      DestType, FD->getTypeSourceInfo(),
                                      SC_None, false/*isInlineSpecified*/,
                                      FD->hasPrototype(),
                                      false/*isConstexprSpecified*/);
          
        if (FD->getQualifier())
          NewFD->setQualifierInfo(FD->getQualifierLoc());

        SmallVector<ParmVarDecl*, 16> Params;
        for (const auto &AI : FT->param_types()) {
          ParmVarDecl *Param =
            S.BuildParmVarDeclForTypedef(FD, Loc, AI);
          Param->setScopeInfo(0, Params.size());
          Params.push_back(Param);
        }
        NewFD->setParams(Params);
        DRE->setDecl(NewFD);
        VD = DRE->getDecl();
      }
    }

    if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD))
      if (MD->isInstance()) {
        ValueKind = VK_RValue;
        Type = S.Context.BoundMemberTy;
      }

    // Function references aren't l-values in C.
    if (!S.getLangOpts().CPlusPlus)
      ValueKind = VK_RValue;

  //  - variables
  } else if (isa<VarDecl>(VD)) {
    if (const ReferenceType *RefTy = Type->getAs<ReferenceType>()) {
      Type = RefTy->getPointeeType();
    } else if (Type->isFunctionType()) {
      S.Diag(E->getExprLoc(), diag::err_unknown_any_var_function_type)
        << VD << E->getSourceRange();
      return ExprError();
    }

  //  - nothing else
  } else {
    S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_decl)
      << VD << E->getSourceRange();
    return ExprError();
  }

  // Modifying the declaration like this is friendly to IR-gen but
  // also really dangerous.
  VD->setType(DestType);
  E->setType(Type);
  E->setValueKind(ValueKind);
  return E;
}

/// Check a cast of an unknown-any type.  We intentionally only
/// trigger this for C-style casts.
ExprResult Sema::checkUnknownAnyCast(SourceRange TypeRange, QualType CastType,
                                     Expr *CastExpr, CastKind &CastKind,
                                     ExprValueKind &VK, CXXCastPath &Path) {
  // Rewrite the casted expression from scratch.
  ExprResult result = RebuildUnknownAnyExpr(*this, CastType).Visit(CastExpr);
  if (!result.isUsable()) return ExprError();

  CastExpr = result.get();
  VK = CastExpr->getValueKind();
  CastKind = CK_NoOp;

  return CastExpr;
}

ExprResult Sema::forceUnknownAnyToType(Expr *E, QualType ToType) {
  return RebuildUnknownAnyExpr(*this, ToType).Visit(E);
}

ExprResult Sema::checkUnknownAnyArg(SourceLocation callLoc,
                                    Expr *arg, QualType &paramType) {
  // If the syntactic form of the argument is not an explicit cast of
  // any sort, just do default argument promotion.
  ExplicitCastExpr *castArg = dyn_cast<ExplicitCastExpr>(arg->IgnoreParens());
  if (!castArg) {
    ExprResult result = DefaultArgumentPromotion(arg);
    if (result.isInvalid()) return ExprError();
    paramType = result.get()->getType();
    return result;
  }

  // Otherwise, use the type that was written in the explicit cast.
  assert(!arg->hasPlaceholderType());
  paramType = castArg->getTypeAsWritten();

  // Copy-initialize a parameter of that type.
  InitializedEntity entity =
    InitializedEntity::InitializeParameter(Context, paramType,
                                           /*consumed*/ false);
  return PerformCopyInitialization(entity, callLoc, arg);
}

static ExprResult diagnoseUnknownAnyExpr(Sema &S, Expr *E) {
  Expr *orig = E;
  unsigned diagID = diag::err_uncasted_use_of_unknown_any;
  while (true) {
    E = E->IgnoreParenImpCasts();
    if (CallExpr *call = dyn_cast<CallExpr>(E)) {
      E = call->getCallee();
      diagID = diag::err_uncasted_call_of_unknown_any;
    } else {
      break;
    }
  }

  SourceLocation loc;
  NamedDecl *d;
  if (DeclRefExpr *ref = dyn_cast<DeclRefExpr>(E)) {
    loc = ref->getLocation();
    d = ref->getDecl();
  } else if (MemberExpr *mem = dyn_cast<MemberExpr>(E)) {
    loc = mem->getMemberLoc();
    d = mem->getMemberDecl();
  } else if (ObjCMessageExpr *msg = dyn_cast<ObjCMessageExpr>(E)) {
    diagID = diag::err_uncasted_call_of_unknown_any;
    loc = msg->getSelectorStartLoc();
    d = msg->getMethodDecl();
    if (!d) {
      S.Diag(loc, diag::err_uncasted_send_to_unknown_any_method)
        << static_cast<unsigned>(msg->isClassMessage()) << msg->getSelector()
        << orig->getSourceRange();
      return ExprError();
    }
  } else {
    S.Diag(E->getExprLoc(), diag::err_unsupported_unknown_any_expr)
      << E->getSourceRange();
    return ExprError();
  }

  S.Diag(loc, diagID) << d << orig->getSourceRange();

  // Never recoverable.
  return ExprError();
}

/// Check for operands with placeholder types and complain if found.
/// Returns true if there was an error and no recovery was possible.
ExprResult Sema::CheckPlaceholderExpr(Expr *E) {
  if (!getLangOpts().CPlusPlus) {
    // C cannot handle TypoExpr nodes on either side of a binop because it
    // doesn't handle dependent types properly, so make sure any TypoExprs have
    // been dealt with before checking the operands.
    ExprResult Result = CorrectDelayedTyposInExpr(E);
    if (!Result.isUsable()) return ExprError();
    E = Result.get();
  }

  const BuiltinType *placeholderType = E->getType()->getAsPlaceholderType();
  if (!placeholderType) return E;

  switch (placeholderType->getKind()) {

  // Overloaded expressions.
  case BuiltinType::Overload: {
    // Try to resolve a single function template specialization.
    // This is obligatory.
    ExprResult result = E;
    if (ResolveAndFixSingleFunctionTemplateSpecialization(result, false)) {
      return result;

    // If that failed, try to recover with a call.
    } else {
      tryToRecoverWithCall(result, PDiag(diag::err_ovl_unresolvable),
                           /*complain*/ true);
      return result;
    }
  }

  // Bound member functions.
  case BuiltinType::BoundMember: {
    ExprResult result = E;
    const Expr *BME = E->IgnoreParens();
    PartialDiagnostic PD = PDiag(diag::err_bound_member_function);
    // Try to give a nicer diagnostic if it is a bound member that we recognize.
    if (isa<CXXPseudoDestructorExpr>(BME)) {
      PD = PDiag(diag::err_dtor_expr_without_call) << /*pseudo-destructor*/ 1;
    } else if (const auto *ME = dyn_cast<MemberExpr>(BME)) {
      if (ME->getMemberNameInfo().getName().getNameKind() ==
          DeclarationName::CXXDestructorName)
        PD = PDiag(diag::err_dtor_expr_without_call) << /*destructor*/ 0;
    }
    tryToRecoverWithCall(result, PD,
                         /*complain*/ true);
    return result;
  }

  // ARC unbridged casts.
  case BuiltinType::ARCUnbridgedCast: {
    Expr *realCast = stripARCUnbridgedCast(E);
    diagnoseARCUnbridgedCast(realCast);
    return realCast;
  }

  // Expressions of unknown type.
  case BuiltinType::UnknownAny:
    return diagnoseUnknownAnyExpr(*this, E);

  // Pseudo-objects.
  case BuiltinType::PseudoObject:
    return checkPseudoObjectRValue(E);

  case BuiltinType::BuiltinFn: {
    // Accept __noop without parens by implicitly converting it to a call expr.
    auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    if (DRE) {
      auto *FD = cast<FunctionDecl>(DRE->getDecl());
      if (FD->getBuiltinID() == Builtin::BI__noop) {
        E = ImpCastExprToType(E, Context.getPointerType(FD->getType()),
                              CK_BuiltinFnToFnPtr).get();
        return new (Context) CallExpr(Context, E, None, Context.IntTy,
                                      VK_RValue, SourceLocation());
      }
    }

    Diag(E->getLocStart(), diag::err_builtin_fn_use);
    return ExprError();
  }

  // Expressions of unknown type.
  case BuiltinType::OMPArraySection:
    Diag(E->getLocStart(), diag::err_omp_array_section_use);
    return ExprError();

  // Everything else should be impossible.
#define BUILTIN_TYPE(Id, SingletonId) \
  case BuiltinType::Id:
#define PLACEHOLDER_TYPE(Id, SingletonId)
#include "clang/AST/BuiltinTypes.def"
    break;
  }

  llvm_unreachable("invalid placeholder type!");
}

bool Sema::CheckCaseExpression(Expr *E) {
  if (E->isTypeDependent())
    return true;
  if (E->isValueDependent() || E->isIntegerConstantExpr(Context))
    return E->getType()->isIntegralOrEnumerationType();
  return false;
}

/// ActOnObjCBoolLiteral - Parse {__objc_yes,__objc_no} literals.
ExprResult
Sema::ActOnObjCBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind == tok::kw___objc_yes || Kind == tok::kw___objc_no) &&
         "Unknown Objective-C Boolean value!");
  QualType BoolT = Context.ObjCBuiltinBoolTy;
  if (!Context.getBOOLDecl()) {
    LookupResult Result(*this, &Context.Idents.get("BOOL"), OpLoc,
                        Sema::LookupOrdinaryName);
    if (LookupName(Result, getCurScope()) && Result.isSingleResult()) {
      NamedDecl *ND = Result.getFoundDecl();
      if (TypedefDecl *TD = dyn_cast<TypedefDecl>(ND)) 
        Context.setBOOLDecl(TD);
    }
  }
  if (Context.getBOOLDecl())
    BoolT = Context.getBOOLType();
  return new (Context)
      ObjCBoolLiteralExpr(Kind == tok::kw___objc_yes, BoolT, OpLoc);
}

//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements semantic analysis for C++ expressions.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "TypeLocBuilder.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace sema;

ParsedType Sema::getDestructorName(SourceLocation TildeLoc,
                                   IdentifierInfo &II,
                                   SourceLocation NameLoc,
                                   Scope *S, CXXScopeSpec &SS,
                                   ParsedType ObjectTypePtr,
                                   bool EnteringContext) {
  // Determine where to perform name lookup.

  // FIXME: This area of the standard is very messy, and the current
  // wording is rather unclear about which scopes we search for the
  // destructor name; see core issues 399 and 555. Issue 399 in
  // particular shows where the current description of destructor name
  // lookup is completely out of line with existing practice, e.g.,
  // this appears to be ill-formed:
  //
  //   namespace N {
  //     template <typename T> struct S {
  //       ~S();
  //     };
  //   }
  //
  //   void f(N::S<int>* s) {
  //     s->N::S<int>::~S();
  //   }
  //
  // See also PR6358 and PR6359.
  // For this reason, we're currently only doing the C++03 version of this
  // code; the C++0x version has to wait until we get a proper spec.
  QualType SearchType;
  DeclContext *LookupCtx = 0;
  bool isDependent = false;
  bool LookInScope = false;

  // If we have an object type, it's because we are in a
  // pseudo-destructor-expression or a member access expression, and
  // we know what type we're looking for.
  if (ObjectTypePtr)
    SearchType = GetTypeFromParser(ObjectTypePtr);

  if (SS.isSet()) {
    NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();

    bool AlreadySearched = false;
    bool LookAtPrefix = true;
    // C++ [basic.lookup.qual]p6:
    //   If a pseudo-destructor-name (5.2.4) contains a nested-name-specifier,
    //   the type-names are looked up as types in the scope designated by the
    //   nested-name-specifier. In a qualified-id of the form:
    //
    //     ::[opt] nested-name-specifier  ~ class-name
    //
    //   where the nested-name-specifier designates a namespace scope, and in
    //   a qualified-id of the form:
    //
    //     ::opt nested-name-specifier class-name ::  ~ class-name
    //
    //   the class-names are looked up as types in the scope designated by
    //   the nested-name-specifier.
    //
    // Here, we check the first case (completely) and determine whether the
    // code below is permitted to look at the prefix of the
    // nested-name-specifier.
    DeclContext *DC = computeDeclContext(SS, EnteringContext);
    if (DC && DC->isFileContext()) {
      AlreadySearched = true;
      LookupCtx = DC;
      isDependent = false;
    } else if (DC && isa<CXXRecordDecl>(DC))
      LookAtPrefix = false;

    // The second case from the C++03 rules quoted further above.
    NestedNameSpecifier *Prefix = 0;
    if (AlreadySearched) {
      // Nothing left to do.
    } else if (LookAtPrefix && (Prefix = NNS->getPrefix())) {
      CXXScopeSpec PrefixSS;
      PrefixSS.Adopt(NestedNameSpecifierLoc(Prefix, SS.location_data()));
      LookupCtx = computeDeclContext(PrefixSS, EnteringContext);
      isDependent = isDependentScopeSpecifier(PrefixSS);
    } else if (ObjectTypePtr) {
      LookupCtx = computeDeclContext(SearchType);
      isDependent = SearchType->isDependentType();
    } else {
      LookupCtx = computeDeclContext(SS, EnteringContext);
      isDependent = LookupCtx && LookupCtx->isDependentContext();
    }

    LookInScope = false;
  } else if (ObjectTypePtr) {
    // C++ [basic.lookup.classref]p3:
    //   If the unqualified-id is ~type-name, the type-name is looked up
    //   in the context of the entire postfix-expression. If the type T
    //   of the object expression is of a class type C, the type-name is
    //   also looked up in the scope of class C. At least one of the
    //   lookups shall find a name that refers to (possibly
    //   cv-qualified) T.
    LookupCtx = computeDeclContext(SearchType);
    isDependent = SearchType->isDependentType();
    assert((isDependent || !SearchType->isIncompleteType()) &&
           "Caller should have completed object type");

    LookInScope = true;
  } else {
    // Perform lookup into the current scope (only).
    LookInScope = true;
  }

  TypeDecl *NonMatchingTypeDecl = 0;
  LookupResult Found(*this, &II, NameLoc, LookupOrdinaryName);
  for (unsigned Step = 0; Step != 2; ++Step) {
    // Look for the name first in the computed lookup context (if we
    // have one) and, if that fails to find a match, in the scope (if
    // we're allowed to look there).
    Found.clear();
    if (Step == 0 && LookupCtx)
      LookupQualifiedName(Found, LookupCtx);
    else if (Step == 1 && LookInScope && S)
      LookupName(Found, S);
    else
      continue;

    // FIXME: Should we be suppressing ambiguities here?
    if (Found.isAmbiguous())
      return ParsedType();

    if (TypeDecl *Type = Found.getAsSingle<TypeDecl>()) {
      QualType T = Context.getTypeDeclType(Type);

      if (SearchType.isNull() || SearchType->isDependentType() ||
          Context.hasSameUnqualifiedType(T, SearchType)) {
        // We found our type!

        return ParsedType::make(T);
      }

      if (!SearchType.isNull())
        NonMatchingTypeDecl = Type;
    }

    // If the name that we found is a class template name, and it is
    // the same name as the template name in the last part of the
    // nested-name-specifier (if present) or the object type, then
    // this is the destructor for that class.
    // FIXME: This is a workaround until we get real drafting for core
    // issue 399, for which there isn't even an obvious direction.
    if (ClassTemplateDecl *Template = Found.getAsSingle<ClassTemplateDecl>()) {
      QualType MemberOfType;
      if (SS.isSet()) {
        if (DeclContext *Ctx = computeDeclContext(SS, EnteringContext)) {
          // Figure out the type of the context, if it has one.
          if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Ctx))
            MemberOfType = Context.getTypeDeclType(Record);
        }
      }
      if (MemberOfType.isNull())
        MemberOfType = SearchType;

      if (MemberOfType.isNull())
        continue;

      // We're referring into a class template specialization. If the
      // class template we found is the same as the template being
      // specialized, we found what we are looking for.
      if (const RecordType *Record = MemberOfType->getAs<RecordType>()) {
        if (ClassTemplateSpecializationDecl *Spec
              = dyn_cast<ClassTemplateSpecializationDecl>(Record->getDecl())) {
          if (Spec->getSpecializedTemplate()->getCanonicalDecl() ==
                Template->getCanonicalDecl())
            return ParsedType::make(MemberOfType);
        }

        continue;
      }

      // We're referring to an unresolved class template
      // specialization. Determine whether we class template we found
      // is the same as the template being specialized or, if we don't
      // know which template is being specialized, that it at least
      // has the same name.
      if (const TemplateSpecializationType *SpecType
            = MemberOfType->getAs<TemplateSpecializationType>()) {
        TemplateName SpecName = SpecType->getTemplateName();

        // The class template we found is the same template being
        // specialized.
        if (TemplateDecl *SpecTemplate = SpecName.getAsTemplateDecl()) {
          if (SpecTemplate->getCanonicalDecl() == Template->getCanonicalDecl())
            return ParsedType::make(MemberOfType);

          continue;
        }

        // The class template we found has the same name as the
        // (dependent) template name being specialized.
        if (DependentTemplateName *DepTemplate
                                    = SpecName.getAsDependentTemplateName()) {
          if (DepTemplate->isIdentifier() &&
              DepTemplate->getIdentifier() == Template->getIdentifier())
            return ParsedType::make(MemberOfType);

          continue;
        }
      }
    }
  }

  if (isDependent) {
    // We didn't find our type, but that's okay: it's dependent
    // anyway.
    
    // FIXME: What if we have no nested-name-specifier?
    QualType T = CheckTypenameType(ETK_None, SourceLocation(),
                                   SS.getWithLocInContext(Context),
                                   II, NameLoc);
    return ParsedType::make(T);
  }

  if (NonMatchingTypeDecl) {
    QualType T = Context.getTypeDeclType(NonMatchingTypeDecl);
    Diag(NameLoc, diag::err_destructor_expr_type_mismatch)
      << T << SearchType;
    Diag(NonMatchingTypeDecl->getLocation(), diag::note_destructor_type_here)
      << T;
  } else if (ObjectTypePtr)
    Diag(NameLoc, diag::err_ident_in_dtor_not_a_type)
      << &II;
  else
    Diag(NameLoc, diag::err_destructor_class_name);

  return ParsedType();
}

ParsedType Sema::getDestructorType(const DeclSpec& DS, ParsedType ObjectType) {
    if (DS.getTypeSpecType() == DeclSpec::TST_error || !ObjectType)
      return ParsedType();
    assert(DS.getTypeSpecType() == DeclSpec::TST_decltype 
           && "only get destructor types from declspecs");
    QualType T = BuildDecltypeType(DS.getRepAsExpr(), DS.getTypeSpecTypeLoc());
    QualType SearchType = GetTypeFromParser(ObjectType);
    if (SearchType->isDependentType() || Context.hasSameUnqualifiedType(SearchType, T)) {
      return ParsedType::make(T);
    }
      
    Diag(DS.getTypeSpecTypeLoc(), diag::err_destructor_expr_type_mismatch)
      << T << SearchType;
    return ParsedType();
}

/// \brief Build a C++ typeid expression with a type operand.
ExprResult Sema::BuildCXXTypeId(QualType TypeInfoType,
                                SourceLocation TypeidLoc,
                                TypeSourceInfo *Operand,
                                SourceLocation RParenLoc) {
  // C++ [expr.typeid]p4:
  //   The top-level cv-qualifiers of the lvalue expression or the type-id
  //   that is the operand of typeid are always ignored.
  //   If the type of the type-id is a class type or a reference to a class
  //   type, the class shall be completely-defined.
  Qualifiers Quals;
  QualType T
    = Context.getUnqualifiedArrayType(Operand->getType().getNonReferenceType(),
                                      Quals);
  if (T->getAs<RecordType>() &&
      RequireCompleteType(TypeidLoc, T, diag::err_incomplete_typeid))
    return ExprError();

  return Owned(new (Context) CXXTypeidExpr(TypeInfoType.withConst(),
                                           Operand,
                                           SourceRange(TypeidLoc, RParenLoc)));
}

/// \brief Build a C++ typeid expression with an expression operand.
ExprResult Sema::BuildCXXTypeId(QualType TypeInfoType,
                                SourceLocation TypeidLoc,
                                Expr *E,
                                SourceLocation RParenLoc) {
  if (E && !E->isTypeDependent()) {
    if (E->getType()->isPlaceholderType()) {
      ExprResult result = CheckPlaceholderExpr(E);
      if (result.isInvalid()) return ExprError();
      E = result.take();
    }

    QualType T = E->getType();
    if (const RecordType *RecordT = T->getAs<RecordType>()) {
      CXXRecordDecl *RecordD = cast<CXXRecordDecl>(RecordT->getDecl());
      // C++ [expr.typeid]p3:
      //   [...] If the type of the expression is a class type, the class
      //   shall be completely-defined.
      if (RequireCompleteType(TypeidLoc, T, diag::err_incomplete_typeid))
        return ExprError();

      // C++ [expr.typeid]p3:
      //   When typeid is applied to an expression other than an glvalue of a
      //   polymorphic class type [...] [the] expression is an unevaluated
      //   operand. [...]
      if (RecordD->isPolymorphic() && E->isGLValue()) {
        // The subexpression is potentially evaluated; switch the context
        // and recheck the subexpression.
        ExprResult Result = TranformToPotentiallyEvaluated(E);
        if (Result.isInvalid()) return ExprError();
        E = Result.take();

        // We require a vtable to query the type at run time.
        MarkVTableUsed(TypeidLoc, RecordD);
      }
    }

    // C++ [expr.typeid]p4:
    //   [...] If the type of the type-id is a reference to a possibly
    //   cv-qualified type, the result of the typeid expression refers to a
    //   std::type_info object representing the cv-unqualified referenced
    //   type.
    Qualifiers Quals;
    QualType UnqualT = Context.getUnqualifiedArrayType(T, Quals);
    if (!Context.hasSameType(T, UnqualT)) {
      T = UnqualT;
      E = ImpCastExprToType(E, UnqualT, CK_NoOp, E->getValueKind()).take();
    }
  }

  return Owned(new (Context) CXXTypeidExpr(TypeInfoType.withConst(),
                                           E,
                                           SourceRange(TypeidLoc, RParenLoc)));
}

/// ActOnCXXTypeidOfType - Parse typeid( type-id ) or typeid (expression);
ExprResult
Sema::ActOnCXXTypeid(SourceLocation OpLoc, SourceLocation LParenLoc,
                     bool isType, void *TyOrExpr, SourceLocation RParenLoc) {
  // Find the std::type_info type.
  if (!getStdNamespace())
    return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));

  if (!CXXTypeInfoDecl) {
    IdentifierInfo *TypeInfoII = &PP.getIdentifierTable().get("type_info");
    LookupResult R(*this, TypeInfoII, SourceLocation(), LookupTagName);
    LookupQualifiedName(R, getStdNamespace());
    CXXTypeInfoDecl = R.getAsSingle<RecordDecl>();
    // Microsoft's typeinfo doesn't have type_info in std but in the global
    // namespace if _HAS_EXCEPTIONS is defined to 0. See PR13153.
    if (!CXXTypeInfoDecl && LangOpts.MicrosoftMode) {
      LookupQualifiedName(R, Context.getTranslationUnitDecl());
      CXXTypeInfoDecl = R.getAsSingle<RecordDecl>();
    }
    if (!CXXTypeInfoDecl)
      return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));
  }

  if (!getLangOpts().RTTI) {
    return ExprError(Diag(OpLoc, diag::err_no_typeid_with_fno_rtti));
  }

  QualType TypeInfoType = Context.getTypeDeclType(CXXTypeInfoDecl);

  if (isType) {
    // The operand is a type; handle it as such.
    TypeSourceInfo *TInfo = 0;
    QualType T = GetTypeFromParser(ParsedType::getFromOpaquePtr(TyOrExpr),
                                   &TInfo);
    if (T.isNull())
      return ExprError();

    if (!TInfo)
      TInfo = Context.getTrivialTypeSourceInfo(T, OpLoc);

    return BuildCXXTypeId(TypeInfoType, OpLoc, TInfo, RParenLoc);
  }

  // The operand is an expression.
  return BuildCXXTypeId(TypeInfoType, OpLoc, (Expr*)TyOrExpr, RParenLoc);
}

/// \brief Build a Microsoft __uuidof expression with a type operand.
ExprResult Sema::BuildCXXUuidof(QualType TypeInfoType,
                                SourceLocation TypeidLoc,
                                TypeSourceInfo *Operand,
                                SourceLocation RParenLoc) {
  if (!Operand->getType()->isDependentType()) {
    if (!CXXUuidofExpr::GetUuidAttrOfType(Operand->getType()))
      return ExprError(Diag(TypeidLoc, diag::err_uuidof_without_guid));
  }

  // FIXME: add __uuidof semantic analysis for type operand.
  return Owned(new (Context) CXXUuidofExpr(TypeInfoType.withConst(),
                                           Operand,
                                           SourceRange(TypeidLoc, RParenLoc)));
}

/// \brief Build a Microsoft __uuidof expression with an expression operand.
ExprResult Sema::BuildCXXUuidof(QualType TypeInfoType,
                                SourceLocation TypeidLoc,
                                Expr *E,
                                SourceLocation RParenLoc) {
  if (!E->getType()->isDependentType()) {
    if (!CXXUuidofExpr::GetUuidAttrOfType(E->getType()) &&
        !E->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull))
      return ExprError(Diag(TypeidLoc, diag::err_uuidof_without_guid));
  }
  // FIXME: add __uuidof semantic analysis for type operand.
  return Owned(new (Context) CXXUuidofExpr(TypeInfoType.withConst(),
                                           E,
                                           SourceRange(TypeidLoc, RParenLoc)));
}

/// ActOnCXXUuidof - Parse __uuidof( type-id ) or __uuidof (expression);
ExprResult
Sema::ActOnCXXUuidof(SourceLocation OpLoc, SourceLocation LParenLoc,
                     bool isType, void *TyOrExpr, SourceLocation RParenLoc) {
  // If MSVCGuidDecl has not been cached, do the lookup.
  if (!MSVCGuidDecl) {
    IdentifierInfo *GuidII = &PP.getIdentifierTable().get("_GUID");
    LookupResult R(*this, GuidII, SourceLocation(), LookupTagName);
    LookupQualifiedName(R, Context.getTranslationUnitDecl());
    MSVCGuidDecl = R.getAsSingle<RecordDecl>();
    if (!MSVCGuidDecl)
      return ExprError(Diag(OpLoc, diag::err_need_header_before_ms_uuidof));
  }

  QualType GuidType = Context.getTypeDeclType(MSVCGuidDecl);

  if (isType) {
    // The operand is a type; handle it as such.
    TypeSourceInfo *TInfo = 0;
    QualType T = GetTypeFromParser(ParsedType::getFromOpaquePtr(TyOrExpr),
                                   &TInfo);
    if (T.isNull())
      return ExprError();

    if (!TInfo)
      TInfo = Context.getTrivialTypeSourceInfo(T, OpLoc);

    return BuildCXXUuidof(GuidType, OpLoc, TInfo, RParenLoc);
  }

  // The operand is an expression.
  return BuildCXXUuidof(GuidType, OpLoc, (Expr*)TyOrExpr, RParenLoc);
}

/// ActOnCXXBoolLiteral - Parse {true,false} literals.
ExprResult
Sema::ActOnCXXBoolLiteral(SourceLocation OpLoc, tok::TokenKind Kind) {
  assert((Kind == tok::kw_true || Kind == tok::kw_false) &&
         "Unknown C++ Boolean value!");
  return Owned(new (Context) CXXBoolLiteralExpr(Kind == tok::kw_true,
                                                Context.BoolTy, OpLoc));
}

/// ActOnCXXNullPtrLiteral - Parse 'nullptr'.
ExprResult
Sema::ActOnCXXNullPtrLiteral(SourceLocation Loc) {
  return Owned(new (Context) CXXNullPtrLiteralExpr(Context.NullPtrTy, Loc));
}

/// ActOnCXXThrow - Parse throw expressions.
ExprResult
Sema::ActOnCXXThrow(Scope *S, SourceLocation OpLoc, Expr *Ex) {
  bool IsThrownVarInScope = false;
  if (Ex) {
    // C++0x [class.copymove]p31:
    //   When certain criteria are met, an implementation is allowed to omit the 
    //   copy/move construction of a class object [...]
    //
    //     - in a throw-expression, when the operand is the name of a 
    //       non-volatile automatic object (other than a function or catch-
    //       clause parameter) whose scope does not extend beyond the end of the 
    //       innermost enclosing try-block (if there is one), the copy/move 
    //       operation from the operand to the exception object (15.1) can be 
    //       omitted by constructing the automatic object directly into the 
    //       exception object
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Ex->IgnoreParens()))
      if (VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (Var->hasLocalStorage() && !Var->getType().isVolatileQualified()) {
          for( ; S; S = S->getParent()) {
            if (S->isDeclScope(Var)) {
              IsThrownVarInScope = true;
              break;
            }
            
            if (S->getFlags() &
                (Scope::FnScope | Scope::ClassScope | Scope::BlockScope |
                 Scope::FunctionPrototypeScope | Scope::ObjCMethodScope |
                 Scope::TryScope))
              break;
          }
        }
      }
  }
  
  return BuildCXXThrow(OpLoc, Ex, IsThrownVarInScope);
}

ExprResult Sema::BuildCXXThrow(SourceLocation OpLoc, Expr *Ex, 
                               bool IsThrownVarInScope) {
  // Don't report an error if 'throw' is used in system headers.
  if (!getLangOpts().CXXExceptions &&
      !getSourceManager().isInSystemHeader(OpLoc))
    Diag(OpLoc, diag::err_exceptions_disabled) << "throw";
  
  if (Ex && !Ex->isTypeDependent()) {
    ExprResult ExRes = CheckCXXThrowOperand(OpLoc, Ex, IsThrownVarInScope);
    if (ExRes.isInvalid())
      return ExprError();
    Ex = ExRes.take();
  }
  
  return Owned(new (Context) CXXThrowExpr(Ex, Context.VoidTy, OpLoc,
                                          IsThrownVarInScope));
}

/// CheckCXXThrowOperand - Validate the operand of a throw.
ExprResult Sema::CheckCXXThrowOperand(SourceLocation ThrowLoc, Expr *E,
                                      bool IsThrownVarInScope) {
  // C++ [except.throw]p3:
  //   A throw-expression initializes a temporary object, called the exception
  //   object, the type of which is determined by removing any top-level
  //   cv-qualifiers from the static type of the operand of throw and adjusting
  //   the type from "array of T" or "function returning T" to "pointer to T"
  //   or "pointer to function returning T", [...]
  if (E->getType().hasQualifiers())
    E = ImpCastExprToType(E, E->getType().getUnqualifiedType(), CK_NoOp,
                          E->getValueKind()).take();

  ExprResult Res = DefaultFunctionArrayConversion(E);
  if (Res.isInvalid())
    return ExprError();
  E = Res.take();

  //   If the type of the exception would be an incomplete type or a pointer
  //   to an incomplete type other than (cv) void the program is ill-formed.
  QualType Ty = E->getType();
  bool isPointer = false;
  if (const PointerType* Ptr = Ty->getAs<PointerType>()) {
    Ty = Ptr->getPointeeType();
    isPointer = true;
  }
  if (!isPointer || !Ty->isVoidType()) {
    if (RequireCompleteType(ThrowLoc, Ty,
                            isPointer? diag::err_throw_incomplete_ptr
                                     : diag::err_throw_incomplete,
                            E->getSourceRange()))
      return ExprError();

    if (RequireNonAbstractType(ThrowLoc, E->getType(),
                               diag::err_throw_abstract_type, E))
      return ExprError();
  }

  // Initialize the exception result.  This implicitly weeds out
  // abstract types or types with inaccessible copy constructors.
  
  // C++0x [class.copymove]p31:
  //   When certain criteria are met, an implementation is allowed to omit the 
  //   copy/move construction of a class object [...]
  //
  //     - in a throw-expression, when the operand is the name of a 
  //       non-volatile automatic object (other than a function or catch-clause 
  //       parameter) whose scope does not extend beyond the end of the 
  //       innermost enclosing try-block (if there is one), the copy/move 
  //       operation from the operand to the exception object (15.1) can be 
  //       omitted by constructing the automatic object directly into the 
  //       exception object
  const VarDecl *NRVOVariable = 0;
  if (IsThrownVarInScope)
    NRVOVariable = getCopyElisionCandidate(QualType(), E, false);
  
  InitializedEntity Entity =
      InitializedEntity::InitializeException(ThrowLoc, E->getType(),
                                             /*NRVO=*/NRVOVariable != 0);
  Res = PerformMoveOrCopyInitialization(Entity, NRVOVariable,
                                        QualType(), E,
                                        IsThrownVarInScope);
  if (Res.isInvalid())
    return ExprError();
  E = Res.take();

  // If the exception has class type, we need additional handling.
  const RecordType *RecordTy = Ty->getAs<RecordType>();
  if (!RecordTy)
    return Owned(E);
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());

  // If we are throwing a polymorphic class type or pointer thereof,
  // exception handling will make use of the vtable.
  MarkVTableUsed(ThrowLoc, RD);

  // If a pointer is thrown, the referenced object will not be destroyed.
  if (isPointer)
    return Owned(E);

  // If the class has a destructor, we must be able to call it.
  if (RD->hasIrrelevantDestructor())
    return Owned(E);

  CXXDestructorDecl *Destructor = LookupDestructor(RD);
  if (!Destructor)
    return Owned(E);

  MarkFunctionReferenced(E->getExprLoc(), Destructor);
  CheckDestructorAccess(E->getExprLoc(), Destructor,
                        PDiag(diag::err_access_dtor_exception) << Ty);
  DiagnoseUseOfDecl(Destructor, E->getExprLoc());
  return Owned(E);
}

QualType Sema::getCurrentThisType() {
  DeclContext *DC = getFunctionLevelDeclContext();
  QualType ThisTy = CXXThisTypeOverride;
  if (CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(DC)) {
    if (method && method->isInstance())
      ThisTy = method->getThisType(Context);
  }
  
  return ThisTy;
}

Sema::CXXThisScopeRAII::CXXThisScopeRAII(Sema &S, 
                                         Decl *ContextDecl,
                                         unsigned CXXThisTypeQuals,
                                         bool Enabled) 
  : S(S), OldCXXThisTypeOverride(S.CXXThisTypeOverride), Enabled(false)
{
  if (!Enabled || !ContextDecl)
    return;
  
  CXXRecordDecl *Record = 0;
  if (ClassTemplateDecl *Template = dyn_cast<ClassTemplateDecl>(ContextDecl))
    Record = Template->getTemplatedDecl();
  else
    Record = cast<CXXRecordDecl>(ContextDecl);
    
  S.CXXThisTypeOverride
    = S.Context.getPointerType(
        S.Context.getRecordType(Record).withCVRQualifiers(CXXThisTypeQuals));
  
  this->Enabled = true;
}


Sema::CXXThisScopeRAII::~CXXThisScopeRAII() {
  if (Enabled) {
    S.CXXThisTypeOverride = OldCXXThisTypeOverride;
  }
}

void Sema::CheckCXXThisCapture(SourceLocation Loc, bool Explicit) {
  // We don't need to capture this in an unevaluated context.
  if (ExprEvalContexts.back().Context == Unevaluated && !Explicit)
    return;

  // Otherwise, check that we can capture 'this'.
  unsigned NumClosures = 0;
  for (unsigned idx = FunctionScopes.size() - 1; idx != 0; idx--) {
    if (CapturingScopeInfo *CSI =
            dyn_cast<CapturingScopeInfo>(FunctionScopes[idx])) {
      if (CSI->CXXThisCaptureIndex != 0) {
        // 'this' is already being captured; there isn't anything more to do.
        break;
      }
      
      if (CSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_LambdaByref ||
          CSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_LambdaByval ||
          CSI->ImpCaptureStyle == CapturingScopeInfo::ImpCap_Block ||
          Explicit) {
        // This closure can capture 'this'; continue looking upwards.
        NumClosures++;
        Explicit = false;
        continue;
      }
      // This context can't implicitly capture 'this'; fail out.
      Diag(Loc, diag::err_this_capture) << Explicit;
      return;
    }
    break;
  }

  // Mark that we're implicitly capturing 'this' in all the scopes we skipped.
  // FIXME: We need to delay this marking in PotentiallyPotentiallyEvaluated
  // contexts.
  for (unsigned idx = FunctionScopes.size() - 1;
       NumClosures; --idx, --NumClosures) {
    CapturingScopeInfo *CSI = cast<CapturingScopeInfo>(FunctionScopes[idx]);
    Expr *ThisExpr = 0;
    QualType ThisTy = getCurrentThisType();
    if (LambdaScopeInfo *LSI = dyn_cast<LambdaScopeInfo>(CSI)) {
      // For lambda expressions, build a field and an initializing expression.
      CXXRecordDecl *Lambda = LSI->Lambda;
      FieldDecl *Field
        = FieldDecl::Create(Context, Lambda, Loc, Loc, 0, ThisTy,
                            Context.getTrivialTypeSourceInfo(ThisTy, Loc),
                            0, false, ICIS_NoInit);
      Field->setImplicit(true);
      Field->setAccess(AS_private);
      Lambda->addDecl(Field);
      ThisExpr = new (Context) CXXThisExpr(Loc, ThisTy, /*isImplicit=*/true);
    }
    bool isNested = NumClosures > 1;
    CSI->addThisCapture(isNested, Loc, ThisTy, ThisExpr);
  }
}

ExprResult Sema::ActOnCXXThis(SourceLocation Loc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  QualType ThisTy = getCurrentThisType();
  if (ThisTy.isNull()) return Diag(Loc, diag::err_invalid_this_use);

  CheckCXXThisCapture(Loc);
  return Owned(new (Context) CXXThisExpr(Loc, ThisTy, /*isImplicit=*/false));
}

bool Sema::isThisOutsideMemberFunctionBody(QualType BaseType) {
  // If we're outside the body of a member function, then we'll have a specified
  // type for 'this'.
  if (CXXThisTypeOverride.isNull())
    return false;
  
  // Determine whether we're looking into a class that's currently being
  // defined.
  CXXRecordDecl *Class = BaseType->getAsCXXRecordDecl();
  return Class && Class->isBeingDefined();
}

ExprResult
Sema::ActOnCXXTypeConstructExpr(ParsedType TypeRep,
                                SourceLocation LParenLoc,
                                MultiExprArg exprs,
                                SourceLocation RParenLoc) {
  if (!TypeRep)
    return ExprError();

  TypeSourceInfo *TInfo;
  QualType Ty = GetTypeFromParser(TypeRep, &TInfo);
  if (!TInfo)
    TInfo = Context.getTrivialTypeSourceInfo(Ty, SourceLocation());

  return BuildCXXTypeConstructExpr(TInfo, LParenLoc, exprs, RParenLoc);
}

/// ActOnCXXTypeConstructExpr - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
ExprResult
Sema::BuildCXXTypeConstructExpr(TypeSourceInfo *TInfo,
                                SourceLocation LParenLoc,
                                MultiExprArg exprs,
                                SourceLocation RParenLoc) {
  QualType Ty = TInfo->getType();
  SourceLocation TyBeginLoc = TInfo->getTypeLoc().getBeginLoc();

  if (Ty->isDependentType() || CallExpr::hasAnyTypeDependentArguments(exprs)) {
    return Owned(CXXUnresolvedConstructExpr::Create(Context, TInfo,
                                                    LParenLoc,
                                                    exprs,
                                                    RParenLoc));
  }

  unsigned NumExprs = exprs.size();
  Expr **Exprs = exprs.data();

  bool ListInitialization = LParenLoc.isInvalid();
  assert((!ListInitialization || (NumExprs == 1 && isa<InitListExpr>(Exprs[0])))
         && "List initialization must have initializer list as expression.");
  SourceRange FullRange = SourceRange(TyBeginLoc,
      ListInitialization ? Exprs[0]->getSourceRange().getEnd() : RParenLoc);

  // C++ [expr.type.conv]p1:
  // If the expression list is a single expression, the type conversion
  // expression is equivalent (in definedness, and if defined in meaning) to the
  // corresponding cast expression.
  if (NumExprs == 1 && !ListInitialization) {
    Expr *Arg = Exprs[0];
    return BuildCXXFunctionalCastExpr(TInfo, LParenLoc, Arg, RParenLoc);
  }

  QualType ElemTy = Ty;
  if (Ty->isArrayType()) {
    if (!ListInitialization)
      return ExprError(Diag(TyBeginLoc,
                            diag::err_value_init_for_array_type) << FullRange);
    ElemTy = Context.getBaseElementType(Ty);
  }

  if (!Ty->isVoidType() &&
      RequireCompleteType(TyBeginLoc, ElemTy,
                          diag::err_invalid_incomplete_type_use, FullRange))
    return ExprError();

  if (RequireNonAbstractType(TyBeginLoc, Ty,
                             diag::err_allocation_of_abstract_type))
    return ExprError();

  InitializedEntity Entity = InitializedEntity::InitializeTemporary(TInfo);
  InitializationKind Kind
    = NumExprs ? ListInitialization
                    ? InitializationKind::CreateDirectList(TyBeginLoc)
                    : InitializationKind::CreateDirect(TyBeginLoc,
                                                       LParenLoc, RParenLoc)
               : InitializationKind::CreateValue(TyBeginLoc,
                                                 LParenLoc, RParenLoc);
  InitializationSequence InitSeq(*this, Entity, Kind, Exprs, NumExprs);
  ExprResult Result = InitSeq.Perform(*this, Entity, Kind, exprs);

  if (!Result.isInvalid() && ListInitialization &&
      isa<InitListExpr>(Result.get())) {
    // If the list-initialization doesn't involve a constructor call, we'll get
    // the initializer-list (with corrected type) back, but that's not what we
    // want, since it will be treated as an initializer list in further
    // processing. Explicitly insert a cast here.
    InitListExpr *List = cast<InitListExpr>(Result.take());
    Result = Owned(CXXFunctionalCastExpr::Create(Context, List->getType(),
                                    Expr::getValueKindForType(TInfo->getType()),
                                                 TInfo, TyBeginLoc, CK_NoOp,
                                                 List, /*Path=*/0, RParenLoc));
  }

  // FIXME: Improve AST representation?
  return Result;
}

/// doesUsualArrayDeleteWantSize - Answers whether the usual
/// operator delete[] for the given type has a size_t parameter.
static bool doesUsualArrayDeleteWantSize(Sema &S, SourceLocation loc,
                                         QualType allocType) {
  const RecordType *record =
    allocType->getBaseElementTypeUnsafe()->getAs<RecordType>();
  if (!record) return false;

  // Try to find an operator delete[] in class scope.

  DeclarationName deleteName =
    S.Context.DeclarationNames.getCXXOperatorName(OO_Array_Delete);
  LookupResult ops(S, deleteName, loc, Sema::LookupOrdinaryName);
  S.LookupQualifiedName(ops, record->getDecl());

  // We're just doing this for information.
  ops.suppressDiagnostics();

  // Very likely: there's no operator delete[].
  if (ops.empty()) return false;

  // If it's ambiguous, it should be illegal to call operator delete[]
  // on this thing, so it doesn't matter if we allocate extra space or not.
  if (ops.isAmbiguous()) return false;

  LookupResult::Filter filter = ops.makeFilter();
  while (filter.hasNext()) {
    NamedDecl *del = filter.next()->getUnderlyingDecl();

    // C++0x [basic.stc.dynamic.deallocation]p2:
    //   A template instance is never a usual deallocation function,
    //   regardless of its signature.
    if (isa<FunctionTemplateDecl>(del)) {
      filter.erase();
      continue;
    }

    // C++0x [basic.stc.dynamic.deallocation]p2:
    //   If class T does not declare [an operator delete[] with one
    //   parameter] but does declare a member deallocation function
    //   named operator delete[] with exactly two parameters, the
    //   second of which has type std::size_t, then this function
    //   is a usual deallocation function.
    if (!cast<CXXMethodDecl>(del)->isUsualDeallocationFunction()) {
      filter.erase();
      continue;
    }
  }
  filter.done();

  if (!ops.isSingleResult()) return false;

  const FunctionDecl *del = cast<FunctionDecl>(ops.getFoundDecl());
  return (del->getNumParams() == 2);
}

/// \brief Parsed a C++ 'new' expression (C++ 5.3.4).
///
/// E.g.:
/// @code new (memory) int[size][4] @endcode
/// or
/// @code ::new Foo(23, "hello") @endcode
///
/// \param StartLoc The first location of the expression.
/// \param UseGlobal True if 'new' was prefixed with '::'.
/// \param PlacementLParen Opening paren of the placement arguments.
/// \param PlacementArgs Placement new arguments.
/// \param PlacementRParen Closing paren of the placement arguments.
/// \param TypeIdParens If the type is in parens, the source range.
/// \param D The type to be allocated, as well as array dimensions.
/// \param Initializer The initializing expression or initializer-list, or null
///   if there is none.
ExprResult
Sema::ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                  SourceLocation PlacementLParen, MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen, SourceRange TypeIdParens,
                  Declarator &D, Expr *Initializer) {
  bool TypeContainsAuto = D.getDeclSpec().getTypeSpecType() == DeclSpec::TST_auto;

  Expr *ArraySize = 0;
  // If the specified type is an array, unwrap it and save the expression.
  if (D.getNumTypeObjects() > 0 &&
      D.getTypeObject(0).Kind == DeclaratorChunk::Array) {
     DeclaratorChunk &Chunk = D.getTypeObject(0);
    if (TypeContainsAuto)
      return ExprError(Diag(Chunk.Loc, diag::err_new_array_of_auto)
        << D.getSourceRange());
    if (Chunk.Arr.hasStatic)
      return ExprError(Diag(Chunk.Loc, diag::err_static_illegal_in_new)
        << D.getSourceRange());
    if (!Chunk.Arr.NumElts)
      return ExprError(Diag(Chunk.Loc, diag::err_array_new_needs_size)
        << D.getSourceRange());

    ArraySize = static_cast<Expr*>(Chunk.Arr.NumElts);
    D.DropFirstTypeObject();
  }

  // Every dimension shall be of constant size.
  if (ArraySize) {
    for (unsigned I = 0, N = D.getNumTypeObjects(); I < N; ++I) {
      if (D.getTypeObject(I).Kind != DeclaratorChunk::Array)
        break;

      DeclaratorChunk::ArrayTypeInfo &Array = D.getTypeObject(I).Arr;
      if (Expr *NumElts = (Expr *)Array.NumElts) {
        if (!NumElts->isTypeDependent() && !NumElts->isValueDependent()) {
          Array.NumElts
            = VerifyIntegerConstantExpression(NumElts, 0,
                                              diag::err_new_array_nonconst)
                .take();
          if (!Array.NumElts)
            return ExprError();
        }
      }
    }
  }

  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, /*Scope=*/0);
  QualType AllocType = TInfo->getType();
  if (D.isInvalidType())
    return ExprError();

  SourceRange DirectInitRange;
  if (ParenListExpr *List = dyn_cast_or_null<ParenListExpr>(Initializer))
    DirectInitRange = List->getSourceRange();

  return BuildCXXNew(SourceRange(StartLoc, D.getLocEnd()), UseGlobal,
                     PlacementLParen,
                     PlacementArgs,
                     PlacementRParen,
                     TypeIdParens,
                     AllocType,
                     TInfo,
                     ArraySize,
                     DirectInitRange,
                     Initializer,
                     TypeContainsAuto);
}

static bool isLegalArrayNewInitializer(CXXNewExpr::InitializationStyle Style,
                                       Expr *Init) {
  if (!Init)
    return true;
  if (ParenListExpr *PLE = dyn_cast<ParenListExpr>(Init))
    return PLE->getNumExprs() == 0;
  if (isa<ImplicitValueInitExpr>(Init))
    return true;
  else if (CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init))
    return !CCE->isListInitialization() &&
           CCE->getConstructor()->isDefaultConstructor();
  else if (Style == CXXNewExpr::ListInit) {
    assert(isa<InitListExpr>(Init) &&
           "Shouldn't create list CXXConstructExprs for arrays.");
    return true;
  }
  return false;
}

ExprResult
Sema::BuildCXXNew(SourceRange Range, bool UseGlobal,
                  SourceLocation PlacementLParen,
                  MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen,
                  SourceRange TypeIdParens,
                  QualType AllocType,
                  TypeSourceInfo *AllocTypeInfo,
                  Expr *ArraySize,
                  SourceRange DirectInitRange,
                  Expr *Initializer,
                  bool TypeMayContainAuto) {
  SourceRange TypeRange = AllocTypeInfo->getTypeLoc().getSourceRange();
  SourceLocation StartLoc = Range.getBegin();

  CXXNewExpr::InitializationStyle initStyle;
  if (DirectInitRange.isValid()) {
    assert(Initializer && "Have parens but no initializer.");
    initStyle = CXXNewExpr::CallInit;
  } else if (Initializer && isa<InitListExpr>(Initializer))
    initStyle = CXXNewExpr::ListInit;
  else {
    // In template instantiation, the initializer could be a CXXDefaultArgExpr
    // unwrapped from a CXXConstructExpr that was implicitly built. There is no
    // particularly sane way we can handle this (especially since it can even
    // occur for array new), so we throw the initializer away and have it be
    // rebuilt.
    if (Initializer && isa<CXXDefaultArgExpr>(Initializer))
      Initializer = 0;
    assert((!Initializer || isa<ImplicitValueInitExpr>(Initializer) ||
            isa<CXXConstructExpr>(Initializer)) &&
           "Initializer expression that cannot have been implicitly created.");
    initStyle = CXXNewExpr::NoInit;
  }

  Expr **Inits = &Initializer;
  unsigned NumInits = Initializer ? 1 : 0;
  if (initStyle == CXXNewExpr::CallInit) {
    if (ParenListExpr *List = dyn_cast<ParenListExpr>(Initializer)) {
      Inits = List->getExprs();
      NumInits = List->getNumExprs();
    } else if (CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Initializer)){
      if (!isa<CXXTemporaryObjectExpr>(CCE)) {
        // Can happen in template instantiation. Since this is just an implicit
        // construction, we just take it apart and rebuild it.
        Inits = CCE->getArgs();
        NumInits = CCE->getNumArgs();
      }
    }
  }

  // C++11 [decl.spec.auto]p6. Deduce the type which 'auto' stands in for.
  AutoType *AT = 0;
  if (TypeMayContainAuto &&
      (AT = AllocType->getContainedAutoType()) && !AT->isDeduced()) {
    if (initStyle == CXXNewExpr::NoInit || NumInits == 0)
      return ExprError(Diag(StartLoc, diag::err_auto_new_requires_ctor_arg)
                       << AllocType << TypeRange);
    if (initStyle == CXXNewExpr::ListInit)
      return ExprError(Diag(Inits[0]->getLocStart(),
                            diag::err_auto_new_requires_parens)
                       << AllocType << TypeRange);
    if (NumInits > 1) {
      Expr *FirstBad = Inits[1];
      return ExprError(Diag(FirstBad->getLocStart(),
                            diag::err_auto_new_ctor_multiple_expressions)
                       << AllocType << TypeRange);
    }
    Expr *Deduce = Inits[0];
    TypeSourceInfo *DeducedType = 0;
    if (DeduceAutoType(AllocTypeInfo, Deduce, DeducedType) == DAR_Failed)
      return ExprError(Diag(StartLoc, diag::err_auto_new_deduction_failure)
                       << AllocType << Deduce->getType()
                       << TypeRange << Deduce->getSourceRange());
    if (!DeducedType)
      return ExprError();

    AllocTypeInfo = DeducedType;
    AllocType = AllocTypeInfo->getType();
  }

  // Per C++0x [expr.new]p5, the type being constructed may be a
  // typedef of an array type.
  if (!ArraySize) {
    if (const ConstantArrayType *Array
                              = Context.getAsConstantArrayType(AllocType)) {
      ArraySize = IntegerLiteral::Create(Context, Array->getSize(),
                                         Context.getSizeType(),
                                         TypeRange.getEnd());
      AllocType = Array->getElementType();
    }
  }

  if (CheckAllocatedType(AllocType, TypeRange.getBegin(), TypeRange))
    return ExprError();

  if (initStyle == CXXNewExpr::ListInit && isStdInitializerList(AllocType, 0)) {
    Diag(AllocTypeInfo->getTypeLoc().getBeginLoc(),
         diag::warn_dangling_std_initializer_list)
        << /*at end of FE*/0 << Inits[0]->getSourceRange();
  }

  // In ARC, infer 'retaining' for the allocated 
  if (getLangOpts().ObjCAutoRefCount &&
      AllocType.getObjCLifetime() == Qualifiers::OCL_None &&
      AllocType->isObjCLifetimeType()) {
    AllocType = Context.getLifetimeQualifiedType(AllocType,
                                    AllocType->getObjCARCImplicitLifetime());
  }

  QualType ResultType = Context.getPointerType(AllocType);
    
  // C++98 5.3.4p6: "The expression in a direct-new-declarator shall have
  //   integral or enumeration type with a non-negative value."
  // C++11 [expr.new]p6: The expression [...] shall be of integral or unscoped
  //   enumeration type, or a class type for which a single non-explicit
  //   conversion function to integral or unscoped enumeration type exists.
  if (ArraySize && !ArraySize->isTypeDependent()) {
    class SizeConvertDiagnoser : public ICEConvertDiagnoser {
      Expr *ArraySize;
      
    public:
      SizeConvertDiagnoser(Expr *ArraySize)
        : ICEConvertDiagnoser(false, false), ArraySize(ArraySize) { }
      
      virtual DiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                               QualType T) {
        return S.Diag(Loc, diag::err_array_size_not_integral)
                 << S.getLangOpts().CPlusPlus0x << T;
      }
      
      virtual DiagnosticBuilder diagnoseIncomplete(Sema &S, SourceLocation Loc,
                                                   QualType T) {
        return S.Diag(Loc, diag::err_array_size_incomplete_type)
                 << T << ArraySize->getSourceRange();
      }
      
      virtual DiagnosticBuilder diagnoseExplicitConv(Sema &S,
                                                     SourceLocation Loc,
                                                     QualType T,
                                                     QualType ConvTy) {
        return S.Diag(Loc, diag::err_array_size_explicit_conversion) << T << ConvTy;
      }
      
      virtual DiagnosticBuilder noteExplicitConv(Sema &S,
                                                 CXXConversionDecl *Conv,
                                                 QualType ConvTy) {
        return S.Diag(Conv->getLocation(), diag::note_array_size_conversion)
                 << ConvTy->isEnumeralType() << ConvTy;
      }
      
      virtual DiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                                  QualType T) {
        return S.Diag(Loc, diag::err_array_size_ambiguous_conversion) << T;
      }
      
      virtual DiagnosticBuilder noteAmbiguous(Sema &S, CXXConversionDecl *Conv,
                                              QualType ConvTy) {
        return S.Diag(Conv->getLocation(), diag::note_array_size_conversion)
                 << ConvTy->isEnumeralType() << ConvTy;
      }
      
      virtual DiagnosticBuilder diagnoseConversion(Sema &S, SourceLocation Loc,
                                                   QualType T,
                                                   QualType ConvTy) {
        return S.Diag(Loc,
                      S.getLangOpts().CPlusPlus0x
                        ? diag::warn_cxx98_compat_array_size_conversion
                        : diag::ext_array_size_conversion)
                 << T << ConvTy->isEnumeralType() << ConvTy;
      }
    } SizeDiagnoser(ArraySize);

    ExprResult ConvertedSize
      = ConvertToIntegralOrEnumerationType(StartLoc, ArraySize, SizeDiagnoser,
                                           /*AllowScopedEnumerations*/ false);
    if (ConvertedSize.isInvalid())
      return ExprError();

    ArraySize = ConvertedSize.take();
    QualType SizeType = ArraySize->getType();
    if (!SizeType->isIntegralOrUnscopedEnumerationType())
      return ExprError();

    // C++98 [expr.new]p7:
    //   The expression in a direct-new-declarator shall have integral type
    //   with a non-negative value.
    //
    // Let's see if this is a constant < 0. If so, we reject it out of
    // hand. Otherwise, if it's not a constant, we must have an unparenthesized
    // array type.
    //
    // Note: such a construct has well-defined semantics in C++11: it throws
    // std::bad_array_new_length.
    if (!ArraySize->isValueDependent()) {
      llvm::APSInt Value;
      // We've already performed any required implicit conversion to integer or
      // unscoped enumeration type.
      if (ArraySize->isIntegerConstantExpr(Value, Context)) {
        if (Value < llvm::APSInt(
                        llvm::APInt::getNullValue(Value.getBitWidth()),
                                 Value.isUnsigned())) {
          if (getLangOpts().CPlusPlus0x)
            Diag(ArraySize->getLocStart(),
                 diag::warn_typecheck_negative_array_new_size)
              << ArraySize->getSourceRange();
          else
            return ExprError(Diag(ArraySize->getLocStart(),
                                  diag::err_typecheck_negative_array_size)
                             << ArraySize->getSourceRange());
        } else if (!AllocType->isDependentType()) {
          unsigned ActiveSizeBits =
            ConstantArrayType::getNumAddressingBits(Context, AllocType, Value);
          if (ActiveSizeBits > ConstantArrayType::getMaxSizeBits(Context)) {
            if (getLangOpts().CPlusPlus0x)
              Diag(ArraySize->getLocStart(),
                   diag::warn_array_new_too_large)
                << Value.toString(10)
                << ArraySize->getSourceRange();
            else
              return ExprError(Diag(ArraySize->getLocStart(),
                                    diag::err_array_too_large)
                               << Value.toString(10)
                               << ArraySize->getSourceRange());
          }
        }
      } else if (TypeIdParens.isValid()) {
        // Can't have dynamic array size when the type-id is in parentheses.
        Diag(ArraySize->getLocStart(), diag::ext_new_paren_array_nonconst)
          << ArraySize->getSourceRange()
          << FixItHint::CreateRemoval(TypeIdParens.getBegin())
          << FixItHint::CreateRemoval(TypeIdParens.getEnd());

        TypeIdParens = SourceRange();
      }
    }

    // Note that we do *not* convert the argument in any way.  It can
    // be signed, larger than size_t, whatever.
  }

  FunctionDecl *OperatorNew = 0;
  FunctionDecl *OperatorDelete = 0;
  Expr **PlaceArgs = PlacementArgs.data();
  unsigned NumPlaceArgs = PlacementArgs.size();

  if (!AllocType->isDependentType() &&
      !Expr::hasAnyTypeDependentArguments(
        llvm::makeArrayRef(PlaceArgs, NumPlaceArgs)) &&
      FindAllocationFunctions(StartLoc,
                              SourceRange(PlacementLParen, PlacementRParen),
                              UseGlobal, AllocType, ArraySize, PlaceArgs,
                              NumPlaceArgs, OperatorNew, OperatorDelete))
    return ExprError();

  // If this is an array allocation, compute whether the usual array
  // deallocation function for the type has a size_t parameter.
  bool UsualArrayDeleteWantsSize = false;
  if (ArraySize && !AllocType->isDependentType())
    UsualArrayDeleteWantsSize
      = doesUsualArrayDeleteWantSize(*this, StartLoc, AllocType);

  SmallVector<Expr *, 8> AllPlaceArgs;
  if (OperatorNew) {
    // Add default arguments, if any.
    const FunctionProtoType *Proto =
      OperatorNew->getType()->getAs<FunctionProtoType>();
    VariadicCallType CallType =
      Proto->isVariadic() ? VariadicFunction : VariadicDoesNotApply;

    if (GatherArgumentsForCall(PlacementLParen, OperatorNew,
                               Proto, 1, PlaceArgs, NumPlaceArgs,
                               AllPlaceArgs, CallType))
      return ExprError();

    NumPlaceArgs = AllPlaceArgs.size();
    if (NumPlaceArgs > 0)
      PlaceArgs = &AllPlaceArgs[0];

    DiagnoseSentinelCalls(OperatorNew, PlacementLParen,
                          PlaceArgs, NumPlaceArgs);

    // FIXME: Missing call to CheckFunctionCall or equivalent
  }

  // Warn if the type is over-aligned and is being allocated by global operator
  // new.
  if (NumPlaceArgs == 0 && OperatorNew && 
      (OperatorNew->isImplicit() ||
       getSourceManager().isInSystemHeader(OperatorNew->getLocStart()))) {
    if (unsigned Align = Context.getPreferredTypeAlign(AllocType.getTypePtr())){
      unsigned SuitableAlign = Context.getTargetInfo().getSuitableAlign();
      if (Align > SuitableAlign)
        Diag(StartLoc, diag::warn_overaligned_type)
            << AllocType
            << unsigned(Align / Context.getCharWidth())
            << unsigned(SuitableAlign / Context.getCharWidth());
    }
  }

  QualType InitType = AllocType;
  // Array 'new' can't have any initializers except empty parentheses.
  // Initializer lists are also allowed, in C++11. Rely on the parser for the
  // dialect distinction.
  if (ResultType->isArrayType() || ArraySize) {
    if (!isLegalArrayNewInitializer(initStyle, Initializer)) {
      SourceRange InitRange(Inits[0]->getLocStart(),
                            Inits[NumInits - 1]->getLocEnd());
      Diag(StartLoc, diag::err_new_array_init_args) << InitRange;
      return ExprError();
    }
    if (InitListExpr *ILE = dyn_cast_or_null<InitListExpr>(Initializer)) {
      // We do the initialization typechecking against the array type
      // corresponding to the number of initializers + 1 (to also check
      // default-initialization).
      unsigned NumElements = ILE->getNumInits() + 1;
      InitType = Context.getConstantArrayType(AllocType,
          llvm::APInt(Context.getTypeSize(Context.getSizeType()), NumElements),
                                              ArrayType::Normal, 0);
    }
  }

  if (!AllocType->isDependentType() &&
      !Expr::hasAnyTypeDependentArguments(
        llvm::makeArrayRef(Inits, NumInits))) {
    // C++11 [expr.new]p15:
    //   A new-expression that creates an object of type T initializes that
    //   object as follows:
    InitializationKind Kind
    //     - If the new-initializer is omitted, the object is default-
    //       initialized (8.5); if no initialization is performed,
    //       the object has indeterminate value
      = initStyle == CXXNewExpr::NoInit
          ? InitializationKind::CreateDefault(TypeRange.getBegin())
    //     - Otherwise, the new-initializer is interpreted according to the
    //       initialization rules of 8.5 for direct-initialization.
          : initStyle == CXXNewExpr::ListInit
              ? InitializationKind::CreateDirectList(TypeRange.getBegin())
              : InitializationKind::CreateDirect(TypeRange.getBegin(),
                                                 DirectInitRange.getBegin(),
                                                 DirectInitRange.getEnd());

    InitializedEntity Entity
      = InitializedEntity::InitializeNew(StartLoc, InitType);
    InitializationSequence InitSeq(*this, Entity, Kind, Inits, NumInits);
    ExprResult FullInit = InitSeq.Perform(*this, Entity, Kind,
                                          MultiExprArg(Inits, NumInits));
    if (FullInit.isInvalid())
      return ExprError();

    // FullInit is our initializer; strip off CXXBindTemporaryExprs, because
    // we don't want the initialized object to be destructed.
    if (CXXBindTemporaryExpr *Binder =
            dyn_cast_or_null<CXXBindTemporaryExpr>(FullInit.get()))
      FullInit = Owned(Binder->getSubExpr());

    Initializer = FullInit.take();
  }

  // Mark the new and delete operators as referenced.
  if (OperatorNew)
    MarkFunctionReferenced(StartLoc, OperatorNew);
  if (OperatorDelete)
    MarkFunctionReferenced(StartLoc, OperatorDelete);

  // C++0x [expr.new]p17:
  //   If the new expression creates an array of objects of class type,
  //   access and ambiguity control are done for the destructor.
  QualType BaseAllocType = Context.getBaseElementType(AllocType);
  if (ArraySize && !BaseAllocType->isDependentType()) {
    if (const RecordType *BaseRecordType = BaseAllocType->getAs<RecordType>()) {
      if (CXXDestructorDecl *dtor = LookupDestructor(
              cast<CXXRecordDecl>(BaseRecordType->getDecl()))) {
        MarkFunctionReferenced(StartLoc, dtor);
        CheckDestructorAccess(StartLoc, dtor, 
                              PDiag(diag::err_access_dtor)
                                << BaseAllocType);
        DiagnoseUseOfDecl(dtor, StartLoc);
      }
    }
  }

  return Owned(new (Context) CXXNewExpr(Context, UseGlobal, OperatorNew,
                                        OperatorDelete,
                                        UsualArrayDeleteWantsSize,
                                   llvm::makeArrayRef(PlaceArgs, NumPlaceArgs),
                                        TypeIdParens,
                                        ArraySize, initStyle, Initializer,
                                        ResultType, AllocTypeInfo,
                                        Range, DirectInitRange));
}

/// \brief Checks that a type is suitable as the allocated type
/// in a new-expression.
bool Sema::CheckAllocatedType(QualType AllocType, SourceLocation Loc,
                              SourceRange R) {
  // C++ 5.3.4p1: "[The] type shall be a complete object type, but not an
  //   abstract class type or array thereof.
  if (AllocType->isFunctionType())
    return Diag(Loc, diag::err_bad_new_type)
      << AllocType << 0 << R;
  else if (AllocType->isReferenceType())
    return Diag(Loc, diag::err_bad_new_type)
      << AllocType << 1 << R;
  else if (!AllocType->isDependentType() &&
           RequireCompleteType(Loc, AllocType, diag::err_new_incomplete_type,R))
    return true;
  else if (RequireNonAbstractType(Loc, AllocType,
                                  diag::err_allocation_of_abstract_type))
    return true;
  else if (AllocType->isVariablyModifiedType())
    return Diag(Loc, diag::err_variably_modified_new_type)
             << AllocType;
  else if (unsigned AddressSpace = AllocType.getAddressSpace())
    return Diag(Loc, diag::err_address_space_qualified_new)
      << AllocType.getUnqualifiedType() << AddressSpace;
  else if (getLangOpts().ObjCAutoRefCount) {
    if (const ArrayType *AT = Context.getAsArrayType(AllocType)) {
      QualType BaseAllocType = Context.getBaseElementType(AT);
      if (BaseAllocType.getObjCLifetime() == Qualifiers::OCL_None &&
          BaseAllocType->isObjCLifetimeType())
        return Diag(Loc, diag::err_arc_new_array_without_ownership)
          << BaseAllocType;
    }
  }
           
  return false;
}

/// \brief Determine whether the given function is a non-placement
/// deallocation function.
static bool isNonPlacementDeallocationFunction(FunctionDecl *FD) {
  if (FD->isInvalidDecl())
    return false;

  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FD))
    return Method->isUsualDeallocationFunction();

  return ((FD->getOverloadedOperator() == OO_Delete ||
           FD->getOverloadedOperator() == OO_Array_Delete) &&
          FD->getNumParams() == 1);
}

/// FindAllocationFunctions - Finds the overloads of operator new and delete
/// that are appropriate for the allocation.
bool Sema::FindAllocationFunctions(SourceLocation StartLoc, SourceRange Range,
                                   bool UseGlobal, QualType AllocType,
                                   bool IsArray, Expr **PlaceArgs,
                                   unsigned NumPlaceArgs,
                                   FunctionDecl *&OperatorNew,
                                   FunctionDecl *&OperatorDelete) {
  // --- Choosing an allocation function ---
  // C++ 5.3.4p8 - 14 & 18
  // 1) If UseGlobal is true, only look in the global scope. Else, also look
  //   in the scope of the allocated class.
  // 2) If an array size is given, look for operator new[], else look for
  //   operator new.
  // 3) The first argument is always size_t. Append the arguments from the
  //   placement form.

  SmallVector<Expr*, 8> AllocArgs(1 + NumPlaceArgs);
  // We don't care about the actual value of this argument.
  // FIXME: Should the Sema create the expression and embed it in the syntax
  // tree? Or should the consumer just recalculate the value?
  IntegerLiteral Size(Context, llvm::APInt::getNullValue(
                      Context.getTargetInfo().getPointerWidth(0)),
                      Context.getSizeType(),
                      SourceLocation());
  AllocArgs[0] = &Size;
  std::copy(PlaceArgs, PlaceArgs + NumPlaceArgs, AllocArgs.begin() + 1);

  // C++ [expr.new]p8:
  //   If the allocated type is a non-array type, the allocation
  //   function's name is operator new and the deallocation function's
  //   name is operator delete. If the allocated type is an array
  //   type, the allocation function's name is operator new[] and the
  //   deallocation function's name is operator delete[].
  DeclarationName NewName = Context.DeclarationNames.getCXXOperatorName(
                                        IsArray ? OO_Array_New : OO_New);
  DeclarationName DeleteName = Context.DeclarationNames.getCXXOperatorName(
                                        IsArray ? OO_Array_Delete : OO_Delete);

  QualType AllocElemType = Context.getBaseElementType(AllocType);

  if (AllocElemType->isRecordType() && !UseGlobal) {
    CXXRecordDecl *Record
      = cast<CXXRecordDecl>(AllocElemType->getAs<RecordType>()->getDecl());
    if (FindAllocationOverload(StartLoc, Range, NewName, &AllocArgs[0],
                          AllocArgs.size(), Record, /*AllowMissing=*/true,
                          OperatorNew))
      return true;
  }
  if (!OperatorNew) {
    // Didn't find a member overload. Look for a global one.
    DeclareGlobalNewDelete();
    DeclContext *TUDecl = Context.getTranslationUnitDecl();
    if (FindAllocationOverload(StartLoc, Range, NewName, &AllocArgs[0],
                          AllocArgs.size(), TUDecl, /*AllowMissing=*/false,
                          OperatorNew))
      return true;
  }

  // We don't need an operator delete if we're running under
  // -fno-exceptions.
  if (!getLangOpts().Exceptions) {
    OperatorDelete = 0;
    return false;
  }

  // FindAllocationOverload can change the passed in arguments, so we need to
  // copy them back.
  if (NumPlaceArgs > 0)
    std::copy(&AllocArgs[1], AllocArgs.end(), PlaceArgs);

  // C++ [expr.new]p19:
  //
  //   If the new-expression begins with a unary :: operator, the
  //   deallocation function's name is looked up in the global
  //   scope. Otherwise, if the allocated type is a class type T or an
  //   array thereof, the deallocation function's name is looked up in
  //   the scope of T. If this lookup fails to find the name, or if
  //   the allocated type is not a class type or array thereof, the
  //   deallocation function's name is looked up in the global scope.
  LookupResult FoundDelete(*this, DeleteName, StartLoc, LookupOrdinaryName);
  if (AllocElemType->isRecordType() && !UseGlobal) {
    CXXRecordDecl *RD
      = cast<CXXRecordDecl>(AllocElemType->getAs<RecordType>()->getDecl());
    LookupQualifiedName(FoundDelete, RD);
  }
  if (FoundDelete.isAmbiguous())
    return true; // FIXME: clean up expressions?

  if (FoundDelete.empty()) {
    DeclareGlobalNewDelete();
    LookupQualifiedName(FoundDelete, Context.getTranslationUnitDecl());
  }

  FoundDelete.suppressDiagnostics();

  SmallVector<std::pair<DeclAccessPair,FunctionDecl*>, 2> Matches;

  // Whether we're looking for a placement operator delete is dictated
  // by whether we selected a placement operator new, not by whether
  // we had explicit placement arguments.  This matters for things like
  //   struct A { void *operator new(size_t, int = 0); ... };
  //   A *a = new A()
  bool isPlacementNew = (NumPlaceArgs > 0 || OperatorNew->param_size() != 1);

  if (isPlacementNew) {
    // C++ [expr.new]p20:
    //   A declaration of a placement deallocation function matches the
    //   declaration of a placement allocation function if it has the
    //   same number of parameters and, after parameter transformations
    //   (8.3.5), all parameter types except the first are
    //   identical. [...]
    //
    // To perform this comparison, we compute the function type that
    // the deallocation function should have, and use that type both
    // for template argument deduction and for comparison purposes.
    //
    // FIXME: this comparison should ignore CC and the like.
    QualType ExpectedFunctionType;
    {
      const FunctionProtoType *Proto
        = OperatorNew->getType()->getAs<FunctionProtoType>();

      SmallVector<QualType, 4> ArgTypes;
      ArgTypes.push_back(Context.VoidPtrTy);
      for (unsigned I = 1, N = Proto->getNumArgs(); I < N; ++I)
        ArgTypes.push_back(Proto->getArgType(I));

      FunctionProtoType::ExtProtoInfo EPI;
      EPI.Variadic = Proto->isVariadic();

      ExpectedFunctionType
        = Context.getFunctionType(Context.VoidTy, ArgTypes.data(),
                                  ArgTypes.size(), EPI);
    }

    for (LookupResult::iterator D = FoundDelete.begin(),
                             DEnd = FoundDelete.end();
         D != DEnd; ++D) {
      FunctionDecl *Fn = 0;
      if (FunctionTemplateDecl *FnTmpl
            = dyn_cast<FunctionTemplateDecl>((*D)->getUnderlyingDecl())) {
        // Perform template argument deduction to try to match the
        // expected function type.
        TemplateDeductionInfo Info(StartLoc);
        if (DeduceTemplateArguments(FnTmpl, 0, ExpectedFunctionType, Fn, Info))
          continue;
      } else
        Fn = cast<FunctionDecl>((*D)->getUnderlyingDecl());

      if (Context.hasSameType(Fn->getType(), ExpectedFunctionType))
        Matches.push_back(std::make_pair(D.getPair(), Fn));
    }
  } else {
    // C++ [expr.new]p20:
    //   [...] Any non-placement deallocation function matches a
    //   non-placement allocation function. [...]
    for (LookupResult::iterator D = FoundDelete.begin(),
                             DEnd = FoundDelete.end();
         D != DEnd; ++D) {
      if (FunctionDecl *Fn = dyn_cast<FunctionDecl>((*D)->getUnderlyingDecl()))
        if (isNonPlacementDeallocationFunction(Fn))
          Matches.push_back(std::make_pair(D.getPair(), Fn));
    }
  }

  // C++ [expr.new]p20:
  //   [...] If the lookup finds a single matching deallocation
  //   function, that function will be called; otherwise, no
  //   deallocation function will be called.
  if (Matches.size() == 1) {
    OperatorDelete = Matches[0].second;

    // C++0x [expr.new]p20:
    //   If the lookup finds the two-parameter form of a usual
    //   deallocation function (3.7.4.2) and that function, considered
    //   as a placement deallocation function, would have been
    //   selected as a match for the allocation function, the program
    //   is ill-formed.
    if (NumPlaceArgs && getLangOpts().CPlusPlus0x &&
        isNonPlacementDeallocationFunction(OperatorDelete)) {
      Diag(StartLoc, diag::err_placement_new_non_placement_delete)
        << SourceRange(PlaceArgs[0]->getLocStart(),
                       PlaceArgs[NumPlaceArgs - 1]->getLocEnd());
      Diag(OperatorDelete->getLocation(), diag::note_previous_decl)
        << DeleteName;
    } else {
      CheckAllocationAccess(StartLoc, Range, FoundDelete.getNamingClass(),
                            Matches[0].first);
    }
  }

  return false;
}

/// FindAllocationOverload - Find an fitting overload for the allocation
/// function in the specified scope.
bool Sema::FindAllocationOverload(SourceLocation StartLoc, SourceRange Range,
                                  DeclarationName Name, Expr** Args,
                                  unsigned NumArgs, DeclContext *Ctx,
                                  bool AllowMissing, FunctionDecl *&Operator,
                                  bool Diagnose) {
  LookupResult R(*this, Name, StartLoc, LookupOrdinaryName);
  LookupQualifiedName(R, Ctx);
  if (R.empty()) {
    if (AllowMissing || !Diagnose)
      return false;
    return Diag(StartLoc, diag::err_ovl_no_viable_function_in_call)
      << Name << Range;
  }

  if (R.isAmbiguous())
    return true;

  R.suppressDiagnostics();

  OverloadCandidateSet Candidates(StartLoc);
  for (LookupResult::iterator Alloc = R.begin(), AllocEnd = R.end();
       Alloc != AllocEnd; ++Alloc) {
    // Even member operator new/delete are implicitly treated as
    // static, so don't use AddMemberCandidate.
    NamedDecl *D = (*Alloc)->getUnderlyingDecl();

    if (FunctionTemplateDecl *FnTemplate = dyn_cast<FunctionTemplateDecl>(D)) {
      AddTemplateOverloadCandidate(FnTemplate, Alloc.getPair(),
                                   /*ExplicitTemplateArgs=*/0,
                                   llvm::makeArrayRef(Args, NumArgs),
                                   Candidates,
                                   /*SuppressUserConversions=*/false);
      continue;
    }

    FunctionDecl *Fn = cast<FunctionDecl>(D);
    AddOverloadCandidate(Fn, Alloc.getPair(),
                         llvm::makeArrayRef(Args, NumArgs), Candidates,
                         /*SuppressUserConversions=*/false);
  }

  // Do the resolution.
  OverloadCandidateSet::iterator Best;
  switch (Candidates.BestViableFunction(*this, StartLoc, Best)) {
  case OR_Success: {
    // Got one!
    FunctionDecl *FnDecl = Best->Function;
    MarkFunctionReferenced(StartLoc, FnDecl);
    // The first argument is size_t, and the first parameter must be size_t,
    // too. This is checked on declaration and can be assumed. (It can't be
    // asserted on, though, since invalid decls are left in there.)
    // Watch out for variadic allocator function.
    unsigned NumArgsInFnDecl = FnDecl->getNumParams();
    for (unsigned i = 0; (i < NumArgs && i < NumArgsInFnDecl); ++i) {
      InitializedEntity Entity = InitializedEntity::InitializeParameter(Context,
                                                       FnDecl->getParamDecl(i));

      if (!Diagnose && !CanPerformCopyInitialization(Entity, Owned(Args[i])))
        return true;

      ExprResult Result
        = PerformCopyInitialization(Entity, SourceLocation(), Owned(Args[i]));
      if (Result.isInvalid())
        return true;

      Args[i] = Result.takeAs<Expr>();
    }

    Operator = FnDecl;

    if (CheckAllocationAccess(StartLoc, Range, R.getNamingClass(),
                              Best->FoundDecl, Diagnose) == AR_inaccessible)
      return true;

    return false;
  }

  case OR_No_Viable_Function:
    if (Diagnose) {
      Diag(StartLoc, diag::err_ovl_no_viable_function_in_call)
        << Name << Range;
      Candidates.NoteCandidates(*this, OCD_AllCandidates,
                                llvm::makeArrayRef(Args, NumArgs));
    }
    return true;

  case OR_Ambiguous:
    if (Diagnose) {
      Diag(StartLoc, diag::err_ovl_ambiguous_call)
        << Name << Range;
      Candidates.NoteCandidates(*this, OCD_ViableCandidates,
                                llvm::makeArrayRef(Args, NumArgs));
    }
    return true;

  case OR_Deleted: {
    if (Diagnose) {
      Diag(StartLoc, diag::err_ovl_deleted_call)
        << Best->Function->isDeleted()
        << Name 
        << getDeletedOrUnavailableSuffix(Best->Function)
        << Range;
      Candidates.NoteCandidates(*this, OCD_AllCandidates,
                                llvm::makeArrayRef(Args, NumArgs));
    }
    return true;
  }
  }
  llvm_unreachable("Unreachable, bad result from BestViableFunction");
}


/// DeclareGlobalNewDelete - Declare the global forms of operator new and
/// delete. These are:
/// @code
///   // C++03:
///   void* operator new(std::size_t) throw(std::bad_alloc);
///   void* operator new[](std::size_t) throw(std::bad_alloc);
///   void operator delete(void *) throw();
///   void operator delete[](void *) throw();
///   // C++0x:
///   void* operator new(std::size_t);
///   void* operator new[](std::size_t);
///   void operator delete(void *);
///   void operator delete[](void *);
/// @endcode
/// C++0x operator delete is implicitly noexcept.
/// Note that the placement and nothrow forms of new are *not* implicitly
/// declared. Their use requires including \<new\>.
void Sema::DeclareGlobalNewDelete() {
  if (GlobalNewDeleteDeclared)
    return;

  // C++ [basic.std.dynamic]p2:
  //   [...] The following allocation and deallocation functions (18.4) are
  //   implicitly declared in global scope in each translation unit of a
  //   program
  //
  //     C++03:
  //     void* operator new(std::size_t) throw(std::bad_alloc);
  //     void* operator new[](std::size_t) throw(std::bad_alloc);
  //     void  operator delete(void*) throw();
  //     void  operator delete[](void*) throw();
  //     C++0x:
  //     void* operator new(std::size_t);
  //     void* operator new[](std::size_t);
  //     void  operator delete(void*);
  //     void  operator delete[](void*);
  //
  //   These implicit declarations introduce only the function names operator
  //   new, operator new[], operator delete, operator delete[].
  //
  // Here, we need to refer to std::bad_alloc, so we will implicitly declare
  // "std" or "bad_alloc" as necessary to form the exception specification.
  // However, we do not make these implicit declarations visible to name
  // lookup.
  // Note that the C++0x versions of operator delete are deallocation functions,
  // and thus are implicitly noexcept.
  if (!StdBadAlloc && !getLangOpts().CPlusPlus0x) {
    // The "std::bad_alloc" class has not yet been declared, so build it
    // implicitly.
    StdBadAlloc = CXXRecordDecl::Create(Context, TTK_Class,
                                        getOrCreateStdNamespace(),
                                        SourceLocation(), SourceLocation(),
                                      &PP.getIdentifierTable().get("bad_alloc"),
                                        0);
    getStdBadAlloc()->setImplicit(true);
  }

  GlobalNewDeleteDeclared = true;

  QualType VoidPtr = Context.getPointerType(Context.VoidTy);
  QualType SizeT = Context.getSizeType();
  bool AssumeSaneOperatorNew = getLangOpts().AssumeSaneOperatorNew;

  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_New),
      VoidPtr, SizeT, AssumeSaneOperatorNew);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Array_New),
      VoidPtr, SizeT, AssumeSaneOperatorNew);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Delete),
      Context.VoidTy, VoidPtr);
  DeclareGlobalAllocationFunction(
      Context.DeclarationNames.getCXXOperatorName(OO_Array_Delete),
      Context.VoidTy, VoidPtr);
}

/// DeclareGlobalAllocationFunction - Declares a single implicit global
/// allocation function if it doesn't already exist.
void Sema::DeclareGlobalAllocationFunction(DeclarationName Name,
                                           QualType Return, QualType Argument,
                                           bool AddMallocAttr) {
  DeclContext *GlobalCtx = Context.getTranslationUnitDecl();

  // Check if this function is already declared.
  {
    DeclContext::lookup_iterator Alloc, AllocEnd;
    for (llvm::tie(Alloc, AllocEnd) = GlobalCtx->lookup(Name);
         Alloc != AllocEnd; ++Alloc) {
      // Only look at non-template functions, as it is the predefined,
      // non-templated allocation function we are trying to declare here.
      if (FunctionDecl *Func = dyn_cast<FunctionDecl>(*Alloc)) {
        QualType InitialParamType =
          Context.getCanonicalType(
            Func->getParamDecl(0)->getType().getUnqualifiedType());
        // FIXME: Do we need to check for default arguments here?
        if (Func->getNumParams() == 1 && InitialParamType == Argument) {
          if(AddMallocAttr && !Func->hasAttr<MallocAttr>())
            Func->addAttr(::new (Context) MallocAttr(SourceLocation(), Context));
          return;
        }
      }
    }
  }

  QualType BadAllocType;
  bool HasBadAllocExceptionSpec
    = (Name.getCXXOverloadedOperator() == OO_New ||
       Name.getCXXOverloadedOperator() == OO_Array_New);
  if (HasBadAllocExceptionSpec && !getLangOpts().CPlusPlus0x) {
    assert(StdBadAlloc && "Must have std::bad_alloc declared");
    BadAllocType = Context.getTypeDeclType(getStdBadAlloc());
  }

  FunctionProtoType::ExtProtoInfo EPI;
  if (HasBadAllocExceptionSpec) {
    if (!getLangOpts().CPlusPlus0x) {
      EPI.ExceptionSpecType = EST_Dynamic;
      EPI.NumExceptions = 1;
      EPI.Exceptions = &BadAllocType;
    }
  } else {
    EPI.ExceptionSpecType = getLangOpts().CPlusPlus0x ?
                                EST_BasicNoexcept : EST_DynamicNone;
  }

  QualType FnType = Context.getFunctionType(Return, &Argument, 1, EPI);
  FunctionDecl *Alloc =
    FunctionDecl::Create(Context, GlobalCtx, SourceLocation(),
                         SourceLocation(), Name,
                         FnType, /*TInfo=*/0, SC_None,
                         SC_None, false, true);
  Alloc->setImplicit();

  if (AddMallocAttr)
    Alloc->addAttr(::new (Context) MallocAttr(SourceLocation(), Context));

  ParmVarDecl *Param = ParmVarDecl::Create(Context, Alloc, SourceLocation(),
                                           SourceLocation(), 0,
                                           Argument, /*TInfo=*/0,
                                           SC_None, SC_None, 0);
  Alloc->setParams(Param);

  // FIXME: Also add this declaration to the IdentifierResolver, but
  // make sure it is at the end of the chain to coincide with the
  // global scope.
  Context.getTranslationUnitDecl()->addDecl(Alloc);
}

bool Sema::FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD,
                                    DeclarationName Name,
                                    FunctionDecl* &Operator, bool Diagnose) {
  LookupResult Found(*this, Name, StartLoc, LookupOrdinaryName);
  // Try to find operator delete/operator delete[] in class scope.
  LookupQualifiedName(Found, RD);

  if (Found.isAmbiguous())
    return true;

  Found.suppressDiagnostics();

  SmallVector<DeclAccessPair,4> Matches;
  for (LookupResult::iterator F = Found.begin(), FEnd = Found.end();
       F != FEnd; ++F) {
    NamedDecl *ND = (*F)->getUnderlyingDecl();

    // Ignore template operator delete members from the check for a usual
    // deallocation function.
    if (isa<FunctionTemplateDecl>(ND))
      continue;

    if (cast<CXXMethodDecl>(ND)->isUsualDeallocationFunction())
      Matches.push_back(F.getPair());
  }

  // There's exactly one suitable operator;  pick it.
  if (Matches.size() == 1) {
    Operator = cast<CXXMethodDecl>(Matches[0]->getUnderlyingDecl());

    if (Operator->isDeleted()) {
      if (Diagnose) {
        Diag(StartLoc, diag::err_deleted_function_use);
        NoteDeletedFunction(Operator);
      }
      return true;
    }

    if (CheckAllocationAccess(StartLoc, SourceRange(), Found.getNamingClass(),
                              Matches[0], Diagnose) == AR_inaccessible)
      return true;

    return false;

  // We found multiple suitable operators;  complain about the ambiguity.
  } else if (!Matches.empty()) {
    if (Diagnose) {
      Diag(StartLoc, diag::err_ambiguous_suitable_delete_member_function_found)
        << Name << RD;

      for (SmallVectorImpl<DeclAccessPair>::iterator
             F = Matches.begin(), FEnd = Matches.end(); F != FEnd; ++F)
        Diag((*F)->getUnderlyingDecl()->getLocation(),
             diag::note_member_declared_here) << Name;
    }
    return true;
  }

  // We did find operator delete/operator delete[] declarations, but
  // none of them were suitable.
  if (!Found.empty()) {
    if (Diagnose) {
      Diag(StartLoc, diag::err_no_suitable_delete_member_function_found)
        << Name << RD;

      for (LookupResult::iterator F = Found.begin(), FEnd = Found.end();
           F != FEnd; ++F)
        Diag((*F)->getUnderlyingDecl()->getLocation(),
             diag::note_member_declared_here) << Name;
    }
    return true;
  }

  // Look for a global declaration.
  DeclareGlobalNewDelete();
  DeclContext *TUDecl = Context.getTranslationUnitDecl();

  CXXNullPtrLiteralExpr Null(Context.VoidPtrTy, SourceLocation());
  Expr* DeallocArgs[1];
  DeallocArgs[0] = &Null;
  if (FindAllocationOverload(StartLoc, SourceRange(), Name,
                             DeallocArgs, 1, TUDecl, !Diagnose,
                             Operator, Diagnose))
    return true;

  assert(Operator && "Did not find a deallocation function!");
  return false;
}

/// ActOnCXXDelete - Parsed a C++ 'delete' expression (C++ 5.3.5), as in:
/// @code ::delete ptr; @endcode
/// or
/// @code delete [] ptr; @endcode
ExprResult
Sema::ActOnCXXDelete(SourceLocation StartLoc, bool UseGlobal,
                     bool ArrayForm, Expr *ExE) {
  // C++ [expr.delete]p1:
  //   The operand shall have a pointer type, or a class type having a single
  //   conversion function to a pointer type. The result has type void.
  //
  // DR599 amends "pointer type" to "pointer to object type" in both cases.

  ExprResult Ex = Owned(ExE);
  FunctionDecl *OperatorDelete = 0;
  bool ArrayFormAsWritten = ArrayForm;
  bool UsualArrayDeleteWantsSize = false;

  if (!Ex.get()->isTypeDependent()) {
    // Perform lvalue-to-rvalue cast, if needed.
    Ex = DefaultLvalueConversion(Ex.take());

    QualType Type = Ex.get()->getType();

    if (const RecordType *Record = Type->getAs<RecordType>()) {
      if (RequireCompleteType(StartLoc, Type,
                              diag::err_delete_incomplete_class_type))
        return ExprError();

      SmallVector<CXXConversionDecl*, 4> ObjectPtrConversions;

      CXXRecordDecl *RD = cast<CXXRecordDecl>(Record->getDecl());
      const UnresolvedSetImpl *Conversions = RD->getVisibleConversionFunctions();
      for (UnresolvedSetImpl::iterator I = Conversions->begin(),
             E = Conversions->end(); I != E; ++I) {
        NamedDecl *D = I.getDecl();
        if (isa<UsingShadowDecl>(D))
          D = cast<UsingShadowDecl>(D)->getTargetDecl();

        // Skip over templated conversion functions; they aren't considered.
        if (isa<FunctionTemplateDecl>(D))
          continue;

        CXXConversionDecl *Conv = cast<CXXConversionDecl>(D);

        QualType ConvType = Conv->getConversionType().getNonReferenceType();
        if (const PointerType *ConvPtrType = ConvType->getAs<PointerType>())
          if (ConvPtrType->getPointeeType()->isIncompleteOrObjectType())
            ObjectPtrConversions.push_back(Conv);
      }
      if (ObjectPtrConversions.size() == 1) {
        // We have a single conversion to a pointer-to-object type. Perform
        // that conversion.
        // TODO: don't redo the conversion calculation.
        ExprResult Res =
          PerformImplicitConversion(Ex.get(),
                            ObjectPtrConversions.front()->getConversionType(),
                                    AA_Converting);
        if (Res.isUsable()) {
          Ex = Res;
          Type = Ex.get()->getType();
        }
      }
      else if (ObjectPtrConversions.size() > 1) {
        Diag(StartLoc, diag::err_ambiguous_delete_operand)
              << Type << Ex.get()->getSourceRange();
        for (unsigned i= 0; i < ObjectPtrConversions.size(); i++)
          NoteOverloadCandidate(ObjectPtrConversions[i]);
        return ExprError();
      }
    }

    if (!Type->isPointerType())
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex.get()->getSourceRange());

    QualType Pointee = Type->getAs<PointerType>()->getPointeeType();
    QualType PointeeElem = Context.getBaseElementType(Pointee);

    if (unsigned AddressSpace = Pointee.getAddressSpace())
      return Diag(Ex.get()->getLocStart(), 
                  diag::err_address_space_qualified_delete)
               << Pointee.getUnqualifiedType() << AddressSpace;

    CXXRecordDecl *PointeeRD = 0;
    if (Pointee->isVoidType() && !isSFINAEContext()) {
      // The C++ standard bans deleting a pointer to a non-object type, which
      // effectively bans deletion of "void*". However, most compilers support
      // this, so we treat it as a warning unless we're in a SFINAE context.
      Diag(StartLoc, diag::ext_delete_void_ptr_operand)
        << Type << Ex.get()->getSourceRange();
    } else if (Pointee->isFunctionType() || Pointee->isVoidType()) {
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex.get()->getSourceRange());
    } else if (!Pointee->isDependentType()) {
      if (!RequireCompleteType(StartLoc, Pointee,
                               diag::warn_delete_incomplete, Ex.get())) {
        if (const RecordType *RT = PointeeElem->getAs<RecordType>())
          PointeeRD = cast<CXXRecordDecl>(RT->getDecl());
      }
    }

    // C++ [expr.delete]p2:
    //   [Note: a pointer to a const type can be the operand of a
    //   delete-expression; it is not necessary to cast away the constness
    //   (5.2.11) of the pointer expression before it is used as the operand
    //   of the delete-expression. ]

    if (Pointee->isArrayType() && !ArrayForm) {
      Diag(StartLoc, diag::warn_delete_array_type)
          << Type << Ex.get()->getSourceRange()
          << FixItHint::CreateInsertion(PP.getLocForEndOfToken(StartLoc), "[]");
      ArrayForm = true;
    }

    DeclarationName DeleteName = Context.DeclarationNames.getCXXOperatorName(
                                      ArrayForm ? OO_Array_Delete : OO_Delete);

    if (PointeeRD) {
      if (!UseGlobal &&
          FindDeallocationFunction(StartLoc, PointeeRD, DeleteName,
                                   OperatorDelete))
        return ExprError();

      // If we're allocating an array of records, check whether the
      // usual operator delete[] has a size_t parameter.
      if (ArrayForm) {
        // If the user specifically asked to use the global allocator,
        // we'll need to do the lookup into the class.
        if (UseGlobal)
          UsualArrayDeleteWantsSize =
            doesUsualArrayDeleteWantSize(*this, StartLoc, PointeeElem);

        // Otherwise, the usual operator delete[] should be the
        // function we just found.
        else if (isa<CXXMethodDecl>(OperatorDelete))
          UsualArrayDeleteWantsSize = (OperatorDelete->getNumParams() == 2);
      }

      if (!PointeeRD->hasIrrelevantDestructor())
        if (CXXDestructorDecl *Dtor = LookupDestructor(PointeeRD)) {
          MarkFunctionReferenced(StartLoc,
                                    const_cast<CXXDestructorDecl*>(Dtor));
          DiagnoseUseOfDecl(Dtor, StartLoc);
        }

      // C++ [expr.delete]p3:
      //   In the first alternative (delete object), if the static type of the
      //   object to be deleted is different from its dynamic type, the static
      //   type shall be a base class of the dynamic type of the object to be
      //   deleted and the static type shall have a virtual destructor or the
      //   behavior is undefined.
      //
      // Note: a final class cannot be derived from, no issue there
      if (PointeeRD->isPolymorphic() && !PointeeRD->hasAttr<FinalAttr>()) {
        CXXDestructorDecl *dtor = PointeeRD->getDestructor();
        if (dtor && !dtor->isVirtual()) {
          if (PointeeRD->isAbstract()) {
            // If the class is abstract, we warn by default, because we're
            // sure the code has undefined behavior.
            Diag(StartLoc, diag::warn_delete_abstract_non_virtual_dtor)
                << PointeeElem;
          } else if (!ArrayForm) {
            // Otherwise, if this is not an array delete, it's a bit suspect,
            // but not necessarily wrong.
            Diag(StartLoc, diag::warn_delete_non_virtual_dtor) << PointeeElem;
          }
        }
      }

    }

    if (!OperatorDelete) {
      // Look for a global declaration.
      DeclareGlobalNewDelete();
      DeclContext *TUDecl = Context.getTranslationUnitDecl();
      Expr *Arg = Ex.get();
      if (!Context.hasSameType(Arg->getType(), Context.VoidPtrTy))
        Arg = ImplicitCastExpr::Create(Context, Context.VoidPtrTy,
                                       CK_BitCast, Arg, 0, VK_RValue);
      if (FindAllocationOverload(StartLoc, SourceRange(), DeleteName,
                                 &Arg, 1, TUDecl, /*AllowMissing=*/false,
                                 OperatorDelete))
        return ExprError();
    }

    MarkFunctionReferenced(StartLoc, OperatorDelete);
    
    // Check access and ambiguity of operator delete and destructor.
    if (PointeeRD) {
      if (CXXDestructorDecl *Dtor = LookupDestructor(PointeeRD)) {
          CheckDestructorAccess(Ex.get()->getExprLoc(), Dtor, 
                      PDiag(diag::err_access_dtor) << PointeeElem);
      }
    }

  }

  return Owned(new (Context) CXXDeleteExpr(Context.VoidTy, UseGlobal, ArrayForm,
                                           ArrayFormAsWritten,
                                           UsualArrayDeleteWantsSize,
                                           OperatorDelete, Ex.take(), StartLoc));
}

/// \brief Check the use of the given variable as a C++ condition in an if,
/// while, do-while, or switch statement.
ExprResult Sema::CheckConditionVariable(VarDecl *ConditionVar,
                                        SourceLocation StmtLoc,
                                        bool ConvertToBoolean) {
  QualType T = ConditionVar->getType();

  // C++ [stmt.select]p2:
  //   The declarator shall not specify a function or an array.
  if (T->isFunctionType())
    return ExprError(Diag(ConditionVar->getLocation(),
                          diag::err_invalid_use_of_function_type)
                       << ConditionVar->getSourceRange());
  else if (T->isArrayType())
    return ExprError(Diag(ConditionVar->getLocation(),
                          diag::err_invalid_use_of_array_type)
                     << ConditionVar->getSourceRange());

  ExprResult Condition =
    Owned(DeclRefExpr::Create(Context, NestedNameSpecifierLoc(),
                              SourceLocation(),
                              ConditionVar,
                              /*enclosing*/ false,
                              ConditionVar->getLocation(),
                              ConditionVar->getType().getNonReferenceType(),
                              VK_LValue));

  MarkDeclRefReferenced(cast<DeclRefExpr>(Condition.get()));

  if (ConvertToBoolean) {
    Condition = CheckBooleanCondition(Condition.take(), StmtLoc);
    if (Condition.isInvalid())
      return ExprError();
  }

  return Condition;
}

/// CheckCXXBooleanCondition - Returns true if a conversion to bool is invalid.
ExprResult Sema::CheckCXXBooleanCondition(Expr *CondExpr) {
  // C++ 6.4p4:
  // The value of a condition that is an initialized declaration in a statement
  // other than a switch statement is the value of the declared variable
  // implicitly converted to type bool. If that conversion is ill-formed, the
  // program is ill-formed.
  // The value of a condition that is an expression is the value of the
  // expression, implicitly converted to bool.
  //
  return PerformContextuallyConvertToBool(CondExpr);
}

/// Helper function to determine whether this is the (deprecated) C++
/// conversion from a string literal to a pointer to non-const char or
/// non-const wchar_t (for narrow and wide string literals,
/// respectively).
bool
Sema::IsStringLiteralToNonConstPointerConversion(Expr *From, QualType ToType) {
  // Look inside the implicit cast, if it exists.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(From))
    From = Cast->getSubExpr();

  // A string literal (2.13.4) that is not a wide string literal can
  // be converted to an rvalue of type "pointer to char"; a wide
  // string literal can be converted to an rvalue of type "pointer
  // to wchar_t" (C++ 4.2p2).
  if (StringLiteral *StrLit = dyn_cast<StringLiteral>(From->IgnoreParens()))
    if (const PointerType *ToPtrType = ToType->getAs<PointerType>())
      if (const BuiltinType *ToPointeeType
          = ToPtrType->getPointeeType()->getAs<BuiltinType>()) {
        // This conversion is considered only when there is an
        // explicit appropriate pointer target type (C++ 4.2p2).
        if (!ToPtrType->getPointeeType().hasQualifiers()) {
          switch (StrLit->getKind()) {
            case StringLiteral::UTF8:
            case StringLiteral::UTF16:
            case StringLiteral::UTF32:
              // We don't allow UTF literals to be implicitly converted
              break;
            case StringLiteral::Ascii:
              return (ToPointeeType->getKind() == BuiltinType::Char_U ||
                      ToPointeeType->getKind() == BuiltinType::Char_S);
            case StringLiteral::Wide:
              return ToPointeeType->isWideCharType();
          }
        }
      }

  return false;
}

static ExprResult BuildCXXCastArgument(Sema &S,
                                       SourceLocation CastLoc,
                                       QualType Ty,
                                       CastKind Kind,
                                       CXXMethodDecl *Method,
                                       DeclAccessPair FoundDecl,
                                       bool HadMultipleCandidates,
                                       Expr *From) {
  switch (Kind) {
  default: llvm_unreachable("Unhandled cast kind!");
  case CK_ConstructorConversion: {
    CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(Method);
    SmallVector<Expr*, 8> ConstructorArgs;

    if (S.CompleteConstructorCall(Constructor, From, CastLoc, ConstructorArgs))
      return ExprError();

    S.CheckConstructorAccess(CastLoc, Constructor,
                             InitializedEntity::InitializeTemporary(Ty),
                             Constructor->getAccess());
    
    ExprResult Result
      = S.BuildCXXConstructExpr(CastLoc, Ty, cast<CXXConstructorDecl>(Method),
                                ConstructorArgs, 
                                HadMultipleCandidates, /*ZeroInit*/ false, 
                                CXXConstructExpr::CK_Complete, SourceRange());
    if (Result.isInvalid())
      return ExprError();

    return S.MaybeBindToTemporary(Result.takeAs<Expr>());
  }

  case CK_UserDefinedConversion: {
    assert(!From->getType()->isPointerType() && "Arg can't have pointer type!");

    // Create an implicit call expr that calls it.
    CXXConversionDecl *Conv = cast<CXXConversionDecl>(Method);
    ExprResult Result = S.BuildCXXMemberCallExpr(From, FoundDecl, Conv,
                                                 HadMultipleCandidates);
    if (Result.isInvalid())
      return ExprError();
    // Record usage of conversion in an implicit cast.
    Result = S.Owned(ImplicitCastExpr::Create(S.Context,
                                              Result.get()->getType(),
                                              CK_UserDefinedConversion,
                                              Result.get(), 0,
                                              Result.get()->getValueKind()));

    S.CheckMemberOperatorAccess(CastLoc, From, /*arg*/ 0, FoundDecl);

    return S.MaybeBindToTemporary(Result.get());
  }
  }
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType using the pre-computed implicit
/// conversion sequence ICS. Returns the converted
/// expression. Action is the kind of conversion we're performing,
/// used in the error message.
ExprResult
Sema::PerformImplicitConversion(Expr *From, QualType ToType,
                                const ImplicitConversionSequence &ICS,
                                AssignmentAction Action, 
                                CheckedConversionKind CCK) {
  switch (ICS.getKind()) {
  case ImplicitConversionSequence::StandardConversion: {
    ExprResult Res = PerformImplicitConversion(From, ToType, ICS.Standard,
                                               Action, CCK);
    if (Res.isInvalid())
      return ExprError();
    From = Res.take();
    break;
  }

  case ImplicitConversionSequence::UserDefinedConversion: {

      FunctionDecl *FD = ICS.UserDefined.ConversionFunction;
      CastKind CastKind;
      QualType BeforeToType;
      assert(FD && "FIXME: aggregate initialization from init list");
      if (const CXXConversionDecl *Conv = dyn_cast<CXXConversionDecl>(FD)) {
        CastKind = CK_UserDefinedConversion;

        // If the user-defined conversion is specified by a conversion function,
        // the initial standard conversion sequence converts the source type to
        // the implicit object parameter of the conversion function.
        BeforeToType = Context.getTagDeclType(Conv->getParent());
      } else {
        const CXXConstructorDecl *Ctor = cast<CXXConstructorDecl>(FD);
        CastKind = CK_ConstructorConversion;
        // Do no conversion if dealing with ... for the first conversion.
        if (!ICS.UserDefined.EllipsisConversion) {
          // If the user-defined conversion is specified by a constructor, the
          // initial standard conversion sequence converts the source type to the
          // type required by the argument of the constructor
          BeforeToType = Ctor->getParamDecl(0)->getType().getNonReferenceType();
        }
      }
      // Watch out for elipsis conversion.
      if (!ICS.UserDefined.EllipsisConversion) {
        ExprResult Res =
          PerformImplicitConversion(From, BeforeToType,
                                    ICS.UserDefined.Before, AA_Converting,
                                    CCK);
        if (Res.isInvalid())
          return ExprError();
        From = Res.take();
      }

      ExprResult CastArg
        = BuildCXXCastArgument(*this,
                               From->getLocStart(),
                               ToType.getNonReferenceType(),
                               CastKind, cast<CXXMethodDecl>(FD),
                               ICS.UserDefined.FoundConversionFunction,
                               ICS.UserDefined.HadMultipleCandidates,
                               From);

      if (CastArg.isInvalid())
        return ExprError();

      From = CastArg.take();

      return PerformImplicitConversion(From, ToType, ICS.UserDefined.After,
                                       AA_Converting, CCK);
  }

  case ImplicitConversionSequence::AmbiguousConversion:
    ICS.DiagnoseAmbiguousConversion(*this, From->getExprLoc(),
                          PDiag(diag::err_typecheck_ambiguous_condition)
                            << From->getSourceRange());
     return ExprError();

  case ImplicitConversionSequence::EllipsisConversion:
    llvm_unreachable("Cannot perform an ellipsis conversion");

  case ImplicitConversionSequence::BadConversion:
    return ExprError();
  }

  // Everything went well.
  return Owned(From);
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType by following the standard
/// conversion sequence SCS. Returns the converted
/// expression. Flavor is the context in which we're performing this
/// conversion, for use in error messages.
ExprResult
Sema::PerformImplicitConversion(Expr *From, QualType ToType,
                                const StandardConversionSequence& SCS,
                                AssignmentAction Action, 
                                CheckedConversionKind CCK) {
  bool CStyle = (CCK == CCK_CStyleCast || CCK == CCK_FunctionalCast);
  
  // Overall FIXME: we are recomputing too many types here and doing far too
  // much extra work. What this means is that we need to keep track of more
  // information that is computed when we try the implicit conversion initially,
  // so that we don't need to recompute anything here.
  QualType FromType = From->getType();
  
  if (SCS.CopyConstructor) {
    // FIXME: When can ToType be a reference type?
    assert(!ToType->isReferenceType());
    if (SCS.Second == ICK_Derived_To_Base) {
      SmallVector<Expr*, 8> ConstructorArgs;
      if (CompleteConstructorCall(cast<CXXConstructorDecl>(SCS.CopyConstructor),
                                  From, /*FIXME:ConstructLoc*/SourceLocation(),
                                  ConstructorArgs))
        return ExprError();
      return BuildCXXConstructExpr(/*FIXME:ConstructLoc*/SourceLocation(),
                                   ToType, SCS.CopyConstructor,
                                   ConstructorArgs,
                                   /*HadMultipleCandidates*/ false,
                                   /*ZeroInit*/ false,
                                   CXXConstructExpr::CK_Complete,
                                   SourceRange());
    }
    return BuildCXXConstructExpr(/*FIXME:ConstructLoc*/SourceLocation(),
                                 ToType, SCS.CopyConstructor,
                                 From, /*HadMultipleCandidates*/ false,
                                 /*ZeroInit*/ false,
                                 CXXConstructExpr::CK_Complete,
                                 SourceRange());
  }

  // Resolve overloaded function references.
  if (Context.hasSameType(FromType, Context.OverloadTy)) {
    DeclAccessPair Found;
    FunctionDecl *Fn = ResolveAddressOfOverloadedFunction(From, ToType,
                                                          true, Found);
    if (!Fn)
      return ExprError();

    if (DiagnoseUseOfDecl(Fn, From->getLocStart()))
      return ExprError();

    From = FixOverloadedFunctionReference(From, Found, Fn);
    FromType = From->getType();
  }

  // Perform the first implicit conversion.
  switch (SCS.First) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Lvalue_To_Rvalue: {
    assert(From->getObjectKind() != OK_ObjCProperty);
    FromType = FromType.getUnqualifiedType();
    ExprResult FromRes = DefaultLvalueConversion(From);
    assert(!FromRes.isInvalid() && "Can't perform deduced conversion?!");
    From = FromRes.take();
    break;
  }

  case ICK_Array_To_Pointer:
    FromType = Context.getArrayDecayedType(FromType);
    From = ImpCastExprToType(From, FromType, CK_ArrayToPointerDecay, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Function_To_Pointer:
    FromType = Context.getPointerType(FromType);
    From = ImpCastExprToType(From, FromType, CK_FunctionToPointerDecay, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  default:
    llvm_unreachable("Improper first standard conversion");
  }

  // Perform the second implicit conversion
  switch (SCS.Second) {
  case ICK_Identity:
    // If both sides are functions (or pointers/references to them), there could
    // be incompatible exception declarations.
    if (CheckExceptionSpecCompatibility(From, ToType))
      return ExprError();
    // Nothing else to do.
    break;

  case ICK_NoReturn_Adjustment:
    // If both sides are functions (or pointers/references to them), there could
    // be incompatible exception declarations.
    if (CheckExceptionSpecCompatibility(From, ToType))
      return ExprError();

    From = ImpCastExprToType(From, ToType, CK_NoOp, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Integral_Promotion:
  case ICK_Integral_Conversion:
    if (ToType->isBooleanType()) {
      assert(FromType->castAs<EnumType>()->getDecl()->isFixed() &&
             SCS.Second == ICK_Integral_Promotion &&
             "only enums with fixed underlying type can promote to bool");
      From = ImpCastExprToType(From, ToType, CK_IntegralToBoolean,
                               VK_RValue, /*BasePath=*/0, CCK).take();
    } else {
      From = ImpCastExprToType(From, ToType, CK_IntegralCast,
                               VK_RValue, /*BasePath=*/0, CCK).take();
    }
    break;

  case ICK_Floating_Promotion:
  case ICK_Floating_Conversion:
    From = ImpCastExprToType(From, ToType, CK_FloatingCast, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Complex_Promotion:
  case ICK_Complex_Conversion: {
    QualType FromEl = From->getType()->getAs<ComplexType>()->getElementType();
    QualType ToEl = ToType->getAs<ComplexType>()->getElementType();
    CastKind CK;
    if (FromEl->isRealFloatingType()) {
      if (ToEl->isRealFloatingType())
        CK = CK_FloatingComplexCast;
      else
        CK = CK_FloatingComplexToIntegralComplex;
    } else if (ToEl->isRealFloatingType()) {
      CK = CK_IntegralComplexToFloatingComplex;
    } else {
      CK = CK_IntegralComplexCast;
    }
    From = ImpCastExprToType(From, ToType, CK, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;
  }

  case ICK_Floating_Integral:
    if (ToType->isRealFloatingType())
      From = ImpCastExprToType(From, ToType, CK_IntegralToFloating, 
                               VK_RValue, /*BasePath=*/0, CCK).take();
    else
      From = ImpCastExprToType(From, ToType, CK_FloatingToIntegral, 
                               VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Compatible_Conversion:
      From = ImpCastExprToType(From, ToType, CK_NoOp, 
                               VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Writeback_Conversion:
  case ICK_Pointer_Conversion: {
    if (SCS.IncompatibleObjC && Action != AA_Casting) {
      // Diagnose incompatible Objective-C conversions
      if (Action == AA_Initializing || Action == AA_Assigning)
        Diag(From->getLocStart(),
             diag::ext_typecheck_convert_incompatible_pointer)
          << ToType << From->getType() << Action
          << From->getSourceRange() << 0;
      else
        Diag(From->getLocStart(),
             diag::ext_typecheck_convert_incompatible_pointer)
          << From->getType() << ToType << Action
          << From->getSourceRange() << 0;

      if (From->getType()->isObjCObjectPointerType() &&
          ToType->isObjCObjectPointerType())
        EmitRelatedResultTypeNote(From);
    } 
    else if (getLangOpts().ObjCAutoRefCount &&
             !CheckObjCARCUnavailableWeakConversion(ToType, 
                                                    From->getType())) {
      if (Action == AA_Initializing)
        Diag(From->getLocStart(), 
             diag::err_arc_weak_unavailable_assign);
      else
        Diag(From->getLocStart(),
             diag::err_arc_convesion_of_weak_unavailable) 
          << (Action == AA_Casting) << From->getType() << ToType 
          << From->getSourceRange();
    }
             
    CastKind Kind = CK_Invalid;
    CXXCastPath BasePath;
    if (CheckPointerConversion(From, ToType, Kind, BasePath, CStyle))
      return ExprError();

    // Make sure we extend blocks if necessary.
    // FIXME: doing this here is really ugly.
    if (Kind == CK_BlockPointerToObjCPointerCast) {
      ExprResult E = From;
      (void) PrepareCastToObjCObjectPointer(E);
      From = E.take();
    }

    From = ImpCastExprToType(From, ToType, Kind, VK_RValue, &BasePath, CCK)
             .take();
    break;
  }

  case ICK_Pointer_Member: {
    CastKind Kind = CK_Invalid;
    CXXCastPath BasePath;
    if (CheckMemberPointerConversion(From, ToType, Kind, BasePath, CStyle))
      return ExprError();
    if (CheckExceptionSpecCompatibility(From, ToType))
      return ExprError();
    From = ImpCastExprToType(From, ToType, Kind, VK_RValue, &BasePath, CCK)
             .take();
    break;
  }

  case ICK_Boolean_Conversion:
    // Perform half-to-boolean conversion via float.
    if (From->getType()->isHalfType()) {
      From = ImpCastExprToType(From, Context.FloatTy, CK_FloatingCast).take();
      FromType = Context.FloatTy;
    }

    From = ImpCastExprToType(From, Context.BoolTy,
                             ScalarTypeToBooleanCastKind(FromType), 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Derived_To_Base: {
    CXXCastPath BasePath;
    if (CheckDerivedToBaseConversion(From->getType(),
                                     ToType.getNonReferenceType(),
                                     From->getLocStart(),
                                     From->getSourceRange(),
                                     &BasePath,
                                     CStyle))
      return ExprError();

    From = ImpCastExprToType(From, ToType.getNonReferenceType(),
                      CK_DerivedToBase, From->getValueKind(),
                      &BasePath, CCK).take();
    break;
  }

  case ICK_Vector_Conversion:
    From = ImpCastExprToType(From, ToType, CK_BitCast, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Vector_Splat:
    From = ImpCastExprToType(From, ToType, CK_VectorSplat, 
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;

  case ICK_Complex_Real:
    // Case 1.  x -> _Complex y
    if (const ComplexType *ToComplex = ToType->getAs<ComplexType>()) {
      QualType ElType = ToComplex->getElementType();
      bool isFloatingComplex = ElType->isRealFloatingType();

      // x -> y
      if (Context.hasSameUnqualifiedType(ElType, From->getType())) {
        // do nothing
      } else if (From->getType()->isRealFloatingType()) {
        From = ImpCastExprToType(From, ElType,
                isFloatingComplex ? CK_FloatingCast : CK_FloatingToIntegral).take();
      } else {
        assert(From->getType()->isIntegerType());
        From = ImpCastExprToType(From, ElType,
                isFloatingComplex ? CK_IntegralToFloating : CK_IntegralCast).take();
      }
      // y -> _Complex y
      From = ImpCastExprToType(From, ToType,
                   isFloatingComplex ? CK_FloatingRealToComplex
                                     : CK_IntegralRealToComplex).take();

    // Case 2.  _Complex x -> y
    } else {
      const ComplexType *FromComplex = From->getType()->getAs<ComplexType>();
      assert(FromComplex);

      QualType ElType = FromComplex->getElementType();
      bool isFloatingComplex = ElType->isRealFloatingType();

      // _Complex x -> x
      From = ImpCastExprToType(From, ElType,
                   isFloatingComplex ? CK_FloatingComplexToReal
                                     : CK_IntegralComplexToReal, 
                               VK_RValue, /*BasePath=*/0, CCK).take();

      // x -> y
      if (Context.hasSameUnqualifiedType(ElType, ToType)) {
        // do nothing
      } else if (ToType->isRealFloatingType()) {
        From = ImpCastExprToType(From, ToType,
                   isFloatingComplex ? CK_FloatingCast : CK_IntegralToFloating, 
                                 VK_RValue, /*BasePath=*/0, CCK).take();
      } else {
        assert(ToType->isIntegerType());
        From = ImpCastExprToType(From, ToType,
                   isFloatingComplex ? CK_FloatingToIntegral : CK_IntegralCast, 
                                 VK_RValue, /*BasePath=*/0, CCK).take();
      }
    }
    break;
  
  case ICK_Block_Pointer_Conversion: {
    From = ImpCastExprToType(From, ToType.getUnqualifiedType(), CK_BitCast,
                             VK_RValue, /*BasePath=*/0, CCK).take();
    break;
  }
      
  case ICK_TransparentUnionConversion: {
    ExprResult FromRes = Owned(From);
    Sema::AssignConvertType ConvTy =
      CheckTransparentUnionArgumentConstraints(ToType, FromRes);
    if (FromRes.isInvalid())
      return ExprError();
    From = FromRes.take();
    assert ((ConvTy == Sema::Compatible) &&
            "Improper transparent union conversion");
    (void)ConvTy;
    break;
  }

  case ICK_Lvalue_To_Rvalue:
  case ICK_Array_To_Pointer:
  case ICK_Function_To_Pointer:
  case ICK_Qualification:
  case ICK_Num_Conversion_Kinds:
    llvm_unreachable("Improper second standard conversion");
  }

  switch (SCS.Third) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Qualification: {
    // The qualification keeps the category of the inner expression, unless the
    // target type isn't a reference.
    ExprValueKind VK = ToType->isReferenceType() ?
                                  From->getValueKind() : VK_RValue;
    From = ImpCastExprToType(From, ToType.getNonLValueExprType(Context),
                             CK_NoOp, VK, /*BasePath=*/0, CCK).take();

    if (SCS.DeprecatedStringLiteralToCharPtr &&
        !getLangOpts().WritableStrings)
      Diag(From->getLocStart(), diag::warn_deprecated_string_literal_conversion)
        << ToType.getNonReferenceType();

    break;
    }

  default:
    llvm_unreachable("Improper third standard conversion");
  }

  // If this conversion sequence involved a scalar -> atomic conversion, perform
  // that conversion now.
  if (const AtomicType *ToAtomic = ToType->getAs<AtomicType>())
    if (Context.hasSameType(ToAtomic->getValueType(), From->getType()))
      From = ImpCastExprToType(From, ToType, CK_NonAtomicToAtomic, VK_RValue, 0,
                               CCK).take();
      
  return Owned(From);
}

ExprResult Sema::ActOnUnaryTypeTrait(UnaryTypeTrait UTT,
                                     SourceLocation KWLoc,
                                     ParsedType Ty,
                                     SourceLocation RParen) {
  TypeSourceInfo *TSInfo;
  QualType T = GetTypeFromParser(Ty, &TSInfo);

  if (!TSInfo)
    TSInfo = Context.getTrivialTypeSourceInfo(T);
  return BuildUnaryTypeTrait(UTT, KWLoc, TSInfo, RParen);
}

/// \brief Check the completeness of a type in a unary type trait.
///
/// If the particular type trait requires a complete type, tries to complete
/// it. If completing the type fails, a diagnostic is emitted and false
/// returned. If completing the type succeeds or no completion was required,
/// returns true.
static bool CheckUnaryTypeTraitTypeCompleteness(Sema &S,
                                                UnaryTypeTrait UTT,
                                                SourceLocation Loc,
                                                QualType ArgTy) {
  // C++0x [meta.unary.prop]p3:
  //   For all of the class templates X declared in this Clause, instantiating
  //   that template with a template argument that is a class template
  //   specialization may result in the implicit instantiation of the template
  //   argument if and only if the semantics of X require that the argument
  //   must be a complete type.
  // We apply this rule to all the type trait expressions used to implement
  // these class templates. We also try to follow any GCC documented behavior
  // in these expressions to ensure portability of standard libraries.
  switch (UTT) {
    // is_complete_type somewhat obviously cannot require a complete type.
  case UTT_IsCompleteType:
    // Fall-through

    // These traits are modeled on the type predicates in C++0x
    // [meta.unary.cat] and [meta.unary.comp]. They are not specified as
    // requiring a complete type, as whether or not they return true cannot be
    // impacted by the completeness of the type.
  case UTT_IsVoid:
  case UTT_IsIntegral:
  case UTT_IsFloatingPoint:
  case UTT_IsArray:
  case UTT_IsPointer:
  case UTT_IsLvalueReference:
  case UTT_IsRvalueReference:
  case UTT_IsMemberFunctionPointer:
  case UTT_IsMemberObjectPointer:
  case UTT_IsEnum:
  case UTT_IsUnion:
  case UTT_IsClass:
  case UTT_IsFunction:
  case UTT_IsReference:
  case UTT_IsArithmetic:
  case UTT_IsFundamental:
  case UTT_IsObject:
  case UTT_IsScalar:
  case UTT_IsCompound:
  case UTT_IsMemberPointer:
    // Fall-through

    // These traits are modeled on type predicates in C++0x [meta.unary.prop]
    // which requires some of its traits to have the complete type. However,
    // the completeness of the type cannot impact these traits' semantics, and
    // so they don't require it. This matches the comments on these traits in
    // Table 49.
  case UTT_IsConst:
  case UTT_IsVolatile:
  case UTT_IsSigned:
  case UTT_IsUnsigned:
    return true;

    // C++0x [meta.unary.prop] Table 49 requires the following traits to be
    // applied to a complete type.
  case UTT_IsTrivial:
  case UTT_IsTriviallyCopyable:
  case UTT_IsStandardLayout:
  case UTT_IsPOD:
  case UTT_IsLiteral:
  case UTT_IsEmpty:
  case UTT_IsPolymorphic:
  case UTT_IsAbstract:
  case UTT_IsInterfaceClass:
    // Fall-through

  // These traits require a complete type.
  case UTT_IsFinal:

    // These trait expressions are designed to help implement predicates in
    // [meta.unary.prop] despite not being named the same. They are specified
    // by both GCC and the Embarcadero C++ compiler, and require the complete
    // type due to the overarching C++0x type predicates being implemented
    // requiring the complete type.
  case UTT_HasNothrowAssign:
  case UTT_HasNothrowConstructor:
  case UTT_HasNothrowCopy:
  case UTT_HasTrivialAssign:
  case UTT_HasTrivialDefaultConstructor:
  case UTT_HasTrivialCopy:
  case UTT_HasTrivialDestructor:
  case UTT_HasVirtualDestructor:
    // Arrays of unknown bound are expressly allowed.
    QualType ElTy = ArgTy;
    if (ArgTy->isIncompleteArrayType())
      ElTy = S.Context.getAsArrayType(ArgTy)->getElementType();

    // The void type is expressly allowed.
    if (ElTy->isVoidType())
      return true;

    return !S.RequireCompleteType(
      Loc, ElTy, diag::err_incomplete_type_used_in_type_trait_expr);
  }
  llvm_unreachable("Type trait not handled by switch");
}

static bool EvaluateUnaryTypeTrait(Sema &Self, UnaryTypeTrait UTT,
                                   SourceLocation KeyLoc, QualType T) {
  assert(!T->isDependentType() && "Cannot evaluate traits of dependent type");

  ASTContext &C = Self.Context;
  switch(UTT) {
    // Type trait expressions corresponding to the primary type category
    // predicates in C++0x [meta.unary.cat].
  case UTT_IsVoid:
    return T->isVoidType();
  case UTT_IsIntegral:
    return T->isIntegralType(C);
  case UTT_IsFloatingPoint:
    return T->isFloatingType();
  case UTT_IsArray:
    return T->isArrayType();
  case UTT_IsPointer:
    return T->isPointerType();
  case UTT_IsLvalueReference:
    return T->isLValueReferenceType();
  case UTT_IsRvalueReference:
    return T->isRValueReferenceType();
  case UTT_IsMemberFunctionPointer:
    return T->isMemberFunctionPointerType();
  case UTT_IsMemberObjectPointer:
    return T->isMemberDataPointerType();
  case UTT_IsEnum:
    return T->isEnumeralType();
  case UTT_IsUnion:
    return T->isUnionType();
  case UTT_IsClass:
    return T->isClassType() || T->isStructureType() || T->isInterfaceType();
  case UTT_IsFunction:
    return T->isFunctionType();

    // Type trait expressions which correspond to the convenient composition
    // predicates in C++0x [meta.unary.comp].
  case UTT_IsReference:
    return T->isReferenceType();
  case UTT_IsArithmetic:
    return T->isArithmeticType() && !T->isEnumeralType();
  case UTT_IsFundamental:
    return T->isFundamentalType();
  case UTT_IsObject:
    return T->isObjectType();
  case UTT_IsScalar:
    // Note: semantic analysis depends on Objective-C lifetime types to be
    // considered scalar types. However, such types do not actually behave
    // like scalar types at run time (since they may require retain/release
    // operations), so we report them as non-scalar.
    if (T->isObjCLifetimeType()) {
      switch (T.getObjCLifetime()) {
      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
        return true;

      case Qualifiers::OCL_Strong:
      case Qualifiers::OCL_Weak:
      case Qualifiers::OCL_Autoreleasing:
        return false;
      }
    }
      
    return T->isScalarType();
  case UTT_IsCompound:
    return T->isCompoundType();
  case UTT_IsMemberPointer:
    return T->isMemberPointerType();

    // Type trait expressions which correspond to the type property predicates
    // in C++0x [meta.unary.prop].
  case UTT_IsConst:
    return T.isConstQualified();
  case UTT_IsVolatile:
    return T.isVolatileQualified();
  case UTT_IsTrivial:
    return T.isTrivialType(Self.Context);
  case UTT_IsTriviallyCopyable:
    return T.isTriviallyCopyableType(Self.Context);
  case UTT_IsStandardLayout:
    return T->isStandardLayoutType();
  case UTT_IsPOD:
    return T.isPODType(Self.Context);
  case UTT_IsLiteral:
    return T->isLiteralType();
  case UTT_IsEmpty:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return !RD->isUnion() && RD->isEmpty();
    return false;
  case UTT_IsPolymorphic:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->isPolymorphic();
    return false;
  case UTT_IsAbstract:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->isAbstract();
    return false;
  case UTT_IsInterfaceClass:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->isInterface();
    return false;
  case UTT_IsFinal:
    if (const CXXRecordDecl *RD = T->getAsCXXRecordDecl())
      return RD->hasAttr<FinalAttr>();
    return false;
  case UTT_IsSigned:
    return T->isSignedIntegerType();
  case UTT_IsUnsigned:
    return T->isUnsignedIntegerType();

    // Type trait expressions which query classes regarding their construction,
    // destruction, and copying. Rather than being based directly on the
    // related type predicates in the standard, they are specified by both
    // GCC[1] and the Embarcadero C++ compiler[2], and Clang implements those
    // specifications.
    //
    //   1: http://gcc.gnu/.org/onlinedocs/gcc/Type-Traits.html
    //   2: http://docwiki.embarcadero.com/RADStudio/XE/en/Type_Trait_Functions_(C%2B%2B0x)_Index
  case UTT_HasTrivialDefaultConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true then the trait is true, else if type is
    //   a cv class or union type (or array thereof) with a trivial default
    //   constructor ([class.ctor]) then the trait is true, else it is false.
    if (T.isPODType(Self.Context))
      return true;
    if (const RecordType *RT =
          C.getBaseElementType(T)->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialDefaultConstructor();
    return false;
  case UTT_HasTrivialCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type then
    //   the trait is true, else if type is a cv class or union type
    //   with a trivial copy constructor ([class.copy]) then the trait
    //   is true, else it is false.
    if (T.isPODType(Self.Context) || T->isReferenceType())
      return true;
    if (const RecordType *RT = T->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialCopyConstructor();
    return false;
  case UTT_HasTrivialAssign:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is const qualified or is a reference type then the
    //   trait is false. Otherwise if __is_pod (type) is true then the
    //   trait is true, else if type is a cv class or union type with
    //   a trivial copy assignment ([class.copy]) then the trait is
    //   true, else it is false.
    // Note: the const and reference restrictions are interesting,
    // given that const and reference members don't prevent a class
    // from having a trivial copy assignment operator (but do cause
    // errors if the copy assignment operator is actually used, q.v.
    // [class.copy]p12).

    if (C.getBaseElementType(T).isConstQualified())
      return false;
    if (T.isPODType(Self.Context))
      return true;
    if (const RecordType *RT = T->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialCopyAssignment();
    return false;
  case UTT_HasTrivialDestructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type
    //   then the trait is true, else if type is a cv class or union
    //   type (or array thereof) with a trivial destructor
    //   ([class.dtor]) then the trait is true, else it is
    //   false.
    if (T.isPODType(Self.Context) || T->isReferenceType())
      return true;
      
    // Objective-C++ ARC: autorelease types don't require destruction.
    if (T->isObjCLifetimeType() && 
        T.getObjCLifetime() == Qualifiers::OCL_Autoreleasing)
      return true;
      
    if (const RecordType *RT =
          C.getBaseElementType(T)->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialDestructor();
    return false;
  // TODO: Propagate nothrowness for implicitly declared special members.
  case UTT_HasNothrowAssign:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is const qualified or is a reference type then the
    //   trait is false. Otherwise if __has_trivial_assign (type)
    //   is true then the trait is true, else if type is a cv class
    //   or union type with copy assignment operators that are known
    //   not to throw an exception then the trait is true, else it is
    //   false.
    if (C.getBaseElementType(T).isConstQualified())
      return false;
    if (T->isReferenceType())
      return false;
    if (T.isPODType(Self.Context) || T->isObjCLifetimeType())
      return true;     
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl* RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialCopyAssignment())
        return true;

      bool FoundAssign = false;
      DeclarationName Name = C.DeclarationNames.getCXXOperatorName(OO_Equal);
      LookupResult Res(Self, DeclarationNameInfo(Name, KeyLoc),
                       Sema::LookupOrdinaryName);
      if (Self.LookupQualifiedName(Res, RD)) {
        Res.suppressDiagnostics();
        for (LookupResult::iterator Op = Res.begin(), OpEnd = Res.end();
             Op != OpEnd; ++Op) {
          if (isa<FunctionTemplateDecl>(*Op))
            continue;
          
          CXXMethodDecl *Operator = cast<CXXMethodDecl>(*Op);
          if (Operator->isCopyAssignmentOperator()) {
            FoundAssign = true;
            const FunctionProtoType *CPT
                = Operator->getType()->getAs<FunctionProtoType>();
            CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
            if (!CPT)
              return false;
            if (!CPT->isNothrow(Self.Context))
              return false;
          }
        }
      }
      
      return FoundAssign;
    }
    return false;
  case UTT_HasNothrowCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __has_trivial_copy (type) is true then the trait is true, else
    //   if type is a cv class or union type with copy constructors that are
    //   known not to throw an exception then the trait is true, else it is
    //   false.
    if (T.isPODType(C) || T->isReferenceType() || T->isObjCLifetimeType())
      return true;
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialCopyConstructor())
        return true;

      bool FoundConstructor = false;
      unsigned FoundTQs;
      DeclContext::lookup_const_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd) = Self.LookupConstructors(RD);
           Con != ConEnd; ++Con) {
        // A template constructor is never a copy constructor.
        // FIXME: However, it may actually be selected at the actual overload
        // resolution point.
        if (isa<FunctionTemplateDecl>(*Con))
          continue;
        CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
        if (Constructor->isCopyConstructor(FoundTQs)) {
          FoundConstructor = true;
          const FunctionProtoType *CPT
              = Constructor->getType()->getAs<FunctionProtoType>();
          CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
          if (!CPT)
            return false;
          // FIXME: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          if (!CPT->isNothrow(Self.Context) || CPT->getNumArgs() > 1)
            return false;
        }
      }

      return FoundConstructor;
    }
    return false;
  case UTT_HasNothrowConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __has_trivial_constructor (type) is true then the trait is
    //   true, else if type is a cv class or union type (or array
    //   thereof) with a default constructor that is known not to
    //   throw an exception then the trait is true, else it is false.
    if (T.isPODType(C) || T->isObjCLifetimeType())
      return true;
    if (const RecordType *RT = C.getBaseElementType(T)->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialDefaultConstructor())
        return true;

      DeclContext::lookup_const_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd) = Self.LookupConstructors(RD);
           Con != ConEnd; ++Con) {
        // FIXME: In C++0x, a constructor template can be a default constructor.
        if (isa<FunctionTemplateDecl>(*Con))
          continue;
        CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
        if (Constructor->isDefaultConstructor()) {
          const FunctionProtoType *CPT
              = Constructor->getType()->getAs<FunctionProtoType>();
          CPT = Self.ResolveExceptionSpec(KeyLoc, CPT);
          if (!CPT)
            return false;
          // TODO: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          return CPT->isNothrow(Self.Context) && CPT->getNumArgs() == 0;
        }
      }
    }
    return false;
  case UTT_HasVirtualDestructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is a class type with a virtual destructor ([class.dtor])
    //   then the trait is true, else it is false.
    if (const RecordType *Record = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(Record->getDecl());
      if (CXXDestructorDecl *Destructor = Self.LookupDestructor(RD))
        return Destructor->isVirtual();
    }
    return false;

    // These type trait expressions are modeled on the specifications for the
    // Embarcadero C++0x type trait functions:
    //   http://docwiki.embarcadero.com/RADStudio/XE/en/Type_Trait_Functions_(C%2B%2B0x)_Index
  case UTT_IsCompleteType:
    // http://docwiki.embarcadero.com/RADStudio/XE/en/Is_complete_type_(typename_T_):
    //   Returns True if and only if T is a complete type at the point of the
    //   function call.
    return !T->isIncompleteType();
  }
  llvm_unreachable("Type trait not covered by switch");
}

ExprResult Sema::BuildUnaryTypeTrait(UnaryTypeTrait UTT,
                                     SourceLocation KWLoc,
                                     TypeSourceInfo *TSInfo,
                                     SourceLocation RParen) {
  QualType T = TSInfo->getType();
  if (!CheckUnaryTypeTraitTypeCompleteness(*this, UTT, KWLoc, T))
    return ExprError();

  bool Value = false;
  if (!T->isDependentType())
    Value = EvaluateUnaryTypeTrait(*this, UTT, KWLoc, T);

  return Owned(new (Context) UnaryTypeTraitExpr(KWLoc, UTT, TSInfo, Value,
                                                RParen, Context.BoolTy));
}

ExprResult Sema::ActOnBinaryTypeTrait(BinaryTypeTrait BTT,
                                      SourceLocation KWLoc,
                                      ParsedType LhsTy,
                                      ParsedType RhsTy,
                                      SourceLocation RParen) {
  TypeSourceInfo *LhsTSInfo;
  QualType LhsT = GetTypeFromParser(LhsTy, &LhsTSInfo);
  if (!LhsTSInfo)
    LhsTSInfo = Context.getTrivialTypeSourceInfo(LhsT);

  TypeSourceInfo *RhsTSInfo;
  QualType RhsT = GetTypeFromParser(RhsTy, &RhsTSInfo);
  if (!RhsTSInfo)
    RhsTSInfo = Context.getTrivialTypeSourceInfo(RhsT);

  return BuildBinaryTypeTrait(BTT, KWLoc, LhsTSInfo, RhsTSInfo, RParen);
}

/// \brief Determine whether T has a non-trivial Objective-C lifetime in
/// ARC mode.
static bool hasNontrivialObjCLifetime(QualType T) {
  switch (T.getObjCLifetime()) {
  case Qualifiers::OCL_ExplicitNone:
    return false;

  case Qualifiers::OCL_Strong:
  case Qualifiers::OCL_Weak:
  case Qualifiers::OCL_Autoreleasing:
    return true;

  case Qualifiers::OCL_None:
    return T->isObjCLifetimeType();
  }

  llvm_unreachable("Unknown ObjC lifetime qualifier");
}

static bool evaluateTypeTrait(Sema &S, TypeTrait Kind, SourceLocation KWLoc,
                              ArrayRef<TypeSourceInfo *> Args,
                              SourceLocation RParenLoc) {
  switch (Kind) {
  case clang::TT_IsTriviallyConstructible: {
    // C++11 [meta.unary.prop]:
    //   is_trivially_constructible is defined as:
    //
    //     is_constructible<T, Args...>::value is true and the variable
    //     definition for is_constructible, as defined below, is known to call no
    //     operation that is not trivial.
    //
    //   The predicate condition for a template specialization 
    //   is_constructible<T, Args...> shall be satisfied if and only if the 
    //   following variable definition would be well-formed for some invented 
    //   variable t:
    //
    //     T t(create<Args>()...);
    if (Args.empty()) {
      S.Diag(KWLoc, diag::err_type_trait_arity)
        << 1 << 1 << 1 << (int)Args.size();
      return false;
    }
    
    bool SawVoid = false;
    for (unsigned I = 0, N = Args.size(); I != N; ++I) {
      if (Args[I]->getType()->isVoidType()) {
        SawVoid = true;
        continue;
      }
      
      if (!Args[I]->getType()->isIncompleteType() &&
        S.RequireCompleteType(KWLoc, Args[I]->getType(), 
          diag::err_incomplete_type_used_in_type_trait_expr))
        return false;
    }
    
    // If any argument was 'void', of course it won't type-check.
    if (SawVoid)
      return false;
    
    llvm::SmallVector<OpaqueValueExpr, 2> OpaqueArgExprs;
    llvm::SmallVector<Expr *, 2> ArgExprs;
    ArgExprs.reserve(Args.size() - 1);
    for (unsigned I = 1, N = Args.size(); I != N; ++I) {
      QualType T = Args[I]->getType();
      if (T->isObjectType() || T->isFunctionType())
        T = S.Context.getRValueReferenceType(T);
      OpaqueArgExprs.push_back(
        OpaqueValueExpr(Args[I]->getTypeLoc().getLocStart(), 
                        T.getNonLValueExprType(S.Context),
                        Expr::getValueKindForType(T)));
      ArgExprs.push_back(&OpaqueArgExprs.back());
    }
    
    // Perform the initialization in an unevaluated context within a SFINAE 
    // trap at translation unit scope.
    EnterExpressionEvaluationContext Unevaluated(S, Sema::Unevaluated);
    Sema::SFINAETrap SFINAE(S, /*AccessCheckingSFINAE=*/true);
    Sema::ContextRAII TUContext(S, S.Context.getTranslationUnitDecl());
    InitializedEntity To(InitializedEntity::InitializeTemporary(Args[0]));
    InitializationKind InitKind(InitializationKind::CreateDirect(KWLoc, KWLoc,
                                                                 RParenLoc));
    InitializationSequence Init(S, To, InitKind, 
                                ArgExprs.begin(), ArgExprs.size());
    if (Init.Failed())
      return false;
    
    ExprResult Result = Init.Perform(S, To, InitKind, ArgExprs);
    if (Result.isInvalid() || SFINAE.hasErrorOccurred())
      return false;

    // Under Objective-C ARC, if the destination has non-trivial Objective-C
    // lifetime, this is a non-trivial construction.
    if (S.getLangOpts().ObjCAutoRefCount &&
        hasNontrivialObjCLifetime(Args[0]->getType().getNonReferenceType()))
      return false;

    // The initialization succeeded; now make sure there are no non-trivial
    // calls.
    return !Result.get()->hasNonTrivialCall(S.Context);
  }
  }
  
  return false;
}

ExprResult Sema::BuildTypeTrait(TypeTrait Kind, SourceLocation KWLoc, 
                                ArrayRef<TypeSourceInfo *> Args, 
                                SourceLocation RParenLoc) {
  bool Dependent = false;
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    if (Args[I]->getType()->isDependentType()) {
      Dependent = true;
      break;
    }
  }
  
  bool Value = false;
  if (!Dependent)
    Value = evaluateTypeTrait(*this, Kind, KWLoc, Args, RParenLoc);
  
  return TypeTraitExpr::Create(Context, Context.BoolTy, KWLoc, Kind,
                               Args, RParenLoc, Value);
}

ExprResult Sema::ActOnTypeTrait(TypeTrait Kind, SourceLocation KWLoc, 
                                ArrayRef<ParsedType> Args, 
                                SourceLocation RParenLoc) {
  llvm::SmallVector<TypeSourceInfo *, 4> ConvertedArgs;
  ConvertedArgs.reserve(Args.size());
  
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    TypeSourceInfo *TInfo;
    QualType T = GetTypeFromParser(Args[I], &TInfo);
    if (!TInfo)
      TInfo = Context.getTrivialTypeSourceInfo(T, KWLoc);
    
    ConvertedArgs.push_back(TInfo);    
  }
  
  return BuildTypeTrait(Kind, KWLoc, ConvertedArgs, RParenLoc);
}

static bool EvaluateBinaryTypeTrait(Sema &Self, BinaryTypeTrait BTT,
                                    QualType LhsT, QualType RhsT,
                                    SourceLocation KeyLoc) {
  assert(!LhsT->isDependentType() && !RhsT->isDependentType() &&
         "Cannot evaluate traits of dependent types");

  switch(BTT) {
  case BTT_IsBaseOf: {
    // C++0x [meta.rel]p2
    // Base is a base class of Derived without regard to cv-qualifiers or
    // Base and Derived are not unions and name the same class type without
    // regard to cv-qualifiers.

    const RecordType *lhsRecord = LhsT->getAs<RecordType>();
    if (!lhsRecord) return false;

    const RecordType *rhsRecord = RhsT->getAs<RecordType>();
    if (!rhsRecord) return false;

    assert(Self.Context.hasSameUnqualifiedType(LhsT, RhsT)
             == (lhsRecord == rhsRecord));

    if (lhsRecord == rhsRecord)
      return !lhsRecord->getDecl()->isUnion();

    // C++0x [meta.rel]p2:
    //   If Base and Derived are class types and are different types
    //   (ignoring possible cv-qualifiers) then Derived shall be a
    //   complete type.
    if (Self.RequireCompleteType(KeyLoc, RhsT, 
                          diag::err_incomplete_type_used_in_type_trait_expr))
      return false;

    return cast<CXXRecordDecl>(rhsRecord->getDecl())
      ->isDerivedFrom(cast<CXXRecordDecl>(lhsRecord->getDecl()));
  }
  case BTT_IsSame:
    return Self.Context.hasSameType(LhsT, RhsT);
  case BTT_TypeCompatible:
    return Self.Context.typesAreCompatible(LhsT.getUnqualifiedType(),
                                           RhsT.getUnqualifiedType());
  case BTT_IsConvertible:
  case BTT_IsConvertibleTo: {
    // C++0x [meta.rel]p4:
    //   Given the following function prototype:
    //
    //     template <class T> 
    //       typename add_rvalue_reference<T>::type create();
    //
    //   the predicate condition for a template specialization 
    //   is_convertible<From, To> shall be satisfied if and only if 
    //   the return expression in the following code would be 
    //   well-formed, including any implicit conversions to the return
    //   type of the function:
    //
    //     To test() { 
    //       return create<From>();
    //     }
    //
    //   Access checking is performed as if in a context unrelated to To and 
    //   From. Only the validity of the immediate context of the expression 
    //   of the return-statement (including conversions to the return type)
    //   is considered.
    //
    // We model the initialization as a copy-initialization of a temporary
    // of the appropriate type, which for this expression is identical to the
    // return statement (since NRVO doesn't apply).

    // Functions aren't allowed to return function or array types.
    if (RhsT->isFunctionType() || RhsT->isArrayType())
      return false;

    // A return statement in a void function must have void type.
    if (RhsT->isVoidType())
      return LhsT->isVoidType();

    // A function definition requires a complete, non-abstract return type.
    if (Self.RequireCompleteType(KeyLoc, RhsT, 0) ||
        Self.RequireNonAbstractType(KeyLoc, RhsT, 0))
      return false;

    // Compute the result of add_rvalue_reference.
    if (LhsT->isObjectType() || LhsT->isFunctionType())
      LhsT = Self.Context.getRValueReferenceType(LhsT);

    // Build a fake source and destination for initialization.
    InitializedEntity To(InitializedEntity::InitializeTemporary(RhsT));
    OpaqueValueExpr From(KeyLoc, LhsT.getNonLValueExprType(Self.Context),
                         Expr::getValueKindForType(LhsT));
    Expr *FromPtr = &From;
    InitializationKind Kind(InitializationKind::CreateCopy(KeyLoc, 
                                                           SourceLocation()));
    
    // Perform the initialization in an unevaluated context within a SFINAE 
    // trap at translation unit scope.
    EnterExpressionEvaluationContext Unevaluated(Self, Sema::Unevaluated);
    Sema::SFINAETrap SFINAE(Self, /*AccessCheckingSFINAE=*/true);
    Sema::ContextRAII TUContext(Self, Self.Context.getTranslationUnitDecl());
    InitializationSequence Init(Self, To, Kind, &FromPtr, 1);
    if (Init.Failed())
      return false;

    ExprResult Result = Init.Perform(Self, To, Kind, FromPtr);
    return !Result.isInvalid() && !SFINAE.hasErrorOccurred();
  }
      
  case BTT_IsTriviallyAssignable: {
    // C++11 [meta.unary.prop]p3:
    //   is_trivially_assignable is defined as:
    //     is_assignable<T, U>::value is true and the assignment, as defined by
    //     is_assignable, is known to call no operation that is not trivial
    //
    //   is_assignable is defined as:
    //     The expression declval<T>() = declval<U>() is well-formed when 
    //     treated as an unevaluated operand (Clause 5).
    //
    //   For both, T and U shall be complete types, (possibly cv-qualified) 
    //   void, or arrays of unknown bound.
    if (!LhsT->isVoidType() && !LhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(KeyLoc, LhsT, 
          diag::err_incomplete_type_used_in_type_trait_expr))
      return false;
    if (!RhsT->isVoidType() && !RhsT->isIncompleteArrayType() &&
        Self.RequireCompleteType(KeyLoc, RhsT, 
          diag::err_incomplete_type_used_in_type_trait_expr))
      return false;

    // cv void is never assignable.
    if (LhsT->isVoidType() || RhsT->isVoidType())
      return false;

    // Build expressions that emulate the effect of declval<T>() and 
    // declval<U>().
    if (LhsT->isObjectType() || LhsT->isFunctionType())
      LhsT = Self.Context.getRValueReferenceType(LhsT);
    if (RhsT->isObjectType() || RhsT->isFunctionType())
      RhsT = Self.Context.getRValueReferenceType(RhsT);
    OpaqueValueExpr Lhs(KeyLoc, LhsT.getNonLValueExprType(Self.Context),
                        Expr::getValueKindForType(LhsT));
    OpaqueValueExpr Rhs(KeyLoc, RhsT.getNonLValueExprType(Self.Context),
                        Expr::getValueKindForType(RhsT));
    
    // Attempt the assignment in an unevaluated context within a SFINAE 
    // trap at translation unit scope.
    EnterExpressionEvaluationContext Unevaluated(Self, Sema::Unevaluated);
    Sema::SFINAETrap SFINAE(Self, /*AccessCheckingSFINAE=*/true);
    Sema::ContextRAII TUContext(Self, Self.Context.getTranslationUnitDecl());
    ExprResult Result = Self.BuildBinOp(/*S=*/0, KeyLoc, BO_Assign, &Lhs, &Rhs);
    if (Result.isInvalid() || SFINAE.hasErrorOccurred())
      return false;

    // Under Objective-C ARC, if the destination has non-trivial Objective-C
    // lifetime, this is a non-trivial assignment.
    if (Self.getLangOpts().ObjCAutoRefCount &&
        hasNontrivialObjCLifetime(LhsT.getNonReferenceType()))
      return false;

    return !Result.get()->hasNonTrivialCall(Self.Context);
  }
  }
  llvm_unreachable("Unknown type trait or not implemented");
}

ExprResult Sema::BuildBinaryTypeTrait(BinaryTypeTrait BTT,
                                      SourceLocation KWLoc,
                                      TypeSourceInfo *LhsTSInfo,
                                      TypeSourceInfo *RhsTSInfo,
                                      SourceLocation RParen) {
  QualType LhsT = LhsTSInfo->getType();
  QualType RhsT = RhsTSInfo->getType();

  if (BTT == BTT_TypeCompatible) {
    if (getLangOpts().CPlusPlus) {
      Diag(KWLoc, diag::err_types_compatible_p_in_cplusplus)
        << SourceRange(KWLoc, RParen);
      return ExprError();
    }
  }

  bool Value = false;
  if (!LhsT->isDependentType() && !RhsT->isDependentType())
    Value = EvaluateBinaryTypeTrait(*this, BTT, LhsT, RhsT, KWLoc);

  // Select trait result type.
  QualType ResultType;
  switch (BTT) {
  case BTT_IsBaseOf:       ResultType = Context.BoolTy; break;
  case BTT_IsConvertible:  ResultType = Context.BoolTy; break;
  case BTT_IsSame:         ResultType = Context.BoolTy; break;
  case BTT_TypeCompatible: ResultType = Context.IntTy; break;
  case BTT_IsConvertibleTo: ResultType = Context.BoolTy; break;
  case BTT_IsTriviallyAssignable: ResultType = Context.BoolTy;
  }

  return Owned(new (Context) BinaryTypeTraitExpr(KWLoc, BTT, LhsTSInfo,
                                                 RhsTSInfo, Value, RParen,
                                                 ResultType));
}

ExprResult Sema::ActOnArrayTypeTrait(ArrayTypeTrait ATT,
                                     SourceLocation KWLoc,
                                     ParsedType Ty,
                                     Expr* DimExpr,
                                     SourceLocation RParen) {
  TypeSourceInfo *TSInfo;
  QualType T = GetTypeFromParser(Ty, &TSInfo);
  if (!TSInfo)
    TSInfo = Context.getTrivialTypeSourceInfo(T);

  return BuildArrayTypeTrait(ATT, KWLoc, TSInfo, DimExpr, RParen);
}

static uint64_t EvaluateArrayTypeTrait(Sema &Self, ArrayTypeTrait ATT,
                                           QualType T, Expr *DimExpr,
                                           SourceLocation KeyLoc) {
  assert(!T->isDependentType() && "Cannot evaluate traits of dependent type");

  switch(ATT) {
  case ATT_ArrayRank:
    if (T->isArrayType()) {
      unsigned Dim = 0;
      while (const ArrayType *AT = Self.Context.getAsArrayType(T)) {
        ++Dim;
        T = AT->getElementType();
      }
      return Dim;
    }
    return 0;

  case ATT_ArrayExtent: {
    llvm::APSInt Value;
    uint64_t Dim;
    if (Self.VerifyIntegerConstantExpression(DimExpr, &Value,
          diag::err_dimension_expr_not_constant_integer,
          false).isInvalid())
      return 0;
    if (Value.isSigned() && Value.isNegative()) {
      Self.Diag(KeyLoc, diag::err_dimension_expr_not_constant_integer)
        << DimExpr->getSourceRange();
      return 0;
    }
    Dim = Value.getLimitedValue();

    if (T->isArrayType()) {
      unsigned D = 0;
      bool Matched = false;
      while (const ArrayType *AT = Self.Context.getAsArrayType(T)) {
        if (Dim == D) {
          Matched = true;
          break;
        }
        ++D;
        T = AT->getElementType();
      }

      if (Matched && T->isArrayType()) {
        if (const ConstantArrayType *CAT = Self.Context.getAsConstantArrayType(T))
          return CAT->getSize().getLimitedValue();
      }
    }
    return 0;
  }
  }
  llvm_unreachable("Unknown type trait or not implemented");
}

ExprResult Sema::BuildArrayTypeTrait(ArrayTypeTrait ATT,
                                     SourceLocation KWLoc,
                                     TypeSourceInfo *TSInfo,
                                     Expr* DimExpr,
                                     SourceLocation RParen) {
  QualType T = TSInfo->getType();

  // FIXME: This should likely be tracked as an APInt to remove any host
  // assumptions about the width of size_t on the target.
  uint64_t Value = 0;
  if (!T->isDependentType())
    Value = EvaluateArrayTypeTrait(*this, ATT, T, DimExpr, KWLoc);

  // While the specification for these traits from the Embarcadero C++
  // compiler's documentation says the return type is 'unsigned int', Clang
  // returns 'size_t'. On Windows, the primary platform for the Embarcadero
  // compiler, there is no difference. On several other platforms this is an
  // important distinction.
  return Owned(new (Context) ArrayTypeTraitExpr(KWLoc, ATT, TSInfo, Value,
                                                DimExpr, RParen,
                                                Context.getSizeType()));
}

ExprResult Sema::ActOnExpressionTrait(ExpressionTrait ET,
                                      SourceLocation KWLoc,
                                      Expr *Queried,
                                      SourceLocation RParen) {
  // If error parsing the expression, ignore.
  if (!Queried)
    return ExprError();

  ExprResult Result = BuildExpressionTrait(ET, KWLoc, Queried, RParen);

  return Result;
}

static bool EvaluateExpressionTrait(ExpressionTrait ET, Expr *E) {
  switch (ET) {
  case ET_IsLValueExpr: return E->isLValue();
  case ET_IsRValueExpr: return E->isRValue();
  }
  llvm_unreachable("Expression trait not covered by switch");
}

ExprResult Sema::BuildExpressionTrait(ExpressionTrait ET,
                                      SourceLocation KWLoc,
                                      Expr *Queried,
                                      SourceLocation RParen) {
  if (Queried->isTypeDependent()) {
    // Delay type-checking for type-dependent expressions.
  } else if (Queried->getType()->isPlaceholderType()) {
    ExprResult PE = CheckPlaceholderExpr(Queried);
    if (PE.isInvalid()) return ExprError();
    return BuildExpressionTrait(ET, KWLoc, PE.take(), RParen);
  }

  bool Value = EvaluateExpressionTrait(ET, Queried);

  return Owned(new (Context) ExpressionTraitExpr(KWLoc, ET, Queried, Value,
                                                 RParen, Context.BoolTy));
}

QualType Sema::CheckPointerToMemberOperands(ExprResult &LHS, ExprResult &RHS,
                                            ExprValueKind &VK,
                                            SourceLocation Loc,
                                            bool isIndirect) {
  assert(!LHS.get()->getType()->isPlaceholderType() &&
         !RHS.get()->getType()->isPlaceholderType() &&
         "placeholders should have been weeded out by now");

  // The LHS undergoes lvalue conversions if this is ->*.
  if (isIndirect) {
    LHS = DefaultLvalueConversion(LHS.take());
    if (LHS.isInvalid()) return QualType();
  }

  // The RHS always undergoes lvalue conversions.
  RHS = DefaultLvalueConversion(RHS.take());
  if (RHS.isInvalid()) return QualType();

  const char *OpSpelling = isIndirect ? "->*" : ".*";
  // C++ 5.5p2
  //   The binary operator .* [p3: ->*] binds its second operand, which shall
  //   be of type "pointer to member of T" (where T is a completely-defined
  //   class type) [...]
  QualType RHSType = RHS.get()->getType();
  const MemberPointerType *MemPtr = RHSType->getAs<MemberPointerType>();
  if (!MemPtr) {
    Diag(Loc, diag::err_bad_memptr_rhs)
      << OpSpelling << RHSType << RHS.get()->getSourceRange();
    return QualType();
  }

  QualType Class(MemPtr->getClass(), 0);

  // Note: C++ [expr.mptr.oper]p2-3 says that the class type into which the
  // member pointer points must be completely-defined. However, there is no
  // reason for this semantic distinction, and the rule is not enforced by
  // other compilers. Therefore, we do not check this property, as it is
  // likely to be considered a defect.

  // C++ 5.5p2
  //   [...] to its first operand, which shall be of class T or of a class of
  //   which T is an unambiguous and accessible base class. [p3: a pointer to
  //   such a class]
  QualType LHSType = LHS.get()->getType();
  if (isIndirect) {
    if (const PointerType *Ptr = LHSType->getAs<PointerType>())
      LHSType = Ptr->getPointeeType();
    else {
      Diag(Loc, diag::err_bad_memptr_lhs)
        << OpSpelling << 1 << LHSType
        << FixItHint::CreateReplacement(SourceRange(Loc), ".*");
      return QualType();
    }
  }

  if (!Context.hasSameUnqualifiedType(Class, LHSType)) {
    // If we want to check the hierarchy, we need a complete type.
    if (RequireCompleteType(Loc, LHSType, diag::err_bad_memptr_lhs,
                            OpSpelling, (int)isIndirect)) {
      return QualType();
    }
    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);
    // FIXME: Would it be useful to print full ambiguity paths, or is that
    // overkill?
    if (!IsDerivedFrom(LHSType, Class, Paths) ||
        Paths.isAmbiguous(Context.getCanonicalType(Class))) {
      Diag(Loc, diag::err_bad_memptr_lhs) << OpSpelling
        << (int)isIndirect << LHS.get()->getType();
      return QualType();
    }
    // Cast LHS to type of use.
    QualType UseType = isIndirect ? Context.getPointerType(Class) : Class;
    ExprValueKind VK = isIndirect ? VK_RValue : LHS.get()->getValueKind();

    CXXCastPath BasePath;
    BuildBasePathArray(Paths, BasePath);
    LHS = ImpCastExprToType(LHS.take(), UseType, CK_DerivedToBase, VK,
                            &BasePath);
  }

  if (isa<CXXScalarValueInitExpr>(RHS.get()->IgnoreParens())) {
    // Diagnose use of pointer-to-member type which when used as
    // the functional cast in a pointer-to-member expression.
    Diag(Loc, diag::err_pointer_to_member_type) << isIndirect;
     return QualType();
  }

  // C++ 5.5p2
  //   The result is an object or a function of the type specified by the
  //   second operand.
  // The cv qualifiers are the union of those in the pointer and the left side,
  // in accordance with 5.5p5 and 5.2.5.
  QualType Result = MemPtr->getPointeeType();
  Result = Context.getCVRQualifiedType(Result, LHSType.getCVRQualifiers());

  // C++0x [expr.mptr.oper]p6:
  //   In a .* expression whose object expression is an rvalue, the program is
  //   ill-formed if the second operand is a pointer to member function with
  //   ref-qualifier &. In a ->* expression or in a .* expression whose object
  //   expression is an lvalue, the program is ill-formed if the second operand
  //   is a pointer to member function with ref-qualifier &&.
  if (const FunctionProtoType *Proto = Result->getAs<FunctionProtoType>()) {
    switch (Proto->getRefQualifier()) {
    case RQ_None:
      // Do nothing
      break;

    case RQ_LValue:
      if (!isIndirect && !LHS.get()->Classify(Context).isLValue())
        Diag(Loc, diag::err_pointer_to_member_oper_value_classify)
          << RHSType << 1 << LHS.get()->getSourceRange();
      break;

    case RQ_RValue:
      if (isIndirect || !LHS.get()->Classify(Context).isRValue())
        Diag(Loc, diag::err_pointer_to_member_oper_value_classify)
          << RHSType << 0 << LHS.get()->getSourceRange();
      break;
    }
  }

  // C++ [expr.mptr.oper]p6:
  //   The result of a .* expression whose second operand is a pointer
  //   to a data member is of the same value category as its
  //   first operand. The result of a .* expression whose second
  //   operand is a pointer to a member function is a prvalue. The
  //   result of an ->* expression is an lvalue if its second operand
  //   is a pointer to data member and a prvalue otherwise.
  if (Result->isFunctionType()) {
    VK = VK_RValue;
    return Context.BoundMemberTy;
  } else if (isIndirect) {
    VK = VK_LValue;
  } else {
    VK = LHS.get()->getValueKind();
  }

  return Result;
}

/// \brief Try to convert a type to another according to C++0x 5.16p3.
///
/// This is part of the parameter validation for the ? operator. If either
/// value operand is a class type, the two operands are attempted to be
/// converted to each other. This function does the conversion in one direction.
/// It returns true if the program is ill-formed and has already been diagnosed
/// as such.
static bool TryClassUnification(Sema &Self, Expr *From, Expr *To,
                                SourceLocation QuestionLoc,
                                bool &HaveConversion,
                                QualType &ToType) {
  HaveConversion = false;
  ToType = To->getType();

  InitializationKind Kind = InitializationKind::CreateCopy(To->getLocStart(),
                                                           SourceLocation());
  // C++0x 5.16p3
  //   The process for determining whether an operand expression E1 of type T1
  //   can be converted to match an operand expression E2 of type T2 is defined
  //   as follows:
  //   -- If E2 is an lvalue:
  bool ToIsLvalue = To->isLValue();
  if (ToIsLvalue) {
    //   E1 can be converted to match E2 if E1 can be implicitly converted to
    //   type "lvalue reference to T2", subject to the constraint that in the
    //   conversion the reference must bind directly to E1.
    QualType T = Self.Context.getLValueReferenceType(ToType);
    InitializedEntity Entity = InitializedEntity::InitializeTemporary(T);

    InitializationSequence InitSeq(Self, Entity, Kind, &From, 1);
    if (InitSeq.isDirectReferenceBinding()) {
      ToType = T;
      HaveConversion = true;
      return false;
    }

    if (InitSeq.isAmbiguous())
      return InitSeq.Diagnose(Self, Entity, Kind, &From, 1);
  }

  //   -- If E2 is an rvalue, or if the conversion above cannot be done:
  //      -- if E1 and E2 have class type, and the underlying class types are
  //         the same or one is a base class of the other:
  QualType FTy = From->getType();
  QualType TTy = To->getType();
  const RecordType *FRec = FTy->getAs<RecordType>();
  const RecordType *TRec = TTy->getAs<RecordType>();
  bool FDerivedFromT = FRec && TRec && FRec != TRec &&
                       Self.IsDerivedFrom(FTy, TTy);
  if (FRec && TRec &&
      (FRec == TRec || FDerivedFromT || Self.IsDerivedFrom(TTy, FTy))) {
    //         E1 can be converted to match E2 if the class of T2 is the
    //         same type as, or a base class of, the class of T1, and
    //         [cv2 > cv1].
    if (FRec == TRec || FDerivedFromT) {
      if (TTy.isAtLeastAsQualifiedAs(FTy)) {
        InitializedEntity Entity = InitializedEntity::InitializeTemporary(TTy);
        InitializationSequence InitSeq(Self, Entity, Kind, &From, 1);
        if (InitSeq) {
          HaveConversion = true;
          return false;
        }

        if (InitSeq.isAmbiguous())
          return InitSeq.Diagnose(Self, Entity, Kind, &From, 1);
      }
    }

    return false;
  }

  //     -- Otherwise: E1 can be converted to match E2 if E1 can be
  //        implicitly converted to the type that expression E2 would have
  //        if E2 were converted to an rvalue (or the type it has, if E2 is
  //        an rvalue).
  //
  // This actually refers very narrowly to the lvalue-to-rvalue conversion, not
  // to the array-to-pointer or function-to-pointer conversions.
  if (!TTy->getAs<TagType>())
    TTy = TTy.getUnqualifiedType();

  InitializedEntity Entity = InitializedEntity::InitializeTemporary(TTy);
  InitializationSequence InitSeq(Self, Entity, Kind, &From, 1);
  HaveConversion = !InitSeq.Failed();
  ToType = TTy;
  if (InitSeq.isAmbiguous())
    return InitSeq.Diagnose(Self, Entity, Kind, &From, 1);

  return false;
}

/// \brief Try to find a common type for two according to C++0x 5.16p5.
///
/// This is part of the parameter validation for the ? operator. If either
/// value operand is a class type, overload resolution is used to find a
/// conversion to a common type.
static bool FindConditionalOverload(Sema &Self, ExprResult &LHS, ExprResult &RHS,
                                    SourceLocation QuestionLoc) {
  Expr *Args[2] = { LHS.get(), RHS.get() };
  OverloadCandidateSet CandidateSet(QuestionLoc);
  Self.AddBuiltinOperatorCandidates(OO_Conditional, QuestionLoc, Args, 2,
                                    CandidateSet);

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(Self, QuestionLoc, Best)) {
    case OR_Success: {
      // We found a match. Perform the conversions on the arguments and move on.
      ExprResult LHSRes =
        Self.PerformImplicitConversion(LHS.get(), Best->BuiltinTypes.ParamTypes[0],
                                       Best->Conversions[0], Sema::AA_Converting);
      if (LHSRes.isInvalid())
        break;
      LHS = LHSRes;

      ExprResult RHSRes =
        Self.PerformImplicitConversion(RHS.get(), Best->BuiltinTypes.ParamTypes[1],
                                       Best->Conversions[1], Sema::AA_Converting);
      if (RHSRes.isInvalid())
        break;
      RHS = RHSRes;
      if (Best->Function)
        Self.MarkFunctionReferenced(QuestionLoc, Best->Function);
      return false;
    }
    
    case OR_No_Viable_Function:

      // Emit a better diagnostic if one of the expressions is a null pointer
      // constant and the other is a pointer type. In this case, the user most
      // likely forgot to take the address of the other expression.
      if (Self.DiagnoseConditionalForNull(LHS.get(), RHS.get(), QuestionLoc))
        return true;

      Self.Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
        << LHS.get()->getType() << RHS.get()->getType()
        << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      return true;

    case OR_Ambiguous:
      Self.Diag(QuestionLoc, diag::err_conditional_ambiguous_ovl)
        << LHS.get()->getType() << RHS.get()->getType()
        << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      // FIXME: Print the possible common types by printing the return types of
      // the viable candidates.
      break;

    case OR_Deleted:
      llvm_unreachable("Conditional operator has only built-in overloads");
  }
  return true;
}

/// \brief Perform an "extended" implicit conversion as returned by
/// TryClassUnification.
static bool ConvertForConditional(Sema &Self, ExprResult &E, QualType T) {
  InitializedEntity Entity = InitializedEntity::InitializeTemporary(T);
  InitializationKind Kind = InitializationKind::CreateCopy(E.get()->getLocStart(),
                                                           SourceLocation());
  Expr *Arg = E.take();
  InitializationSequence InitSeq(Self, Entity, Kind, &Arg, 1);
  ExprResult Result = InitSeq.Perform(Self, Entity, Kind, Arg);
  if (Result.isInvalid())
    return true;

  E = Result;
  return false;
}

/// \brief Check the operands of ?: under C++ semantics.
///
/// See C++ [expr.cond]. Note that LHS is never null, even for the GNU x ?: y
/// extension. In this case, LHS == Cond. (But they're not aliases.)
QualType Sema::CXXCheckConditionalOperands(ExprResult &Cond, ExprResult &LHS,
                                           ExprResult &RHS, ExprValueKind &VK,
                                           ExprObjectKind &OK,
                                           SourceLocation QuestionLoc) {
  // FIXME: Handle C99's complex types, vector types, block pointers and Obj-C++
  // interface pointers.

  // C++11 [expr.cond]p1
  //   The first expression is contextually converted to bool.
  if (!Cond.get()->isTypeDependent()) {
    ExprResult CondRes = CheckCXXBooleanCondition(Cond.take());
    if (CondRes.isInvalid())
      return QualType();
    Cond = CondRes;
  }

  // Assume r-value.
  VK = VK_RValue;
  OK = OK_Ordinary;

  // Either of the arguments dependent?
  if (LHS.get()->isTypeDependent() || RHS.get()->isTypeDependent())
    return Context.DependentTy;

  // C++11 [expr.cond]p2
  //   If either the second or the third operand has type (cv) void, ...
  QualType LTy = LHS.get()->getType();
  QualType RTy = RHS.get()->getType();
  bool LVoid = LTy->isVoidType();
  bool RVoid = RTy->isVoidType();
  if (LVoid || RVoid) {
    //   ... then the [l2r] conversions are performed on the second and third
    //   operands ...
    LHS = DefaultFunctionArrayLvalueConversion(LHS.take());
    RHS = DefaultFunctionArrayLvalueConversion(RHS.take());
    if (LHS.isInvalid() || RHS.isInvalid())
      return QualType();

    // Finish off the lvalue-to-rvalue conversion by copy-initializing a
    // temporary if necessary. DefaultFunctionArrayLvalueConversion doesn't
    // do this part for us.
    ExprResult &NonVoid = LVoid ? RHS : LHS;
    if (NonVoid.get()->getType()->isRecordType() &&
        NonVoid.get()->isGLValue()) {
      if (RequireNonAbstractType(QuestionLoc, NonVoid.get()->getType(),
                             diag::err_allocation_of_abstract_type))
        return QualType();
      InitializedEntity Entity =
          InitializedEntity::InitializeTemporary(NonVoid.get()->getType());
      NonVoid = PerformCopyInitialization(Entity, SourceLocation(), NonVoid);
      if (NonVoid.isInvalid())
        return QualType();
    }

    LTy = LHS.get()->getType();
    RTy = RHS.get()->getType();

    //   ... and one of the following shall hold:
    //   -- The second or the third operand (but not both) is a throw-
    //      expression; the result is of the type of the other and is a prvalue.
    bool LThrow = isa<CXXThrowExpr>(LHS.get());
    bool RThrow = isa<CXXThrowExpr>(RHS.get());
    if (LThrow && !RThrow)
      return RTy;
    if (RThrow && !LThrow)
      return LTy;

    //   -- Both the second and third operands have type void; the result is of
    //      type void and is a prvalue.
    if (LVoid && RVoid)
      return Context.VoidTy;

    // Neither holds, error.
    Diag(QuestionLoc, diag::err_conditional_void_nonvoid)
      << (LVoid ? RTy : LTy) << (LVoid ? 0 : 1)
      << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
    return QualType();
  }

  // Neither is void.

  // C++11 [expr.cond]p3
  //   Otherwise, if the second and third operand have different types, and
  //   either has (cv) class type [...] an attempt is made to convert each of
  //   those operands to the type of the other.
  if (!Context.hasSameType(LTy, RTy) &&
      (LTy->isRecordType() || RTy->isRecordType())) {
    ImplicitConversionSequence ICSLeftToRight, ICSRightToLeft;
    // These return true if a single direction is already ambiguous.
    QualType L2RType, R2LType;
    bool HaveL2R, HaveR2L;
    if (TryClassUnification(*this, LHS.get(), RHS.get(), QuestionLoc, HaveL2R, L2RType))
      return QualType();
    if (TryClassUnification(*this, RHS.get(), LHS.get(), QuestionLoc, HaveR2L, R2LType))
      return QualType();

    //   If both can be converted, [...] the program is ill-formed.
    if (HaveL2R && HaveR2L) {
      Diag(QuestionLoc, diag::err_conditional_ambiguous)
        << LTy << RTy << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
      return QualType();
    }

    //   If exactly one conversion is possible, that conversion is applied to
    //   the chosen operand and the converted operands are used in place of the
    //   original operands for the remainder of this section.
    if (HaveL2R) {
      if (ConvertForConditional(*this, LHS, L2RType) || LHS.isInvalid())
        return QualType();
      LTy = LHS.get()->getType();
    } else if (HaveR2L) {
      if (ConvertForConditional(*this, RHS, R2LType) || RHS.isInvalid())
        return QualType();
      RTy = RHS.get()->getType();
    }
  }

  // C++11 [expr.cond]p3
  //   if both are glvalues of the same value category and the same type except
  //   for cv-qualification, an attempt is made to convert each of those
  //   operands to the type of the other.
  ExprValueKind LVK = LHS.get()->getValueKind();
  ExprValueKind RVK = RHS.get()->getValueKind();
  if (!Context.hasSameType(LTy, RTy) &&
      Context.hasSameUnqualifiedType(LTy, RTy) &&
      LVK == RVK && LVK != VK_RValue) {
    // Since the unqualified types are reference-related and we require the
    // result to be as if a reference bound directly, the only conversion
    // we can perform is to add cv-qualifiers.
    Qualifiers LCVR = Qualifiers::fromCVRMask(LTy.getCVRQualifiers());
    Qualifiers RCVR = Qualifiers::fromCVRMask(RTy.getCVRQualifiers());
    if (RCVR.isStrictSupersetOf(LCVR)) {
      LHS = ImpCastExprToType(LHS.take(), RTy, CK_NoOp, LVK);
      LTy = LHS.get()->getType();
    }
    else if (LCVR.isStrictSupersetOf(RCVR)) {
      RHS = ImpCastExprToType(RHS.take(), LTy, CK_NoOp, RVK);
      RTy = RHS.get()->getType();
    }
  }

  // C++11 [expr.cond]p4
  //   If the second and third operands are glvalues of the same value
  //   category and have the same type, the result is of that type and
  //   value category and it is a bit-field if the second or the third
  //   operand is a bit-field, or if both are bit-fields.
  // We only extend this to bitfields, not to the crazy other kinds of
  // l-values.
  bool Same = Context.hasSameType(LTy, RTy);
  if (Same && LVK == RVK && LVK != VK_RValue &&
      LHS.get()->isOrdinaryOrBitFieldObject() &&
      RHS.get()->isOrdinaryOrBitFieldObject()) {
    VK = LHS.get()->getValueKind();
    if (LHS.get()->getObjectKind() == OK_BitField ||
        RHS.get()->getObjectKind() == OK_BitField)
      OK = OK_BitField;
    return LTy;
  }

  // C++11 [expr.cond]p5
  //   Otherwise, the result is a prvalue. If the second and third operands
  //   do not have the same type, and either has (cv) class type, ...
  if (!Same && (LTy->isRecordType() || RTy->isRecordType())) {
    //   ... overload resolution is used to determine the conversions (if any)
    //   to be applied to the operands. If the overload resolution fails, the
    //   program is ill-formed.
    if (FindConditionalOverload(*this, LHS, RHS, QuestionLoc))
      return QualType();
  }

  // C++11 [expr.cond]p6
  //   Lvalue-to-rvalue, array-to-pointer, and function-to-pointer standard
  //   conversions are performed on the second and third operands.
  LHS = DefaultFunctionArrayLvalueConversion(LHS.take());
  RHS = DefaultFunctionArrayLvalueConversion(RHS.take());
  if (LHS.isInvalid() || RHS.isInvalid())
    return QualType();
  LTy = LHS.get()->getType();
  RTy = RHS.get()->getType();

  //   After those conversions, one of the following shall hold:
  //   -- The second and third operands have the same type; the result
  //      is of that type. If the operands have class type, the result
  //      is a prvalue temporary of the result type, which is
  //      copy-initialized from either the second operand or the third
  //      operand depending on the value of the first operand.
  if (Context.getCanonicalType(LTy) == Context.getCanonicalType(RTy)) {
    if (LTy->isRecordType()) {
      // The operands have class type. Make a temporary copy.
      if (RequireNonAbstractType(QuestionLoc, LTy,
                                 diag::err_allocation_of_abstract_type))
        return QualType();
      InitializedEntity Entity = InitializedEntity::InitializeTemporary(LTy);

      ExprResult LHSCopy = PerformCopyInitialization(Entity,
                                                     SourceLocation(),
                                                     LHS);
      if (LHSCopy.isInvalid())
        return QualType();

      ExprResult RHSCopy = PerformCopyInitialization(Entity,
                                                     SourceLocation(),
                                                     RHS);
      if (RHSCopy.isInvalid())
        return QualType();

      LHS = LHSCopy;
      RHS = RHSCopy;
    }

    return LTy;
  }

  // Extension: conditional operator involving vector types.
  if (LTy->isVectorType() || RTy->isVectorType())
    return CheckVectorOperands(LHS, RHS, QuestionLoc, /*isCompAssign*/false);

  //   -- The second and third operands have arithmetic or enumeration type;
  //      the usual arithmetic conversions are performed to bring them to a
  //      common type, and the result is of that type.
  if (LTy->isArithmeticType() && RTy->isArithmeticType()) {
    UsualArithmeticConversions(LHS, RHS);
    if (LHS.isInvalid() || RHS.isInvalid())
      return QualType();
    return LHS.get()->getType();
  }

  //   -- The second and third operands have pointer type, or one has pointer
  //      type and the other is a null pointer constant, or both are null
  //      pointer constants, at least one of which is non-integral; pointer
  //      conversions and qualification conversions are performed to bring them
  //      to their composite pointer type. The result is of the composite
  //      pointer type.
  //   -- The second and third operands have pointer to member type, or one has
  //      pointer to member type and the other is a null pointer constant;
  //      pointer to member conversions and qualification conversions are
  //      performed to bring them to a common type, whose cv-qualification
  //      shall match the cv-qualification of either the second or the third
  //      operand. The result is of the common type.
  bool NonStandardCompositeType = false;
  QualType Composite = FindCompositePointerType(QuestionLoc, LHS, RHS,
                              isSFINAEContext()? 0 : &NonStandardCompositeType);
  if (!Composite.isNull()) {
    if (NonStandardCompositeType)
      Diag(QuestionLoc,
           diag::ext_typecheck_cond_incompatible_operands_nonstandard)
        << LTy << RTy << Composite
        << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();

    return Composite;
  }

  // Similarly, attempt to find composite type of two objective-c pointers.
  Composite = FindCompositeObjCPointerType(LHS, RHS, QuestionLoc);
  if (!Composite.isNull())
    return Composite;

  // Check if we are using a null with a non-pointer type.
  if (DiagnoseConditionalForNull(LHS.get(), RHS.get(), QuestionLoc))
    return QualType();

  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHS.get()->getType() << RHS.get()->getType()
    << LHS.get()->getSourceRange() << RHS.get()->getSourceRange();
  return QualType();
}

/// \brief Find a merged pointer type and convert the two expressions to it.
///
/// This finds the composite pointer type (or member pointer type) for @p E1
/// and @p E2 according to C++11 5.9p2. It converts both expressions to this
/// type and returns it.
/// It does not emit diagnostics.
///
/// \param Loc The location of the operator requiring these two expressions to
/// be converted to the composite pointer type.
///
/// If \p NonStandardCompositeType is non-NULL, then we are permitted to find
/// a non-standard (but still sane) composite type to which both expressions
/// can be converted. When such a type is chosen, \c *NonStandardCompositeType
/// will be set true.
QualType Sema::FindCompositePointerType(SourceLocation Loc,
                                        Expr *&E1, Expr *&E2,
                                        bool *NonStandardCompositeType) {
  if (NonStandardCompositeType)
    *NonStandardCompositeType = false;

  assert(getLangOpts().CPlusPlus && "This function assumes C++");
  QualType T1 = E1->getType(), T2 = E2->getType();

  // C++11 5.9p2
  //   Pointer conversions and qualification conversions are performed on
  //   pointer operands to bring them to their composite pointer type. If
  //   one operand is a null pointer constant, the composite pointer type is
  //   std::nullptr_t if the other operand is also a null pointer constant or,
  //   if the other operand is a pointer, the type of the other operand.
  if (!T1->isAnyPointerType() && !T1->isMemberPointerType() &&
      !T2->isAnyPointerType() && !T2->isMemberPointerType()) {
    if (T1->isNullPtrType() &&
        E2->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
      E2 = ImpCastExprToType(E2, T1, CK_NullToPointer).take();
      return T1;
    }
    if (T2->isNullPtrType() &&
        E1->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
      E1 = ImpCastExprToType(E1, T2, CK_NullToPointer).take();
      return T2;
    }
    return QualType();
  }

  if (E1->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    if (T2->isMemberPointerType())
      E1 = ImpCastExprToType(E1, T2, CK_NullToMemberPointer).take();
    else
      E1 = ImpCastExprToType(E1, T2, CK_NullToPointer).take();
    return T2;
  }
  if (E2->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    if (T1->isMemberPointerType())
      E2 = ImpCastExprToType(E2, T1, CK_NullToMemberPointer).take();
    else
      E2 = ImpCastExprToType(E2, T1, CK_NullToPointer).take();
    return T1;
  }

  // Now both have to be pointers or member pointers.
  if ((!T1->isPointerType() && !T1->isMemberPointerType()) ||
      (!T2->isPointerType() && !T2->isMemberPointerType()))
    return QualType();

  //   Otherwise, of one of the operands has type "pointer to cv1 void," then
  //   the other has type "pointer to cv2 T" and the composite pointer type is
  //   "pointer to cv12 void," where cv12 is the union of cv1 and cv2.
  //   Otherwise, the composite pointer type is a pointer type similar to the
  //   type of one of the operands, with a cv-qualification signature that is
  //   the union of the cv-qualification signatures of the operand types.
  // In practice, the first part here is redundant; it's subsumed by the second.
  // What we do here is, we build the two possible composite types, and try the
  // conversions in both directions. If only one works, or if the two composite
  // types are the same, we have succeeded.
  // FIXME: extended qualifiers?
  typedef SmallVector<unsigned, 4> QualifierVector;
  QualifierVector QualifierUnion;
  typedef SmallVector<std::pair<const Type *, const Type *>, 4>
      ContainingClassVector;
  ContainingClassVector MemberOfClass;
  QualType Composite1 = Context.getCanonicalType(T1),
           Composite2 = Context.getCanonicalType(T2);
  unsigned NeedConstBefore = 0;
  do {
    const PointerType *Ptr1, *Ptr2;
    if ((Ptr1 = Composite1->getAs<PointerType>()) &&
        (Ptr2 = Composite2->getAs<PointerType>())) {
      Composite1 = Ptr1->getPointeeType();
      Composite2 = Ptr2->getPointeeType();

      // If we're allowed to create a non-standard composite type, keep track
      // of where we need to fill in additional 'const' qualifiers.
      if (NonStandardCompositeType &&
          Composite1.getCVRQualifiers() != Composite2.getCVRQualifiers())
        NeedConstBefore = QualifierUnion.size();

      QualifierUnion.push_back(
                 Composite1.getCVRQualifiers() | Composite2.getCVRQualifiers());
      MemberOfClass.push_back(std::make_pair((const Type *)0, (const Type *)0));
      continue;
    }

    const MemberPointerType *MemPtr1, *MemPtr2;
    if ((MemPtr1 = Composite1->getAs<MemberPointerType>()) &&
        (MemPtr2 = Composite2->getAs<MemberPointerType>())) {
      Composite1 = MemPtr1->getPointeeType();
      Composite2 = MemPtr2->getPointeeType();

      // If we're allowed to create a non-standard composite type, keep track
      // of where we need to fill in additional 'const' qualifiers.
      if (NonStandardCompositeType &&
          Composite1.getCVRQualifiers() != Composite2.getCVRQualifiers())
        NeedConstBefore = QualifierUnion.size();

      QualifierUnion.push_back(
                 Composite1.getCVRQualifiers() | Composite2.getCVRQualifiers());
      MemberOfClass.push_back(std::make_pair(MemPtr1->getClass(),
                                             MemPtr2->getClass()));
      continue;
    }

    // FIXME: block pointer types?

    // Cannot unwrap any more types.
    break;
  } while (true);

  if (NeedConstBefore && NonStandardCompositeType) {
    // Extension: Add 'const' to qualifiers that come before the first qualifier
    // mismatch, so that our (non-standard!) composite type meets the
    // requirements of C++ [conv.qual]p4 bullet 3.
    for (unsigned I = 0; I != NeedConstBefore; ++I) {
      if ((QualifierUnion[I] & Qualifiers::Const) == 0) {
        QualifierUnion[I] = QualifierUnion[I] | Qualifiers::Const;
        *NonStandardCompositeType = true;
      }
    }
  }

  // Rewrap the composites as pointers or member pointers with the union CVRs.
  ContainingClassVector::reverse_iterator MOC
    = MemberOfClass.rbegin();
  for (QualifierVector::reverse_iterator
         I = QualifierUnion.rbegin(),
         E = QualifierUnion.rend();
       I != E; (void)++I, ++MOC) {
    Qualifiers Quals = Qualifiers::fromCVRMask(*I);
    if (MOC->first && MOC->second) {
      // Rebuild member pointer type
      Composite1 = Context.getMemberPointerType(
                                    Context.getQualifiedType(Composite1, Quals),
                                    MOC->first);
      Composite2 = Context.getMemberPointerType(
                                    Context.getQualifiedType(Composite2, Quals),
                                    MOC->second);
    } else {
      // Rebuild pointer type
      Composite1
        = Context.getPointerType(Context.getQualifiedType(Composite1, Quals));
      Composite2
        = Context.getPointerType(Context.getQualifiedType(Composite2, Quals));
    }
  }

  // Try to convert to the first composite pointer type.
  InitializedEntity Entity1
    = InitializedEntity::InitializeTemporary(Composite1);
  InitializationKind Kind
    = InitializationKind::CreateCopy(Loc, SourceLocation());
  InitializationSequence E1ToC1(*this, Entity1, Kind, &E1, 1);
  InitializationSequence E2ToC1(*this, Entity1, Kind, &E2, 1);

  if (E1ToC1 && E2ToC1) {
    // Conversion to Composite1 is viable.
    if (!Context.hasSameType(Composite1, Composite2)) {
      // Composite2 is a different type from Composite1. Check whether
      // Composite2 is also viable.
      InitializedEntity Entity2
        = InitializedEntity::InitializeTemporary(Composite2);
      InitializationSequence E1ToC2(*this, Entity2, Kind, &E1, 1);
      InitializationSequence E2ToC2(*this, Entity2, Kind, &E2, 1);
      if (E1ToC2 && E2ToC2) {
        // Both Composite1 and Composite2 are viable and are different;
        // this is an ambiguity.
        return QualType();
      }
    }

    // Convert E1 to Composite1
    ExprResult E1Result
      = E1ToC1.Perform(*this, Entity1, Kind, E1);
    if (E1Result.isInvalid())
      return QualType();
    E1 = E1Result.takeAs<Expr>();

    // Convert E2 to Composite1
    ExprResult E2Result
      = E2ToC1.Perform(*this, Entity1, Kind, E2);
    if (E2Result.isInvalid())
      return QualType();
    E2 = E2Result.takeAs<Expr>();

    return Composite1;
  }

  // Check whether Composite2 is viable.
  InitializedEntity Entity2
    = InitializedEntity::InitializeTemporary(Composite2);
  InitializationSequence E1ToC2(*this, Entity2, Kind, &E1, 1);
  InitializationSequence E2ToC2(*this, Entity2, Kind, &E2, 1);
  if (!E1ToC2 || !E2ToC2)
    return QualType();

  // Convert E1 to Composite2
  ExprResult E1Result
    = E1ToC2.Perform(*this, Entity2, Kind, E1);
  if (E1Result.isInvalid())
    return QualType();
  E1 = E1Result.takeAs<Expr>();

  // Convert E2 to Composite2
  ExprResult E2Result
    = E2ToC2.Perform(*this, Entity2, Kind, E2);
  if (E2Result.isInvalid())
    return QualType();
  E2 = E2Result.takeAs<Expr>();

  return Composite2;
}

ExprResult Sema::MaybeBindToTemporary(Expr *E) {
  if (!E)
    return ExprError();

  assert(!isa<CXXBindTemporaryExpr>(E) && "Double-bound temporary?");

  // If the result is a glvalue, we shouldn't bind it.
  if (!E->isRValue())
    return Owned(E);

  // In ARC, calls that return a retainable type can return retained,
  // in which case we have to insert a consuming cast.
  if (getLangOpts().ObjCAutoRefCount &&
      E->getType()->isObjCRetainableType()) {

    bool ReturnsRetained;

    // For actual calls, we compute this by examining the type of the
    // called value.
    if (CallExpr *Call = dyn_cast<CallExpr>(E)) {
      Expr *Callee = Call->getCallee()->IgnoreParens();
      QualType T = Callee->getType();

      if (T == Context.BoundMemberTy) {
        // Handle pointer-to-members.
        if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Callee))
          T = BinOp->getRHS()->getType();
        else if (MemberExpr *Mem = dyn_cast<MemberExpr>(Callee))
          T = Mem->getMemberDecl()->getType();
      }
      
      if (const PointerType *Ptr = T->getAs<PointerType>())
        T = Ptr->getPointeeType();
      else if (const BlockPointerType *Ptr = T->getAs<BlockPointerType>())
        T = Ptr->getPointeeType();
      else if (const MemberPointerType *MemPtr = T->getAs<MemberPointerType>())
        T = MemPtr->getPointeeType();
      
      const FunctionType *FTy = T->getAs<FunctionType>();
      assert(FTy && "call to value not of function type?");
      ReturnsRetained = FTy->getExtInfo().getProducesResult();

    // ActOnStmtExpr arranges things so that StmtExprs of retainable
    // type always produce a +1 object.
    } else if (isa<StmtExpr>(E)) {
      ReturnsRetained = true;

    // We hit this case with the lambda conversion-to-block optimization;
    // we don't want any extra casts here.
    } else if (isa<CastExpr>(E) &&
               isa<BlockExpr>(cast<CastExpr>(E)->getSubExpr())) {
      return Owned(E);

    // For message sends and property references, we try to find an
    // actual method.  FIXME: we should infer retention by selector in
    // cases where we don't have an actual method.
    } else {
      ObjCMethodDecl *D = 0;
      if (ObjCMessageExpr *Send = dyn_cast<ObjCMessageExpr>(E)) {
        D = Send->getMethodDecl();
      } else if (ObjCBoxedExpr *BoxedExpr = dyn_cast<ObjCBoxedExpr>(E)) {
        D = BoxedExpr->getBoxingMethod();
      } else if (ObjCArrayLiteral *ArrayLit = dyn_cast<ObjCArrayLiteral>(E)) {
        D = ArrayLit->getArrayWithObjectsMethod();
      } else if (ObjCDictionaryLiteral *DictLit
                                        = dyn_cast<ObjCDictionaryLiteral>(E)) {
        D = DictLit->getDictWithObjectsMethod();
      }

      ReturnsRetained = (D && D->hasAttr<NSReturnsRetainedAttr>());

      // Don't do reclaims on performSelector calls; despite their
      // return type, the invoked method doesn't necessarily actually
      // return an object.
      if (!ReturnsRetained &&
          D && D->getMethodFamily() == OMF_performSelector)
        return Owned(E);
    }

    // Don't reclaim an object of Class type.
    if (!ReturnsRetained && E->getType()->isObjCARCImplicitlyUnretainedType())
      return Owned(E);

    ExprNeedsCleanups = true;

    CastKind ck = (ReturnsRetained ? CK_ARCConsumeObject
                                   : CK_ARCReclaimReturnedObject);
    return Owned(ImplicitCastExpr::Create(Context, E->getType(), ck, E, 0,
                                          VK_RValue));
  }

  if (!getLangOpts().CPlusPlus)
    return Owned(E);

  // Search for the base element type (cf. ASTContext::getBaseElementType) with
  // a fast path for the common case that the type is directly a RecordType.
  const Type *T = Context.getCanonicalType(E->getType().getTypePtr());
  const RecordType *RT = 0;
  while (!RT) {
    switch (T->getTypeClass()) {
    case Type::Record:
      RT = cast<RecordType>(T);
      break;
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::DependentSizedArray:
      T = cast<ArrayType>(T)->getElementType().getTypePtr();
      break;
    default:
      return Owned(E);
    }
  }

  // That should be enough to guarantee that this type is complete, if we're
  // not processing a decltype expression.
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
  if (RD->isInvalidDecl() || RD->isDependentContext())
    return Owned(E);

  bool IsDecltype = ExprEvalContexts.back().IsDecltype;
  CXXDestructorDecl *Destructor = IsDecltype ? 0 : LookupDestructor(RD);

  if (Destructor) {
    MarkFunctionReferenced(E->getExprLoc(), Destructor);
    CheckDestructorAccess(E->getExprLoc(), Destructor,
                          PDiag(diag::err_access_dtor_temp)
                            << E->getType());
    DiagnoseUseOfDecl(Destructor, E->getExprLoc());

    // If destructor is trivial, we can avoid the extra copy.
    if (Destructor->isTrivial())
      return Owned(E);

    // We need a cleanup, but we don't need to remember the temporary.
    ExprNeedsCleanups = true;
  }

  CXXTemporary *Temp = CXXTemporary::Create(Context, Destructor);
  CXXBindTemporaryExpr *Bind = CXXBindTemporaryExpr::Create(Context, Temp, E);

  if (IsDecltype)
    ExprEvalContexts.back().DelayedDecltypeBinds.push_back(Bind);

  return Owned(Bind);
}

ExprResult
Sema::MaybeCreateExprWithCleanups(ExprResult SubExpr) {
  if (SubExpr.isInvalid())
    return ExprError();

  return Owned(MaybeCreateExprWithCleanups(SubExpr.take()));
}

Expr *Sema::MaybeCreateExprWithCleanups(Expr *SubExpr) {
  assert(SubExpr && "sub expression can't be null!");

  CleanupVarDeclMarking();

  unsigned FirstCleanup = ExprEvalContexts.back().NumCleanupObjects;
  assert(ExprCleanupObjects.size() >= FirstCleanup);
  assert(ExprNeedsCleanups || ExprCleanupObjects.size() == FirstCleanup);
  if (!ExprNeedsCleanups)
    return SubExpr;

  ArrayRef<ExprWithCleanups::CleanupObject> Cleanups
    = llvm::makeArrayRef(ExprCleanupObjects.begin() + FirstCleanup,
                         ExprCleanupObjects.size() - FirstCleanup);

  Expr *E = ExprWithCleanups::Create(Context, SubExpr, Cleanups);
  DiscardCleanupsInEvaluationContext();

  return E;
}

Stmt *Sema::MaybeCreateStmtWithCleanups(Stmt *SubStmt) {
  assert(SubStmt && "sub statement can't be null!");

  CleanupVarDeclMarking();

  if (!ExprNeedsCleanups)
    return SubStmt;

  // FIXME: In order to attach the temporaries, wrap the statement into
  // a StmtExpr; currently this is only used for asm statements.
  // This is hacky, either create a new CXXStmtWithTemporaries statement or
  // a new AsmStmtWithTemporaries.
  CompoundStmt *CompStmt = new (Context) CompoundStmt(Context, &SubStmt, 1,
                                                      SourceLocation(),
                                                      SourceLocation());
  Expr *E = new (Context) StmtExpr(CompStmt, Context.VoidTy, SourceLocation(),
                                   SourceLocation());
  return MaybeCreateExprWithCleanups(E);
}

/// Process the expression contained within a decltype. For such expressions,
/// certain semantic checks on temporaries are delayed until this point, and
/// are omitted for the 'topmost' call in the decltype expression. If the
/// topmost call bound a temporary, strip that temporary off the expression.
ExprResult Sema::ActOnDecltypeExpression(Expr *E) {
  ExpressionEvaluationContextRecord &Rec = ExprEvalContexts.back();
  assert(Rec.IsDecltype && "not in a decltype expression");

  // C++11 [expr.call]p11:
  //   If a function call is a prvalue of object type,
  // -- if the function call is either
  //   -- the operand of a decltype-specifier, or
  //   -- the right operand of a comma operator that is the operand of a
  //      decltype-specifier,
  //   a temporary object is not introduced for the prvalue.

  // Recursively rebuild ParenExprs and comma expressions to strip out the
  // outermost CXXBindTemporaryExpr, if any.
  if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    ExprResult SubExpr = ActOnDecltypeExpression(PE->getSubExpr());
    if (SubExpr.isInvalid())
      return ExprError();
    if (SubExpr.get() == PE->getSubExpr())
      return Owned(E);
    return ActOnParenExpr(PE->getLParen(), PE->getRParen(), SubExpr.take());
  }
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    if (BO->getOpcode() == BO_Comma) {
      ExprResult RHS = ActOnDecltypeExpression(BO->getRHS());
      if (RHS.isInvalid())
        return ExprError();
      if (RHS.get() == BO->getRHS())
        return Owned(E);
      return Owned(new (Context) BinaryOperator(BO->getLHS(), RHS.take(),
                                                BO_Comma, BO->getType(),
                                                BO->getValueKind(),
                                                BO->getObjectKind(),
                                                BO->getOperatorLoc(),
                                                BO->isFPContractable()));
    }
  }

  CXXBindTemporaryExpr *TopBind = dyn_cast<CXXBindTemporaryExpr>(E);
  if (TopBind)
    E = TopBind->getSubExpr();

  // Disable the special decltype handling now.
  Rec.IsDecltype = false;

  // In MS mode, don't perform any extra checking of call return types within a
  // decltype expression.
  if (getLangOpts().MicrosoftMode)
    return Owned(E);

  // Perform the semantic checks we delayed until this point.
  CallExpr *TopCall = dyn_cast<CallExpr>(E);
  for (unsigned I = 0, N = Rec.DelayedDecltypeCalls.size(); I != N; ++I) {
    CallExpr *Call = Rec.DelayedDecltypeCalls[I];
    if (Call == TopCall)
      continue;

    if (CheckCallReturnType(Call->getCallReturnType(),
                            Call->getLocStart(),
                            Call, Call->getDirectCallee()))
      return ExprError();
  }

  // Now all relevant types are complete, check the destructors are accessible
  // and non-deleted, and annotate them on the temporaries.
  for (unsigned I = 0, N = Rec.DelayedDecltypeBinds.size(); I != N; ++I) {
    CXXBindTemporaryExpr *Bind = Rec.DelayedDecltypeBinds[I];
    if (Bind == TopBind)
      continue;

    CXXTemporary *Temp = Bind->getTemporary();

    CXXRecordDecl *RD =
      Bind->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
    CXXDestructorDecl *Destructor = LookupDestructor(RD);
    Temp->setDestructor(Destructor);

    MarkFunctionReferenced(Bind->getExprLoc(), Destructor);
    CheckDestructorAccess(Bind->getExprLoc(), Destructor,
                          PDiag(diag::err_access_dtor_temp)
                            << Bind->getType());
    DiagnoseUseOfDecl(Destructor, Bind->getExprLoc());

    // We need a cleanup, but we don't need to remember the temporary.
    ExprNeedsCleanups = true;
  }

  // Possibly strip off the top CXXBindTemporaryExpr.
  return Owned(E);
}

ExprResult
Sema::ActOnStartCXXMemberReference(Scope *S, Expr *Base, SourceLocation OpLoc,
                                   tok::TokenKind OpKind, ParsedType &ObjectType,
                                   bool &MayBePseudoDestructor) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Base);
  if (Result.isInvalid()) return ExprError();
  Base = Result.get();

  Result = CheckPlaceholderExpr(Base);
  if (Result.isInvalid()) return ExprError();
  Base = Result.take();

  QualType BaseType = Base->getType();
  MayBePseudoDestructor = false;
  if (BaseType->isDependentType()) {
    // If we have a pointer to a dependent type and are using the -> operator,
    // the object type is the type that the pointer points to. We might still
    // have enough information about that type to do something useful.
    if (OpKind == tok::arrow)
      if (const PointerType *Ptr = BaseType->getAs<PointerType>())
        BaseType = Ptr->getPointeeType();

    ObjectType = ParsedType::make(BaseType);
    MayBePseudoDestructor = true;
    return Owned(Base);
  }

  // C++ [over.match.oper]p8:
  //   [...] When operator->returns, the operator-> is applied  to the value
  //   returned, with the original second operand.
  if (OpKind == tok::arrow) {
    // The set of types we've considered so far.
    llvm::SmallPtrSet<CanQualType,8> CTypes;
    SmallVector<SourceLocation, 8> Locations;
    CTypes.insert(Context.getCanonicalType(BaseType));

    while (BaseType->isRecordType()) {
      Result = BuildOverloadedArrowExpr(S, Base, OpLoc);
      if (Result.isInvalid())
        return ExprError();
      Base = Result.get();
      if (CXXOperatorCallExpr *OpCall = dyn_cast<CXXOperatorCallExpr>(Base))
        Locations.push_back(OpCall->getDirectCallee()->getLocation());
      BaseType = Base->getType();
      CanQualType CBaseType = Context.getCanonicalType(BaseType);
      if (!CTypes.insert(CBaseType)) {
        Diag(OpLoc, diag::err_operator_arrow_circular);
        for (unsigned i = 0; i < Locations.size(); i++)
          Diag(Locations[i], diag::note_declared_at);
        return ExprError();
      }
    }

    if (BaseType->isPointerType() || BaseType->isObjCObjectPointerType())
      BaseType = BaseType->getPointeeType();
  }

  // Objective-C properties allow "." access on Objective-C pointer types,
  // so adjust the base type to the object type itself.
  if (BaseType->isObjCObjectPointerType())
    BaseType = BaseType->getPointeeType();
  
  // C++ [basic.lookup.classref]p2:
  //   [...] If the type of the object expression is of pointer to scalar
  //   type, the unqualified-id is looked up in the context of the complete
  //   postfix-expression.
  //
  // This also indicates that we could be parsing a pseudo-destructor-name.
  // Note that Objective-C class and object types can be pseudo-destructor
  // expressions or normal member (ivar or property) access expressions.
  if (BaseType->isObjCObjectOrInterfaceType()) {
    MayBePseudoDestructor = true;
  } else if (!BaseType->isRecordType()) {
    ObjectType = ParsedType();
    MayBePseudoDestructor = true;
    return Owned(Base);
  }

  // The object type must be complete (or dependent), or
  // C++11 [expr.prim.general]p3:
  //   Unlike the object expression in other contexts, *this is not required to
  //   be of complete type for purposes of class member access (5.2.5) outside 
  //   the member function body.
  if (!BaseType->isDependentType() &&
      !isThisOutsideMemberFunctionBody(BaseType) &&
      RequireCompleteType(OpLoc, BaseType, diag::err_incomplete_member_access))
    return ExprError();

  // C++ [basic.lookup.classref]p2:
  //   If the id-expression in a class member access (5.2.5) is an
  //   unqualified-id, and the type of the object expression is of a class
  //   type C (or of pointer to a class type C), the unqualified-id is looked
  //   up in the scope of class C. [...]
  ObjectType = ParsedType::make(BaseType);
  return Base;
}

ExprResult Sema::DiagnoseDtorReference(SourceLocation NameLoc,
                                                   Expr *MemExpr) {
  SourceLocation ExpectedLParenLoc = PP.getLocForEndOfToken(NameLoc);
  Diag(MemExpr->getLocStart(), diag::err_dtor_expr_without_call)
    << isa<CXXPseudoDestructorExpr>(MemExpr)
    << FixItHint::CreateInsertion(ExpectedLParenLoc, "()");

  return ActOnCallExpr(/*Scope*/ 0,
                       MemExpr,
                       /*LPLoc*/ ExpectedLParenLoc,
                       MultiExprArg(),
                       /*RPLoc*/ ExpectedLParenLoc);
}

static bool CheckArrow(Sema& S, QualType& ObjectType, Expr *&Base, 
                   tok::TokenKind& OpKind, SourceLocation OpLoc) {
  if (Base->hasPlaceholderType()) {
    ExprResult result = S.CheckPlaceholderExpr(Base);
    if (result.isInvalid()) return true;
    Base = result.take();
  }
  ObjectType = Base->getType();

  // C++ [expr.pseudo]p2:
  //   The left-hand side of the dot operator shall be of scalar type. The
  //   left-hand side of the arrow operator shall be of pointer to scalar type.
  //   This scalar type is the object type.
  // Note that this is rather different from the normal handling for the
  // arrow operator.
  if (OpKind == tok::arrow) {
    if (const PointerType *Ptr = ObjectType->getAs<PointerType>()) {
      ObjectType = Ptr->getPointeeType();
    } else if (!Base->isTypeDependent()) {
      // The user wrote "p->" when she probably meant "p."; fix it.
      S.Diag(OpLoc, diag::err_typecheck_member_reference_suggestion)
        << ObjectType << true
        << FixItHint::CreateReplacement(OpLoc, ".");
      if (S.isSFINAEContext())
        return true;

      OpKind = tok::period;
    }
  }

  return false;
}

ExprResult Sema::BuildPseudoDestructorExpr(Expr *Base,
                                           SourceLocation OpLoc,
                                           tok::TokenKind OpKind,
                                           const CXXScopeSpec &SS,
                                           TypeSourceInfo *ScopeTypeInfo,
                                           SourceLocation CCLoc,
                                           SourceLocation TildeLoc,
                                         PseudoDestructorTypeStorage Destructed,
                                           bool HasTrailingLParen) {
  TypeSourceInfo *DestructedTypeInfo = Destructed.getTypeSourceInfo();

  QualType ObjectType;
  if (CheckArrow(*this, ObjectType, Base, OpKind, OpLoc))
    return ExprError();

  if (!ObjectType->isDependentType() && !ObjectType->isScalarType() &&
      !ObjectType->isVectorType()) {
    if (getLangOpts().MicrosoftMode && ObjectType->isVoidType())
      Diag(OpLoc, diag::ext_pseudo_dtor_on_void) << Base->getSourceRange();
    else
      Diag(OpLoc, diag::err_pseudo_dtor_base_not_scalar)
        << ObjectType << Base->getSourceRange();
    return ExprError();
  }

  // C++ [expr.pseudo]p2:
  //   [...] The cv-unqualified versions of the object type and of the type
  //   designated by the pseudo-destructor-name shall be the same type.
  if (DestructedTypeInfo) {
    QualType DestructedType = DestructedTypeInfo->getType();
    SourceLocation DestructedTypeStart
      = DestructedTypeInfo->getTypeLoc().getLocalSourceRange().getBegin();
    if (!DestructedType->isDependentType() && !ObjectType->isDependentType()) {
      if (!Context.hasSameUnqualifiedType(DestructedType, ObjectType)) {
        Diag(DestructedTypeStart, diag::err_pseudo_dtor_type_mismatch)
          << ObjectType << DestructedType << Base->getSourceRange()
          << DestructedTypeInfo->getTypeLoc().getLocalSourceRange();

        // Recover by setting the destructed type to the object type.
        DestructedType = ObjectType;
        DestructedTypeInfo = Context.getTrivialTypeSourceInfo(ObjectType,
                                                           DestructedTypeStart);
        Destructed = PseudoDestructorTypeStorage(DestructedTypeInfo);
      } else if (DestructedType.getObjCLifetime() != 
                                                ObjectType.getObjCLifetime()) {
        
        if (DestructedType.getObjCLifetime() == Qualifiers::OCL_None) {
          // Okay: just pretend that the user provided the correctly-qualified
          // type.
        } else {
          Diag(DestructedTypeStart, diag::err_arc_pseudo_dtor_inconstant_quals)
            << ObjectType << DestructedType << Base->getSourceRange()
            << DestructedTypeInfo->getTypeLoc().getLocalSourceRange();
        }
        
        // Recover by setting the destructed type to the object type.
        DestructedType = ObjectType;
        DestructedTypeInfo = Context.getTrivialTypeSourceInfo(ObjectType,
                                                           DestructedTypeStart);
        Destructed = PseudoDestructorTypeStorage(DestructedTypeInfo);
      }
    }
  }

  // C++ [expr.pseudo]p2:
  //   [...] Furthermore, the two type-names in a pseudo-destructor-name of the
  //   form
  //
  //     ::[opt] nested-name-specifier[opt] type-name :: ~ type-name
  //
  //   shall designate the same scalar type.
  if (ScopeTypeInfo) {
    QualType ScopeType = ScopeTypeInfo->getType();
    if (!ScopeType->isDependentType() && !ObjectType->isDependentType() &&
        !Context.hasSameUnqualifiedType(ScopeType, ObjectType)) {

      Diag(ScopeTypeInfo->getTypeLoc().getLocalSourceRange().getBegin(),
           diag::err_pseudo_dtor_type_mismatch)
        << ObjectType << ScopeType << Base->getSourceRange()
        << ScopeTypeInfo->getTypeLoc().getLocalSourceRange();

      ScopeType = QualType();
      ScopeTypeInfo = 0;
    }
  }

  Expr *Result
    = new (Context) CXXPseudoDestructorExpr(Context, Base,
                                            OpKind == tok::arrow, OpLoc,
                                            SS.getWithLocInContext(Context),
                                            ScopeTypeInfo,
                                            CCLoc,
                                            TildeLoc,
                                            Destructed);

  if (HasTrailingLParen)
    return Owned(Result);

  return DiagnoseDtorReference(Destructed.getLocation(), Result);
}

ExprResult Sema::ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
                                           SourceLocation OpLoc,
                                           tok::TokenKind OpKind,
                                           CXXScopeSpec &SS,
                                           UnqualifiedId &FirstTypeName,
                                           SourceLocation CCLoc,
                                           SourceLocation TildeLoc,
                                           UnqualifiedId &SecondTypeName,
                                           bool HasTrailingLParen) {
  assert((FirstTypeName.getKind() == UnqualifiedId::IK_TemplateId ||
          FirstTypeName.getKind() == UnqualifiedId::IK_Identifier) &&
         "Invalid first type name in pseudo-destructor");
  assert((SecondTypeName.getKind() == UnqualifiedId::IK_TemplateId ||
          SecondTypeName.getKind() == UnqualifiedId::IK_Identifier) &&
         "Invalid second type name in pseudo-destructor");

  QualType ObjectType;
  if (CheckArrow(*this, ObjectType, Base, OpKind, OpLoc))
    return ExprError();

  // Compute the object type that we should use for name lookup purposes. Only
  // record types and dependent types matter.
  ParsedType ObjectTypePtrForLookup;
  if (!SS.isSet()) {
    if (ObjectType->isRecordType())
      ObjectTypePtrForLookup = ParsedType::make(ObjectType);
    else if (ObjectType->isDependentType())
      ObjectTypePtrForLookup = ParsedType::make(Context.DependentTy);
  }

  // Convert the name of the type being destructed (following the ~) into a
  // type (with source-location information).
  QualType DestructedType;
  TypeSourceInfo *DestructedTypeInfo = 0;
  PseudoDestructorTypeStorage Destructed;
  if (SecondTypeName.getKind() == UnqualifiedId::IK_Identifier) {
    ParsedType T = getTypeName(*SecondTypeName.Identifier,
                               SecondTypeName.StartLocation,
                               S, &SS, true, false, ObjectTypePtrForLookup);
    if (!T &&
        ((SS.isSet() && !computeDeclContext(SS, false)) ||
         (!SS.isSet() && ObjectType->isDependentType()))) {
      // The name of the type being destroyed is a dependent name, and we
      // couldn't find anything useful in scope. Just store the identifier and
      // it's location, and we'll perform (qualified) name lookup again at
      // template instantiation time.
      Destructed = PseudoDestructorTypeStorage(SecondTypeName.Identifier,
                                               SecondTypeName.StartLocation);
    } else if (!T) {
      Diag(SecondTypeName.StartLocation,
           diag::err_pseudo_dtor_destructor_non_type)
        << SecondTypeName.Identifier << ObjectType;
      if (isSFINAEContext())
        return ExprError();

      // Recover by assuming we had the right type all along.
      DestructedType = ObjectType;
    } else
      DestructedType = GetTypeFromParser(T, &DestructedTypeInfo);
  } else {
    // Resolve the template-id to a type.
    TemplateIdAnnotation *TemplateId = SecondTypeName.TemplateId;
    ASTTemplateArgsPtr TemplateArgsPtr(TemplateId->getTemplateArgs(),
                                       TemplateId->NumArgs);
    TypeResult T = ActOnTemplateIdType(TemplateId->SS,
                                       TemplateId->TemplateKWLoc,
                                       TemplateId->Template,
                                       TemplateId->TemplateNameLoc,
                                       TemplateId->LAngleLoc,
                                       TemplateArgsPtr,
                                       TemplateId->RAngleLoc);
    if (T.isInvalid() || !T.get()) {
      // Recover by assuming we had the right type all along.
      DestructedType = ObjectType;
    } else
      DestructedType = GetTypeFromParser(T.get(), &DestructedTypeInfo);
  }

  // If we've performed some kind of recovery, (re-)build the type source
  // information.
  if (!DestructedType.isNull()) {
    if (!DestructedTypeInfo)
      DestructedTypeInfo = Context.getTrivialTypeSourceInfo(DestructedType,
                                                  SecondTypeName.StartLocation);
    Destructed = PseudoDestructorTypeStorage(DestructedTypeInfo);
  }

  // Convert the name of the scope type (the type prior to '::') into a type.
  TypeSourceInfo *ScopeTypeInfo = 0;
  QualType ScopeType;
  if (FirstTypeName.getKind() == UnqualifiedId::IK_TemplateId ||
      FirstTypeName.Identifier) {
    if (FirstTypeName.getKind() == UnqualifiedId::IK_Identifier) {
      ParsedType T = getTypeName(*FirstTypeName.Identifier,
                                 FirstTypeName.StartLocation,
                                 S, &SS, true, false, ObjectTypePtrForLookup);
      if (!T) {
        Diag(FirstTypeName.StartLocation,
             diag::err_pseudo_dtor_destructor_non_type)
          << FirstTypeName.Identifier << ObjectType;

        if (isSFINAEContext())
          return ExprError();

        // Just drop this type. It's unnecessary anyway.
        ScopeType = QualType();
      } else
        ScopeType = GetTypeFromParser(T, &ScopeTypeInfo);
    } else {
      // Resolve the template-id to a type.
      TemplateIdAnnotation *TemplateId = FirstTypeName.TemplateId;
      ASTTemplateArgsPtr TemplateArgsPtr(TemplateId->getTemplateArgs(),
                                         TemplateId->NumArgs);
      TypeResult T = ActOnTemplateIdType(TemplateId->SS,
                                         TemplateId->TemplateKWLoc,
                                         TemplateId->Template,
                                         TemplateId->TemplateNameLoc,
                                         TemplateId->LAngleLoc,
                                         TemplateArgsPtr,
                                         TemplateId->RAngleLoc);
      if (T.isInvalid() || !T.get()) {
        // Recover by dropping this type.
        ScopeType = QualType();
      } else
        ScopeType = GetTypeFromParser(T.get(), &ScopeTypeInfo);
    }
  }

  if (!ScopeType.isNull() && !ScopeTypeInfo)
    ScopeTypeInfo = Context.getTrivialTypeSourceInfo(ScopeType,
                                                  FirstTypeName.StartLocation);


  return BuildPseudoDestructorExpr(Base, OpLoc, OpKind, SS,
                                   ScopeTypeInfo, CCLoc, TildeLoc,
                                   Destructed, HasTrailingLParen);
}

ExprResult Sema::ActOnPseudoDestructorExpr(Scope *S, Expr *Base,
                                           SourceLocation OpLoc,
                                           tok::TokenKind OpKind,
                                           SourceLocation TildeLoc, 
                                           const DeclSpec& DS,
                                           bool HasTrailingLParen) {
  QualType ObjectType;
  if (CheckArrow(*this, ObjectType, Base, OpKind, OpLoc))
    return ExprError();

  QualType T = BuildDecltypeType(DS.getRepAsExpr(), DS.getTypeSpecTypeLoc());

  TypeLocBuilder TLB;
  DecltypeTypeLoc DecltypeTL = TLB.push<DecltypeTypeLoc>(T);
  DecltypeTL.setNameLoc(DS.getTypeSpecTypeLoc());
  TypeSourceInfo *DestructedTypeInfo = TLB.getTypeSourceInfo(Context, T);
  PseudoDestructorTypeStorage Destructed(DestructedTypeInfo);

  return BuildPseudoDestructorExpr(Base, OpLoc, OpKind, CXXScopeSpec(),
                                   0, SourceLocation(), TildeLoc,
                                   Destructed, HasTrailingLParen);
}

ExprResult Sema::BuildCXXMemberCallExpr(Expr *E, NamedDecl *FoundDecl,
                                        CXXConversionDecl *Method,
                                        bool HadMultipleCandidates) {
  if (Method->getParent()->isLambda() &&
      Method->getConversionType()->isBlockPointerType()) {
    // This is a lambda coversion to block pointer; check if the argument
    // is a LambdaExpr.
    Expr *SubE = E;
    CastExpr *CE = dyn_cast<CastExpr>(SubE);
    if (CE && CE->getCastKind() == CK_NoOp)
      SubE = CE->getSubExpr();
    SubE = SubE->IgnoreParens();
    if (CXXBindTemporaryExpr *BE = dyn_cast<CXXBindTemporaryExpr>(SubE))
      SubE = BE->getSubExpr();
    if (isa<LambdaExpr>(SubE)) {
      // For the conversion to block pointer on a lambda expression, we
      // construct a special BlockLiteral instead; this doesn't really make
      // a difference in ARC, but outside of ARC the resulting block literal
      // follows the normal lifetime rules for block literals instead of being
      // autoreleased.
      DiagnosticErrorTrap Trap(Diags);
      ExprResult Exp = BuildBlockForLambdaConversion(E->getExprLoc(),
                                                     E->getExprLoc(),
                                                     Method, E);
      if (Exp.isInvalid())
        Diag(E->getExprLoc(), diag::note_lambda_to_block_conv);
      return Exp;
    }
  }
      

  ExprResult Exp = PerformObjectArgumentInitialization(E, /*Qualifier=*/0,
                                          FoundDecl, Method);
  if (Exp.isInvalid())
    return true;

  MemberExpr *ME =
      new (Context) MemberExpr(Exp.take(), /*IsArrow=*/false, Method,
                               SourceLocation(), Context.BoundMemberTy,
                               VK_RValue, OK_Ordinary);
  if (HadMultipleCandidates)
    ME->setHadMultipleCandidates(true);

  QualType ResultType = Method->getResultType();
  ExprValueKind VK = Expr::getValueKindForType(ResultType);
  ResultType = ResultType.getNonLValueExprType(Context);

  MarkFunctionReferenced(Exp.get()->getLocStart(), Method);
  CXXMemberCallExpr *CE =
    new (Context) CXXMemberCallExpr(Context, ME, MultiExprArg(), ResultType, VK,
                                    Exp.get()->getLocEnd());
  return CE;
}

ExprResult Sema::BuildCXXNoexceptExpr(SourceLocation KeyLoc, Expr *Operand,
                                      SourceLocation RParen) {
  CanThrowResult CanThrow = canThrow(Operand);
  return Owned(new (Context) CXXNoexceptExpr(Context.BoolTy, Operand,
                                             CanThrow, KeyLoc, RParen));
}

ExprResult Sema::ActOnNoexceptExpr(SourceLocation KeyLoc, SourceLocation,
                                   Expr *Operand, SourceLocation RParen) {
  return BuildCXXNoexceptExpr(KeyLoc, Operand, RParen);
}

static bool IsSpecialDiscardedValue(Expr *E) {
  // In C++11, discarded-value expressions of a certain form are special,
  // according to [expr]p10:
  //   The lvalue-to-rvalue conversion (4.1) is applied only if the
  //   expression is an lvalue of volatile-qualified type and it has
  //   one of the following forms:
  E = E->IgnoreParens();

  //   - id-expression (5.1.1),
  if (isa<DeclRefExpr>(E))
    return true;

  //   - subscripting (5.2.1),
  if (isa<ArraySubscriptExpr>(E))
    return true;

  //   - class member access (5.2.5),
  if (isa<MemberExpr>(E))
    return true;

  //   - indirection (5.3.1),
  if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E))
    if (UO->getOpcode() == UO_Deref)
      return true;

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    //   - pointer-to-member operation (5.5),
    if (BO->isPtrMemOp())
      return true;

    //   - comma expression (5.18) where the right operand is one of the above.
    if (BO->getOpcode() == BO_Comma)
      return IsSpecialDiscardedValue(BO->getRHS());
  }

  //   - conditional expression (5.16) where both the second and the third
  //     operands are one of the above, or
  if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(E))
    return IsSpecialDiscardedValue(CO->getTrueExpr()) &&
           IsSpecialDiscardedValue(CO->getFalseExpr());
  // The related edge case of "*x ?: *x".
  if (BinaryConditionalOperator *BCO =
          dyn_cast<BinaryConditionalOperator>(E)) {
    if (OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(BCO->getTrueExpr()))
      return IsSpecialDiscardedValue(OVE->getSourceExpr()) &&
             IsSpecialDiscardedValue(BCO->getFalseExpr());
  }

  // Objective-C++ extensions to the rule.
  if (isa<PseudoObjectExpr>(E) || isa<ObjCIvarRefExpr>(E))
    return true;

  return false;
}

/// Perform the conversions required for an expression used in a
/// context that ignores the result.
ExprResult Sema::IgnoredValueConversions(Expr *E) {
  if (E->hasPlaceholderType()) {
    ExprResult result = CheckPlaceholderExpr(E);
    if (result.isInvalid()) return Owned(E);
    E = result.take();
  }

  // C99 6.3.2.1:
  //   [Except in specific positions,] an lvalue that does not have
  //   array type is converted to the value stored in the
  //   designated object (and is no longer an lvalue).
  if (E->isRValue()) {
    // In C, function designators (i.e. expressions of function type)
    // are r-values, but we still want to do function-to-pointer decay
    // on them.  This is both technically correct and convenient for
    // some clients.
    if (!getLangOpts().CPlusPlus && E->getType()->isFunctionType())
      return DefaultFunctionArrayConversion(E);

    return Owned(E);
  }

  if (getLangOpts().CPlusPlus)  {
    // The C++11 standard defines the notion of a discarded-value expression;
    // normally, we don't need to do anything to handle it, but if it is a
    // volatile lvalue with a special form, we perform an lvalue-to-rvalue
    // conversion.
    if (getLangOpts().CPlusPlus0x && E->isGLValue() &&
        E->getType().isVolatileQualified() &&
        IsSpecialDiscardedValue(E)) {
      ExprResult Res = DefaultLvalueConversion(E);
      if (Res.isInvalid())
        return Owned(E);
      E = Res.take();
    }
    return Owned(E);
  }

  // GCC seems to also exclude expressions of incomplete enum type.
  if (const EnumType *T = E->getType()->getAs<EnumType>()) {
    if (!T->getDecl()->isComplete()) {
      // FIXME: stupid workaround for a codegen bug!
      E = ImpCastExprToType(E, Context.VoidTy, CK_ToVoid).take();
      return Owned(E);
    }
  }

  ExprResult Res = DefaultFunctionArrayLvalueConversion(E);
  if (Res.isInvalid())
    return Owned(E);
  E = Res.take();

  if (!E->getType()->isVoidType())
    RequireCompleteType(E->getExprLoc(), E->getType(),
                        diag::err_incomplete_type);
  return Owned(E);
}

ExprResult Sema::ActOnFinishFullExpr(Expr *FE, SourceLocation CC) {
  ExprResult FullExpr = Owned(FE);

  if (!FullExpr.get())
    return ExprError();

  if (DiagnoseUnexpandedParameterPack(FullExpr.get()))
    return ExprError();

  // Top-level message sends default to 'id' when we're in a debugger.
  if (getLangOpts().DebuggerCastResultToId &&
      FullExpr.get()->getType() == Context.UnknownAnyTy &&
      isa<ObjCMessageExpr>(FullExpr.get())) {
    FullExpr = forceUnknownAnyToType(FullExpr.take(), Context.getObjCIdType());
    if (FullExpr.isInvalid())
      return ExprError();
  }
  
  FullExpr = CheckPlaceholderExpr(FullExpr.take());
  if (FullExpr.isInvalid())
    return ExprError();

  FullExpr = IgnoredValueConversions(FullExpr.take());
  if (FullExpr.isInvalid())
    return ExprError();

  CheckImplicitConversions(FullExpr.get(), CC);
  return MaybeCreateExprWithCleanups(FullExpr);
}

StmtResult Sema::ActOnFinishFullStmt(Stmt *FullStmt) {
  if (!FullStmt) return StmtError();

  return MaybeCreateStmtWithCleanups(FullStmt);
}

Sema::IfExistsResult 
Sema::CheckMicrosoftIfExistsSymbol(Scope *S,
                                   CXXScopeSpec &SS,
                                   const DeclarationNameInfo &TargetNameInfo) {
  DeclarationName TargetName = TargetNameInfo.getName();
  if (!TargetName)
    return IER_DoesNotExist;
  
  // If the name itself is dependent, then the result is dependent.
  if (TargetName.isDependentName())
    return IER_Dependent;
  
  // Do the redeclaration lookup in the current scope.
  LookupResult R(*this, TargetNameInfo, Sema::LookupAnyName,
                 Sema::NotForRedeclaration);
  LookupParsedName(R, S, &SS);
  R.suppressDiagnostics();
  
  switch (R.getResultKind()) {
  case LookupResult::Found:
  case LookupResult::FoundOverloaded:
  case LookupResult::FoundUnresolvedValue:
  case LookupResult::Ambiguous:
    return IER_Exists;
    
  case LookupResult::NotFound:
    return IER_DoesNotExist;
    
  case LookupResult::NotFoundInCurrentInstantiation:
    return IER_Dependent;
  }

  llvm_unreachable("Invalid LookupResult Kind!");
}

Sema::IfExistsResult 
Sema::CheckMicrosoftIfExistsSymbol(Scope *S, SourceLocation KeywordLoc,
                                   bool IsIfExists, CXXScopeSpec &SS,
                                   UnqualifiedId &Name) {
  DeclarationNameInfo TargetNameInfo = GetNameFromUnqualifiedId(Name);
  
  // Check for unexpanded parameter packs.
  SmallVector<UnexpandedParameterPack, 4> Unexpanded;
  collectUnexpandedParameterPacks(SS, Unexpanded);
  collectUnexpandedParameterPacks(TargetNameInfo, Unexpanded);
  if (!Unexpanded.empty()) {
    DiagnoseUnexpandedParameterPacks(KeywordLoc,
                                     IsIfExists? UPPC_IfExists 
                                               : UPPC_IfNotExists, 
                                     Unexpanded);
    return IER_Error;
  }
  
  return CheckMicrosoftIfExistsSymbol(S, SS, TargetNameInfo);
}

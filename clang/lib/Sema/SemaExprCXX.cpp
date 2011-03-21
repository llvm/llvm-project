//===--- SemaExprCXX.cpp - Semantic Analysis for Expressions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
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
  bool isUnevaluatedOperand = true;
  if (E && !E->isTypeDependent()) {
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
      if (RecordD->isPolymorphic() && E->Classify(Context).isGLValue()) {
        isUnevaluatedOperand = false;

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
      ImpCastExprToType(E, UnqualT, CK_NoOp, CastCategory(E));
    }
  }

  // If this is an unevaluated operand, clear out the set of
  // declaration references we have been computing and eliminate any
  // temporaries introduced in its computation.
  if (isUnevaluatedOperand)
    ExprEvalContexts.back().Context = Unevaluated;

  return Owned(new (Context) CXXTypeidExpr(TypeInfoType.withConst(),
                                           E,
                                           SourceRange(TypeidLoc, RParenLoc)));
}

/// ActOnCXXTypeidOfType - Parse typeid( type-id ) or typeid (expression);
ExprResult
Sema::ActOnCXXTypeid(SourceLocation OpLoc, SourceLocation LParenLoc,
                     bool isType, void *TyOrExpr, SourceLocation RParenLoc) {
  // Find the std::type_info type.
  if (!StdNamespace)
    return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));

  if (!CXXTypeInfoDecl) {
    IdentifierInfo *TypeInfoII = &PP.getIdentifierTable().get("type_info");
    LookupResult R(*this, TypeInfoII, SourceLocation(), LookupTagName);
    LookupQualifiedName(R, getStdNamespace());
    CXXTypeInfoDecl = R.getAsSingle<RecordDecl>();
    if (!CXXTypeInfoDecl)
      return ExprError(Diag(OpLoc, diag::err_need_header_before_typeid));
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

/// Retrieve the UuidAttr associated with QT.
static UuidAttr *GetUuidAttrOfType(QualType QT) {
  // Optionally remove one level of pointer, reference or array indirection.
  const Type *Ty = QT.getTypePtr();;
  if (QT->isPointerType() || QT->isReferenceType())
    Ty = QT->getPointeeType().getTypePtr();
  else if (QT->isArrayType())
    Ty = cast<ArrayType>(QT)->getElementType().getTypePtr();

  // Loop all class definition and declaration looking for an uuid attribute.
  CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
  while (RD) {
    if (UuidAttr *Uuid = RD->getAttr<UuidAttr>())
      return Uuid;
    RD = RD->getPreviousDeclaration();
  }
  return 0;
}

/// \brief Build a Microsoft __uuidof expression with a type operand.
ExprResult Sema::BuildCXXUuidof(QualType TypeInfoType,
                                SourceLocation TypeidLoc,
                                TypeSourceInfo *Operand,
                                SourceLocation RParenLoc) {
  if (!Operand->getType()->isDependentType()) {
    if (!GetUuidAttrOfType(Operand->getType()))
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
    if (!GetUuidAttrOfType(E->getType()) &&
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
Sema::ActOnCXXThrow(SourceLocation OpLoc, Expr *Ex) {
  // Don't report an error if 'throw' is used in system headers.
  if (!getLangOptions().CXXExceptions &&
      !getSourceManager().isInSystemHeader(OpLoc))
    Diag(OpLoc, diag::err_exceptions_disabled) << "throw";

  if (Ex && !Ex->isTypeDependent() && CheckCXXThrowOperand(OpLoc, Ex))
    return ExprError();
  return Owned(new (Context) CXXThrowExpr(Ex, Context.VoidTy, OpLoc));
}

/// CheckCXXThrowOperand - Validate the operand of a throw.
bool Sema::CheckCXXThrowOperand(SourceLocation ThrowLoc, Expr *&E) {
  // C++ [except.throw]p3:
  //   A throw-expression initializes a temporary object, called the exception
  //   object, the type of which is determined by removing any top-level
  //   cv-qualifiers from the static type of the operand of throw and adjusting
  //   the type from "array of T" or "function returning T" to "pointer to T"
  //   or "pointer to function returning T", [...]
  if (E->getType().hasQualifiers())
    ImpCastExprToType(E, E->getType().getUnqualifiedType(), CK_NoOp,
                      CastCategory(E));

  DefaultFunctionArrayConversion(E);

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
                            PDiag(isPointer ? diag::err_throw_incomplete_ptr
                                            : diag::err_throw_incomplete)
                              << E->getSourceRange()))
      return true;

    if (RequireNonAbstractType(ThrowLoc, E->getType(),
                               PDiag(diag::err_throw_abstract_type)
                                 << E->getSourceRange()))
      return true;
  }

  // Initialize the exception result.  This implicitly weeds out
  // abstract types or types with inaccessible copy constructors.
  const VarDecl *NRVOVariable = getCopyElisionCandidate(QualType(), E, false);

  // FIXME: Determine whether we can elide this copy per C++0x [class.copy]p32.
  InitializedEntity Entity =
      InitializedEntity::InitializeException(ThrowLoc, E->getType(),
                                             /*NRVO=*/false);
  ExprResult Res = PerformMoveOrCopyInitialization(Entity, NRVOVariable,
                                                   QualType(), E);
  if (Res.isInvalid())
    return true;
  E = Res.takeAs<Expr>();

  // If the exception has class type, we need additional handling.
  const RecordType *RecordTy = Ty->getAs<RecordType>();
  if (!RecordTy)
    return false;
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());

  // If we are throwing a polymorphic class type or pointer thereof,
  // exception handling will make use of the vtable.
  MarkVTableUsed(ThrowLoc, RD);

  // If a pointer is thrown, the referenced object will not be destroyed.
  if (isPointer)
    return false;

  // If the class has a non-trivial destructor, we must be able to call it.
  if (RD->hasTrivialDestructor())
    return false;

  CXXDestructorDecl *Destructor
    = const_cast<CXXDestructorDecl*>(LookupDestructor(RD));
  if (!Destructor)
    return false;

  MarkDeclarationReferenced(E->getExprLoc(), Destructor);
  CheckDestructorAccess(E->getExprLoc(), Destructor,
                        PDiag(diag::err_access_dtor_exception) << Ty);
  return false;
}

CXXMethodDecl *Sema::tryCaptureCXXThis() {
  // Ignore block scopes: we can capture through them.
  // Ignore nested enum scopes: we'll diagnose non-constant expressions
  // where they're invalid, and other uses are legitimate.
  // Don't ignore nested class scopes: you can't use 'this' in a local class.
  DeclContext *DC = CurContext;
  while (true) {
    if (isa<BlockDecl>(DC)) DC = cast<BlockDecl>(DC)->getDeclContext();
    else if (isa<EnumDecl>(DC)) DC = cast<EnumDecl>(DC)->getDeclContext();
    else break;
  }

  // If we're not in an instance method, error out.
  CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(DC);
  if (!method || !method->isInstance())
    return 0;

  // Mark that we're closing on 'this' in all the block scopes, if applicable.
  for (unsigned idx = FunctionScopes.size() - 1;
       isa<BlockScopeInfo>(FunctionScopes[idx]);
       --idx)
    cast<BlockScopeInfo>(FunctionScopes[idx])->CapturesCXXThis = true;

  return method;
}

ExprResult Sema::ActOnCXXThis(SourceLocation loc) {
  /// C++ 9.3.2: In the body of a non-static member function, the keyword this
  /// is a non-lvalue expression whose value is the address of the object for
  /// which the function is called.

  CXXMethodDecl *method = tryCaptureCXXThis();
  if (!method) return Diag(loc, diag::err_invalid_this_use);

  return Owned(new (Context) CXXThisExpr(loc, method->getThisType(Context),
                                         /*isImplicit=*/false));
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
  unsigned NumExprs = exprs.size();
  Expr **Exprs = (Expr**)exprs.get();
  SourceLocation TyBeginLoc = TInfo->getTypeLoc().getBeginLoc();
  SourceRange FullRange = SourceRange(TyBeginLoc, RParenLoc);

  if (Ty->isDependentType() ||
      CallExpr::hasAnyTypeDependentArguments(Exprs, NumExprs)) {
    exprs.release();

    return Owned(CXXUnresolvedConstructExpr::Create(Context, TInfo,
                                                    LParenLoc,
                                                    Exprs, NumExprs,
                                                    RParenLoc));
  }

  if (Ty->isArrayType())
    return ExprError(Diag(TyBeginLoc,
                          diag::err_value_init_for_array_type) << FullRange);
  if (!Ty->isVoidType() &&
      RequireCompleteType(TyBeginLoc, Ty,
                          PDiag(diag::err_invalid_incomplete_type_use)
                            << FullRange))
    return ExprError();

  if (RequireNonAbstractType(TyBeginLoc, Ty,
                             diag::err_allocation_of_abstract_type))
    return ExprError();


  // C++ [expr.type.conv]p1:
  // If the expression list is a single expression, the type conversion
  // expression is equivalent (in definedness, and if defined in meaning) to the
  // corresponding cast expression.
  //
  if (NumExprs == 1) {
    CastKind Kind = CK_Invalid;
    ExprValueKind VK = VK_RValue;
    CXXCastPath BasePath;
    if (CheckCastTypes(TInfo->getTypeLoc().getSourceRange(), Ty, Exprs[0],
                       Kind, VK, BasePath,
                       /*FunctionalStyle=*/true))
      return ExprError();

    exprs.release();

    return Owned(CXXFunctionalCastExpr::Create(Context,
                                               Ty.getNonLValueExprType(Context),
                                               VK, TInfo, TyBeginLoc, Kind,
                                               Exprs[0], &BasePath,
                                               RParenLoc));
  }

  InitializedEntity Entity = InitializedEntity::InitializeTemporary(TInfo);
  InitializationKind Kind
    = NumExprs ? InitializationKind::CreateDirect(TyBeginLoc,
                                                  LParenLoc, RParenLoc)
               : InitializationKind::CreateValue(TyBeginLoc,
                                                 LParenLoc, RParenLoc);
  InitializationSequence InitSeq(*this, Entity, Kind, Exprs, NumExprs);
  ExprResult Result = InitSeq.Perform(*this, Entity, Kind, move(exprs));

  // FIXME: Improve AST representation?
  return move(Result);
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

/// ActOnCXXNew - Parsed a C++ 'new' expression (C++ 5.3.4), as in e.g.:
/// @code new (memory) int[size][4] @endcode
/// or
/// @code ::new Foo(23, "hello") @endcode
/// For the interpretation of this heap of arguments, consult the base version.
ExprResult
Sema::ActOnCXXNew(SourceLocation StartLoc, bool UseGlobal,
                  SourceLocation PlacementLParen, MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen, SourceRange TypeIdParens,
                  Declarator &D, SourceLocation ConstructorLParen,
                  MultiExprArg ConstructorArgs,
                  SourceLocation ConstructorRParen) {
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
        if (!NumElts->isTypeDependent() && !NumElts->isValueDependent() &&
            !NumElts->isIntegerConstantExpr(Context)) {
          Diag(D.getTypeObject(I).Loc, diag::err_new_array_nonconst)
            << NumElts->getSourceRange();
          return ExprError();
        }
      }
    }
  }

  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, /*Scope=*/0, /*OwnedDecl=*/0,
                                               /*AllowAuto=*/true);
  QualType AllocType = TInfo->getType();
  if (D.isInvalidType())
    return ExprError();

  return BuildCXXNew(StartLoc, UseGlobal,
                     PlacementLParen,
                     move(PlacementArgs),
                     PlacementRParen,
                     TypeIdParens,
                     AllocType,
                     TInfo,
                     ArraySize,
                     ConstructorLParen,
                     move(ConstructorArgs),
                     ConstructorRParen,
                     TypeContainsAuto);
}

ExprResult
Sema::BuildCXXNew(SourceLocation StartLoc, bool UseGlobal,
                  SourceLocation PlacementLParen,
                  MultiExprArg PlacementArgs,
                  SourceLocation PlacementRParen,
                  SourceRange TypeIdParens,
                  QualType AllocType,
                  TypeSourceInfo *AllocTypeInfo,
                  Expr *ArraySize,
                  SourceLocation ConstructorLParen,
                  MultiExprArg ConstructorArgs,
                  SourceLocation ConstructorRParen,
                  bool TypeMayContainAuto) {
  SourceRange TypeRange = AllocTypeInfo->getTypeLoc().getSourceRange();

  // C++0x [decl.spec.auto]p6. Deduce the type which 'auto' stands in for.
  if (TypeMayContainAuto && AllocType->getContainedAutoType()) {
    if (ConstructorArgs.size() == 0)
      return ExprError(Diag(StartLoc, diag::err_auto_new_requires_ctor_arg)
                       << AllocType << TypeRange);
    if (ConstructorArgs.size() != 1) {
      Expr *FirstBad = ConstructorArgs.get()[1];
      return ExprError(Diag(FirstBad->getSourceRange().getBegin(),
                            diag::err_auto_new_ctor_multiple_expressions)
                       << AllocType << TypeRange);
    }
    TypeSourceInfo *DeducedType = 0;
    if (!DeduceAutoType(AllocTypeInfo, ConstructorArgs.get()[0], DeducedType))
      return ExprError(Diag(StartLoc, diag::err_auto_new_deduction_failure)
                       << AllocType
                       << ConstructorArgs.get()[0]->getType()
                       << TypeRange
                       << ConstructorArgs.get()[0]->getSourceRange());
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

  QualType ResultType = Context.getPointerType(AllocType);

  // C++ 5.3.4p6: "The expression in a direct-new-declarator shall have integral
  //   or enumeration type with a non-negative value."
  if (ArraySize && !ArraySize->isTypeDependent()) {

    QualType SizeType = ArraySize->getType();

    ExprResult ConvertedSize
      = ConvertToIntegralOrEnumerationType(StartLoc, ArraySize,
                                       PDiag(diag::err_array_size_not_integral),
                                     PDiag(diag::err_array_size_incomplete_type)
                                       << ArraySize->getSourceRange(),
                               PDiag(diag::err_array_size_explicit_conversion),
                                       PDiag(diag::note_array_size_conversion),
                               PDiag(diag::err_array_size_ambiguous_conversion),
                                       PDiag(diag::note_array_size_conversion),
                          PDiag(getLangOptions().CPlusPlus0x? 0
                                            : diag::ext_array_size_conversion));
    if (ConvertedSize.isInvalid())
      return ExprError();

    ArraySize = ConvertedSize.take();
    SizeType = ArraySize->getType();
    if (!SizeType->isIntegralOrUnscopedEnumerationType())
      return ExprError();

    // Let's see if this is a constant < 0. If so, we reject it out of hand.
    // We don't care about special rules, so we tell the machinery it's not
    // evaluated - it gives us a result in more cases.
    if (!ArraySize->isValueDependent()) {
      llvm::APSInt Value;
      if (ArraySize->isIntegerConstantExpr(Value, Context, 0, false)) {
        if (Value < llvm::APSInt(
                        llvm::APInt::getNullValue(Value.getBitWidth()),
                                 Value.isUnsigned()))
          return ExprError(Diag(ArraySize->getSourceRange().getBegin(),
                                diag::err_typecheck_negative_array_size)
            << ArraySize->getSourceRange());

        if (!AllocType->isDependentType()) {
          unsigned ActiveSizeBits
            = ConstantArrayType::getNumAddressingBits(Context, AllocType, Value);
          if (ActiveSizeBits > ConstantArrayType::getMaxSizeBits(Context)) {
            Diag(ArraySize->getSourceRange().getBegin(),
                 diag::err_array_too_large)
              << Value.toString(10)
              << ArraySize->getSourceRange();
            return ExprError();
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

    ImpCastExprToType(ArraySize, Context.getSizeType(),
                      CK_IntegralCast);
  }

  FunctionDecl *OperatorNew = 0;
  FunctionDecl *OperatorDelete = 0;
  Expr **PlaceArgs = (Expr**)PlacementArgs.get();
  unsigned NumPlaceArgs = PlacementArgs.size();

  if (!AllocType->isDependentType() &&
      !Expr::hasAnyTypeDependentArguments(PlaceArgs, NumPlaceArgs) &&
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

  llvm::SmallVector<Expr *, 8> AllPlaceArgs;
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
  }

  bool Init = ConstructorLParen.isValid();
  // --- Choosing a constructor ---
  CXXConstructorDecl *Constructor = 0;
  Expr **ConsArgs = (Expr**)ConstructorArgs.get();
  unsigned NumConsArgs = ConstructorArgs.size();
  ASTOwningVector<Expr*> ConvertedConstructorArgs(*this);

  // Array 'new' can't have any initializers.
  if (NumConsArgs && (ResultType->isArrayType() || ArraySize)) {
    SourceRange InitRange(ConsArgs[0]->getLocStart(),
                          ConsArgs[NumConsArgs - 1]->getLocEnd());

    Diag(StartLoc, diag::err_new_array_init_args) << InitRange;
    return ExprError();
  }

  if (!AllocType->isDependentType() &&
      !Expr::hasAnyTypeDependentArguments(ConsArgs, NumConsArgs)) {
    // C++0x [expr.new]p15:
    //   A new-expression that creates an object of type T initializes that
    //   object as follows:
    InitializationKind Kind
    //     - If the new-initializer is omitted, the object is default-
    //       initialized (8.5); if no initialization is performed,
    //       the object has indeterminate value
      = !Init? InitializationKind::CreateDefault(TypeRange.getBegin())
    //     - Otherwise, the new-initializer is interpreted according to the
    //       initialization rules of 8.5 for direct-initialization.
             : InitializationKind::CreateDirect(TypeRange.getBegin(),
                                                ConstructorLParen,
                                                ConstructorRParen);

    InitializedEntity Entity
      = InitializedEntity::InitializeNew(StartLoc, AllocType);
    InitializationSequence InitSeq(*this, Entity, Kind, ConsArgs, NumConsArgs);
    ExprResult FullInit = InitSeq.Perform(*this, Entity, Kind,
                                                move(ConstructorArgs));
    if (FullInit.isInvalid())
      return ExprError();

    // FullInit is our initializer; walk through it to determine if it's a
    // constructor call, which CXXNewExpr handles directly.
    if (Expr *FullInitExpr = (Expr *)FullInit.get()) {
      if (CXXBindTemporaryExpr *Binder
            = dyn_cast<CXXBindTemporaryExpr>(FullInitExpr))
        FullInitExpr = Binder->getSubExpr();
      if (CXXConstructExpr *Construct
                    = dyn_cast<CXXConstructExpr>(FullInitExpr)) {
        Constructor = Construct->getConstructor();
        for (CXXConstructExpr::arg_iterator A = Construct->arg_begin(),
                                         AEnd = Construct->arg_end();
             A != AEnd; ++A)
          ConvertedConstructorArgs.push_back(*A);
      } else {
        // Take the converted initializer.
        ConvertedConstructorArgs.push_back(FullInit.release());
      }
    } else {
      // No initialization required.
    }

    // Take the converted arguments and use them for the new expression.
    NumConsArgs = ConvertedConstructorArgs.size();
    ConsArgs = (Expr **)ConvertedConstructorArgs.take();
  }

  // Mark the new and delete operators as referenced.
  if (OperatorNew)
    MarkDeclarationReferenced(StartLoc, OperatorNew);
  if (OperatorDelete)
    MarkDeclarationReferenced(StartLoc, OperatorDelete);

  // FIXME: Also check that the destructor is accessible. (C++ 5.3.4p16)

  PlacementArgs.release();
  ConstructorArgs.release();

  return Owned(new (Context) CXXNewExpr(Context, UseGlobal, OperatorNew,
                                        PlaceArgs, NumPlaceArgs, TypeIdParens,
                                        ArraySize, Constructor, Init,
                                        ConsArgs, NumConsArgs, OperatorDelete,
                                        UsualArrayDeleteWantsSize,
                                        ResultType, AllocTypeInfo,
                                        StartLoc,
                                        Init ? ConstructorRParen :
                                               TypeRange.getEnd(),
                                        ConstructorLParen, ConstructorRParen));
}

/// CheckAllocatedType - Checks that a type is suitable as the allocated type
/// in a new-expression.
/// dimension off and stores the size expression in ArraySize.
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
           RequireCompleteType(Loc, AllocType,
                               PDiag(diag::err_new_incomplete_type)
                                 << R))
    return true;
  else if (RequireNonAbstractType(Loc, AllocType,
                                  diag::err_allocation_of_abstract_type))
    return true;
  else if (AllocType->isVariablyModifiedType())
    return Diag(Loc, diag::err_variably_modified_new_type)
             << AllocType;

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

  llvm::SmallVector<Expr*, 8> AllocArgs(1 + NumPlaceArgs);
  // We don't care about the actual value of this argument.
  // FIXME: Should the Sema create the expression and embed it in the syntax
  // tree? Or should the consumer just recalculate the value?
  IntegerLiteral Size(Context, llvm::APInt::getNullValue(
                      Context.Target.getPointerWidth(0)),
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
  if (!getLangOptions().Exceptions) {
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

  llvm::SmallVector<std::pair<DeclAccessPair,FunctionDecl*>, 2> Matches;

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

      llvm::SmallVector<QualType, 4> ArgTypes;
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
        TemplateDeductionInfo Info(Context, StartLoc);
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
    if (NumPlaceArgs && getLangOptions().CPlusPlus0x &&
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
                                  bool AllowMissing, FunctionDecl *&Operator) {
  LookupResult R(*this, Name, StartLoc, LookupOrdinaryName);
  LookupQualifiedName(R, Ctx);
  if (R.empty()) {
    if (AllowMissing)
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
                                   /*ExplicitTemplateArgs=*/0, Args, NumArgs,
                                   Candidates,
                                   /*SuppressUserConversions=*/false);
      continue;
    }

    FunctionDecl *Fn = cast<FunctionDecl>(D);
    AddOverloadCandidate(Fn, Alloc.getPair(), Args, NumArgs, Candidates,
                         /*SuppressUserConversions=*/false);
  }

  // Do the resolution.
  OverloadCandidateSet::iterator Best;
  switch (Candidates.BestViableFunction(*this, StartLoc, Best)) {
  case OR_Success: {
    // Got one!
    FunctionDecl *FnDecl = Best->Function;
    MarkDeclarationReferenced(StartLoc, FnDecl);
    // The first argument is size_t, and the first parameter must be size_t,
    // too. This is checked on declaration and can be assumed. (It can't be
    // asserted on, though, since invalid decls are left in there.)
    // Watch out for variadic allocator function.
    unsigned NumArgsInFnDecl = FnDecl->getNumParams();
    for (unsigned i = 0; (i < NumArgs && i < NumArgsInFnDecl); ++i) {
      ExprResult Result
        = PerformCopyInitialization(InitializedEntity::InitializeParameter(
                                                       Context,
                                                       FnDecl->getParamDecl(i)),
                                    SourceLocation(),
                                    Owned(Args[i]));
      if (Result.isInvalid())
        return true;

      Args[i] = Result.takeAs<Expr>();
    }
    Operator = FnDecl;
    CheckAllocationAccess(StartLoc, Range, R.getNamingClass(), Best->FoundDecl);
    return false;
  }

  case OR_No_Viable_Function:
    Diag(StartLoc, diag::err_ovl_no_viable_function_in_call)
      << Name << Range;
    Candidates.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    return true;

  case OR_Ambiguous:
    Diag(StartLoc, diag::err_ovl_ambiguous_call)
      << Name << Range;
    Candidates.NoteCandidates(*this, OCD_ViableCandidates, Args, NumArgs);
    return true;

  case OR_Deleted:
    Diag(StartLoc, diag::err_ovl_deleted_call)
      << Best->Function->isDeleted()
      << Name 
      << Best->Function->getMessageUnavailableAttr(
             !Best->Function->isDeleted())
      << Range;
    Candidates.NoteCandidates(*this, OCD_AllCandidates, Args, NumArgs);
    return true;
  }
  assert(false && "Unreachable, bad result from BestViableFunction");
  return true;
}


/// DeclareGlobalNewDelete - Declare the global forms of operator new and
/// delete. These are:
/// @code
///   void* operator new(std::size_t) throw(std::bad_alloc);
///   void* operator new[](std::size_t) throw(std::bad_alloc);
///   void operator delete(void *) throw();
///   void operator delete[](void *) throw();
/// @endcode
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
  //     void* operator new(std::size_t) throw(std::bad_alloc);
  //     void* operator new[](std::size_t) throw(std::bad_alloc);
  //     void  operator delete(void*) throw();
  //     void  operator delete[](void*) throw();
  //
  //   These implicit declarations introduce only the function names operator
  //   new, operator new[], operator delete, operator delete[].
  //
  // Here, we need to refer to std::bad_alloc, so we will implicitly declare
  // "std" or "bad_alloc" as necessary to form the exception specification.
  // However, we do not make these implicit declarations visible to name
  // lookup.
  if (!StdBadAlloc) {
    // The "std::bad_alloc" class has not yet been declared, so build it
    // implicitly.
    StdBadAlloc = CXXRecordDecl::Create(Context, TTK_Class,
                                        getOrCreateStdNamespace(),
                                        SourceLocation(),
                                      &PP.getIdentifierTable().get("bad_alloc"),
                                        SourceLocation(), 0);
    getStdBadAlloc()->setImplicit(true);
  }

  GlobalNewDeleteDeclared = true;

  QualType VoidPtr = Context.getPointerType(Context.VoidTy);
  QualType SizeT = Context.getSizeType();
  bool AssumeSaneOperatorNew = getLangOptions().AssumeSaneOperatorNew;

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
  if (HasBadAllocExceptionSpec) {
    assert(StdBadAlloc && "Must have std::bad_alloc declared");
    BadAllocType = Context.getTypeDeclType(getStdBadAlloc());
  }

  FunctionProtoType::ExtProtoInfo EPI;
  EPI.HasExceptionSpec = true;
  if (HasBadAllocExceptionSpec) {
    EPI.NumExceptions = 1;
    EPI.Exceptions = &BadAllocType;
  }

  QualType FnType = Context.getFunctionType(Return, &Argument, 1, EPI);
  FunctionDecl *Alloc =
    FunctionDecl::Create(Context, GlobalCtx, SourceLocation(), Name,
                         FnType, /*TInfo=*/0, SC_None,
                         SC_None, false, true);
  Alloc->setImplicit();

  if (AddMallocAttr)
    Alloc->addAttr(::new (Context) MallocAttr(SourceLocation(), Context));

  ParmVarDecl *Param = ParmVarDecl::Create(Context, Alloc, SourceLocation(),
                                           0, Argument, /*TInfo=*/0,
                                           SC_None,
                                           SC_None, 0);
  Alloc->setParams(&Param, 1);

  // FIXME: Also add this declaration to the IdentifierResolver, but
  // make sure it is at the end of the chain to coincide with the
  // global scope.
  Context.getTranslationUnitDecl()->addDecl(Alloc);
}

bool Sema::FindDeallocationFunction(SourceLocation StartLoc, CXXRecordDecl *RD,
                                    DeclarationName Name,
                                    FunctionDecl* &Operator) {
  LookupResult Found(*this, Name, StartLoc, LookupOrdinaryName);
  // Try to find operator delete/operator delete[] in class scope.
  LookupQualifiedName(Found, RD);

  if (Found.isAmbiguous())
    return true;

  Found.suppressDiagnostics();

  llvm::SmallVector<DeclAccessPair,4> Matches;
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
    CheckAllocationAccess(StartLoc, SourceRange(), Found.getNamingClass(),
                          Matches[0]);
    return false;

  // We found multiple suitable operators;  complain about the ambiguity.
  } else if (!Matches.empty()) {
    Diag(StartLoc, diag::err_ambiguous_suitable_delete_member_function_found)
      << Name << RD;

    for (llvm::SmallVectorImpl<DeclAccessPair>::iterator
           F = Matches.begin(), FEnd = Matches.end(); F != FEnd; ++F)
      Diag((*F)->getUnderlyingDecl()->getLocation(),
           diag::note_member_declared_here) << Name;
    return true;
  }

  // We did find operator delete/operator delete[] declarations, but
  // none of them were suitable.
  if (!Found.empty()) {
    Diag(StartLoc, diag::err_no_suitable_delete_member_function_found)
      << Name << RD;

    for (LookupResult::iterator F = Found.begin(), FEnd = Found.end();
         F != FEnd; ++F)
      Diag((*F)->getUnderlyingDecl()->getLocation(),
           diag::note_member_declared_here) << Name;

    return true;
  }

  // Look for a global declaration.
  DeclareGlobalNewDelete();
  DeclContext *TUDecl = Context.getTranslationUnitDecl();

  CXXNullPtrLiteralExpr Null(Context.VoidPtrTy, SourceLocation());
  Expr* DeallocArgs[1];
  DeallocArgs[0] = &Null;
  if (FindAllocationOverload(StartLoc, SourceRange(), Name,
                             DeallocArgs, 1, TUDecl, /*AllowMissing=*/false,
                             Operator))
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
                     bool ArrayForm, Expr *Ex) {
  // C++ [expr.delete]p1:
  //   The operand shall have a pointer type, or a class type having a single
  //   conversion function to a pointer type. The result has type void.
  //
  // DR599 amends "pointer type" to "pointer to object type" in both cases.

  FunctionDecl *OperatorDelete = 0;
  bool ArrayFormAsWritten = ArrayForm;
  bool UsualArrayDeleteWantsSize = false;

  if (!Ex->isTypeDependent()) {
    QualType Type = Ex->getType();

    if (const RecordType *Record = Type->getAs<RecordType>()) {
      if (RequireCompleteType(StartLoc, Type,
                              PDiag(diag::err_delete_incomplete_class_type)))
        return ExprError();

      llvm::SmallVector<CXXConversionDecl*, 4> ObjectPtrConversions;

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
        if (!PerformImplicitConversion(Ex,
                            ObjectPtrConversions.front()->getConversionType(),
                                      AA_Converting)) {
          Type = Ex->getType();
        }
      }
      else if (ObjectPtrConversions.size() > 1) {
        Diag(StartLoc, diag::err_ambiguous_delete_operand)
              << Type << Ex->getSourceRange();
        for (unsigned i= 0; i < ObjectPtrConversions.size(); i++)
          NoteOverloadCandidate(ObjectPtrConversions[i]);
        return ExprError();
      }
    }

    if (!Type->isPointerType())
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex->getSourceRange());

    QualType Pointee = Type->getAs<PointerType>()->getPointeeType();
    if (Pointee->isVoidType() && !isSFINAEContext()) {
      // The C++ standard bans deleting a pointer to a non-object type, which
      // effectively bans deletion of "void*". However, most compilers support
      // this, so we treat it as a warning unless we're in a SFINAE context.
      Diag(StartLoc, diag::ext_delete_void_ptr_operand)
        << Type << Ex->getSourceRange();
    } else if (Pointee->isFunctionType() || Pointee->isVoidType())
      return ExprError(Diag(StartLoc, diag::err_delete_operand)
        << Type << Ex->getSourceRange());
    else if (!Pointee->isDependentType() &&
             RequireCompleteType(StartLoc, Pointee,
                                 PDiag(diag::warn_delete_incomplete)
                                   << Ex->getSourceRange()))
      return ExprError();

    // C++ [expr.delete]p2:
    //   [Note: a pointer to a const type can be the operand of a
    //   delete-expression; it is not necessary to cast away the constness
    //   (5.2.11) of the pointer expression before it is used as the operand
    //   of the delete-expression. ]
    ImpCastExprToType(Ex, Context.getPointerType(Context.VoidTy),
                      CK_NoOp);

    if (Pointee->isArrayType() && !ArrayForm) {
      Diag(StartLoc, diag::warn_delete_array_type)
          << Type << Ex->getSourceRange()
          << FixItHint::CreateInsertion(PP.getLocForEndOfToken(StartLoc), "[]");
      ArrayForm = true;
    }

    DeclarationName DeleteName = Context.DeclarationNames.getCXXOperatorName(
                                      ArrayForm ? OO_Array_Delete : OO_Delete);

    QualType PointeeElem = Context.getBaseElementType(Pointee);
    if (const RecordType *RT = PointeeElem->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());

      if (!UseGlobal &&
          FindDeallocationFunction(StartLoc, RD, DeleteName, OperatorDelete))
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

      if (!RD->hasTrivialDestructor())
        if (CXXDestructorDecl *Dtor = LookupDestructor(RD)) {
          MarkDeclarationReferenced(StartLoc,
                                    const_cast<CXXDestructorDecl*>(Dtor));
          DiagnoseUseOfDecl(Dtor, StartLoc);
        }
    }

    if (!OperatorDelete) {
      // Look for a global declaration.
      DeclareGlobalNewDelete();
      DeclContext *TUDecl = Context.getTranslationUnitDecl();
      if (FindAllocationOverload(StartLoc, SourceRange(), DeleteName,
                                 &Ex, 1, TUDecl, /*AllowMissing=*/false,
                                 OperatorDelete))
        return ExprError();
    }

    MarkDeclarationReferenced(StartLoc, OperatorDelete);
    
    // Check access and ambiguity of operator delete and destructor.
    if (const RecordType *RT = PointeeElem->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (CXXDestructorDecl *Dtor = LookupDestructor(RD)) {
          CheckDestructorAccess(Ex->getExprLoc(), Dtor, 
                      PDiag(diag::err_access_dtor) << PointeeElem);
      }
    }

  }

  return Owned(new (Context) CXXDeleteExpr(Context.VoidTy, UseGlobal, ArrayForm,
                                           ArrayFormAsWritten,
                                           UsualArrayDeleteWantsSize,
                                           OperatorDelete, Ex, StartLoc));
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

  Expr *Condition = DeclRefExpr::Create(Context, NestedNameSpecifierLoc(), 
                                        ConditionVar,
                                        ConditionVar->getLocation(),
                            ConditionVar->getType().getNonReferenceType(),
                                        VK_LValue);
  if (ConvertToBoolean && CheckBooleanCondition(Condition, StmtLoc))
    return ExprError();

  return Owned(Condition);
}

/// CheckCXXBooleanCondition - Returns true if a conversion to bool is invalid.
bool Sema::CheckCXXBooleanCondition(Expr *&CondExpr) {
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
        if (!ToPtrType->getPointeeType().hasQualifiers() &&
            ((StrLit->isWide() && ToPointeeType->isWideCharType()) ||
             (!StrLit->isWide() &&
              (ToPointeeType->getKind() == BuiltinType::Char_U ||
               ToPointeeType->getKind() == BuiltinType::Char_S))))
          return true;
      }

  return false;
}

static ExprResult BuildCXXCastArgument(Sema &S,
                                       SourceLocation CastLoc,
                                       QualType Ty,
                                       CastKind Kind,
                                       CXXMethodDecl *Method,
                                       NamedDecl *FoundDecl,
                                       Expr *From) {
  switch (Kind) {
  default: assert(0 && "Unhandled cast kind!");
  case CK_ConstructorConversion: {
    ASTOwningVector<Expr*> ConstructorArgs(S);

    if (S.CompleteConstructorCall(cast<CXXConstructorDecl>(Method),
                                  MultiExprArg(&From, 1),
                                  CastLoc, ConstructorArgs))
      return ExprError();

    ExprResult Result =
    S.BuildCXXConstructExpr(CastLoc, Ty, cast<CXXConstructorDecl>(Method),
                            move_arg(ConstructorArgs),
                            /*ZeroInit*/ false, CXXConstructExpr::CK_Complete,
                            SourceRange());
    if (Result.isInvalid())
      return ExprError();

    return S.MaybeBindToTemporary(Result.takeAs<Expr>());
  }

  case CK_UserDefinedConversion: {
    assert(!From->getType()->isPointerType() && "Arg can't have pointer type!");

    // Create an implicit call expr that calls it.
    ExprResult Result = S.BuildCXXMemberCallExpr(From, FoundDecl, Method);
    if (Result.isInvalid())
      return ExprError();

    return S.MaybeBindToTemporary(Result.get());
  }
  }
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType using the pre-computed implicit
/// conversion sequence ICS. Returns true if there was an error, false
/// otherwise. The expression From is replaced with the converted
/// expression. Action is the kind of conversion we're performing,
/// used in the error message.
bool
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const ImplicitConversionSequence &ICS,
                                AssignmentAction Action, bool CStyle) {
  switch (ICS.getKind()) {
  case ImplicitConversionSequence::StandardConversion:
    if (PerformImplicitConversion(From, ToType, ICS.Standard, Action,
                                  CStyle))
      return true;
    break;

  case ImplicitConversionSequence::UserDefinedConversion: {

      FunctionDecl *FD = ICS.UserDefined.ConversionFunction;
      CastKind CastKind;
      QualType BeforeToType;
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
        if (PerformImplicitConversion(From, BeforeToType,
                                      ICS.UserDefined.Before, AA_Converting,
                                      CStyle))
          return true;
      }

      ExprResult CastArg
        = BuildCXXCastArgument(*this,
                               From->getLocStart(),
                               ToType.getNonReferenceType(),
                               CastKind, cast<CXXMethodDecl>(FD),
                               ICS.UserDefined.FoundConversionFunction,
                               From);

      if (CastArg.isInvalid())
        return true;

      From = CastArg.takeAs<Expr>();

      return PerformImplicitConversion(From, ToType, ICS.UserDefined.After,
                                       AA_Converting, CStyle);
  }

  case ImplicitConversionSequence::AmbiguousConversion:
    ICS.DiagnoseAmbiguousConversion(*this, From->getExprLoc(),
                          PDiag(diag::err_typecheck_ambiguous_condition)
                            << From->getSourceRange());
     return true;

  case ImplicitConversionSequence::EllipsisConversion:
    assert(false && "Cannot perform an ellipsis conversion");
    return false;

  case ImplicitConversionSequence::BadConversion:
    return true;
  }

  // Everything went well.
  return false;
}

/// PerformImplicitConversion - Perform an implicit conversion of the
/// expression From to the type ToType by following the standard
/// conversion sequence SCS. Returns true if there was an error, false
/// otherwise. The expression From is replaced with the converted
/// expression. Flavor is the context in which we're performing this
/// conversion, for use in error messages.
bool
Sema::PerformImplicitConversion(Expr *&From, QualType ToType,
                                const StandardConversionSequence& SCS,
                                AssignmentAction Action, bool CStyle) {
  // Overall FIXME: we are recomputing too many types here and doing far too
  // much extra work. What this means is that we need to keep track of more
  // information that is computed when we try the implicit conversion initially,
  // so that we don't need to recompute anything here.
  QualType FromType = From->getType();

  if (SCS.CopyConstructor) {
    // FIXME: When can ToType be a reference type?
    assert(!ToType->isReferenceType());
    if (SCS.Second == ICK_Derived_To_Base) {
      ASTOwningVector<Expr*> ConstructorArgs(*this);
      if (CompleteConstructorCall(cast<CXXConstructorDecl>(SCS.CopyConstructor),
                                  MultiExprArg(*this, &From, 1),
                                  /*FIXME:ConstructLoc*/SourceLocation(),
                                  ConstructorArgs))
        return true;
      ExprResult FromResult =
        BuildCXXConstructExpr(/*FIXME:ConstructLoc*/SourceLocation(),
                              ToType, SCS.CopyConstructor,
                              move_arg(ConstructorArgs),
                              /*ZeroInit*/ false,
                              CXXConstructExpr::CK_Complete,
                              SourceRange());
      if (FromResult.isInvalid())
        return true;
      From = FromResult.takeAs<Expr>();
      return false;
    }
    ExprResult FromResult =
      BuildCXXConstructExpr(/*FIXME:ConstructLoc*/SourceLocation(),
                            ToType, SCS.CopyConstructor,
                            MultiExprArg(*this, &From, 1),
                            /*ZeroInit*/ false,
                            CXXConstructExpr::CK_Complete,
                            SourceRange());

    if (FromResult.isInvalid())
      return true;

    From = FromResult.takeAs<Expr>();
    return false;
  }

  // Resolve overloaded function references.
  if (Context.hasSameType(FromType, Context.OverloadTy)) {
    DeclAccessPair Found;
    FunctionDecl *Fn = ResolveAddressOfOverloadedFunction(From, ToType,
                                                          true, Found);
    if (!Fn)
      return true;

    if (DiagnoseUseOfDecl(Fn, From->getSourceRange().getBegin()))
      return true;

    From = FixOverloadedFunctionReference(From, Found, Fn);
    FromType = From->getType();
  }

  // Perform the first implicit conversion.
  switch (SCS.First) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Lvalue_To_Rvalue:
    // Should this get its own ICK?
    if (From->getObjectKind() == OK_ObjCProperty) {
      ConvertPropertyForRValue(From);
      if (!From->isGLValue()) break;
    }

    // Check for trivial buffer overflows.
    CheckArrayAccess(From);

    FromType = FromType.getUnqualifiedType();
    From = ImplicitCastExpr::Create(Context, FromType, CK_LValueToRValue,
                                    From, 0, VK_RValue);
    break;

  case ICK_Array_To_Pointer:
    FromType = Context.getArrayDecayedType(FromType);
    ImpCastExprToType(From, FromType, CK_ArrayToPointerDecay);
    break;

  case ICK_Function_To_Pointer:
    FromType = Context.getPointerType(FromType);
    ImpCastExprToType(From, FromType, CK_FunctionToPointerDecay);
    break;

  default:
    assert(false && "Improper first standard conversion");
    break;
  }

  // Perform the second implicit conversion
  switch (SCS.Second) {
  case ICK_Identity:
    // If both sides are functions (or pointers/references to them), there could
    // be incompatible exception declarations.
    if (CheckExceptionSpecCompatibility(From, ToType))
      return true;
    // Nothing else to do.
    break;

  case ICK_NoReturn_Adjustment:
    // If both sides are functions (or pointers/references to them), there could
    // be incompatible exception declarations.
    if (CheckExceptionSpecCompatibility(From, ToType))
      return true;

    ImpCastExprToType(From, ToType, CK_NoOp);
    break;

  case ICK_Integral_Promotion:
  case ICK_Integral_Conversion:
    ImpCastExprToType(From, ToType, CK_IntegralCast);
    break;

  case ICK_Floating_Promotion:
  case ICK_Floating_Conversion:
    ImpCastExprToType(From, ToType, CK_FloatingCast);
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
    ImpCastExprToType(From, ToType, CK);
    break;
  }

  case ICK_Floating_Integral:
    if (ToType->isRealFloatingType())
      ImpCastExprToType(From, ToType, CK_IntegralToFloating);
    else
      ImpCastExprToType(From, ToType, CK_FloatingToIntegral);
    break;

  case ICK_Compatible_Conversion:
    ImpCastExprToType(From, ToType, CK_NoOp);
    break;

  case ICK_Pointer_Conversion: {
    if (SCS.IncompatibleObjC && Action != AA_Casting) {
      // Diagnose incompatible Objective-C conversions
      Diag(From->getSourceRange().getBegin(),
           diag::ext_typecheck_convert_incompatible_pointer)
        << From->getType() << ToType << Action
        << From->getSourceRange();
    }

    CastKind Kind = CK_Invalid;
    CXXCastPath BasePath;
    if (CheckPointerConversion(From, ToType, Kind, BasePath, CStyle))
      return true;
    ImpCastExprToType(From, ToType, Kind, VK_RValue, &BasePath);
    break;
  }

  case ICK_Pointer_Member: {
    CastKind Kind = CK_Invalid;
    CXXCastPath BasePath;
    if (CheckMemberPointerConversion(From, ToType, Kind, BasePath, CStyle))
      return true;
    if (CheckExceptionSpecCompatibility(From, ToType))
      return true;
    ImpCastExprToType(From, ToType, Kind, VK_RValue, &BasePath);
    break;
  }
  case ICK_Boolean_Conversion: {
    CastKind Kind = CK_Invalid;
    switch (FromType->getScalarTypeKind()) {
    case Type::STK_Pointer: Kind = CK_PointerToBoolean; break;
    case Type::STK_MemberPointer: Kind = CK_MemberPointerToBoolean; break;
    case Type::STK_Bool: llvm_unreachable("bool -> bool conversion?");
    case Type::STK_Integral: Kind = CK_IntegralToBoolean; break;
    case Type::STK_Floating: Kind = CK_FloatingToBoolean; break;
    case Type::STK_IntegralComplex: Kind = CK_IntegralComplexToBoolean; break;
    case Type::STK_FloatingComplex: Kind = CK_FloatingComplexToBoolean; break;
    }

    ImpCastExprToType(From, Context.BoolTy, Kind);
    break;
  }

  case ICK_Derived_To_Base: {
    CXXCastPath BasePath;
    if (CheckDerivedToBaseConversion(From->getType(),
                                     ToType.getNonReferenceType(),
                                     From->getLocStart(),
                                     From->getSourceRange(),
                                     &BasePath,
                                     CStyle))
      return true;

    ImpCastExprToType(From, ToType.getNonReferenceType(),
                      CK_DerivedToBase, CastCategory(From),
                      &BasePath);
    break;
  }

  case ICK_Vector_Conversion:
    ImpCastExprToType(From, ToType, CK_BitCast);
    break;

  case ICK_Vector_Splat:
    ImpCastExprToType(From, ToType, CK_VectorSplat);
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
        ImpCastExprToType(From, ElType,
                isFloatingComplex ? CK_FloatingCast : CK_FloatingToIntegral);
      } else {
        assert(From->getType()->isIntegerType());
        ImpCastExprToType(From, ElType,
                isFloatingComplex ? CK_IntegralToFloating : CK_IntegralCast);
      }
      // y -> _Complex y
      ImpCastExprToType(From, ToType,
                   isFloatingComplex ? CK_FloatingRealToComplex
                                     : CK_IntegralRealToComplex);

    // Case 2.  _Complex x -> y
    } else {
      const ComplexType *FromComplex = From->getType()->getAs<ComplexType>();
      assert(FromComplex);

      QualType ElType = FromComplex->getElementType();
      bool isFloatingComplex = ElType->isRealFloatingType();

      // _Complex x -> x
      ImpCastExprToType(From, ElType,
                   isFloatingComplex ? CK_FloatingComplexToReal
                                     : CK_IntegralComplexToReal);

      // x -> y
      if (Context.hasSameUnqualifiedType(ElType, ToType)) {
        // do nothing
      } else if (ToType->isRealFloatingType()) {
        ImpCastExprToType(From, ToType,
                isFloatingComplex ? CK_FloatingCast : CK_IntegralToFloating);
      } else {
        assert(ToType->isIntegerType());
        ImpCastExprToType(From, ToType,
                isFloatingComplex ? CK_FloatingToIntegral : CK_IntegralCast);
      }
    }
    break;
  
  case ICK_Block_Pointer_Conversion: {
      ImpCastExprToType(From, ToType.getUnqualifiedType(), CK_BitCast, VK_RValue);
      break;
    }
      
  case ICK_Lvalue_To_Rvalue:
  case ICK_Array_To_Pointer:
  case ICK_Function_To_Pointer:
  case ICK_Qualification:
  case ICK_Num_Conversion_Kinds:
    assert(false && "Improper second standard conversion");
    break;
  }

  switch (SCS.Third) {
  case ICK_Identity:
    // Nothing to do.
    break;

  case ICK_Qualification: {
    // The qualification keeps the category of the inner expression, unless the
    // target type isn't a reference.
    ExprValueKind VK = ToType->isReferenceType() ?
                                  CastCategory(From) : VK_RValue;
    ImpCastExprToType(From, ToType.getNonLValueExprType(Context),
                      CK_NoOp, VK);

    if (SCS.DeprecatedStringLiteralToCharPtr)
      Diag(From->getLocStart(), diag::warn_deprecated_string_literal_conversion)
        << ToType.getNonReferenceType();

    break;
    }

  default:
    assert(false && "Improper third standard conversion");
    break;
  }

  return false;
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

static bool EvaluateUnaryTypeTrait(Sema &Self, UnaryTypeTrait UTT, QualType T,
                                   SourceLocation KeyLoc) {
  // FIXME: For many of these traits, we need a complete type before we can 
  // check these properties.
  assert(!T->isDependentType() &&
         "Cannot evaluate traits for dependent types.");
  ASTContext &C = Self.Context;
  switch(UTT) {
  default: assert(false && "Unknown type trait or not implemented");
  case UTT_IsPOD: return T->isPODType();
  case UTT_IsLiteral: return T->isLiteralType();
  case UTT_IsClass: // Fallthrough
  case UTT_IsUnion:
    if (const RecordType *Record = T->getAs<RecordType>()) {
      bool Union = Record->getDecl()->isUnion();
      return UTT == UTT_IsUnion ? Union : !Union;
    }
    return false;
  case UTT_IsEnum: return T->isEnumeralType();
  case UTT_IsPolymorphic:
    if (const RecordType *Record = T->getAs<RecordType>()) {
      // Type traits are only parsed in C++, so we've got CXXRecords.
      return cast<CXXRecordDecl>(Record->getDecl())->isPolymorphic();
    }
    return false;
  case UTT_IsAbstract:
    if (const RecordType *RT = T->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->isAbstract();
    return false;
  case UTT_IsEmpty:
    if (const RecordType *Record = T->getAs<RecordType>()) {
      return !Record->getDecl()->isUnion()
          && cast<CXXRecordDecl>(Record->getDecl())->isEmpty();
    }
    return false;
  case UTT_HasTrivialConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true then the trait is true, else if type is
    //   a cv class or union type (or array thereof) with a trivial default
    //   constructor ([class.ctor]) then the trait is true, else it is false.
    if (T->isPODType())
      return true;
    if (const RecordType *RT =
          C.getBaseElementType(T)->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialConstructor();
    return false;
  case UTT_HasTrivialCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type then
    //   the trait is true, else if type is a cv class or union type
    //   with a trivial copy constructor ([class.copy]) then the trait
    //   is true, else it is false.
    if (T->isPODType() || T->isReferenceType())
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
    if (T->isPODType())
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
    if (T->isPODType() || T->isReferenceType())
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
    if (T->isPODType())
      return true;
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl* RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialCopyAssignment())
        return true;

      bool FoundAssign = false;
      bool AllNoThrow = true;
      DeclarationName Name = C.DeclarationNames.getCXXOperatorName(OO_Equal);
      LookupResult Res(Self, DeclarationNameInfo(Name, KeyLoc),
                       Sema::LookupOrdinaryName);
      if (Self.LookupQualifiedName(Res, RD)) {
        for (LookupResult::iterator Op = Res.begin(), OpEnd = Res.end();
             Op != OpEnd; ++Op) {
          CXXMethodDecl *Operator = cast<CXXMethodDecl>(*Op);
          if (Operator->isCopyAssignmentOperator()) {
            FoundAssign = true;
            const FunctionProtoType *CPT
                = Operator->getType()->getAs<FunctionProtoType>();
            if (!CPT->hasEmptyExceptionSpec()) {
              AllNoThrow = false;
              break;
            }
          }
        }
      }

      return FoundAssign && AllNoThrow;
    }
    return false;
  case UTT_HasNothrowCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __has_trivial_copy (type) is true then the trait is true, else
    //   if type is a cv class or union type with copy constructors that are
    //   known not to throw an exception then the trait is true, else it is
    //   false.
    if (T->isPODType() || T->isReferenceType())
      return true;
    if (const RecordType *RT = T->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialCopyConstructor())
        return true;

      bool FoundConstructor = false;
      bool AllNoThrow = true;
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
          // TODO: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          if (!CPT->hasEmptyExceptionSpec() || CPT->getNumArgs() > 1) {
            AllNoThrow = false;
            break;
          }
        }
      }

      return FoundConstructor && AllNoThrow;
    }
    return false;
  case UTT_HasNothrowConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __has_trivial_constructor (type) is true then the trait is
    //   true, else if type is a cv class or union type (or array
    //   thereof) with a default constructor that is known not to
    //   throw an exception then the trait is true, else it is false.
    if (T->isPODType())
      return true;
    if (const RecordType *RT = C.getBaseElementType(T)->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasTrivialConstructor())
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
          // TODO: check whether evaluating default arguments can throw.
          // For now, we'll be conservative and assume that they can throw.
          return CPT->hasEmptyExceptionSpec() && CPT->getNumArgs() == 0;
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
  }
}

ExprResult Sema::BuildUnaryTypeTrait(UnaryTypeTrait UTT,
                                     SourceLocation KWLoc,
                                     TypeSourceInfo *TSInfo,
                                     SourceLocation RParen) {
  QualType T = TSInfo->getType();

  // According to http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html
  // all traits except __is_class, __is_enum and __is_union require a the type
  // to be complete, an array of unknown bound, or void.
  if (UTT != UTT_IsClass && UTT != UTT_IsEnum && UTT != UTT_IsUnion) {
    QualType E = T;
    if (T->isIncompleteArrayType())
      E = Context.getAsArrayType(T)->getElementType();
    if (!T->isVoidType() &&
        RequireCompleteType(KWLoc, E,
                            diag::err_incomplete_type_used_in_type_trait_expr))
      return ExprError();
  }

  bool Value = false;
  if (!T->isDependentType())
    Value = EvaluateUnaryTypeTrait(*this, UTT, T, KWLoc);

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

static bool EvaluateBinaryTypeTrait(Sema &Self, BinaryTypeTrait BTT,
                                    QualType LhsT, QualType RhsT,
                                    SourceLocation KeyLoc) {
  assert((!LhsT->isDependentType() || RhsT->isDependentType()) &&
         "Cannot evaluate traits for dependent types.");

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

  case BTT_TypeCompatible:
    return Self.Context.typesAreCompatible(LhsT.getUnqualifiedType(),
                                           RhsT.getUnqualifiedType());
      
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
    if (LhsT->isObjectType() || LhsT->isFunctionType())
      LhsT = Self.Context.getRValueReferenceType(LhsT);
    
    InitializedEntity To(InitializedEntity::InitializeTemporary(RhsT));
    OpaqueValueExpr From(KeyLoc, LhsT.getNonLValueExprType(Self.Context),
                         Expr::getValueKindForType(LhsT));
    Expr *FromPtr = &From;
    InitializationKind Kind(InitializationKind::CreateCopy(KeyLoc, 
                                                           SourceLocation()));
    
    // Perform the initialization within a SFINAE trap at translation unit 
    // scope.
    Sema::SFINAETrap SFINAE(Self, /*AccessCheckingSFINAE=*/true);
    Sema::ContextRAII TUContext(Self, Self.Context.getTranslationUnitDecl());
    InitializationSequence Init(Self, To, Kind, &FromPtr, 1);
    if (Init.getKind() == InitializationSequence::FailedSequence)
      return false;

    ExprResult Result = Init.Perform(Self, To, Kind, MultiExprArg(&FromPtr, 1));
    return !Result.isInvalid() && !SFINAE.hasErrorOccurred();
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
    if (getLangOptions().CPlusPlus) {
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
  case BTT_TypeCompatible: ResultType = Context.IntTy; break;
  case BTT_IsConvertibleTo: ResultType = Context.BoolTy; break;
  }

  return Owned(new (Context) BinaryTypeTraitExpr(KWLoc, BTT, LhsTSInfo,
                                                 RhsTSInfo, Value, RParen,
                                                 ResultType));
}

QualType Sema::CheckPointerToMemberOperands(Expr *&lex, Expr *&rex,
                                            ExprValueKind &VK,
                                            SourceLocation Loc,
                                            bool isIndirect) {
  const char *OpSpelling = isIndirect ? "->*" : ".*";
  // C++ 5.5p2
  //   The binary operator .* [p3: ->*] binds its second operand, which shall
  //   be of type "pointer to member of T" (where T is a completely-defined
  //   class type) [...]
  QualType RType = rex->getType();
  const MemberPointerType *MemPtr = RType->getAs<MemberPointerType>();
  if (!MemPtr) {
    Diag(Loc, diag::err_bad_memptr_rhs)
      << OpSpelling << RType << rex->getSourceRange();
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
  QualType LType = lex->getType();
  if (isIndirect) {
    if (const PointerType *Ptr = LType->getAs<PointerType>())
      LType = Ptr->getPointeeType();
    else {
      Diag(Loc, diag::err_bad_memptr_lhs)
        << OpSpelling << 1 << LType
        << FixItHint::CreateReplacement(SourceRange(Loc), ".*");
      return QualType();
    }
  }

  if (!Context.hasSameUnqualifiedType(Class, LType)) {
    // If we want to check the hierarchy, we need a complete type.
    if (RequireCompleteType(Loc, LType, PDiag(diag::err_bad_memptr_lhs)
        << OpSpelling << (int)isIndirect)) {
      return QualType();
    }
    CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);
    // FIXME: Would it be useful to print full ambiguity paths, or is that
    // overkill?
    if (!IsDerivedFrom(LType, Class, Paths) ||
        Paths.isAmbiguous(Context.getCanonicalType(Class))) {
      Diag(Loc, diag::err_bad_memptr_lhs) << OpSpelling
        << (int)isIndirect << lex->getType();
      return QualType();
    }
    // Cast LHS to type of use.
    QualType UseType = isIndirect ? Context.getPointerType(Class) : Class;
    ExprValueKind VK =
        isIndirect ? VK_RValue : CastCategory(lex);

    CXXCastPath BasePath;
    BuildBasePathArray(Paths, BasePath);
    ImpCastExprToType(lex, UseType, CK_DerivedToBase, VK, &BasePath);
  }

  if (isa<CXXScalarValueInitExpr>(rex->IgnoreParens())) {
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
  // FIXME: This returns a dereferenced member function pointer as a normal
  // function type. However, the only operation valid on such functions is
  // calling them. There's also a GCC extension to get a function pointer to the
  // thing, which is another complication, because this type - unlike the type
  // that is the result of this expression - takes the class as the first
  // argument.
  // We probably need a "MemberFunctionClosureType" or something like that.
  QualType Result = MemPtr->getPointeeType();
  Result = Context.getCVRQualifiedType(Result, LType.getCVRQualifiers());

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
      if (!isIndirect && !lex->Classify(Context).isLValue())
        Diag(Loc, diag::err_pointer_to_member_oper_value_classify)
          << RType << 1 << lex->getSourceRange();
      break;

    case RQ_RValue:
      if (isIndirect || !lex->Classify(Context).isRValue())
        Diag(Loc, diag::err_pointer_to_member_oper_value_classify)
          << RType << 0 << lex->getSourceRange();
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
  if (Result->isFunctionType())
    VK = VK_RValue;
  else if (isIndirect)
    VK = VK_LValue;
  else
    VK = lex->getValueKind();

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
        if (InitSeq.getKind() != InitializationSequence::FailedSequence) {
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
  HaveConversion = InitSeq.getKind() != InitializationSequence::FailedSequence;
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
static bool FindConditionalOverload(Sema &Self, Expr *&LHS, Expr *&RHS,
                                    SourceLocation QuestionLoc) {
  Expr *Args[2] = { LHS, RHS };
  OverloadCandidateSet CandidateSet(QuestionLoc);
  Self.AddBuiltinOperatorCandidates(OO_Conditional, QuestionLoc, Args, 2,
                                    CandidateSet);

  OverloadCandidateSet::iterator Best;
  switch (CandidateSet.BestViableFunction(Self, QuestionLoc, Best)) {
    case OR_Success:
      // We found a match. Perform the conversions on the arguments and move on.
      if (Self.PerformImplicitConversion(LHS, Best->BuiltinTypes.ParamTypes[0],
                                         Best->Conversions[0], Sema::AA_Converting) ||
          Self.PerformImplicitConversion(RHS, Best->BuiltinTypes.ParamTypes[1],
                                         Best->Conversions[1], Sema::AA_Converting))
        break;
      if (Best->Function)
        Self.MarkDeclarationReferenced(QuestionLoc, Best->Function);
      return false;

    case OR_No_Viable_Function:

      // Emit a better diagnostic if one of the expressions is a null pointer
      // constant and the other is a pointer type. In this case, the user most
      // likely forgot to take the address of the other expression.
      if (Self.DiagnoseConditionalForNull(LHS, RHS, QuestionLoc))
        return true;

      Self.Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
        << LHS->getType() << RHS->getType()
        << LHS->getSourceRange() << RHS->getSourceRange();
      return true;

    case OR_Ambiguous:
      Self.Diag(QuestionLoc, diag::err_conditional_ambiguous_ovl)
        << LHS->getType() << RHS->getType()
        << LHS->getSourceRange() << RHS->getSourceRange();
      // FIXME: Print the possible common types by printing the return types of
      // the viable candidates.
      break;

    case OR_Deleted:
      assert(false && "Conditional operator has only built-in overloads");
      break;
  }
  return true;
}

/// \brief Perform an "extended" implicit conversion as returned by
/// TryClassUnification.
static bool ConvertForConditional(Sema &Self, Expr *&E, QualType T) {
  InitializedEntity Entity = InitializedEntity::InitializeTemporary(T);
  InitializationKind Kind = InitializationKind::CreateCopy(E->getLocStart(),
                                                           SourceLocation());
  InitializationSequence InitSeq(Self, Entity, Kind, &E, 1);
  ExprResult Result = InitSeq.Perform(Self, Entity, Kind, MultiExprArg(&E, 1));
  if (Result.isInvalid())
    return true;

  E = Result.takeAs<Expr>();
  return false;
}

/// \brief Check the operands of ?: under C++ semantics.
///
/// See C++ [expr.cond]. Note that LHS is never null, even for the GNU x ?: y
/// extension. In this case, LHS == Cond. (But they're not aliases.)
QualType Sema::CXXCheckConditionalOperands(Expr *&Cond, Expr *&LHS, Expr *&RHS,
                                           ExprValueKind &VK, ExprObjectKind &OK,
                                           SourceLocation QuestionLoc) {
  // FIXME: Handle C99's complex types, vector types, block pointers and Obj-C++
  // interface pointers.

  // C++0x 5.16p1
  //   The first expression is contextually converted to bool.
  if (!Cond->isTypeDependent()) {
    if (CheckCXXBooleanCondition(Cond))
      return QualType();
  }

  // Assume r-value.
  VK = VK_RValue;
  OK = OK_Ordinary;

  // Either of the arguments dependent?
  if (LHS->isTypeDependent() || RHS->isTypeDependent())
    return Context.DependentTy;

  // C++0x 5.16p2
  //   If either the second or the third operand has type (cv) void, ...
  QualType LTy = LHS->getType();
  QualType RTy = RHS->getType();
  bool LVoid = LTy->isVoidType();
  bool RVoid = RTy->isVoidType();
  if (LVoid || RVoid) {
    //   ... then the [l2r] conversions are performed on the second and third
    //   operands ...
    DefaultFunctionArrayLvalueConversion(LHS);
    DefaultFunctionArrayLvalueConversion(RHS);
    LTy = LHS->getType();
    RTy = RHS->getType();

    //   ... and one of the following shall hold:
    //   -- The second or the third operand (but not both) is a throw-
    //      expression; the result is of the type of the other and is an rvalue.
    bool LThrow = isa<CXXThrowExpr>(LHS);
    bool RThrow = isa<CXXThrowExpr>(RHS);
    if (LThrow && !RThrow)
      return RTy;
    if (RThrow && !LThrow)
      return LTy;

    //   -- Both the second and third operands have type void; the result is of
    //      type void and is an rvalue.
    if (LVoid && RVoid)
      return Context.VoidTy;

    // Neither holds, error.
    Diag(QuestionLoc, diag::err_conditional_void_nonvoid)
      << (LVoid ? RTy : LTy) << (LVoid ? 0 : 1)
      << LHS->getSourceRange() << RHS->getSourceRange();
    return QualType();
  }

  // Neither is void.

  // C++0x 5.16p3
  //   Otherwise, if the second and third operand have different types, and
  //   either has (cv) class type, and attempt is made to convert each of those
  //   operands to the other.
  if (!Context.hasSameType(LTy, RTy) &&
      (LTy->isRecordType() || RTy->isRecordType())) {
    ImplicitConversionSequence ICSLeftToRight, ICSRightToLeft;
    // These return true if a single direction is already ambiguous.
    QualType L2RType, R2LType;
    bool HaveL2R, HaveR2L;
    if (TryClassUnification(*this, LHS, RHS, QuestionLoc, HaveL2R, L2RType))
      return QualType();
    if (TryClassUnification(*this, RHS, LHS, QuestionLoc, HaveR2L, R2LType))
      return QualType();

    //   If both can be converted, [...] the program is ill-formed.
    if (HaveL2R && HaveR2L) {
      Diag(QuestionLoc, diag::err_conditional_ambiguous)
        << LTy << RTy << LHS->getSourceRange() << RHS->getSourceRange();
      return QualType();
    }

    //   If exactly one conversion is possible, that conversion is applied to
    //   the chosen operand and the converted operands are used in place of the
    //   original operands for the remainder of this section.
    if (HaveL2R) {
      if (ConvertForConditional(*this, LHS, L2RType))
        return QualType();
      LTy = LHS->getType();
    } else if (HaveR2L) {
      if (ConvertForConditional(*this, RHS, R2LType))
        return QualType();
      RTy = RHS->getType();
    }
  }

  // C++0x 5.16p4
  //   If the second and third operands are glvalues of the same value
  //   category and have the same type, the result is of that type and
  //   value category and it is a bit-field if the second or the third
  //   operand is a bit-field, or if both are bit-fields.
  // We only extend this to bitfields, not to the crazy other kinds of
  // l-values.
  bool Same = Context.hasSameType(LTy, RTy);
  if (Same &&
      LHS->isGLValue() &&
      LHS->getValueKind() == RHS->getValueKind() &&
      LHS->isOrdinaryOrBitFieldObject() &&
      RHS->isOrdinaryOrBitFieldObject()) {
    VK = LHS->getValueKind();
    if (LHS->getObjectKind() == OK_BitField ||
        RHS->getObjectKind() == OK_BitField)
      OK = OK_BitField;
    return LTy;
  }

  // C++0x 5.16p5
  //   Otherwise, the result is an rvalue. If the second and third operands
  //   do not have the same type, and either has (cv) class type, ...
  if (!Same && (LTy->isRecordType() || RTy->isRecordType())) {
    //   ... overload resolution is used to determine the conversions (if any)
    //   to be applied to the operands. If the overload resolution fails, the
    //   program is ill-formed.
    if (FindConditionalOverload(*this, LHS, RHS, QuestionLoc))
      return QualType();
  }

  // C++0x 5.16p6
  //   LValue-to-rvalue, array-to-pointer, and function-to-pointer standard
  //   conversions are performed on the second and third operands.
  DefaultFunctionArrayLvalueConversion(LHS);
  DefaultFunctionArrayLvalueConversion(RHS);
  LTy = LHS->getType();
  RTy = RHS->getType();

  //   After those conversions, one of the following shall hold:
  //   -- The second and third operands have the same type; the result
  //      is of that type. If the operands have class type, the result
  //      is a prvalue temporary of the result type, which is
  //      copy-initialized from either the second operand or the third
  //      operand depending on the value of the first operand.
  if (Context.getCanonicalType(LTy) == Context.getCanonicalType(RTy)) {
    if (LTy->isRecordType()) {
      // The operands have class type. Make a temporary copy.
      InitializedEntity Entity = InitializedEntity::InitializeTemporary(LTy);
      ExprResult LHSCopy = PerformCopyInitialization(Entity,
                                                     SourceLocation(),
                                                     Owned(LHS));
      if (LHSCopy.isInvalid())
        return QualType();

      ExprResult RHSCopy = PerformCopyInitialization(Entity,
                                                     SourceLocation(),
                                                     Owned(RHS));
      if (RHSCopy.isInvalid())
        return QualType();

      LHS = LHSCopy.takeAs<Expr>();
      RHS = RHSCopy.takeAs<Expr>();
    }

    return LTy;
  }

  // Extension: conditional operator involving vector types.
  if (LTy->isVectorType() || RTy->isVectorType())
    return CheckVectorOperands(QuestionLoc, LHS, RHS);

  //   -- The second and third operands have arithmetic or enumeration type;
  //      the usual arithmetic conversions are performed to bring them to a
  //      common type, and the result is of that type.
  if (LTy->isArithmeticType() && RTy->isArithmeticType()) {
    UsualArithmeticConversions(LHS, RHS);
    return LHS->getType();
  }

  //   -- The second and third operands have pointer type, or one has pointer
  //      type and the other is a null pointer constant; pointer conversions
  //      and qualification conversions are performed to bring them to their
  //      composite pointer type. The result is of the composite pointer type.
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
        << LHS->getSourceRange() << RHS->getSourceRange();

    return Composite;
  }

  // Similarly, attempt to find composite type of two objective-c pointers.
  Composite = FindCompositeObjCPointerType(LHS, RHS, QuestionLoc);
  if (!Composite.isNull())
    return Composite;

  // Check if we are using a null with a non-pointer type.
  if (DiagnoseConditionalForNull(LHS, RHS, QuestionLoc))
    return QualType();

  Diag(QuestionLoc, diag::err_typecheck_cond_incompatible_operands)
    << LHS->getType() << RHS->getType()
    << LHS->getSourceRange() << RHS->getSourceRange();
  return QualType();
}

/// \brief Find a merged pointer type and convert the two expressions to it.
///
/// This finds the composite pointer type (or member pointer type) for @p E1
/// and @p E2 according to C++0x 5.9p2. It converts both expressions to this
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

  assert(getLangOptions().CPlusPlus && "This function assumes C++");
  QualType T1 = E1->getType(), T2 = E2->getType();

  if (!T1->isAnyPointerType() && !T1->isMemberPointerType() &&
      !T2->isAnyPointerType() && !T2->isMemberPointerType())
   return QualType();

  // C++0x 5.9p2
  //   Pointer conversions and qualification conversions are performed on
  //   pointer operands to bring them to their composite pointer type. If
  //   one operand is a null pointer constant, the composite pointer type is
  //   the type of the other operand.
  if (E1->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    if (T2->isMemberPointerType())
      ImpCastExprToType(E1, T2, CK_NullToMemberPointer);
    else
      ImpCastExprToType(E1, T2, CK_NullToPointer);
    return T2;
  }
  if (E2->isNullPointerConstant(Context, Expr::NPC_ValueDependentIsNull)) {
    if (T1->isMemberPointerType())
      ImpCastExprToType(E2, T1, CK_NullToMemberPointer);
    else
      ImpCastExprToType(E2, T1, CK_NullToPointer);
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
  typedef llvm::SmallVector<unsigned, 4> QualifierVector;
  QualifierVector QualifierUnion;
  typedef llvm::SmallVector<std::pair<const Type *, const Type *>, 4>
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
      = E1ToC1.Perform(*this, Entity1, Kind, MultiExprArg(*this,&E1,1));
    if (E1Result.isInvalid())
      return QualType();
    E1 = E1Result.takeAs<Expr>();

    // Convert E2 to Composite1
    ExprResult E2Result
      = E2ToC1.Perform(*this, Entity1, Kind, MultiExprArg(*this,&E2,1));
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
    = E1ToC2.Perform(*this, Entity2, Kind, MultiExprArg(*this, &E1, 1));
  if (E1Result.isInvalid())
    return QualType();
  E1 = E1Result.takeAs<Expr>();

  // Convert E2 to Composite2
  ExprResult E2Result
    = E2ToC2.Perform(*this, Entity2, Kind, MultiExprArg(*this, &E2, 1));
  if (E2Result.isInvalid())
    return QualType();
  E2 = E2Result.takeAs<Expr>();

  return Composite2;
}

ExprResult Sema::MaybeBindToTemporary(Expr *E) {
  if (!E)
    return ExprError();

  if (!Context.getLangOptions().CPlusPlus)
    return Owned(E);

  assert(!isa<CXXBindTemporaryExpr>(E) && "Double-bound temporary?");

  const RecordType *RT = E->getType()->getAs<RecordType>();
  if (!RT)
    return Owned(E);

  // If the result is a glvalue, we shouldn't bind it.
  if (E->Classify(Context).isGLValue())
    return Owned(E);

  // That should be enough to guarantee that this type is complete.
  // If it has a trivial destructor, we can avoid the extra copy.
  CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
  if (RD->isInvalidDecl() || RD->hasTrivialDestructor())
    return Owned(E);

  CXXTemporary *Temp = CXXTemporary::Create(Context, LookupDestructor(RD));
  ExprTemporaries.push_back(Temp);
  if (CXXDestructorDecl *Destructor = LookupDestructor(RD)) {
    MarkDeclarationReferenced(E->getExprLoc(), Destructor);
    CheckDestructorAccess(E->getExprLoc(), Destructor,
                          PDiag(diag::err_access_dtor_temp)
                            << E->getType());
  }
  // FIXME: Add the temporary to the temporaries vector.
  return Owned(CXXBindTemporaryExpr::Create(Context, Temp, E));
}

Expr *Sema::MaybeCreateExprWithCleanups(Expr *SubExpr) {
  assert(SubExpr && "sub expression can't be null!");

  unsigned FirstTemporary = ExprEvalContexts.back().NumTemporaries;
  assert(ExprTemporaries.size() >= FirstTemporary);
  if (ExprTemporaries.size() == FirstTemporary)
    return SubExpr;

  Expr *E = ExprWithCleanups::Create(Context, SubExpr,
                                     &ExprTemporaries[FirstTemporary],
                                     ExprTemporaries.size() - FirstTemporary);
  ExprTemporaries.erase(ExprTemporaries.begin() + FirstTemporary,
                        ExprTemporaries.end());

  return E;
}

ExprResult
Sema::MaybeCreateExprWithCleanups(ExprResult SubExpr) {
  if (SubExpr.isInvalid())
    return ExprError();

  return Owned(MaybeCreateExprWithCleanups(SubExpr.take()));
}

Stmt *Sema::MaybeCreateStmtWithCleanups(Stmt *SubStmt) {
  assert(SubStmt && "sub statement can't be null!");

  unsigned FirstTemporary = ExprEvalContexts.back().NumTemporaries;
  assert(ExprTemporaries.size() >= FirstTemporary);
  if (ExprTemporaries.size() == FirstTemporary)
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

ExprResult
Sema::ActOnStartCXXMemberReference(Scope *S, Expr *Base, SourceLocation OpLoc,
                                   tok::TokenKind OpKind, ParsedType &ObjectType,
                                   bool &MayBePseudoDestructor) {
  // Since this might be a postfix expression, get rid of ParenListExprs.
  ExprResult Result = MaybeConvertParenListExprToParenExpr(S, Base);
  if (Result.isInvalid()) return ExprError();
  Base = Result.get();

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
    llvm::SmallVector<SourceLocation, 8> Locations;
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

    if (BaseType->isPointerType())
      BaseType = BaseType->getPointeeType();
  }

  // We could end up with various non-record types here, such as extended
  // vector types or Objective-C interfaces. Just return early and let
  // ActOnMemberReferenceExpr do the work.
  if (!BaseType->isRecordType()) {
    // C++ [basic.lookup.classref]p2:
    //   [...] If the type of the object expression is of pointer to scalar
    //   type, the unqualified-id is looked up in the context of the complete
    //   postfix-expression.
    //
    // This also indicates that we should be parsing a
    // pseudo-destructor-name.
    ObjectType = ParsedType();
    MayBePseudoDestructor = true;
    return Owned(Base);
  }

  // The object type must be complete (or dependent).
  if (!BaseType->isDependentType() &&
      RequireCompleteType(OpLoc, BaseType,
                          PDiag(diag::err_incomplete_member_access)))
    return ExprError();

  // C++ [basic.lookup.classref]p2:
  //   If the id-expression in a class member access (5.2.5) is an
  //   unqualified-id, and the type of the object expression is of a class
  //   type C (or of pointer to a class type C), the unqualified-id is looked
  //   up in the scope of class C. [...]
  ObjectType = ParsedType::make(BaseType);
  return move(Base);
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

  // C++ [expr.pseudo]p2:
  //   The left-hand side of the dot operator shall be of scalar type. The
  //   left-hand side of the arrow operator shall be of pointer to scalar type.
  //   This scalar type is the object type.
  QualType ObjectType = Base->getType();
  if (OpKind == tok::arrow) {
    if (const PointerType *Ptr = ObjectType->getAs<PointerType>()) {
      ObjectType = Ptr->getPointeeType();
    } else if (!Base->isTypeDependent()) {
      // The user wrote "p->" when she probably meant "p."; fix it.
      Diag(OpLoc, diag::err_typecheck_member_reference_suggestion)
        << ObjectType << true
        << FixItHint::CreateReplacement(OpLoc, ".");
      if (isSFINAEContext())
        return ExprError();

      OpKind = tok::period;
    }
  }

  if (!ObjectType->isDependentType() && !ObjectType->isScalarType()) {
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
    if (!DestructedType->isDependentType() && !ObjectType->isDependentType() &&
        !Context.hasSameUnqualifiedType(DestructedType, ObjectType)) {
      Diag(DestructedTypeStart, diag::err_pseudo_dtor_type_mismatch)
        << ObjectType << DestructedType << Base->getSourceRange()
        << DestructedTypeInfo->getTypeLoc().getLocalSourceRange();

      // Recover by setting the destructed type to the object type.
      DestructedType = ObjectType;
      DestructedTypeInfo = Context.getTrivialTypeSourceInfo(ObjectType,
                                                           DestructedTypeStart);
      Destructed = PseudoDestructorTypeStorage(DestructedTypeInfo);
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

  // C++ [expr.pseudo]p2:
  //   The left-hand side of the dot operator shall be of scalar type. The
  //   left-hand side of the arrow operator shall be of pointer to scalar type.
  //   This scalar type is the object type.
  QualType ObjectType = Base->getType();
  if (OpKind == tok::arrow) {
    if (const PointerType *Ptr = ObjectType->getAs<PointerType>()) {
      ObjectType = Ptr->getPointeeType();
    } else if (!ObjectType->isDependentType()) {
      // The user wrote "p->" when she probably meant "p."; fix it.
      Diag(OpLoc, diag::err_typecheck_member_reference_suggestion)
        << ObjectType << true
        << FixItHint::CreateReplacement(OpLoc, ".");
      if (isSFINAEContext())
        return ExprError();

      OpKind = tok::period;
    }
  }

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
    ASTTemplateArgsPtr TemplateArgsPtr(*this,
                                       TemplateId->getTemplateArgs(),
                                       TemplateId->NumArgs);
    TypeResult T = ActOnTemplateIdType(TemplateId->SS,
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
      ASTTemplateArgsPtr TemplateArgsPtr(*this,
                                         TemplateId->getTemplateArgs(),
                                         TemplateId->NumArgs);
      TypeResult T = ActOnTemplateIdType(TemplateId->SS,
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

ExprResult Sema::BuildCXXMemberCallExpr(Expr *Exp, NamedDecl *FoundDecl,
                                        CXXMethodDecl *Method) {
  if (PerformObjectArgumentInitialization(Exp, /*Qualifier=*/0,
                                          FoundDecl, Method))
    return true;

  MemberExpr *ME =
      new (Context) MemberExpr(Exp, /*IsArrow=*/false, Method,
                               SourceLocation(), Method->getType(),
                               VK_RValue, OK_Ordinary);
  QualType ResultType = Method->getResultType();
  ExprValueKind VK = Expr::getValueKindForType(ResultType);
  ResultType = ResultType.getNonLValueExprType(Context);

  MarkDeclarationReferenced(Exp->getLocStart(), Method);
  CXXMemberCallExpr *CE =
    new (Context) CXXMemberCallExpr(Context, ME, 0, 0, ResultType, VK,
                                    Exp->getLocEnd());
  return CE;
}

ExprResult Sema::BuildCXXNoexceptExpr(SourceLocation KeyLoc, Expr *Operand,
                                      SourceLocation RParen) {
  return Owned(new (Context) CXXNoexceptExpr(Context.BoolTy, Operand,
                                             Operand->CanThrow(Context),
                                             KeyLoc, RParen));
}

ExprResult Sema::ActOnNoexceptExpr(SourceLocation KeyLoc, SourceLocation,
                                   Expr *Operand, SourceLocation RParen) {
  return BuildCXXNoexceptExpr(KeyLoc, Operand, RParen);
}

/// Perform the conversions required for an expression used in a
/// context that ignores the result.
void Sema::IgnoredValueConversions(Expr *&E) {
  // C99 6.3.2.1:
  //   [Except in specific positions,] an lvalue that does not have
  //   array type is converted to the value stored in the
  //   designated object (and is no longer an lvalue).
  if (E->isRValue()) return;

  // We always want to do this on ObjC property references.
  if (E->getObjectKind() == OK_ObjCProperty) {
    ConvertPropertyForRValue(E);
    if (E->isRValue()) return;
  }

  // Otherwise, this rule does not apply in C++, at least not for the moment.
  if (getLangOptions().CPlusPlus) return;

  // GCC seems to also exclude expressions of incomplete enum type.
  if (const EnumType *T = E->getType()->getAs<EnumType>()) {
    if (!T->getDecl()->isComplete()) {
      // FIXME: stupid workaround for a codegen bug!
      ImpCastExprToType(E, Context.VoidTy, CK_ToVoid);
      return;
    }
  }

  DefaultFunctionArrayLvalueConversion(E);
  if (!E->getType()->isVoidType())
    RequireCompleteType(E->getExprLoc(), E->getType(),
                        diag::err_incomplete_type);
}

ExprResult Sema::ActOnFinishFullExpr(Expr *FullExpr) {
  if (!FullExpr)
    return ExprError();

  if (DiagnoseUnexpandedParameterPack(FullExpr))
    return ExprError();

  // 13.4.1 ... An overloaded function name shall not be used without arguments 
  //         in contexts other than those listed [i.e list of targets].
  //  
  //  void foo(); void foo(int);
  //  template<class T> void fooT(); template<class T> void fooT(int);
  
  //  Therefore these should error:
  //  foo; 
  //  fooT<int>;
  
  if (FullExpr->getType() == Context.OverloadTy) {
    if (!ResolveSingleFunctionTemplateSpecialization(FullExpr, 
                                                     /* Complain */ false)) {
      OverloadExpr* OvlExpr = OverloadExpr::find(FullExpr).Expression; 
      Diag(FullExpr->getLocStart(), diag::err_addr_ovl_ambiguous)
        << OvlExpr->getName();
      NoteAllOverloadCandidates(OvlExpr);
      return ExprError();
    }  
  }


  IgnoredValueConversions(FullExpr);
  CheckImplicitConversions(FullExpr);
  
  return MaybeCreateExprWithCleanups(FullExpr);
}

StmtResult Sema::ActOnFinishFullStmt(Stmt *FullStmt) {
  if (!FullStmt) return StmtError();

  return MaybeCreateStmtWithCleanups(FullStmt);
}

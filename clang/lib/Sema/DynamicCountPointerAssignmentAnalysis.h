//===--- DynamicCountPointerAssignmentAnalysis.h --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements interface of dynamic count pointer assignment analysis
//  for -fbounds-safety.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_DYNAMIC_COUNT_POINTER_ASSIGNMENT_ANALYSIS_H
#define LLVM_CLANG_SEMA_DYNAMIC_COUNT_POINTER_ASSIGNMENT_ANALYSIS_H

#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/SaveAndRestore.h"

namespace clang {

class DynamicCountPointerAssignmentAnalysis {
  Sema &SemaRef;
  Decl *dcl;
public:
  DynamicCountPointerAssignmentAnalysis(Sema &SemaRef, Decl *dcl)
      : SemaRef(SemaRef), dcl(dcl) {}

  void run();

  static RecordDecl *computeFlexBaseKey(Expr *InE,
                                        llvm::raw_string_ostream *OS);
};

/// CopyExpr simply duplicates expressions up to OpaqueValueExpr, which are
/// used as-is.
class CopyExpr : public TreeTransform<CopyExpr> {
  /// List of declarations that should be substituted for a given expression.
  /// This is meant to assist with bounds checking on CountAttributedType
  /// by allowing users to replace the reference declaration with the right
  /// expression.
  llvm::SmallVector<std::pair<ValueDecl *, Expr *>, 1> Replacements;
  bool InSyntacticTransformation = false;

  ValueDecl *SelectFirstDecl(ValueDecl *Decl) {
    if (auto *VD = dyn_cast<VarDecl>(Decl)) {
      Decl = VD->getFirstDecl();
    } else if (auto *FD = dyn_cast<FunctionDecl>(Decl)) {
      Decl = FD->getFirstDecl();
    }
    return Decl;
  }
public:
  using TreeTransform::TreeTransform;

  bool AlwaysRebuild() { return true; }

  void AddDeclSubstitution(ValueDecl *DeclToChange,
                           OpaqueValueExpr *ValueToUse) {
    UnsafelyAddDeclSubstitution(DeclToChange, ValueToUse);
  }

  // UnsafelyAddDeclSubstitution lets you replace declaration references with
  // any expression that you like better, including expressions that would
  // unsafely be used in multiple places in the AST. Prefer AddDeclSubstitution
  // unless you expect the result to not actually be used in the final AST.
  void UnsafelyAddDeclSubstitution(ValueDecl *DeclToChange, Expr *ValueToUse) {
    Replacements.emplace_back(SelectFirstDecl(DeclToChange), ValueToUse);
  }

  ExprResult TransformOpaqueValueExpr(OpaqueValueExpr *E) {
    if (RemovedOVEs.count(E))
      return TreeTransform::TransformExpr(E->getSourceExpr());
    // Don't copy expressions inside opaque values.
    return E;
  }

  bool HasDeclReplacement(ValueDecl *VD) {
    for (auto &Pair : Replacements) {
      if (Pair.first == VD) {
        return true;
      }
    }
    return false;
  }

  ExprResult RebuildDeclRefExpr(NestedNameSpecifierLoc QualifierLoc,
                                ValueDecl *VD,
                                const DeclarationNameInfo &NameInfo,
                                NamedDecl *Found,
                                TemplateArgumentListInfo *TemplateArgs) {
    VD = SelectFirstDecl(VD);
    for (auto &Pair : Replacements) {
      if (Pair.first == VD) {
        // Rebuild the replacement expression. To avoid recursion, we
        // use a new instance of CopyExpr.
        return CopyExpr(getSema()).TransformExpr(Pair.second);
      }
    }

    // TreeTransform::RebuildDeclRefExpr below does semantic analysis, and
    // well-formed programs cannot have references to fields in C. However, in
    // -fbounds-safety we can define a struct:
    //   struct {
    //     void *p1;
    //     void *__ended_by(p1) p2;
    //     void *__ended_by(p2-1) p3;
    //   };
    // p2 in p2-1 will be bounds-safety-promoted to __bidi_indexable during
    // LValueToRValue. This will try to create copies of DeclRefExprs to p1 and
    // p2, which would trigger an assert "building reference to field in C?' in
    // TreeTransform::RebuildDeclRefExpr. To avoid this, we build the
    // DeclRefExprs here manually.
    if (isa<FieldDecl>(VD)) {
      // Copy-paste from Sema::BuildDeclarationNameExpr, but without assert.
      QualType Ty = VD->getType().getNonReferenceType();
      ExprValueKind VK = VK_LValue;
      CXXScopeSpec SS;
      SS.Adopt(QualifierLoc);
      return getSema().BuildDeclRefExpr(VD, Ty, VK, NameInfo, &SS, Found,
                                        /*TemplateKWLoc=*/SourceLocation(),
                                        TemplateArgs);
    }

    return TreeTransform::RebuildDeclRefExpr(QualifierLoc, VD, NameInfo, Found,
                                             TemplateArgs);
  }

  /// Super class returns the syntactic form only. Override it to ensure that
  /// TransformMemberExpr can access the semantic form.
  ExprResult TransformInitListExpr(InitListExpr *E) {
    if (InSyntacticTransformation)
      return TreeTransform::TransformInitListExpr(E);
    assert(E->isSemanticForm());
    bool InitChanged = false;

    EnterExpressionEvaluationContext Context(
        getSema(), EnterExpressionEvaluationContext::InitList);

    SmallVector<Expr *, 4> Inits;
    if (TransformExprs(E->getInits(), E->getNumInits(), false, Inits,
                       &InitChanged))
      return ExprError();

    ExprResult Res =
        RebuildInitList(E->getLBraceLoc(), Inits, E->getRBraceLoc());
    if (Res.isUsable()) {
      SaveAndRestore SAR(InSyntacticTransformation, true);
      ExprResult SyntacticRes = TreeTransform::TransformInitListExpr(E);
      if (SyntacticRes.isUsable()) {
        auto SyntacticIL = cast<InitListExpr>(SyntacticRes.get());
        assert(SyntacticIL->isSyntacticForm());
        cast<InitListExpr>(Res.get())->setSyntacticForm(SyntacticIL);
      }
    }
    return Res;
  }

  ExprResult RebuildMemberExpr(Expr *Base, SourceLocation OpLoc,
                                bool isArrow,
                                NestedNameSpecifierLoc QualifierLoc,
                                SourceLocation TemplateKWLoc,
                                const DeclarationNameInfo &MemberNameInfo,
                                ValueDecl *Member,
                                NamedDecl *FoundDecl,
                          const TemplateArgumentListInfo *ExplicitTemplateArgs,
                                NamedDecl *FirstQualifierInScope) {
    if (auto IL = dyn_cast<InitListExpr>(Base);
        IL && !InSyntacticTransformation) {
      // We need field initializers to align with the struct type.
      assert(IL->isSemanticForm());
      auto *VD = cast<ValueDecl>(Member);
      Expr *Val = IL->getInitForField(VD);
      assert(Val && !isa<DesignatedInitExpr>(Val));
      return Val;
    }

    // XXX: this assumes that you'll only see the field accessed on the
    // specific (and implicit) object you're interested in. It's not a problem
    // with the current user-facing rules, which prohibit count expressions
    // from referencing a field off of any other object, but if the rules
    // change or the compiler starts synthesizing expressions that do so, this
    // would need to change in difficult ways.
    for (auto &Pair : Replacements) {
      if (Pair.first == Member) {
        // Rebuild the replacement expression. To avoid recursion, we
        // use a new instance of CopyExpr.
        return CopyExpr(getSema()).TransformExpr(Pair.second);
      }
    }
    return TreeTransform::RebuildMemberExpr(
        Base, OpLoc, isArrow, QualifierLoc, TemplateKWLoc, MemberNameInfo,
        Member, FoundDecl, ExplicitTemplateArgs, FirstQualifierInScope);
  }
};

class ForceRebuild : public TreeTransform<ForceRebuild> {
public:
  using BaseTransform = TreeTransform<ForceRebuild>;
  bool AlwaysRebuild() { return true; }
  ForceRebuild(Sema &SemaRef) : TreeTransform<ForceRebuild>(SemaRef) {}
  ExprResult TransformOpaqueValueExpr(OpaqueValueExpr *E) {
    if (RemovedOVEs.count(E))
      return TransformExpr(E->getSourceExpr());
    return E;
  }

  // Workaround upstream behavior of TreeTransform.
  //
  // Normally rebuilding `CompoundLiteralExpr` runs
  // `getSema().BuildCompoundLiteralExpr` which reruns Sema checks. That
  // function checks `CurContext->isFunctionOrMethod();` which won't necessarily
  // get the Context this CompoundLiteralExpr was originally built in which
  // means we can emit spurious `diag::err_init_element_not_constant`
  // diagnostics. To workaround this we construct the expression directly using
  // the information on `CompoundLiteralExpr::isFileScope()`.
  //
  // FIXME: Is this a bug upstream or intentional behavior?
  ExprResult RebuildCompoundLiteralExpr(SourceLocation LParenLoc,
                                        TypeSourceInfo *TInfo,
                                        SourceLocation RParenLoc, Expr *Init,
                                        bool IsFileScope) {

    QualType literalType = TInfo->getType();
    // Copied from Sema::BuildCompoundLiteralExpr
    ExprValueKind VK = (getSema().getLangOpts().CPlusPlus &&
                        !(IsFileScope && literalType->isArrayType()))
                           ? VK_PRValue
                           : VK_LValue;

    auto *E = new (getSema().getASTContext()) CompoundLiteralExpr(
        LParenLoc, TInfo, TInfo->getType(), VK, Init, IsFileScope);
    return E;
  }
};

/// Create a new expression in terms of pre-registered RHS of assignments by
/// replacing a DeclRef or MemberExpr with corresponding RHS.
///
/// We are performing checks before any of the assignments are done. This means,
/// we can't simply load the updated \c f->cnt to perform count checks like this
/// \f[0 <= f->cnt + 2 <= bounds_of(new_ptr)\f] since we only have the old \c
/// f->cnt. Therefore, we create a new count expression in terms of \c new_cnt
/// by replacing \c f->cnt, which is \c new_cnt + 2 in the example.
///
/// \code
///  struct Foo { int *__counted_by(cnt + 2) ptr; int cnt; };
///  void Test(struct Foo *f) {
///    f->ptr = new_ptr;
///    f->cnt = new_cnt;
/// \endcode
struct ReplaceDeclRefWithRHS : public TreeTransform<ReplaceDeclRefWithRHS> {
  using BaseTransform = TreeTransform<ReplaceDeclRefWithRHS>;
  using ExprLevelPair = std::pair<Expr *, unsigned>;
  using DeclToNewValueTy = llvm::DenseMap<const ValueDecl *, ExprLevelPair>;

  llvm::SmallPtrSet<const Expr *, 2> ReplacingValues;
  unsigned LevelToUnwrap = 0;
  DeclToNewValueTy &DeclToNewValue;
  Expr *MemberBase = nullptr;

  explicit ReplaceDeclRefWithRHS(Sema &SemaRef,
                                 DeclToNewValueTy &DeclToNewValue)
      : BaseTransform(SemaRef), DeclToNewValue(DeclToNewValue) {}

  bool AlwaysRebuild() { return true; }

  ExprResult TransformOpaqueValueExpr(OpaqueValueExpr *E) { return Owned(E); }

  /// If we are replacing the value pointed to by DeclRef, we unwrap dereference
  /// operator(s) from the original bound expression.
  ///
  /// In the following example,  we replace \c *out_cnt the argument of \c
  /// __counted_by with \c new_cnt, not just \c out_cnt.
  ///
  /// \code
  /// void foo(int *__counted_by(*out_cnt) *out_buf, int *out_cnt) {
  ///   *out_buf = new_buf;
  ///   *out_cnt = new_cnt;
  ///  }
  /// \endcode
  ExprResult TransformUnaryOperator(UnaryOperator *E) {
    if (E->getOpcode() != UO_Deref)
      return BaseTransform::TransformUnaryOperator(E);

    ExprResult SubExpr = TransformExpr(E->getSubExpr());
    if (SubExpr.isInvalid())
      return ExprError();

    if (LevelToUnwrap != 0) {
      --LevelToUnwrap;
      return SubExpr;
    }

    return RebuildUnaryOperator(E->getOperatorLoc(), E->getOpcode(),
                                SubExpr.get());
  }

  ExprResult TransformDeclRefExpr(DeclRefExpr *E) {
    auto It = DeclToNewValue.find(E->getDecl());
    if (It == DeclToNewValue.end()) {
      if (auto *Field = dyn_cast<FieldDecl>(E->getDecl())) {
        assert(MemberBase);
        bool IsArrow = MemberBase->getType()->isPointerType();
        return MemberExpr::CreateImplicit(getSema().Context, MemberBase,
                                          IsArrow, Field, Field->getType(),
                                          VK_LValue, OK_Ordinary);
      }
      return BaseTransform::TransformDeclRefExpr(E);
    }
    // Clone the new value.
    return Replace(It->second.first, It->second.second);
  }

  ExprResult TransformMemberExpr(MemberExpr *E) {
    auto It = DeclToNewValue.find(E->getMemberDecl());
    if (It == DeclToNewValue.end())
      return BaseTransform::TransformMemberExpr(E);
    return Replace(It->second.first, It->second.second);
  }

  ExprResult Replace(Expr *New, unsigned Level) {
    ExprResult Repl = ForceRebuild(SemaRef).TransformExpr(New);

    if (const Expr *E = Repl.get())
      ReplacingValues.insert(E);

    assert(LevelToUnwrap == 0);
    LevelToUnwrap = Level;
    return Repl;
  }

  const llvm::SmallPtrSetImpl<const Expr *> &GetReplacingValues() const {
    return ReplacingValues;
  }
};

class FlexibleArrayMemberUtils {
  Sema &SemaRef;

public:
  FlexibleArrayMemberUtils(Sema &SemaRef) : SemaRef(SemaRef)
  { }

  /// If RD is not a union and has a flexible array member with a dynamic count,
  /// return true and fill PathToFlex with every field decl towards the flexible
  /// field, and CountDecls with the list of declarations involved in the count
  /// expression. Return false otherwise.
  bool Find(RecordDecl *RD, llvm::SmallVectorImpl<FieldDecl *> &PathToFlex,
            ArrayRef<TypeCoupledDeclRefInfo> &CountDecls);

  RecordDecl *GetFlexibleRecord(QualType QT);

  Expr *SelectFlexibleObject(
      const llvm::SmallVectorImpl<FieldDecl *> &PathToFlex, Expr *Base);

  ExprResult BuildCountExpr(FieldDecl *FlexibleField,
                            const ArrayRef<TypeCoupledDeclRefInfo> CountDecls,
                            Expr *Base,
                            llvm::SmallVectorImpl<OpaqueValueExpr *> &OVEs,
                            CopyExpr *Copy = nullptr);
};

/// Function return type or argument may have a '__counted_by()' attribute with
/// the count expression expressed with ParmVarDecl. Similarly to instantiating
/// count expression of struct fields expressed by FieldDecl, we replace
/// DeclRefExpr of ParmVarDecl with the actual argument of the function call.
class TransformDynamicCountWithFunctionArgument
    : public TreeTransform<TransformDynamicCountWithFunctionArgument> {
  typedef TreeTransform<TransformDynamicCountWithFunctionArgument>
      BaseTransform;
  const SmallVectorImpl<Expr *> &ActualArgs;
  unsigned FirstParam = 0;

public:
  TransformDynamicCountWithFunctionArgument(Sema &SemaRef,
                                            const SmallVectorImpl<Expr *> &Args,
                                            unsigned FirstParam = 0);

  ExprResult TransformDeclRefExpr(DeclRefExpr *E);
};

} // end namespace clang

#endif // LLVM_CLANG_SEMA_DYNAMIC_COUNT_POINTER_ASSIGNMENT_ANALYSIS_H

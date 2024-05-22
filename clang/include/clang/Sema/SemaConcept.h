//===-- SemaConcept.h - Semantic Analysis for Constraints and Concepts ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//  This file provides semantic analysis for C++ constraints and concepts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMACONCEPT_H
#define LLVM_CLANG_SEMA_SEMACONCEPT_H
#include "clang/AST/ASTConcept.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <utility>

namespace clang {

struct AtomicConstraint {
  const Expr *ConstraintExpr;
  std::optional<ArrayRef<TemplateArgumentLoc>> ParameterMapping;

  AtomicConstraint(Sema &S, const Expr *ConstraintExpr) :
      ConstraintExpr(ConstraintExpr) { };

  bool hasMatchingParameterMapping(ASTContext &C,
                                   const AtomicConstraint &Other) const {
    if (!ParameterMapping != !Other.ParameterMapping)
      return false;
    if (!ParameterMapping)
      return true;
    if (ParameterMapping->size() != Other.ParameterMapping->size())
      return false;

    for (unsigned I = 0, S = ParameterMapping->size(); I < S; ++I) {
      llvm::FoldingSetNodeID IDA, IDB;
      C.getCanonicalTemplateArgument((*ParameterMapping)[I].getArgument())
          .Profile(IDA, C);
      C.getCanonicalTemplateArgument((*Other.ParameterMapping)[I].getArgument())
          .Profile(IDB, C);
      if (IDA != IDB)
        return false;
    }
    return true;
  }

  bool subsumes(ASTContext &C, const AtomicConstraint &Other) const {
    // C++ [temp.constr.order] p2
    //   - an atomic constraint A subsumes another atomic constraint B
    //     if and only if the A and B are identical [...]
    //
    // C++ [temp.constr.atomic] p2
    //   Two atomic constraints are identical if they are formed from the
    //   same expression and the targets of the parameter mappings are
    //   equivalent according to the rules for expressions [...]

    // We do not actually substitute the parameter mappings into the
    // constraint expressions, therefore the constraint expressions are
    // the originals, and comparing them will suffice.
    if (ConstraintExpr != Other.ConstraintExpr)
      return false;

    // Check that the parameter lists are identical
    return hasMatchingParameterMapping(C, Other);
  }
};

/// \brief A normalized constraint, as defined in C++ [temp.constr.normal], is
/// either an atomic constraint, a conjunction of normalized constraints or a
/// disjunction of normalized constraints.
struct NormalizedConstraint {
  friend class Sema;
  friend class SemaConcept;

  enum CompoundConstraintKind { CCK_Conjunction, CCK_Disjunction };

  using CompoundConstraint = llvm::PointerIntPair<
      std::pair<NormalizedConstraint, NormalizedConstraint> *, 1,
      CompoundConstraintKind>;

  llvm::PointerUnion<AtomicConstraint *, CompoundConstraint> Constraint;

  NormalizedConstraint(AtomicConstraint *C): Constraint{C} { };
  NormalizedConstraint(ASTContext &C, NormalizedConstraint LHS,
                       NormalizedConstraint RHS, CompoundConstraintKind Kind)
      : Constraint{CompoundConstraint{
            new (C) std::pair<NormalizedConstraint, NormalizedConstraint>{
                std::move(LHS), std::move(RHS)}, Kind}} { };

  NormalizedConstraint(ASTContext &C, const NormalizedConstraint &Other) {
    if (Other.isAtomic()) {
      Constraint = new (C) AtomicConstraint(*Other.getAtomicConstraint());
    } else {
      Constraint = CompoundConstraint(
          new (C) std::pair<NormalizedConstraint, NormalizedConstraint>{
              NormalizedConstraint(C, Other.getLHS()),
              NormalizedConstraint(C, Other.getRHS())},
              Other.getCompoundKind());
    }
  }
  NormalizedConstraint(NormalizedConstraint &&Other):
      Constraint(Other.Constraint) {
    Other.Constraint = nullptr;
  }
  NormalizedConstraint &operator=(const NormalizedConstraint &Other) = delete;
  NormalizedConstraint &operator=(NormalizedConstraint &&Other) {
    if (&Other != this) {
      NormalizedConstraint Temp(std::move(Other));
      std::swap(Constraint, Temp.Constraint);
    }
    return *this;
  }

  CompoundConstraintKind getCompoundKind() const {
    assert(!isAtomic() && "getCompoundKind called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getInt();
  }

  bool isAtomic() const { return Constraint.is<AtomicConstraint *>(); }

  NormalizedConstraint &getLHS() const {
    assert(!isAtomic() && "getLHS called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getPointer()->first;
  }

  NormalizedConstraint &getRHS() const {
    assert(!isAtomic() && "getRHS called on atomic constraint.");
    return Constraint.get<CompoundConstraint>().getPointer()->second;
  }

  AtomicConstraint *getAtomicConstraint() const {
    assert(isAtomic() &&
           "getAtomicConstraint called on non-atomic constraint.");
    return Constraint.get<AtomicConstraint *>();
  }

private:
  static std::optional<NormalizedConstraint>
  fromConstraintExprs(Sema &S, NamedDecl *D, ArrayRef<const Expr *> E);
  static std::optional<NormalizedConstraint>
  fromConstraintExpr(Sema &S, NamedDecl *D, const Expr *E);
};

class LocalInstantiationScope;
class LookupResult;
class MultiLevelTemplateArgumentList;

// A struct to represent the 'new' declaration, which is either itself just
// the named decl, or the important information we need about it in order to
// do constraint comparisons.
class TemplateCompareNewDeclInfo {
  const NamedDecl *ND = nullptr;
  const DeclContext *DC = nullptr;
  const DeclContext *LexicalDC = nullptr;
  SourceLocation Loc;

public:
  TemplateCompareNewDeclInfo(const NamedDecl *ND) : ND(ND) {}
  TemplateCompareNewDeclInfo(const DeclContext *DeclCtx,
                             const DeclContext *LexicalDeclCtx,
                             SourceLocation Loc)

      : DC(DeclCtx), LexicalDC(LexicalDeclCtx), Loc(Loc) {
    assert(DC && LexicalDC &&
           "Constructor only for cases where we have the information to put "
           "in here");
  }

  // If this was constructed with no information, we cannot do substitution
  // for constraint comparison, so make sure we can check that.
  bool isInvalid() const { return !ND && !DC; }

  const NamedDecl *getDecl() const { return ND; }

  bool ContainsDecl(const NamedDecl *ND) const { return this->ND == ND; }

  const DeclContext *getLexicalDeclContext() const {
    return ND ? ND->getLexicalDeclContext() : LexicalDC;
  }

  const DeclContext *getDeclContext() const {
    return ND ? ND->getDeclContext() : DC;
  }

  SourceLocation getLocation() const { return ND ? ND->getLocation() : Loc; }
};

class SemaConcept : public SemaBase {
public:
  SemaConcept(Sema &S);
  ~SemaConcept();

  void PushSatisfactionStackEntry(const NamedDecl *D,
                                  const llvm::FoldingSetNodeID &ID) {
    const NamedDecl *Can = cast<NamedDecl>(D->getCanonicalDecl());
    SatisfactionStack.emplace_back(Can, ID);
  }

  void PopSatisfactionStackEntry() { SatisfactionStack.pop_back(); }

  bool SatisfactionStackContains(const NamedDecl *D,
                                 const llvm::FoldingSetNodeID &ID) const {
    const NamedDecl *Can = cast<NamedDecl>(D->getCanonicalDecl());
    return llvm::find(SatisfactionStack, SatisfactionStackEntryTy{Can, ID}) !=
           SatisfactionStack.end();
  }

  using SatisfactionStackEntryTy =
      std::pair<const NamedDecl *, llvm::FoldingSetNodeID>;

  // Resets the current SatisfactionStack for cases where we are instantiating
  // constraints as a 'side effect' of normal instantiation in a way that is not
  // indicative of recursive definition.
  class SatisfactionStackResetRAII {
    llvm::SmallVector<SatisfactionStackEntryTy, 10> BackupSatisfactionStack;
    Sema &SemaRef;

  public:
    SatisfactionStackResetRAII(Sema &S);
    ~SatisfactionStackResetRAII();
  };

  void SwapSatisfactionStack(
      llvm::SmallVectorImpl<SatisfactionStackEntryTy> &NewSS) {
    SatisfactionStack.swap(NewSS);
  }

  /// Check whether the given expression is a valid constraint expression.
  /// A diagnostic is emitted if it is not, false is returned, and
  /// PossibleNonPrimary will be set to true if the failure might be due to a
  /// non-primary expression being used as an atomic constraint.
  bool CheckConstraintExpression(const Expr *CE, Token NextToken = Token(),
                                 bool *PossibleNonPrimary = nullptr,
                                 bool IsTrailingRequiresClause = false);

  /// \brief Check whether the given list of constraint expressions are
  /// satisfied (as if in a 'conjunction') given template arguments.
  /// \param Template the template-like entity that triggered the constraints
  /// check (either a concept or a constrained entity).
  /// \param ConstraintExprs a list of constraint expressions, treated as if
  /// they were 'AND'ed together.
  /// \param TemplateArgLists the list of template arguments to substitute into
  /// the constraint expression.
  /// \param TemplateIDRange The source range of the template id that
  /// caused the constraints check.
  /// \param Satisfaction if true is returned, will contain details of the
  /// satisfaction, with enough information to diagnose an unsatisfied
  /// expression.
  /// \returns true if an error occurred and satisfaction could not be checked,
  /// false otherwise.
  bool CheckConstraintSatisfaction(
      const NamedDecl *Template, ArrayRef<const Expr *> ConstraintExprs,
      const MultiLevelTemplateArgumentList &TemplateArgLists,
      SourceRange TemplateIDRange, ConstraintSatisfaction &Satisfaction) {
    llvm::SmallVector<Expr *, 4> Converted;
    return CheckConstraintSatisfaction(Template, ConstraintExprs, Converted,
                                       TemplateArgLists, TemplateIDRange,
                                       Satisfaction);
  }

  /// \brief Check whether the given list of constraint expressions are
  /// satisfied (as if in a 'conjunction') given template arguments.
  /// Additionally, takes an empty list of Expressions which is populated with
  /// the instantiated versions of the ConstraintExprs.
  /// \param Template the template-like entity that triggered the constraints
  /// check (either a concept or a constrained entity).
  /// \param ConstraintExprs a list of constraint expressions, treated as if
  /// they were 'AND'ed together.
  /// \param ConvertedConstraints a out parameter that will get populated with
  /// the instantiated version of the ConstraintExprs if we successfully checked
  /// satisfaction.
  /// \param TemplateArgList the multi-level list of template arguments to
  /// substitute into the constraint expression. This should be relative to the
  /// top-level (hence multi-level), since we need to instantiate fully at the
  /// time of checking.
  /// \param TemplateIDRange The source range of the template id that
  /// caused the constraints check.
  /// \param Satisfaction if true is returned, will contain details of the
  /// satisfaction, with enough information to diagnose an unsatisfied
  /// expression.
  /// \returns true if an error occurred and satisfaction could not be checked,
  /// false otherwise.
  bool CheckConstraintSatisfaction(
      const NamedDecl *Template, ArrayRef<const Expr *> ConstraintExprs,
      llvm::SmallVectorImpl<Expr *> &ConvertedConstraints,
      const MultiLevelTemplateArgumentList &TemplateArgList,
      SourceRange TemplateIDRange, ConstraintSatisfaction &Satisfaction);

  /// \brief Check whether the given non-dependent constraint expression is
  /// satisfied. Returns false and updates Satisfaction with the satisfaction
  /// verdict if successful, emits a diagnostic and returns true if an error
  /// occurred and satisfaction could not be determined.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool CheckConstraintSatisfaction(const Expr *ConstraintExpr,
                                   ConstraintSatisfaction &Satisfaction);

  /// Check whether the given function decl's trailing requires clause is
  /// satisfied, if any. Returns false and updates Satisfaction with the
  /// satisfaction verdict if successful, emits a diagnostic and returns true if
  /// an error occurred and satisfaction could not be determined.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool CheckFunctionConstraints(const FunctionDecl *FD,
                                ConstraintSatisfaction &Satisfaction,
                                SourceLocation UsageLoc = SourceLocation(),
                                bool ForOverloadResolution = false);

  // Calculates whether two constraint expressions are equal irrespective of a
  // difference in 'depth'. This takes a pair of optional 'NamedDecl's 'Old' and
  // 'New', which are the "source" of the constraint, since this is necessary
  // for figuring out the relative 'depth' of the constraint. The depth of the
  // 'primary template' and the 'instantiated from' templates aren't necessarily
  // the same, such as a case when one is a 'friend' defined in a class.
  bool AreConstraintExpressionsEqual(const NamedDecl *Old,
                                     const Expr *OldConstr,
                                     const TemplateCompareNewDeclInfo &New,
                                     const Expr *NewConstr);

  // Calculates whether the friend function depends on an enclosing template for
  // the purposes of [temp.friend] p9.
  bool FriendConstraintsDependOnEnclosingTemplate(const FunctionDecl *FD);

  /// \brief Ensure that the given template arguments satisfy the constraints
  /// associated with the given template, emitting a diagnostic if they do not.
  ///
  /// \param Template The template to which the template arguments are being
  /// provided.
  ///
  /// \param TemplateArgs The converted, canonicalized template arguments.
  ///
  /// \param TemplateIDRange The source range of the template id that
  /// caused the constraints check.
  ///
  /// \returns true if the constrains are not satisfied or could not be checked
  /// for satisfaction, false if the constraints are satisfied.
  bool EnsureTemplateArgumentListConstraints(
      TemplateDecl *Template,
      const MultiLevelTemplateArgumentList &TemplateArgs,
      SourceRange TemplateIDRange);

  bool CheckInstantiatedFunctionTemplateConstraints(
      SourceLocation PointOfInstantiation, FunctionDecl *Decl,
      ArrayRef<TemplateArgument> TemplateArgs,
      ConstraintSatisfaction &Satisfaction);

  /// \brief Emit diagnostics explaining why a constraint expression was deemed
  /// unsatisfied.
  /// \param First whether this is the first time an unsatisfied constraint is
  /// diagnosed for this error.
  void DiagnoseUnsatisfiedConstraint(const ConstraintSatisfaction &Satisfaction,
                                     bool First = true);

  /// \brief Emit diagnostics explaining why a constraint expression was deemed
  /// unsatisfied.
  void
  DiagnoseUnsatisfiedConstraint(const ASTConstraintSatisfaction &Satisfaction,
                                bool First = true);

  const NormalizedConstraint *getNormalizedAssociatedConstraints(
      NamedDecl *ConstrainedDecl, ArrayRef<const Expr *> AssociatedConstraints);

  /// \brief Check whether the given declaration's associated constraints are
  /// at least as constrained than another declaration's according to the
  /// partial ordering of constraints.
  ///
  /// \param Result If no error occurred, receives the result of true if D1 is
  /// at least constrained than D2, and false otherwise.
  ///
  /// \returns true if an error occurred, false otherwise.
  bool IsAtLeastAsConstrained(NamedDecl *D1, MutableArrayRef<const Expr *> AC1,
                              NamedDecl *D2, MutableArrayRef<const Expr *> AC2,
                              bool &Result);

  /// If D1 was not at least as constrained as D2, but would've been if a pair
  /// of atomic constraints involved had been declared in a concept and not
  /// repeated in two separate places in code.
  /// \returns true if such a diagnostic was emitted, false otherwise.
  bool MaybeEmitAmbiguousAtomicConstraintsDiagnostic(
      NamedDecl *D1, ArrayRef<const Expr *> AC1, NamedDecl *D2,
      ArrayRef<const Expr *> AC2);

  /// Used by SetupConstraintCheckingTemplateArgumentsAndScope to recursively(in
  /// the case of lambdas) set up the LocalInstantiationScope of the current
  /// function.
  bool
  SetupConstraintScope(FunctionDecl *FD,
                       std::optional<ArrayRef<TemplateArgument>> TemplateArgs,
                       const MultiLevelTemplateArgumentList &MLTAL,
                       LocalInstantiationScope &Scope);

  RequiresExprBodyDecl *
  ActOnStartRequiresExpr(SourceLocation RequiresKWLoc,
                         ArrayRef<ParmVarDecl *> LocalParameters,
                         Scope *BodyScope);
  void ActOnFinishRequiresExpr();
  concepts::Requirement *ActOnSimpleRequirement(Expr *E);
  concepts::Requirement *ActOnTypeRequirement(SourceLocation TypenameKWLoc,
                                              CXXScopeSpec &SS,
                                              SourceLocation NameLoc,
                                              const IdentifierInfo *TypeName,
                                              TemplateIdAnnotation *TemplateId);
  concepts::Requirement *ActOnCompoundRequirement(Expr *E,
                                                  SourceLocation NoexceptLoc);
  concepts::Requirement *ActOnCompoundRequirement(
      Expr *E, SourceLocation NoexceptLoc, CXXScopeSpec &SS,
      TemplateIdAnnotation *TypeConstraint, unsigned Depth);
  concepts::Requirement *ActOnNestedRequirement(Expr *Constraint);
  concepts::ExprRequirement *BuildExprRequirement(
      Expr *E, bool IsSatisfied, SourceLocation NoexceptLoc,
      concepts::ExprRequirement::ReturnTypeRequirement ReturnTypeRequirement);
  concepts::ExprRequirement *BuildExprRequirement(
      concepts::Requirement::SubstitutionDiagnostic *ExprSubstDiag,
      bool IsSatisfied, SourceLocation NoexceptLoc,
      concepts::ExprRequirement::ReturnTypeRequirement ReturnTypeRequirement);
  concepts::TypeRequirement *BuildTypeRequirement(TypeSourceInfo *Type);
  concepts::TypeRequirement *BuildTypeRequirement(
      concepts::Requirement::SubstitutionDiagnostic *SubstDiag);
  concepts::NestedRequirement *BuildNestedRequirement(Expr *E);
  concepts::NestedRequirement *
  BuildNestedRequirement(StringRef InvalidConstraintEntity,
                         const ASTConstraintSatisfaction &Satisfaction);
  ExprResult ActOnRequiresExpr(SourceLocation RequiresKWLoc,
                               RequiresExprBodyDecl *Body,
                               SourceLocation LParenLoc,
                               ArrayRef<ParmVarDecl *> LocalParameters,
                               SourceLocation RParenLoc,
                               ArrayRef<concepts::Requirement *> Requirements,
                               SourceLocation ClosingBraceLoc);

  bool CheckTypeConstraint(TemplateIdAnnotation *TypeConstraint);

  bool ActOnTypeConstraint(const CXXScopeSpec &SS,
                           TemplateIdAnnotation *TypeConstraint,
                           TemplateTypeParmDecl *ConstrainedParameter,
                           SourceLocation EllipsisLoc);
  bool BuildTypeConstraint(const CXXScopeSpec &SS,
                           TemplateIdAnnotation *TypeConstraint,
                           TemplateTypeParmDecl *ConstrainedParameter,
                           SourceLocation EllipsisLoc,
                           bool AllowUnexpandedPack);

  bool AttachTypeConstraint(NestedNameSpecifierLoc NS,
                            DeclarationNameInfo NameInfo,
                            ConceptDecl *NamedConcept, NamedDecl *FoundDecl,
                            const TemplateArgumentListInfo *TemplateArgs,
                            TemplateTypeParmDecl *ConstrainedParameter,
                            SourceLocation EllipsisLoc);

  bool AttachTypeConstraint(AutoTypeLoc TL,
                            NonTypeTemplateParmDecl *NewConstrainedParm,
                            NonTypeTemplateParmDecl *OrigConstrainedParm,
                            SourceLocation EllipsisLoc);

  ExprResult
  CheckConceptTemplateId(const CXXScopeSpec &SS, SourceLocation TemplateKWLoc,
                         const DeclarationNameInfo &ConceptNameInfo,
                         NamedDecl *FoundDecl, ConceptDecl *NamedConcept,
                         const TemplateArgumentListInfo *TemplateArgs);

  Decl *ActOnConceptDefinition(Scope *S,
                               MultiTemplateParamsArg TemplateParameterLists,
                               const IdentifierInfo *Name,
                               SourceLocation NameLoc, Expr *ConstraintExpr,
                               const ParsedAttributesView &Attrs);

  void CheckConceptRedefinition(ConceptDecl *NewDecl, LookupResult &Previous,
                                bool &AddToScope);

  void ActOnStartTrailingRequiresClause(Scope *S, Declarator &D);
  ExprResult ActOnFinishTrailingRequiresClause(ExprResult ConstraintExpr);
  ExprResult ActOnRequiresClause(ExprResult ConstraintExpr);

  void CheckConstrainedAuto(const AutoType *AutoT, SourceLocation Loc);

  /// Returns the more constrained function according to the rules of
  /// partial ordering by constraints (C++ [temp.constr.order]).
  ///
  /// \param FD1 the first function
  ///
  /// \param FD2 the second function
  ///
  /// \returns the more constrained function. If neither function is
  /// more constrained, returns NULL.
  FunctionDecl *getMoreConstrainedFunction(FunctionDecl *FD1,
                                           FunctionDecl *FD2);

private:
  /// Caches pairs of template-like decls whose associated constraints were
  /// checked for subsumption and whether or not the first's constraints did in
  /// fact subsume the second's.
  llvm::DenseMap<std::pair<NamedDecl *, NamedDecl *>, bool> SubsumptionCache;
  /// Caches the normalized associated constraints of declarations (concepts or
  /// constrained declarations). If an error occurred while normalizing the
  /// associated constraints of the template or concept, nullptr will be cached
  /// here.
  llvm::DenseMap<NamedDecl *, NormalizedConstraint *> NormalizationCache;

  llvm::ContextualFoldingSet<ConstraintSatisfaction, const ASTContext &>
      SatisfactionCache;

  // The current stack of constraint satisfactions, so we can exit-early.
  llvm::SmallVector<SatisfactionStackEntryTy, 10> SatisfactionStack;
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMACONCEPT_H

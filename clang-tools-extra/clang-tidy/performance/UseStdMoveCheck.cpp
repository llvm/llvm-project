//===--- UseStdMoveCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStdMoveCheck.h"
#include "../utils/ExprSequence.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {
AST_MATCHER(QualType, isLValueReferenceType) {
  return Node->isLValueReferenceType();
}

AST_MATCHER(DeclRefExpr, refersToEnclosingVariableOrCapture) {
  return Node.refersToEnclosingVariableOrCapture();
}

AST_MATCHER_P(Expr, hasValueSource,
              ast_matchers::internal::Matcher<DeclRefExpr>, Inner) {
  const Expr *Value = Node.IgnoreParenImpCasts();
  while (const auto *Comma = dyn_cast<BinaryOperator>(Value)) {
    if (Comma->getOpcode() != BO_Comma)
      break;
    Value = Comma->getRHS()->IgnoreParenImpCasts();
  }
  const auto *Reference = dyn_cast<DeclRefExpr>(Value);
  return Reference && Inner.matches(*Reference, Finder, Builder);
}

AST_MATCHER(CXXOperatorCallExpr, isCopyAssignmentOperator) {
  if (const auto *MD = dyn_cast_or_null<CXXMethodDecl>(Node.getDirectCallee()))
    return MD->isCopyAssignmentOperator();
  return false;
}

} // namespace

// Follow wrappers so that, for example, a const reference initializer and a
// parenthesized reference initializer are treated just like `T &ref = value`.
static bool
mayEscape(const DeclRefExpr *Reference, ASTContext &Context,
          const llvm::DenseMap<const VarDecl *, const VarDecl *> &Aliases) {
  SmallVector<const Expr *, 4> WorkList{Reference};
  while (!WorkList.empty()) {
    const Expr *Expression = WorkList.pop_back_val();
    for (const DynTypedNode &Parent : Context.getParents(*Expression)) {
      if (const auto *Variable = Parent.get<VarDecl>()) {
        if ((Variable->getType()->isReferenceType() &&
             !Aliases.contains(Variable)) ||
            Variable->getType()->isPointerType())
          return true;
        continue;
      }
      const auto *ParentExpr = Parent.get<Expr>();
      if (!ParentExpr)
        continue;
      if (isa<LambdaExpr>(ParentExpr))
        return true;
      if (const auto *Init = dyn_cast<InitListExpr>(ParentExpr)) {
        // The syntactic form can reference the source directly even when the
        // semantic form contains a value copy. Only the latter binds storage.
        if (Init->getSemanticForm())
          continue;
        // Parent maps can associate a syntactic operand with the semantic
        // list, bypassing the intervening copy constructor. Only a direct
        // glvalue or pointer element can retain access to the source.
        if ((Expression->isGLValue() ||
             Expression->getType()->isPointerType()) &&
            llvm::any_of(Init->inits(), [&](const Expr *Element) {
              return Element && Element->IgnoreParenImpCasts() ==
                                    Expression->IgnoreParenImpCasts();
            }))
          return true;
        continue;
      }
      if (isa<CXXThrowExpr, CXXNewExpr>(ParentExpr) &&
          Expression->getType()->isPointerType())
        return true;
      if (const auto *Unary = dyn_cast<UnaryOperator>(ParentExpr);
          Unary && Unary->getOpcode() == UO_AddrOf) {
        WorkList.push_back(Unary);
        continue;
      }
      if (const auto *Assignment = dyn_cast<BinaryOperator>(ParentExpr);
          Assignment && Assignment->isAssignmentOp() &&
          Assignment->getRHS() == Expression &&
          Expression->getType()->isPointerType())
        return true;
      if (isa<BinaryOperator, UnaryOperator>(ParentExpr) &&
          (ParentExpr->isGLValue() || ParentExpr->getType()->isPointerType())) {
        WorkList.push_back(ParentExpr);
        continue;
      }
      if (isa<ParenExpr, CastExpr, ExprWithCleanups, MaterializeTemporaryExpr,
              CXXBindTemporaryExpr, MemberExpr, AbstractConditionalOperator>(
              ParentExpr)) {
        WorkList.push_back(ParentExpr);
        continue;
      }
      if (const auto *Construction = dyn_cast<CXXConstructExpr>(ParentExpr)) {
        const CXXConstructorDecl *Constructor = Construction->getConstructor();
        if (Constructor->isCopyConstructor())
          continue;
        for (unsigned I = 0; I < Construction->getNumArgs(); ++I)
          if (Construction->getArg(I) == Expression &&
              (Expression->getType()->isPointerType() ||
               I >= Constructor->getNumParams() ||
               Constructor->getParamDecl(I)->getType()->isReferenceType()))
            return true;
      }
      if (const auto *Call = dyn_cast<CallExpr>(ParentExpr)) {
        if (const auto *Operator = dyn_cast<CXXOperatorCallExpr>(Call);
            Operator && Operator->getOperator() == OO_Amp &&
            Operator->getNumArgs() == 1)
          return true;
        if (const auto *MemberCall = dyn_cast<CXXMemberCallExpr>(Call);
            MemberCall &&
            MemberCall->getImplicitObjectArgument()->IgnoreParenImpCasts() ==
                Reference &&
            (MemberCall->isGLValue() || MemberCall->getType()->isPointerType()))
          WorkList.push_back(MemberCall);
        const FunctionDecl *Callee = Call->getDirectCallee();
        if (const auto *Operator = dyn_cast<CXXOperatorCallExpr>(Call);
            Operator && isa_and_nonnull<CXXMethodDecl>(Callee) &&
            Operator->getArg(0) == Expression &&
            (Operator->isGLValue() || Operator->getType()->isPointerType()))
          WorkList.push_back(Operator);
        if (const auto *Method = dyn_cast_or_null<CXXMethodDecl>(Callee);
            Method && Method->isCopyAssignmentOperator())
          continue;
        unsigned Offset = isa<CXXOperatorCallExpr>(Call) && Callee &&
                                  isa<CXXMethodDecl>(Callee)
                              ? 1
                              : 0;
        for (unsigned I = Offset; I < Call->getNumArgs(); ++I) {
          if (Call->getArg(I) != Expression)
            continue;
          if (Callee && I - Offset < Callee->getNumParams() &&
              Callee->getParamDecl(I - Offset)->hasAttr<NoEscapeAttr>())
            continue;
          // Unknown callees may retain a reference to the argument.
          if (Expression->getType()->isPointerType() || !Callee ||
              I - Offset >= Callee->getNumParams() ||
              Callee->getParamDecl(I - Offset)->getType()->isReferenceType())
            return true;
        }
      }
    }
  }
  return false;
}

static const VarDecl *initializingVariable(const Expr *Expression,
                                           ASTContext &Context) {
  while (true) {
    const DynTypedNodeList Parents = Context.getParents(*Expression);
    if (Parents.size() != 1)
      return nullptr;
    if (const auto *Variable = Parents[0].get<VarDecl>())
      return Variable;
    const auto *Parent = Parents[0].get<Expr>();
    if (!Parent || !isa<ExprWithCleanups, CXXBindTemporaryExpr,
                        MaterializeTemporaryExpr, ImplicitCastExpr>(Parent))
      return nullptr;
    Expression = Parent;
  }
}

// Keep the destination in the source's lexical scope so that moving ownership
// into an inner block cannot release a resource before the source leaves scope.
static const Stmt *variableScope(const VarDecl *Variable, ASTContext &Context) {
  // Lambda parameters can have both the call operator and the lambda itself
  // as AST parents. Their lexical scope is nevertheless the callable's body.
  if (isa<ParmVarDecl>(Variable)) {
    const auto *Function =
        dyn_cast_or_null<FunctionDecl>(Variable->getParentFunctionOrMethod());
    return Function ? Function->getBody() : nullptr;
  }
  DynTypedNode Node = DynTypedNode::create(*Variable);
  while (true) {
    const DynTypedNodeList Parents = Context.getParents(Node);
    if (Parents.size() != 1)
      return nullptr;
    Node = Parents[0];
    if (const auto *Scope = Node.get<Stmt>();
        Scope && isa<CompoundStmt, ForStmt, CXXForRangeStmt, IfStmt, SwitchStmt,
                     WhileStmt, CXXCatchStmt>(Scope))
      return Scope;
    if (const auto *Function = Node.get<FunctionDecl>())
      return Function->getBody();
  }
}

static bool cannotThrow(const FunctionDecl *Function) {
  return Function->getType()->castAs<FunctionProtoType>()->canThrow() ==
         CT_Cannot;
}

// This deliberately handles only ordinary, unambiguously applicable special
// members. General counterfactual overload resolution would require Sema.
static const CXXMethodDecl *findMove(const CXXMethodDecl *Copy,
                                     const Expr *Target = nullptr) {
  const CXXRecordDecl *Record = Copy->getParent()->getDefinition();
  if (!Record || Copy->isTrivial() || Record->isInvalidDecl())
    return nullptr;
  const CXXMethodDecl *Move = nullptr;
  for (const CXXMethodDecl *Method : Record->methods()) {
    const auto *Constructor = dyn_cast<CXXConstructorDecl>(Method);
    if (Copy->isCopyAssignmentOperator()
            ? !Method->isMoveAssignmentOperator()
            : !Constructor || !Constructor->isMoveConstructor())
      continue;
    QualType Pointee = Method->getParamDecl(0)->getType()->getPointeeType();
    if (Pointee.isNull() || Pointee.isConstQualified() ||
        Pointee.isVolatileQualified())
      continue;
    if (Target &&
        ((Target->getType().isConstQualified() && !Method->isConst()) ||
         (Target->getType().isVolatileQualified() && !Method->isVolatile()) ||
         (Method->getRefQualifier() == RQ_RValue && Target->isLValue()) ||
         (Method->getRefQualifier() == RQ_LValue && !Target->isLValue())))
      continue;
    if (Move || Method->isDeleted() || Method->isVariadic() ||
        Method->isExplicitObjectMemberFunction() ||
        Method->getAccess() != AS_public ||
        Method->getTrailingRequiresClause() || Method->isInvalidDecl())
      return nullptr;
    Move = Method;
  }
  return Move;
}

static bool contains(const Stmt *Parent, const Stmt *Child) {
  if (Parent == Child)
    return true;
  return Parent && llvm::any_of(Parent->children(), [&](const Stmt *S) {
           return contains(S, Child);
         });
}

// EH edges can put unordered operands into distinct CFG blocks. Check their
// common expression, rather than trusting the CFG's chosen evaluation order.
static bool unorderedWith(const Expr *Copy, const DeclRefExpr *Reference,
                          ASTContext &Context) {
  const Expr *Expression = Copy;
  while (true) {
    const DynTypedNodeList Parents = Context.getParents(*Expression);
    if (Parents.size() != 1)
      return true;
    const auto *Parent = Parents[0].get<Expr>();
    if (!Parent)
      return false;
    if (const auto *Call = dyn_cast<CallExpr>(Parent)) {
      if (const auto *Member = dyn_cast<CXXMemberCallExpr>(Call);
          Member && contains(Member->getImplicitObjectArgument(), Reference))
        return true;
      for (const Expr *Arg : Call->arguments())
        if (!contains(Arg, Copy) && contains(Arg, Reference))
          return true;
    } else if (const auto *Construction = dyn_cast<CXXConstructExpr>(Parent)) {
      if (!Construction->isListInitialization())
        for (const Expr *Arg : Construction->arguments())
          if (!contains(Arg, Copy) && contains(Arg, Reference))
            return true;
    } else if (const auto *Binary = dyn_cast<BinaryOperator>(Parent)) {
      if (Binary->getOpcode() != BO_Comma && Binary->getOpcode() != BO_LAnd &&
          Binary->getOpcode() != BO_LOr &&
          !(Binary->isAssignmentOp() && Context.getLangOpts().CPlusPlus17))
        if ((contains(Binary->getLHS(), Copy) &&
             contains(Binary->getRHS(), Reference)) ||
            (contains(Binary->getRHS(), Copy) &&
             contains(Binary->getLHS(), Reference)))
          return true;
    }
    Expression = Parent;
  }
}

struct UseStdMoveCheck::FunctionAnalysis
    : RecursiveASTVisitor<UseStdMoveCheck::FunctionAnalysis> {
  std::unique_ptr<CFG> Graph;
  std::unique_ptr<utils::ExprSequence> Sequence;
  std::unique_ptr<utils::StmtToBlockMap> BlockMap;
  llvm::DenseMap<const VarDecl *, SmallVector<const DeclRefExpr *, 8>>
      References;
  llvm::DenseMap<const VarDecl *, const VarDecl *> Aliases;
  SmallVector<const DeclStmt *, 8> Declarations;
  SmallVector<const CallExpr *, 8> Calls;
  llvm::DenseMap<const VarDecl *, bool> Escaped;

  bool TraverseLambdaExpr(LambdaExpr *Lambda) {
    // Capture initializers execute in this function; the body does not. In
    // particular, a by-value capture's body refers to different storage.
    for (Expr *Init : Lambda->capture_inits())
      TraverseStmt(Init);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *Reference) {
    if (const auto *Variable = dyn_cast<VarDecl>(Reference->getDecl()))
      References[Variable].push_back(Reference);
    return true;
  }

  bool VisitVarDecl(VarDecl *Variable) {
    if (Variable->hasLocalStorage() &&
        Variable->getType()->isLValueReferenceType() && Variable->hasInit())
      if (const auto *Reference =
              dyn_cast<DeclRefExpr>(Variable->getInit()->IgnoreParenImpCasts()))
        if (const auto *Source = dyn_cast<VarDecl>(Reference->getDecl()))
          Aliases.try_emplace(Variable, Source);
    return true;
  }

  bool VisitDeclStmt(DeclStmt *Declaration) {
    Declarations.push_back(Declaration);
    return true;
  }

  bool VisitCallExpr(CallExpr *Call) {
    Calls.push_back(Call);
    return true;
  }

  const VarDecl *root(const VarDecl *Variable) const {
    llvm::SmallPtrSet<const VarDecl *, 8> Seen;
    while (Aliases.contains(Variable) && Seen.insert(Variable).second)
      Variable = Aliases.lookup(Variable);
    return Variable;
  }

  void collectAliases(ASTContext &Context) {
    decltype(References) Merged;
    for (const auto &[Declaration, Refs] : References)
      for (const DeclRefExpr *Reference : Refs)
        if (!ExprMutationAnalyzer::isUnevaluated(Reference, Context))
          Merged[root(Declaration)].push_back(Reference);
    References = std::move(Merged);
  }

  ArrayRef<const DeclRefExpr *> referencesTo(const VarDecl *Variable) const {
    auto Found = References.find(Variable);
    return Found == References.end() ? ArrayRef<const DeclRefExpr *>()
                                     : Found->second;
  }

  bool mayHaveEscaped(const VarDecl *Variable, ASTContext &Context) {
    auto [Found, Inserted] = Escaped.try_emplace(Variable, false);
    if (Inserted)
      for (const DeclRefExpr *Reference : referencesTo(Variable))
        if (mayEscape(Reference, Context, Aliases)) {
          Found->second = true;
          break;
        }
    return Found->second;
  }

  bool isLastUse(const Expr *Copy, const DeclRefExpr *Source,
                 ASTContext &Context) const {
    const auto *Variable = cast<VarDecl>(Source->getDecl());
    const ArrayRef<const DeclRefExpr *> Refs = referencesTo(Variable);
    const CFGBlock *CopyBlock = BlockMap->blockContainingStmt(Copy);
    // The CFG can include constructor initializers, while this analysis only
    // collects uses from the body. Do not reason about an untracked copy site.
    if (!CopyBlock || !llvm::is_contained(Refs, Source))
      return false;
    for (const DeclRefExpr *Reference : Refs)
      if (Reference != Source && unorderedWith(Copy, Reference, Context))
        return false;

    SmallVector<const Stmt *, 8> Restorations;
    llvm::SmallPtrSet<const DeclRefExpr *, 8> RestorationRefs;
    for (const DeclStmt *Declaration : Declarations)
      for (const Decl *D : Declaration->decls())
        if (D == Variable)
          Restorations.push_back(Declaration);
    for (const CallExpr *Call : Calls) {
      const auto *Method =
          dyn_cast_or_null<CXXMethodDecl>(Call->getDirectCallee());
      if (!Method || !cannotThrow(Method))
        continue;
      const Expr *Object = nullptr;
      if (const auto *Assignment = dyn_cast<CXXOperatorCallExpr>(Call);
          Assignment && (Method->isCopyAssignmentOperator() ||
                         Method->isMoveAssignmentOperator())) {
        Object = Assignment->getArg(0);
        // Only an independent RHS restores the old value. An arbitrary
        // reference-taking call is not a restoration contract.
        if (llvm::any_of(Refs, [&](const DeclRefExpr *Ref) {
              return contains(Assignment->getArg(1), Ref);
            }))
          continue;
      } else if (const auto *Member = dyn_cast<CXXMemberCallExpr>(Call);
                 Member && Method->hasAttr<ReinitializesAttr>()) {
        Object = Member->getImplicitObjectArgument();
        if (llvm::any_of(Refs, [&](const DeclRefExpr *Ref) {
              return llvm::any_of(Call->arguments(), [&](const Expr *Arg) {
                return contains(Arg, Ref);
              });
            }))
          continue;
      }
      const auto *Reference =
          Object ? dyn_cast<DeclRefExpr>(Object->IgnoreParenImpCasts())
                 : nullptr;
      if (Reference && Reference->getDecl() == Variable) {
        Restorations.push_back(Call);
        RestorationRefs.insert(Reference);
      }
    }

    llvm::SmallPtrSet<const CFGBlock *, 16> Visited;
    SmallVector<std::pair<const CFGBlock *, bool>, 16> WorkList{
        {CopyBlock, true}};
    while (!WorkList.empty()) {
      const auto [Block, First] = WorkList.pop_back_val();
      if (!First && !Visited.insert(Block).second)
        continue;
      SmallVector<const Stmt *, 4> Saving;
      for (const Stmt *Restoration : Restorations)
        if (BlockMap->blockContainingStmt(Restoration) == Block &&
            (!First || Sequence->inSequence(Copy, Restoration)))
          Saving.push_back(Restoration);
      for (const DeclRefExpr *Reference : Refs) {
        if ((First && Reference == Source) ||
            RestorationRefs.contains(Reference))
          continue;
        const CFGBlock *UseBlock = BlockMap->blockContainingStmt(Reference);
        if (!UseBlock)
          return false;
        if (UseBlock != Block ||
            (First && Sequence->inSequence(Reference, Copy)))
          continue;
        if (!llvm::any_of(Saving, [&](const Stmt *Restoration) {
              return Sequence->inSequence(Restoration, Reference);
            }))
          return false;
      }
      if (Saving.empty())
        for (const auto &Successor : Block->succs())
          if (Successor)
            WorkList.emplace_back(Successor, false);
    }
    return true;
  }

  bool handlerUses(const Expr *Copy, const VarDecl *Variable,
                   ASTContext &Context) const {
    const ArrayRef<const DeclRefExpr *> Refs = referencesTo(Variable);
    DynTypedNode Node = DynTypedNode::create(*Copy);
    while (true) {
      const DynTypedNodeList Parents = Context.getParents(Node);
      if (Parents.size() != 1)
        return true;
      Node = Parents[0];
      if (const auto *Try = Node.get<CXXTryStmt>())
        for (unsigned I = 0; I < Try->getNumHandlers(); ++I)
          if (llvm::any_of(Refs, [&](const DeclRefExpr *Ref) {
                return contains(Try->getHandler(I), Ref);
              }))
            return true;
      if (Node.get<FunctionDecl>())
        return false;
    }
  }
};

UseStdMoveCheck::UseStdMoveCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

UseStdMoveCheck::~UseStdMoveCheck() = default;

void UseStdMoveCheck::registerPPCallbacks(const SourceManager &SM,
                                          Preprocessor *PP,
                                          Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseStdMoveCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

void UseStdMoveCheck::onEndOfTranslationUnit() {
  AnalysisCache.clear();
  Diagnosed.clear();
}

void UseStdMoveCheck::registerMatchers(MatchFinder *Finder) {
  const auto Source =
      declRefExpr(to(varDecl(hasLocalStorage(),
                             hasType(qualType(unless(anyOf(
                                 isLValueReferenceType(), isConstQualified(),
                                 isVolatileQualified())))))),
                  hasType(qualType(unless(
                      anyOf(isConstQualified(), isVolatileQualified())))),
                  unless(refersToEnclosingVariableOrCapture()))
          .bind("source");
  Finder->addMatcher(cxxOperatorCallExpr(isCopyAssignmentOperator(),
                                         hasArgument(1, hasValueSource(Source)))
                         .bind("assignment"),
                     this);
  Finder->addMatcher(
      cxxConstructExpr(hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
                       hasArgument(0, hasValueSource(Source)))
          .bind("construction"),
      this);
}

UseStdMoveCheck::FunctionAnalysis *
UseStdMoveCheck::getFunctionAnalysis(const FunctionDecl *FD,
                                     ASTContext *Context) {
  auto [Found, Inserted] = AnalysisCache.try_emplace(FD);
  if (!Inserted)
    return Found->second.get();
  CFG::BuildOptions Options;
  Options.AddEHEdges = true;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  Options.setAllAlwaysAdd();
  auto Analysis = std::make_unique<FunctionAnalysis>();
  Analysis->Graph = CFG::buildCFG(FD, FD->getBody(), Context, Options);
  if (!Analysis->Graph)
    return nullptr;
  Analysis->Sequence = std::make_unique<utils::ExprSequence>(
      Analysis->Graph.get(), FD->getBody(), Context);
  Analysis->BlockMap =
      std::make_unique<utils::StmtToBlockMap>(Analysis->Graph.get(), Context);
  Analysis->TraverseStmt(FD->getBody());
  Analysis->collectAliases(*Context);
  Found->second = std::move(Analysis);
  return Found->second.get();
}

void UseStdMoveCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Construction =
      Result.Nodes.getNodeAs<CXXConstructExpr>("construction");
  const auto *Assignment =
      Result.Nodes.getNodeAs<CXXOperatorCallExpr>("assignment");
  const Expr *Copy =
      Construction ? static_cast<const Expr *>(Construction) : Assignment;
  const auto *Source = Result.Nodes.getNodeAs<DeclRefExpr>("source");
  const auto *Variable = cast<VarDecl>(Source->getDecl());
  const auto *Function =
      dyn_cast_or_null<FunctionDecl>(Variable->getParentFunctionOrMethod());
  if (!Function || !Function->hasBody() || Function->isInvalidDecl() ||
      Function->isDependentContext() ||
      isa<CoroutineBodyStmt>(Function->getBody()) ||
      ExprMutationAnalyzer::isUnevaluated(Copy, *Result.Context))
    return;
  const auto *CopyMethod =
      Construction ? Construction->getConstructor()
                   : cast<CXXMethodDecl>(Assignment->getDirectCallee());
  const Expr *TargetExpr = Assignment ? Assignment->getArg(0) : nullptr;
  const CXXMethodDecl *Move = findMove(CopyMethod, TargetExpr);
  if (!Move || (Construction && Construction->isElidable()) ||
      !Result.Context->hasSameUnqualifiedType(
          Result.Context->getCanonicalTagType(CopyMethod->getParent()),
          Variable->getType().getNonReferenceType()))
    return;
  // A copy constructor can accept additional explicit arguments. Replacing
  // only its first argument does not establish applicability of the move.
  if (Construction)
    for (unsigned I = 1; I < Construction->getNumArgs(); ++I)
      if (!isa<CXXDefaultArgExpr>(Construction->getArg(I)))
        return;
  const Expr *Argument =
      Construction ? Construction->getArg(0) : Assignment->getArg(1);
  if (!Argument->isLValue())
    return;
  if (!AllowedTypes.empty() &&
      !match(qualType(hasDeclaration(
                 namedDecl(matchers::matchesAnyListedRegexName(AllowedTypes)))),
             Variable->getType().getNonReferenceType().getCanonicalType(),
             *Result.Context)
           .empty())
    return;

  const VarDecl *Target =
      Construction ? initializingVariable(Construction, *Result.Context)
                   : nullptr;
  if (Target == Variable)
    return;
  const auto *TargetReference =
      TargetExpr ? dyn_cast<DeclRefExpr>(TargetExpr->IgnoreParenImpCasts())
                 : nullptr;
  if (TargetReference && TargetReference->getDecl() == Variable)
    return;
  const auto *MoveConstructor = dyn_cast<CXXConstructorDecl>(Move);
  if (MoveConstructor &&
      (!Target || Target->getInitStyle() == VarDecl::CInit) &&
      (MoveConstructor->isExplicit() ||
       MoveConstructor->getExplicitSpecifier().getKind() ==
           ExplicitSpecKind::Unresolved))
    return;

  FunctionAnalysis *Analysis = getFunctionAnalysis(Function, Result.Context);
  if (!Analysis || Analysis->mayHaveEscaped(Variable, *Result.Context) ||
      !Analysis->isLastUse(Copy, Source, *Result.Context))
    return;
  // A cast, alias, conditional expression, or member call can still designate
  // the source. Reject receivers depending on any tracked alias of it.
  if (TargetExpr && llvm::any_of(Analysis->referencesTo(Variable),
                                 [&](const DeclRefExpr *Ref) {
                                   return contains(TargetExpr, Ref);
                                 }))
    return;
  const bool NewExceptionEdge = cannotThrow(CopyMethod) && !cannotThrow(Move);
  if (NewExceptionEdge &&
      Analysis->handlerUses(Copy, Variable, *Result.Context))
    return;

  const SourceLocation Location =
      Result.SourceManager->getExpansionLoc(Source->getBeginLoc());
  // Different specializations can share the same spelling. A diagnostic never
  // authorizes rewriting other, unobserved instantiations.
  if (!Diagnosed.emplace(Location.getRawEncoding(), Variable->getName().str())
           .second)
    return;
  DiagnosticBuilder Diagnostic = diag(Location, "'%0' could be moved here")
                                 << Variable->getName();
  if (NewExceptionEdge || Source->getBeginLoc().isMacroID() ||
      Source->getEndLoc().isMacroID() || Copy->getBeginLoc().isMacroID() ||
      Copy->getEndLoc().isMacroID() ||
      Function->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
      Variable->getType()->isRValueReferenceType())
    return;
  if (Construction) {
    const Stmt *Scope = variableScope(Variable, *Result.Context);
    if (!Target || Target->isInitCapture() || !Target->hasLocalStorage() ||
        !Scope || variableScope(Target, *Result.Context) != Scope)
      return;
    if (Construction->isListInitialization())
      for (const CXXConstructorDecl *Constructor :
           CopyMethod->getParent()->ctors())
        if (Constructor->getNumParams() != 0 &&
            utils::type_traits::isStdInitializerList(
                Constructor->getParamDecl(0)->getType()))
          return;
  }
  const SourceManager &SM = *Result.SourceManager;
  const CharSourceRange Range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(Source->getSourceRange()), SM,
      getLangOpts());
  if (Range.isInvalid())
    return;
  Diagnostic << FixItHint::CreateInsertion(Range.getBegin(), "std::move(")
             << FixItHint::CreateInsertion(Range.getEnd(), ")")
             << Inserter.createIncludeInsertion(SM.getFileID(Range.getBegin()),
                                                "<utility>");
}

} // namespace clang::tidy::performance

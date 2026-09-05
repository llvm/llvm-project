//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "../utils/ExprSequence.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <memory>
#include <optional>

using namespace clang::ast_matchers;
using namespace clang::tidy::utils;

namespace clang::tidy::bugprone {

namespace {

const auto DefaultSharedPointers = "::std::shared_ptr;::boost::shared_ptr";
const auto DefaultUniquePointers = "::std::unique_ptr";
const auto DefaultDefaultDeleters = "::std::default_delete";

} // namespace

// Remove wrappers that do not carry semantic load for classifying the value:
// brackets, implicit casts, temporary objects, cleanup nodes.
static const clang::Expr *stripWrappers(const clang::Expr *E) {
  while (E) {
    const clang::Expr *Prev = E;
    E = E->IgnoreParens();
    switch (E->getStmtClass()) {
    case clang::Stmt::ImplicitCastExprClass:
      E = cast<clang::ImplicitCastExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::ExprWithCleanupsClass:
      E = cast<clang::ExprWithCleanups>(E)->getSubExpr();
      break;
    case clang::Stmt::MaterializeTemporaryExprClass:
      E = cast<clang::MaterializeTemporaryExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::CXXBindTemporaryExprClass:
      E = cast<clang::CXXBindTemporaryExpr>(E)->getSubExpr();
      break;
    case clang::Stmt::ConstantExprClass:
      E = cast<clang::ConstantExpr>(E)->getSubExpr();
      break;
    default:
      break;
    }
    if (E == Prev)
      break;
  }
  return E;
}

/// A matcher fragment for the constructor of an owning smart pointer that
/// takes a raw pointer, e.g. `std::unique_ptr<T>(p)` / `std::shared_ptr<T>(p)`.
/// Copy/move constructors (whose argument is another smart pointer, not a
/// raw pointer) never match `hasType(pointerType())` on the referenced
/// variable, so they're naturally excluded.
static auto smartPtrCtorTakingRawPointer() {
  // TODO: smart pointer names must be loaded from options
  return cxxConstructExpr(hasDeclaration(cxxConstructorDecl(ofClass(
      cxxRecordDecl(hasAnyName("::std::unique_ptr", "::std::shared_ptr"))))));
}

static auto smartPtrResetTakingRawPointer() {
  // TODO: smart pointer names must be loaded from options
  static const auto IsSharedPtr = hasAnyName("::std::shared_ptr");
  static const auto IsUniquePtr = hasAnyName("::std::unique_ptr");
  static const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);
  static const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

  return cxxMemberCallExpr(
      on(hasType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))),
      callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))));
}

namespace {
/// Contains information about a second "ownership transfer" of a raw
/// pointer that already belongs to a smart pointer.
struct OwnershipTransfer {
  // The Expr used as the argument of the second (problematic)
  // smart-pointer construction.
  const Expr *DeclRef; // TODO: rename

  // TODO: change the comment
  // The CXXConstructExpr of that second construction (used to print the
  // smart-pointer type in the diagnostic).
  const Expr *ConstructOrResetExpr;

  // Is the order in which the two constructions are evaluated undefined?
  bool EvaluationOrderUndefined = false;

  // Does the second transfer happen in a later loop iteration than the
  // first one?
  bool UseHappensInLaterLoopIteration = false;
};

/// Finds a second ownership transfer of a raw-pointer variable that already
/// owns a smart pointer (and maintains state required by the various
/// internal helper functions). Structured the same way as
/// `UseAfterMoveFinder` in `UseAfterMoveCheck.cpp`.
class OwnershipTransferFinder {
public:
  explicit OwnershipTransferFinder(ASTContext *TheContext)
      : Context(TheContext) {}

  // Within the given code block, finds the first ownership transfer of
  // 'RawPtrVar' that occurs after 'FirstTransfer' (the construct expression
  // that first handed the pointer to a smart pointer). Returns std::nullopt
  // if none is found.
  std::optional<OwnershipTransfer> find(Stmt *CodeBlock,
                                        const Expr *FirstTransfer,
                                        const DeclRefExpr *RawPtrVar);

private:
  std::optional<OwnershipTransfer> findInternal(const CFGBlock *Block,
                                                const Expr *FirstTransfer,
                                                const ValueDecl *RawPtrVar);

  void getOwnershipTransfers(
      const CFGBlock *Block, const Decl *RawPtrVar,
      SmallVectorImpl<std::pair<const DeclRefExpr *, const Expr *>> *Transfers);

  void getReinits(const CFGBlock *Block, const ValueDecl *RawPtrVar,
                  llvm::SmallPtrSetImpl<const Stmt *> *Stmts);

  ASTContext *Context;
  std::unique_ptr<ExprSequence> Sequence;
  std::unique_ptr<StmtToBlockMap> BlockMap;
  llvm::SmallPtrSet<const CFGBlock *, 8> Visited;
};
} // namespace

std::optional<OwnershipTransfer>
OwnershipTransferFinder::find(Stmt *CodeBlock, const Expr *FirstTransfer,
                              const DeclRefExpr *RawPtrVar) {
  // Same rationale as UseAfterMoveCheck: build the CFG manually (rather than
  // via an AnalysisDeclContext) so this also works for lambda bodies, and
  // include implicit/temporary destructors so [[noreturn]] destructors are
  // handled correctly by the control-flow analysis.
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  std::unique_ptr<CFG> TheCFG =
      CFG::buildCFG(nullptr, CodeBlock, Context, Options);
  if (!TheCFG)
    return std::nullopt;

  Sequence = std::make_unique<ExprSequence>(TheCFG.get(), CodeBlock, Context);
  BlockMap = std::make_unique<StmtToBlockMap>(TheCFG.get(), Context);
  Visited.clear();

  const CFGBlock *FirstBlock = BlockMap->blockContainingStmt(FirstTransfer);
  if (!FirstBlock) {
    // Can happen if FirstTransfer is in a constructor initializer.
    FirstBlock = &TheCFG->getEntry();
  }

  auto TheTransfer =
      findInternal(FirstBlock, FirstTransfer, RawPtrVar->getDecl());

  if (TheTransfer) {
    if (const CFGBlock *UseBlock =
            BlockMap->blockContainingStmt(TheTransfer->DeclRef)) {
      // Same reasoning as UseAfterMoveCheck: figure out whether the second
      // transfer can only happen in a later loop iteration than the first.
      CFGReverseBlockReachabilityAnalysis CFA(*TheCFG);
      TheTransfer->UseHappensInLaterLoopIteration =
          UseBlock == FirstBlock ? Visited.contains(UseBlock)
                                 : CFA.isReachable(UseBlock, FirstBlock);
    }
  }
  return TheTransfer;
}

std::optional<OwnershipTransfer>
OwnershipTransferFinder::findInternal(const CFGBlock *Block,
                                      const Expr *FirstTransfer,
                                      const ValueDecl *RawPtrVar) {
  if (Visited.contains(Block))
    return std::nullopt;

  // Mark the block as visited, except if this is the block containing the
  // very first transfer and it's being visited for the first time -- mirrors
  // UseAfterMoveFinder's handling of the initial std::move() block.
  if (!FirstTransfer)
    Visited.insert(Block);

  SmallVector<std::pair<const DeclRefExpr *, const Expr *>, 1> Transfers;
  llvm::SmallPtrSet<const Stmt *, 1> Reinits;
  getOwnershipTransfers(Block, RawPtrVar, &Transfers);
  getReinits(Block, RawPtrVar, &Reinits);

  // A reassignment of the raw pointer (e.g. `ptr = new B();` or
  // `ptr = nullptr;`) only protects a transfer if it doesn't itself
  // potentially happen after the first transfer -- otherwise we can't be
  // sure the pointer was reset before the second construction ran.
  SmallVector<const Stmt *, 1> ReinitsToDelete;
  for (const Stmt *Reinit : Reinits)
    if (FirstTransfer && Sequence->potentiallyAfter(FirstTransfer, Reinit))
      ReinitsToDelete.push_back(Reinit);
  for (const Stmt *Reinit : ReinitsToDelete)
    Reinits.erase(Reinit);

  for (const auto &[DeclRef, ConstructOrResetExpr] : Transfers) {
    // Never match a transfer against itself.
    if (ConstructOrResetExpr == FirstTransfer)
      continue;

    if (!FirstTransfer ||
        Sequence->potentiallyAfter(ConstructOrResetExpr, FirstTransfer)) {
      // Does this transfer have a "saving" reinit -- i.e. one that
      // definitely (not just potentially) happens before it?
      bool HaveSavingReinit = false;
      for (const Stmt *Reinit : Reinits)
        if (!Sequence->potentiallyAfter(Reinit, ConstructOrResetExpr))
          HaveSavingReinit = true;

      if (!HaveSavingReinit) {
        OwnershipTransfer Result;
        Result.DeclRef = DeclRef;
        Result.ConstructOrResetExpr = ConstructOrResetExpr;

        // Same order-of-evaluation caveat as UseAfterMoveCheck: if the
        // first transfer could also potentially come after this one, the
        // relative order between them is unspecified.
        Result.EvaluationOrderUndefined =
            FirstTransfer != nullptr &&
            Sequence->potentiallyAfter(FirstTransfer, ConstructOrResetExpr);

        return Result;
      }
    }
  }

  // If the pointer wasn't reassigned in this block, keep looking in
  // successor blocks (branches, loop bodies, etc.).
  if (Reinits.empty()) {
    for (const auto &Succ : Block->succs()) {
      if (Succ) {
        if (auto Found = findInternal(Succ, nullptr, RawPtrVar))
          return Found;
      }
    }
  }

  return std::nullopt;
}

void OwnershipTransferFinder::getOwnershipTransfers(
    const CFGBlock *Block, const Decl *RawPtrVar,
    SmallVectorImpl<std::pair<const DeclRefExpr *, const Expr *>> *Transfers) {
  Transfers->clear();

  const auto DeclRefMatcher =
      declRefExpr(hasDeclaration(equalsNode(RawPtrVar))).bind("declref");
  const auto TransferMatcher = anyOf(
      cxxConstructExpr(smartPtrCtorTakingRawPointer(),
                       hasArgument(0, ignoringParenImpCasts(DeclRefMatcher)))
          .bind("construct"),
      cxxMemberCallExpr(smartPtrResetTakingRawPointer(),
                        hasArgument(0, ignoringParenImpCasts(DeclRefMatcher)))
          .bind("reset"));

  for (const auto &Elem : *Block) {
    std::optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    const SmallVector<BoundNodes, 1> Matches =
        match(findAll(expr(TransferMatcher)), *S->getStmt(), *Context);

    for (const auto &Match : Matches) {
      const auto *DeclRef = Match.getNodeAs<DeclRefExpr>("declref");
      const auto *ConstructExpr =
          Match.getNodeAs<CXXConstructExpr>("construct");
      const auto *ResetExpr = Match.getNodeAs<CXXMemberCallExpr>("reset");
      const Expr *ConstructOrResetExpr =
          ConstructExpr ? static_cast<const Expr *>(ConstructExpr)
                        : static_cast<const Expr *>(ResetExpr);
      if (DeclRef && ConstructOrResetExpr &&
          BlockMap->blockContainingStmt(DeclRef) == Block)
        Transfers->push_back({DeclRef, ConstructOrResetExpr});
    }
  }

  llvm::sort(*Transfers, [](const auto &A, const auto &B) {
    return A.first->getExprLoc() < B.first->getExprLoc();
  });
}

void OwnershipTransferFinder::getReinits(
    const CFGBlock *Block, const ValueDecl *RawPtrVar,
    llvm::SmallPtrSetImpl<const Stmt *> *Stmts) {
  Stmts->clear();

  // Reassigning the raw-pointer variable itself (to a new object, or to
  // null) means it's no longer the same pointer, so any smart-pointer
  // construction after this point refers to a different object and is not
  // a double-deletion risk. Redeclaring the variable inside the block (e.g.
  // via a shadowing DeclStmt in a nested scope) has the same effect.
  const auto DeclRefMatcher =
      declRefExpr(hasDeclaration(equalsNode(RawPtrVar)));
  const auto ReinitMatcher =
      stmt(anyOf(binaryOperation(hasOperatorName("="),
                                 hasLHS(ignoringParenImpCasts(DeclRefMatcher))),
                 declStmt(hasDescendant(equalsNode(RawPtrVar)))))
          .bind("reinit");

  for (const auto &Elem : *Block) {
    std::optional<CFGStmt> S = Elem.getAs<CFGStmt>();
    if (!S)
      continue;

    const SmallVector<BoundNodes, 1> Matches =
        match(findAll(ReinitMatcher), *S->getStmt(), *Context);

    for (const auto &Match : Matches) {
      const auto *TheStmt = Match.getNodeAs<Stmt>("reinit");
      if (TheStmt && BlockMap->blockContainingStmt(TheStmt) == Block)
        Stmts->insert(TheStmt);
    }
  }
}

static void emitDiagnostic(const ASTContext *Context,
                           const OwnershipTransfer &Transfer,
                           ClangTidyCheck *Check) {
  const SourceLocation UseLoc = Transfer.DeclRef->getBeginLoc();
  if (UseLoc.isInvalid())
    return;
  if (const auto *SmartPtrCtor =
          dyn_cast<const CXXConstructExpr>(Transfer.ConstructOrResetExpr)) {
    Check->diag(UseLoc, "passing a raw pointer %0 to %1 constructor may cause "
                        "double deletion")
        << Transfer.DeclRef->getType() << SmartPtrCtor->getType();

  } else if (const auto *ResetCall = dyn_cast<const CXXMemberCallExpr>(
                 Transfer.ConstructOrResetExpr)) {
    Check->diag(
        UseLoc,
        "passing a raw pointer %0 to %1 reset method may cause double deletion")
        << Transfer.DeclRef->getType() << ResetCall->getObjectType();
  }

  if (Transfer.EvaluationOrderUndefined) {
    Check->diag(UseLoc,
                "the two smart-pointer constructions are unsequenced, i.e. "
                "there is no guarantee about the order in which they are "
                "evaluated",
                DiagnosticIDs::Note);
  } else if (Transfer.UseHappensInLaterLoopIteration) {
    Check->diag(UseLoc,
                "the second construction happens in a later loop iteration "
                "than the first",
                DiagnosticIDs::Note);
  }
}

class SmartPtrInitializationCheckImpl {
public:
  explicit SmartPtrInitializationCheckImpl(SmartPtrInitializationCheck &Check)
      : Check(Check) {}
  virtual ~SmartPtrInitializationCheckImpl() = default;
  virtual void registerMatchers(ast_matchers::MatchFinder *Finder) = 0;
  virtual void check(const ast_matchers::MatchFinder::MatchResult &Result) = 0;
  virtual std::optional<TraversalKind> getCheckTraversalKind() const;
  virtual bool isStrictMode() = 0;

protected:
  SmartPtrInitializationCheck &Check;
};

class SmartPtrInitializationCheckPermissiveMode
    : public SmartPtrInitializationCheckImpl {
public:
  using SmartPtrInitializationCheckImpl::SmartPtrInitializationCheckImpl;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    // hasType(pointerType()) inspects the *sugared* type node as spelled at
    // the variable's declaration. For `A *first`, that node already is a
    // PointerType, so it matches directly. But for `ptr_a_t first` (a `using
    // ptr_a_t = A*;` alias), the sugared node is a TypedefType wrapping a
    // PointerType -- dyn_cast<PointerType> on it fails, so the whole matcher
    // silently never fires for aliased declarations.
    // hasUnqualifiedDesugaredType strips typedefs/using-aliases (and other
    // sugar) before testing, exactly like the reference
    // `bugprone-use-after-move` check does for its own
    // standard-container/smart-pointer type matchers.
    const auto RawPtrArg =
        declRefExpr(
            to(varDecl(hasType(hasUnqualifiedDesugaredType(pointerType())))
                   .bind("raw-ptr-var")))
            .bind("arg");

    // Mirrors the shape of UseAfterMoveCheck's matcher: find the construct
    // expression, then walk up to whichever kind of body contains it so we
    // know what to build a CFG for.
    Finder->addMatcher(
        traverse(
            TK_AsIs,
            expr(anyOf(
                     cxxConstructExpr(
                         smartPtrCtorTakingRawPointer(),
                         hasArgument(0, ignoringParenImpCasts(RawPtrArg)),
                         anyOf(hasAncestor(compoundStmt(hasParent(
                                   lambdaExpr().bind("containing-lambda")))),
                               hasAncestor(functionDecl(anyOf(
                                   cxxConstructorDecl().bind("containing-ctor"),
                                   functionDecl().bind("containing-func")))))),
                     cxxMemberCallExpr(
                         smartPtrResetTakingRawPointer(),
                         hasArgument(0, ignoringParenImpCasts(RawPtrArg)),
                         anyOf(hasAncestor(compoundStmt(hasParent(
                                   lambdaExpr().bind("containing-lambda")))),
                               hasAncestor(functionDecl(anyOf(
                                   cxxConstructorDecl().bind("containing-ctor"),
                                   functionDecl().bind("containing-func"))))))))
                .bind("transfer-call")),
        &Check);

    // TODO: smart pointer names must be loaded from options
    const auto IsSharedPtr = hasAnyName("::std::shared_ptr");
    const auto IsUniquePtr = hasAnyName("::std::unique_ptr");
    const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);

    const auto IsSharedPtrRecord = cxxRecordDecl(IsSharedPtr);
    const auto IsUniquePtrRecord = cxxRecordDecl(IsUniquePtr);
    const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

    const auto SmartPtrGetCallMatcher = cxxMemberCallExpr(
        callee(cxxMethodDecl(hasName("get"))),
        on(hasType(hasUnqualifiedDesugaredType(recordType(
            hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))));

    // Search for `std::shared_ptr(this);` or `std::shared_ptr(other_sp.get());`
    const auto SmartPtrConstructorMatcher =
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(ofClass(IsSmartPtrRecord))),
            hasArgument(0, anyOf(ignoringParenCasts(cxxThisExpr()),
                                 ignoringParenCasts(SmartPtrGetCallMatcher))))
            .bind("dangerous-ctor");

    // Search for `sp.reset(this);` or `sp.reset(other_sp.get())`
    const auto ResetCallWithThisMatcher =
        cxxMemberCallExpr(
            on(hasType(hasUnqualifiedDesugaredType(recordType(
                hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))),
            callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))),
            hasArgument(0, anyOf(ignoringParenCasts(cxxThisExpr()),
                                 ignoringParenCasts(SmartPtrGetCallMatcher))))
            .bind("dangerous-reset");

    Finder->addMatcher(traverse(TK_AsIs, SmartPtrConstructorMatcher), &Check);
    Finder->addMatcher(traverse(TK_AsIs, ResetCallWithThisMatcher), &Check);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *CtorWithThisExpr =
        Result.Nodes.getNodeAs<CXXConstructExpr>("dangerous-ctor");
    const auto *ResetWithThisExpr =
        Result.Nodes.getNodeAs<CXXMemberCallExpr>("dangerous-reset");
    if (CtorWithThisExpr) {
      const Expr *PointerArg = stripWrappers(CtorWithThisExpr->getArg(0));
      if (!PointerArg)
        return;
      emitDiagnostic(Result.Context, {PointerArg, CtorWithThisExpr}, &Check);
    } else if (ResetWithThisExpr) {
      const Expr *PointerArg = stripWrappers(ResetWithThisExpr->getArg(0));
      if (!PointerArg)
        return;
      emitDiagnostic(Result.Context, {PointerArg, ResetWithThisExpr}, &Check);
    } else
      checkFlowSensitive(Result);
  }

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

  bool isStrictMode() override { return false; }

private:
  void
  checkFlowSensitive(const ast_matchers::MatchFinder::MatchResult &Result) {
    const auto *ContainingCtor =
        Result.Nodes.getNodeAs<CXXConstructorDecl>("containing-ctor");
    const auto *ContainingLambda =
        Result.Nodes.getNodeAs<LambdaExpr>("containing-lambda");
    const auto *ContainingFunc =
        Result.Nodes.getNodeAs<FunctionDecl>("containing-func");
    const auto *TransferCall = Result.Nodes.getNodeAs<Expr>("transfer-call");
    const auto *Arg = Result.Nodes.getNodeAs<DeclRefExpr>("arg");

    if (!TransferCall || !Arg)
      return;

    // Only locals/parameters are in scope for this per-function CFG analysis.
    if (!Arg->getDecl()->getDeclContext()->isFunctionOrMethod())
      return;

    Stmt *CodeBlock = nullptr;
    if (ContainingCtor)
      CodeBlock = ContainingCtor->getBody();
    else if (ContainingLambda)
      CodeBlock = ContainingLambda->getBody();
    else if (ContainingFunc)
      CodeBlock = ContainingFunc->getBody();

    if (!CodeBlock)
      return;

    OwnershipTransferFinder Finder(Result.Context);
    if (auto Transfer = Finder.find(CodeBlock, TransferCall, Arg))
      emitDiagnostic(Result.Context, *Transfer, &Check);
  }
};

class SmartPtrInitializationCheckStrictMode
    : public SmartPtrInitializationCheckImpl {
public:
  using SmartPtrInitializationCheckImpl::SmartPtrInitializationCheckImpl;

  static StatementMatcher releaseCallMatcher() {
    return cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));
  }

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    const auto IsSharedPtr = hasAnyName(Check.SharedPointers);
    const auto IsUniquePtr = hasAnyName(Check.UniquePointers);
    const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);
    const auto IsDefaultDeleter = hasAnyName(Check.DefaultDeleters);

    const auto IsSharedPtrRecord = cxxRecordDecl(IsSharedPtr);
    const auto IsUniquePtrRecord = cxxRecordDecl(IsUniquePtr);
    const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

    // Array automatically decays to pointer
    const auto PointerArg =
        expr(anyOf(hasType(pointerType()), hasType(arrayType())));

    // Matcher for unique_ptr types with custom deleters
    auto UniquePtrWithCustomDeleter = classTemplateSpecializationDecl(
        IsUniquePtr, templateArgumentCountIs(2),
        hasTemplateArgument(
            1, refersToType(
                   unless(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                       classTemplateSpecializationDecl(IsDefaultDeleter))))))));

    // Matcher for shared_ptr with custom deleter in constructor
    // Check if the second argument is NOT std::default_delete
    auto SharedPtrWithCustomDeleter = allOf(
        hasDeclaration(cxxConstructorDecl(ofClass(IsSharedPtrRecord))),
        hasArgument(
            1, ignoringParenCasts(unless(hasType(hasUnqualifiedDesugaredType(
                   recordType(hasDeclaration(classTemplateSpecializationDecl(
                       IsDefaultDeleter)))))))));

    // Matcher for smart pointer constructors
    const auto HasCustomDeleter = anyOf(
        SharedPtrWithCustomDeleter,
        allOf(hasType(hasUnqualifiedDesugaredType(
                  recordType(hasDeclaration(UniquePtrWithCustomDeleter)))),
              hasDeclaration(cxxConstructorDecl(ofClass(IsUniquePtrRecord)))));

    const auto AllowedArguments =
        anyOf(ignoringParenCasts(cxxNewExpr()),
              ignoringParenCasts(releaseCallMatcher()));

    const auto OptionalCondOp =
        optionally(ignoringParenCasts(conditionalOperator().bind("cond-op")));

    const auto SmartPtrConstructorMatcher =
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(ofClass(IsSmartPtrRecord))),
            hasArgument(0, PointerArg), unless(HasCustomDeleter),
            unless(hasArgument(0, AllowedArguments)),
            hasArgument(0, OptionalCondOp))
            .bind("ctor");

    // For reset() - we need to check the type of the smart pointer
    // If it's shared_ptr with custom deleter (2+ args in constructor)
    // or unique_ptr with custom deleter type
    const auto SmartPtrWithCustomDeleterType = anyOf(
        // shared_ptr with custom deleter - check if the type has a second
        // template argument that is NOT std::default_delete
        classTemplateSpecializationDecl(
            IsSharedPtr, templateArgumentCountIs(2),
            hasTemplateArgument(
                1, refersToType(unless(hasUnqualifiedDesugaredType(recordType(
                       hasDeclaration(classTemplateSpecializationDecl(
                           IsDefaultDeleter)))))))),
        UniquePtrWithCustomDeleter);

    const auto HasCustomDeleterInReset =
        anyOf(on(hasType(hasUnqualifiedDesugaredType(
                  recordType(hasDeclaration(SmartPtrWithCustomDeleterType))))),
              // Also check if reset call has 2 arguments (second is deleter)
              // but we can't easily check if it's default_delete without
              // matching the function parameters, so we'll skip this case
              hasArgument(1, anything()));

    // Actually, for simplicity, let's just check if the smart pointer type
    // has a custom deleter. If it does, we skip the warning.
    const auto SmartPtrWithDefaultDeleter = classTemplateSpecializationDecl(
        IsSmartPtr,
        anyOf(
            // shared_ptr with default deleter (1 template arg or 2nd is
            // default_delete)
            allOf(IsSharedPtr,
                  anyOf(templateArgumentCountIs(1),
                        allOf(templateArgumentCountIs(2),
                              hasTemplateArgument(
                                  1, refersToType(hasUnqualifiedDesugaredType(
                                         recordType(hasDeclaration(
                                             classTemplateSpecializationDecl(
                                                 IsDefaultDeleter))))))))),
            // unique_ptr with default deleter
            allOf(IsUniquePtr,
                  anyOf(templateArgumentCountIs(1),
                        allOf(templateArgumentCountIs(2),
                              hasTemplateArgument(
                                  1, refersToType(hasUnqualifiedDesugaredType(
                                         recordType(hasDeclaration(
                                             classTemplateSpecializationDecl(
                                                 IsDefaultDeleter)))))))))));

    const auto ResetCallMatcher =
        cxxMemberCallExpr(
            on(hasType(hasUnqualifiedDesugaredType(
                recordType(hasDeclaration(SmartPtrWithDefaultDeleter))))),
            callee(cxxMethodDecl(ofClass(IsSmartPtrRecord), hasName("reset"))),
            hasArgument(0, PointerArg),
            unless(hasArgument(0, AllowedArguments)),
            hasArgument(0, OptionalCondOp))
            .bind("reset");

    Finder->addMatcher(SmartPtrConstructorMatcher, &Check);
    Finder->addMatcher(ResetCallMatcher, &Check);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
    const auto *Reset = Result.Nodes.getNodeAs<CXXMemberCallExpr>("reset");
    const auto *Cond = Result.Nodes.getNodeAs<ConditionalOperator>("cond-op");
    const Expr *ConstructorOrMember = Ctor;
    if (!ConstructorOrMember)
      ConstructorOrMember = Reset;

    if (ConstructorOrMember)
      checkInternal(*Result.Context, ConstructorOrMember, Cond);
  }

  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return std::nullopt;
  }

  bool isStrictMode() override { return true; }

private:
  void checkInternal(ASTContext &Context, const Expr *ConstructorOrMember,
                     const ConditionalOperator *Cond) {
    if (Cond && validateConditionalOperator(Context, Cond))
      return;
    Check.emitDiagnostic(ConstructorOrMember);
  }

  bool validateConditionalOperator(ASTContext &Context,
                                   const ConditionalOperator *Cond) {
    assert(Cond);

    static const StatementMatcher Matcher =
        anyOf(integerLiteral(equals(0)), cxxNullPtrLiteralExpr(), cxxNewExpr(),
              releaseCallMatcher());

    const auto IsValidExpr = [&](const Expr *E) -> bool {
      if (!E)
        return false;

      E = E->IgnoreParenCasts();

      // If this is a nested ternary operator, we check recursively.
      if (const auto *NestedCond = dyn_cast<ConditionalOperator>(E))
        return validateConditionalOperator(Context, NestedCond);

      // Otherwise, we check through the matcher
      const auto Matches = match(Matcher, *E, Context);
      return !Matches.empty();
    };

    return IsValidExpr(Cond->getTrueExpr()) &&
           IsValidExpr(Cond->getFalseExpr());
  }
};

static std::unique_ptr<SmartPtrInitializationCheckImpl>
makeImpl(bool StrictMode, SmartPtrInitializationCheck &Check) {
  if (StrictMode)
    return std::make_unique<SmartPtrInitializationCheckStrictMode>(Check);
  return std::make_unique<SmartPtrInitializationCheckPermissiveMode>(Check);
}

SmartPtrInitializationCheck::SmartPtrInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointers(utils::options::parseStringList(
          Options.get("SharedPointers", DefaultSharedPointers))),
      UniquePointers(utils::options::parseStringList(
          Options.get("UniquePointers", DefaultUniquePointers))),
      DefaultDeleters(utils::options::parseStringList(
          Options.get("DefaultDeleters", DefaultDefaultDeleters))),
      Impl(makeImpl(Options.get("StrictMode", false), *this)) {}

SmartPtrInitializationCheck::~SmartPtrInitializationCheck() = default;

void SmartPtrInitializationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SharedPointers",
                utils::options::serializeStringList(SharedPointers));
  Options.store(Opts, "UniquePointers",
                utils::options::serializeStringList(UniquePointers));
  Options.store(Opts, "DefaultDeleters",
                utils::options::serializeStringList(DefaultDeleters));
  Options.store(Opts, "StrictMode", Impl->isStrictMode());
}

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  Impl->registerMatchers(Finder);
}

void SmartPtrInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  Impl->check(Result);
}

std::optional<TraversalKind>
SmartPtrInitializationCheck::getCheckTraversalKind() const {
  return Impl->getCheckTraversalKind();
}

void SmartPtrInitializationCheck::emitDiagnostic(
    const Expr *ConstructorOrMember) {
  if (const auto *SmartPtrCtor =
          dyn_cast<const CXXConstructExpr>(ConstructorOrMember)) {
    const Expr *PointerArg = stripWrappers(SmartPtrCtor->getArg(0));
    if (!PointerArg)
      return;
    const SourceLocation Loc = PointerArg->getBeginLoc();
    if (Loc.isInvalid())
      return;
    diag(Loc, "passing a raw pointer %0 to %1 constructor may cause "
              "double deletion")
        << PointerArg->getType() << SmartPtrCtor->getType();
  } else if (const auto *ResetCall =
                 dyn_cast<const CXXMemberCallExpr>(ConstructorOrMember)) {
    const Expr *PointerArg = stripWrappers(ResetCall->getArg(0));
    if (!PointerArg)
      return;
    const SourceLocation Loc = PointerArg->getBeginLoc();
    if (Loc.isInvalid())
      return;
    diag(
        Loc,
        "passing a raw pointer %0 to %1 reset method may cause double deletion")
        << PointerArg->getType() << ResetCall->getObjectType();
  }
}

} // namespace clang::tidy::bugprone

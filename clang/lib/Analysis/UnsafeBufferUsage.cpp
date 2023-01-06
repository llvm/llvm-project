//===- UnsafeBufferUsage.cpp - Replace pointers with modern C++ -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace clang;
using namespace ast_matchers;

namespace {
// Because the analysis revolves around variables and their types, we'll need to
// track uses of variables (aka DeclRefExprs).
using DeclUseList = SmallVector<const DeclRefExpr *, 1>;

// Convenience typedef.
using FixItList = SmallVector<FixItHint, 4>;

// Defined below.
class Strategy;
} // namespace

// Because we're dealing with raw pointers, let's define what we mean by that.
static auto hasPointerType() {
    return hasType(hasCanonicalType(pointerType()));
}

namespace {
/// Gadget is an individual operation in the code that may be of interest to
/// this analysis. Each (non-abstract) subclass corresponds to a specific
/// rigid AST structure that constitutes an operation on a pointer-type object.
/// Discovery of a gadget in the code corresponds to claiming that we understand
/// what this part of code is doing well enough to potentially improve it.
/// Gadgets can be unsafe (immediately deserving a warning) or safe (not
/// deserving a warning per se, but affecting our decision-making process
/// nonetheless).
class Gadget {
public:
  enum class Kind {
#define GADGET(x) x,
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
  };

  /// Common type of ASTMatchers used for discovering gadgets.
  /// Useful for implementing the static matcher() methods
  /// that are expected from all non-abstract subclasses.
  using Matcher = decltype(stmt());

  Gadget(Kind K) : K(K) {}

  Kind getKind() const { return K; }

  virtual bool isSafe() const = 0;
  virtual const Stmt *getBaseStmt() const = 0;

  /// Returns the list of pointer-type variables on which this gadget performs
  /// its operation. Typically, there's only one variable. This isn't a list
  /// of all DeclRefExprs in the gadget's AST!
  virtual DeclUseList getClaimedVarUseSites() const = 0;

  /// Returns a fixit that would fix the current gadget according to
  /// the current strategy. Returns None if the fix cannot be produced;
  /// returns an empty list if no fixes are necessary.
  virtual std::optional<FixItList> getFixits(const Strategy &) const {
    return std::nullopt;
  }

  virtual ~Gadget() = default;

private:
  Kind K;
};

using GadgetList = std::vector<std::unique_ptr<Gadget>>;

/// Unsafe gadgets correspond to unsafe code patterns that warrants
/// an immediate warning.
class UnsafeGadget : public Gadget {
public:
  UnsafeGadget(Kind K) : Gadget(K) {}

  static bool classof(const Gadget *G) { return G->isSafe(); }
  bool isSafe() const final { return false; }
};

/// Safe gadgets correspond to code patterns that aren't unsafe but need to be
/// properly recognized in order to emit correct warnings and fixes over unsafe
/// gadgets. For example, if a raw pointer-type variable is replaced by
/// a safe C++ container, every use of such variable may need to be
/// carefully considered and possibly updated.
class SafeGadget : public Gadget {
public:
  SafeGadget(Kind K) : Gadget(K) {}

  static bool classof(const Gadget *G) { return !G->isSafe(); }
  bool isSafe() const final { return true; }
};

/// An increment of a pointer-type value is unsafe as it may run the pointer
/// out of bounds.
class IncrementGadget : public UnsafeGadget {
  static constexpr const char *const OpTag = "op";
  const UnaryOperator *Op;

public:
  IncrementGadget(const MatchFinder::MatchResult &Result)
      : UnsafeGadget(Kind::Increment),
        Op(Result.Nodes.getNodeAs<UnaryOperator>(OpTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::Increment;
  }

  static Matcher matcher() {
    return stmt(unaryOperator(
      hasOperatorName("++"),
      hasUnaryOperand(ignoringParenImpCasts(hasPointerType()))
    ).bind(OpTag));
  }

  const UnaryOperator *getBaseStmt() const override { return Op; }

  DeclUseList getClaimedVarUseSites() const override {
    SmallVector<const DeclRefExpr *, 2> Uses;
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(Op->getSubExpr()->IgnoreParenImpCasts())) {
      Uses.push_back(DRE);
    }

    return std::move(Uses);
  }
};

/// A decrement of a pointer-type value is unsafe as it may run the pointer
/// out of bounds.
class DecrementGadget : public UnsafeGadget {
  static constexpr const char *const OpTag = "op";
  const UnaryOperator *Op;

public:
  DecrementGadget(const MatchFinder::MatchResult &Result)
      : UnsafeGadget(Kind::Decrement),
        Op(Result.Nodes.getNodeAs<UnaryOperator>(OpTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::Decrement;
  }

  static Matcher matcher() {
    return stmt(unaryOperator(
      hasOperatorName("--"),
      hasUnaryOperand(ignoringParenImpCasts(hasPointerType()))
    ).bind(OpTag));
  }

  const UnaryOperator *getBaseStmt() const override { return Op; }

  DeclUseList getClaimedVarUseSites() const override {
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(Op->getSubExpr()->IgnoreParenImpCasts())) {
      return {DRE};
    }

    return {};
  }
};

/// Array subscript expressions on raw pointers as if they're arrays. Unsafe as
/// it doesn't have any bounds checks for the array.
class ArraySubscriptGadget : public UnsafeGadget {
  static constexpr const char *const ArraySubscrTag = "arraySubscr";
  const ArraySubscriptExpr *ASE;

public:
  ArraySubscriptGadget(const MatchFinder::MatchResult &Result)
      : UnsafeGadget(Kind::ArraySubscript),
        ASE(Result.Nodes.getNodeAs<ArraySubscriptExpr>(ArraySubscrTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::ArraySubscript;
  }

  static Matcher matcher() {
    // FIXME: What if the index is integer literal 0? Should this be
    // a safe gadget in this case?
    return stmt(arraySubscriptExpr(hasBase(ignoringParenImpCasts(hasPointerType())),
                                   unless(hasIndex(integerLiteral(equals(0)))))
                .bind(ArraySubscrTag));
  }

  const ArraySubscriptExpr *getBaseStmt() const override { return ASE; }

  DeclUseList getClaimedVarUseSites() const override {
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreParenImpCasts())) {
      return {DRE};
    }

    return {};
  }
};

/// A pointer arithmetic expression of one of the forms:
///  \code
///  ptr + n | n + ptr | ptr - n | ptr += n | ptr -= n
///  \endcode
class PointerArithmeticGadget : public UnsafeGadget {
  static constexpr const char *const PointerArithmeticTag = "ptrAdd";
  static constexpr const char *const PointerArithmeticPointerTag = "ptrAddPtr";
  const BinaryOperator *PA; // pointer arithmetic expression
  const Expr * Ptr;         // the pointer expression in `PA`

public:
    PointerArithmeticGadget(const MatchFinder::MatchResult &Result)
      : UnsafeGadget(Kind::PointerArithmetic),
        PA(Result.Nodes.getNodeAs<BinaryOperator>(PointerArithmeticTag)),
        Ptr(Result.Nodes.getNodeAs<Expr>(PointerArithmeticPointerTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::PointerArithmetic;
  }

  static Matcher matcher() {
    auto HasIntegerType = anyOf(
          hasType(isInteger()), hasType(enumType()));
    auto PtrAtRight = allOf(hasOperatorName("+"),
                            hasRHS(expr(hasPointerType()).bind(PointerArithmeticPointerTag)),
                            hasLHS(HasIntegerType));
    auto PtrAtLeft = allOf(
           anyOf(hasOperatorName("+"), hasOperatorName("-"),
                 hasOperatorName("+="), hasOperatorName("-=")),
           hasLHS(expr(hasPointerType()).bind(PointerArithmeticPointerTag)),
           hasRHS(HasIntegerType));

    return stmt(binaryOperator(anyOf(PtrAtLeft, PtrAtRight)).bind(PointerArithmeticTag));
  }

  const Stmt *getBaseStmt() const override { return PA; }

  DeclUseList getClaimedVarUseSites() const override {
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(Ptr->IgnoreParenImpCasts())) {
      return {DRE};
    }

    return {};
  }
  // FIXME: pointer adding zero should be fine
  //FIXME: this gadge will need a fix-it
};
} // namespace

namespace {
// An auxiliary tracking facility for the fixit analysis. It helps connect
// declarations to its and make sure we've covered all uses with our analysis
// before we try to fix the declaration.
class DeclUseTracker {
  using UseSetTy = SmallSet<const DeclRefExpr *, 16>;
  using DefMapTy = DenseMap<const VarDecl *, const DeclStmt *>;

  // Allocate on the heap for easier move.
  std::unique_ptr<UseSetTy> Uses{std::make_unique<UseSetTy>()};
  DefMapTy Defs{};

public:
  DeclUseTracker() = default;
  DeclUseTracker(const DeclUseTracker &) = delete; // Let's avoid copies.
  DeclUseTracker(DeclUseTracker &&) = default;

  // Start tracking a freshly discovered DRE.
  void discoverUse(const DeclRefExpr *DRE) { Uses->insert(DRE); }

  // Stop tracking the DRE as it's been fully figured out.
  void claimUse(const DeclRefExpr *DRE) {
    assert(Uses->count(DRE) &&
           "DRE not found or claimed by multiple matchers!");
    Uses->erase(DRE);
  }

  // A variable is unclaimed if at least one use is unclaimed.
  bool hasUnclaimedUses(const VarDecl *VD) const {
    // FIXME: Can this be less linear? Maybe maintain a map from VDs to DREs?
    return any_of(*Uses, [VD](const DeclRefExpr *DRE) {
      return DRE->getDecl()->getCanonicalDecl() == VD->getCanonicalDecl();
    });
  }

  void discoverDecl(const DeclStmt *DS) {
    for (const Decl *D : DS->decls()) {
      if (const auto *VD = dyn_cast<VarDecl>(D)) {
        // FIXME: Assertion temporarily disabled due to a bug in
        // ASTMatcher internal behavior in presence of GNU
        // statement-expressions. We need to properly investigate this
        // because it can screw up our algorithm in other ways.
        // assert(Defs.count(VD) == 0 && "Definition already discovered!");
        Defs[VD] = DS;
      }
    }
  }

  const DeclStmt *lookupDecl(const VarDecl *VD) const {
    auto It = Defs.find(VD);
    assert(It != Defs.end() && "Definition never discovered!");
    return It->second;
  }
};
} // namespace

namespace {
// Strategy is a map from variables to the way we plan to emit fixes for
// these variables. It is figured out gradually by trying different fixes
// for different variables depending on gadgets in which these variables
// participate.
class Strategy {
public:
  enum class Kind {
    Wontfix,    // We don't plan to emit a fixit for this variable.
    Span,       // We recommend replacing the variable with std::span.
    Iterator,   // We recommend replacing the variable with std::span::iterator.
    Array,      // We recommend replacing the variable with std::array.
    Vector      // We recommend replacing the variable with std::vector.
  };

private:
  using MapTy = llvm::DenseMap<const VarDecl *, Kind>;

  MapTy Map;

public:
  Strategy() = default;
  Strategy(const Strategy &) = delete; // Let's avoid copies.
  Strategy(Strategy &&) = default;

  void set(const VarDecl *VD, Kind K) {
    Map[VD] = K;
  }

  Kind lookup(const VarDecl *VD) const {
    auto I = Map.find(VD);
    if (I == Map.end())
      return Kind::Wontfix;

    return I->second;
  }
};
} // namespace

/// Scan the function and return a list of gadgets found with provided kits.
static std::pair<GadgetList, DeclUseTracker> findGadgets(const Decl *D) {

  struct GadgetFinderCallback : MatchFinder::MatchCallback {
    GadgetList Gadgets;
    DeclUseTracker Tracker;

    void run(const MatchFinder::MatchResult &Result) override {
      // In debug mode, assert that we've found exactly one gadget.
      // This helps us avoid conflicts in .bind() tags.
#if NDEBUG
#define NEXT return
#else
      [[maybe_unused]] int numFound = 0;
#define NEXT ++numFound
#endif

      if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("any_dre")) {
        Tracker.discoverUse(DRE);
        NEXT;
      }

      if (const auto *DS = Result.Nodes.getNodeAs<DeclStmt>("any_ds")) {
        Tracker.discoverDecl(DS);
        NEXT;
      }

      // Figure out which matcher we've found, and call the appropriate
      // subclass constructor.
      // FIXME: Can we do this more logarithmically?
#define GADGET(name)                                                           \
      if (Result.Nodes.getNodeAs<Stmt>(#name)) {                               \
        Gadgets.push_back(std::make_unique<name ## Gadget>(Result));           \
        NEXT;                                                                  \
      }
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"

      assert(numFound >= 1 && "Gadgets not found in match result!");
      assert(numFound <= 1 && "Conflicting bind tags in gadgets!");
    }
  };

  MatchFinder M;
  GadgetFinderCallback CB;

  // clang-format off
  M.addMatcher(
    stmt(forEachDescendant(
      stmt(anyOf(
        // Add Gadget::matcher() for every gadget in the registry.
#define GADGET(x)                                                              \
        x ## Gadget::matcher().bind(#x),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
        // In parallel, match all DeclRefExprs so that to find out
        // whether there are any uncovered by gadgets.
        declRefExpr(hasPointerType(), to(varDecl())).bind("any_dre"),
        // Also match DeclStmts because we'll need them when fixing
        // their underlying VarDecls that otherwise don't have
        // any backreferences to DeclStmts.
        declStmt().bind("any_ds")
      ))
      // FIXME: Idiomatically there should be a forCallable(equalsNode(D))
      // here, to make sure that the statement actually belongs to the
      // function and not to a nested function. However, forCallable uses
      // ParentMap which can't be used before the AST is fully constructed.
      // The original problem doesn't sound like it needs ParentMap though,
      // maybe there's a more direct solution?
    )),
    &CB
  );
  // clang-format on

  M.match(*D->getBody(), D->getASTContext());

  // Gadgets "claim" variables they're responsible for. Once this loop finishes,
  // the tracker will only track DREs that weren't claimed by any gadgets,
  // i.e. not understood by the analysis.
  for (const auto &G : CB.Gadgets) {
    for (const auto *DRE : G->getClaimedVarUseSites()) {
      CB.Tracker.claimUse(DRE);
    }
  }

  return {std::move(CB.Gadgets), std::move(CB.Tracker)};
}

void clang::checkUnsafeBufferUsage(const Decl *D,
                                   UnsafeBufferUsageHandler &Handler) {
  assert(D && D->getBody());

  SmallSet<const VarDecl *, 8> WarnedDecls;

  auto [Gadgets, Tracker] = findGadgets(D);

  DenseMap<const VarDecl *, std::vector<const Gadget *>> Map;

  // First, let's sort gadgets by variables. If some gadgets cover more than one
  // variable, they'll appear more than once in the map.
  for (const auto &G : Gadgets) {
    DeclUseList ClaimedVarUseSites = G->getClaimedVarUseSites();

    // Populate the map.
    bool Pushed = false;
    for (const DeclRefExpr *DRE : ClaimedVarUseSites) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        Map[VD].push_back(G.get());
        Pushed = true;
      }
    }

    if (!Pushed && !G->isSafe()) {
      // We won't return to this gadget later. Emit the warning right away.
      Handler.handleUnsafeOperation(G->getBaseStmt());
      continue;
    }
  }

  Strategy S;

  for (const auto &[VD, VDGadgets] : Map) {

    // If the variable has no unsafe gadgets, skip it entirely.
    if (!any_of(VDGadgets, [](const Gadget *G) { return !G->isSafe(); }))
      continue;

    std::optional<FixItList> Fixes;

    // Avoid suggesting fixes if not all uses of the variable are identified
    // as known gadgets.
    // FIXME: Support parameter variables as well.
    if (!Tracker.hasUnclaimedUses(VD) && VD->isLocalVarDecl()) {
      // Choose the appropriate strategy. FIXME: We should try different
      // strategies.
      S.set(VD, Strategy::Kind::Span);

      // Check if it works.
      // FIXME: This isn't sufficient (or even correct) when a gadget has
      // already produced a fixit for a different variable i.e. it was mentioned
      // in the map twice (or more). In such case the correct thing to do is
      // to undo the previous fix first, and then if we can't produce the new
      // fix for both variables, revert to the old one.
      Fixes = FixItList{};
      for (const Gadget *G : VDGadgets) {
        std::optional<FixItList> F = G->getFixits(S);
        if (!F) {
          Fixes = std::nullopt;
          break;
        }

        for (auto &&Fixit: *F)
          Fixes->push_back(std::move(Fixit));
      }
    }

    if (Fixes) {
      // If we reach this point, the strategy is applicable.
      Handler.handleFixableVariable(VD, std::move(*Fixes));
    } else {
      // The strategy has failed. Emit the warning without the fixit.
      S.set(VD, Strategy::Kind::Wontfix);
      for (const Gadget *G : VDGadgets) {
        if (!G->isSafe()) {
          Handler.handleUnsafeOperation(G->getBaseStmt());
        }
      }
    }
  }
}

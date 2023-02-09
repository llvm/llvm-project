//===- UnsafeBufferUsage.cpp - Replace pointers with modern C++ -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>

using namespace llvm;
using namespace clang;
using namespace ast_matchers;

namespace clang::ast_matchers {
// A `RecursiveASTVisitor` that traverses all descendants of a given node "n"
// except for those belonging to a different callable of "n".
class MatchDescendantVisitor
    : public RecursiveASTVisitor<MatchDescendantVisitor> {
public:
  typedef RecursiveASTVisitor<MatchDescendantVisitor> VisitorBase;

  // Creates an AST visitor that matches `Matcher` on all
  // descendants of a given node "n" except for the ones
  // belonging to a different callable of "n".
  MatchDescendantVisitor(const internal::DynTypedMatcher *Matcher,
                         internal::ASTMatchFinder *Finder,
                         internal::BoundNodesTreeBuilder *Builder,
                         internal::ASTMatchFinder::BindKind Bind)
      : Matcher(Matcher), Finder(Finder), Builder(Builder), Bind(Bind),
        Matches(false) {}

  // Returns true if a match is found in a subtree of `DynNode`, which belongs
  // to the same callable of `DynNode`.
  bool findMatch(const DynTypedNode &DynNode) {
    Matches = false;
    if (const Stmt *StmtNode = DynNode.get<Stmt>()) {
      TraverseStmt(const_cast<Stmt *>(StmtNode));
      *Builder = ResultBindings;
      return Matches;
    }
    return false;
  }

  // The following are overriding methods from the base visitor class.
  // They are public only to allow CRTP to work. They are *not *part
  // of the public API of this class.

  // For the matchers so far used in safe buffers, we only need to match
  // `Stmt`s.  To override more as needed.

  bool TraverseDecl(Decl *Node) {
    if (!Node)
      return true;
    if (!match(*Node))
      return false;
    // To skip callables:
    if (isa<FunctionDecl, BlockDecl, ObjCMethodDecl>(Node))
      return true;
    // Traverse descendants
    return VisitorBase::TraverseDecl(Node);
  }

  bool TraverseStmt(Stmt *Node, DataRecursionQueue *Queue = nullptr) {
    if (!Node)
      return true;
    if (!match(*Node))
      return false;
    // To skip callables:
    if (isa<LambdaExpr>(Node))
      return true;
    return VisitorBase::TraverseStmt(Node);
  }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const {
    // TODO: let's ignore implicit code for now
    return false;
  }

private:
  // Sets 'Matched' to true if 'Matcher' matches 'Node'
  //
  // Returns 'true' if traversal should continue after this function
  // returns, i.e. if no match is found or 'Bind' is 'BK_All'.
  template <typename T> bool match(const T &Node) {
    internal::BoundNodesTreeBuilder RecursiveBuilder(*Builder);

    if (Matcher->matches(DynTypedNode::create(Node), Finder,
                         &RecursiveBuilder)) {
      ResultBindings.addMatch(RecursiveBuilder);
      Matches = true;
      if (Bind != internal::ASTMatchFinder::BK_All)
        return false; // Abort as soon as a match is found.
    }
    return true;
  }

  const internal::DynTypedMatcher *const Matcher;
  internal::ASTMatchFinder *const Finder;
  internal::BoundNodesTreeBuilder *const Builder;
  internal::BoundNodesTreeBuilder ResultBindings;
  const internal::ASTMatchFinder::BindKind Bind;
  bool Matches;
};

AST_MATCHER_P(Stmt, forEveryDescendant, internal::Matcher<Stmt>, innerMatcher) {
  const DynTypedMatcher &DTM = static_cast<DynTypedMatcher>(innerMatcher);
  
  MatchDescendantVisitor Visitor(&DTM, Finder, Builder, ASTMatchFinder::BK_All);  
  return Visitor.findMatch(DynTypedNode::create(Node));
}

// Matches a `Stmt` node iff the node is in a safe-buffer opt-out region
AST_MATCHER_P(Stmt, notInSafeBufferOptOut, const UnsafeBufferUsageHandler *, Handler) {
  return !Handler->isSafeBufferOptOut(Node.getBeginLoc());
}

AST_MATCHER_P(CastExpr, castSubExpr, internal::Matcher<Expr>, innerMatcher) {
  return innerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
}

// Returns a matcher that matches any expression 'e' such that `innerMatcher`
// matches 'e' and 'e' is in an Unspecified Lvalue Context.
static internal::Matcher<Stmt>
isInUnspecifiedLvalueContext(internal::Matcher<Expr> innerMatcher) {
  return implicitCastExpr(hasCastKind(CastKind::CK_LValueToRValue),
                          castSubExpr(innerMatcher));
  // FIXME: add assignmentTo context...
}
} // namespace clang::ast_matchers

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

static auto hasArrayType() {
    return hasType(hasCanonicalType(arrayType()));
}

namespace {
/// Gadget is an individual operation in the code that may be of interest to
/// this analysis. Each (non-abstract) subclass corresponds to a specific
/// rigid AST structure that constitutes an operation on a pointer-type object.
/// Discovery of a gadget in the code corresponds to claiming that we understand
/// what this part of code is doing well enough to potentially improve it.
/// Gadgets can be warning (immediately deserving a warning) or fixable (not
/// always deserving a warning per se, but requires our attention to identify
/// it warrants a fixit).
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

  virtual bool isWarningGadget() const = 0;
  virtual const Stmt *getBaseStmt() const = 0;

  /// Returns the list of pointer-type variables on which this gadget performs
  /// its operation. Typically, there's only one variable. This isn't a list
  /// of all DeclRefExprs in the gadget's AST!
  virtual DeclUseList getClaimedVarUseSites() const = 0;

  virtual ~Gadget() = default;

private:
  Kind K;
};


/// Warning gadgets correspond to unsafe code patterns that warrants
/// an immediate warning.
class WarningGadget : public Gadget {
public:
  WarningGadget(Kind K) : Gadget(K) {}

  static bool classof(const Gadget *G) { return G->isWarningGadget(); }
  bool isWarningGadget() const final { return true; }
};

/// Fixable gadgets correspond to code patterns that aren't always unsafe but need to be
/// properly recognized in order to emit fixes. For example, if a raw pointer-type
/// variable is replaced by a safe C++ container, every use of such variable must be
/// carefully considered and possibly updated.
class FixableGadget : public Gadget {
public:
  FixableGadget(Kind K) : Gadget(K) {}

  static bool classof(const Gadget *G) { return !G->isWarningGadget(); }
  bool isWarningGadget() const final { return false; }

  /// Returns a fixit that would fix the current gadget according to
  /// the current strategy. Returns None if the fix cannot be produced;
  /// returns an empty list if no fixes are necessary.
  virtual std::optional<FixItList> getFixits(const Strategy &) const {
    return std::nullopt;
  }
};

using FixableGadgetList = std::vector<std::unique_ptr<FixableGadget>>;
using WarningGadgetList = std::vector<std::unique_ptr<WarningGadget>>;

/// An increment of a pointer-type value is unsafe as it may run the pointer
/// out of bounds.
class IncrementGadget : public WarningGadget {
  static constexpr const char *const OpTag = "op";
  const UnaryOperator *Op;

public:
  IncrementGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::Increment),
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
class DecrementGadget : public WarningGadget {
  static constexpr const char *const OpTag = "op";
  const UnaryOperator *Op;

public:
  DecrementGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::Decrement),
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
class ArraySubscriptGadget : public WarningGadget {
  static constexpr const char *const ArraySubscrTag = "ArraySubscript";
  const ArraySubscriptExpr *ASE;

public:
  ArraySubscriptGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::ArraySubscript),
        ASE(Result.Nodes.getNodeAs<ArraySubscriptExpr>(ArraySubscrTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::ArraySubscript;
  }

  static Matcher matcher() {
    // FIXME: What if the index is integer literal 0? Should this be
    // a safe gadget in this case?
      // clang-format off
      return stmt(arraySubscriptExpr(
            hasBase(ignoringParenImpCasts(
              anyOf(hasPointerType(), hasArrayType()))),
            unless(hasIndex(integerLiteral(equals(0)))))
            .bind(ArraySubscrTag));
      // clang-format on
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
class PointerArithmeticGadget : public WarningGadget {
  static constexpr const char *const PointerArithmeticTag = "ptrAdd";
  static constexpr const char *const PointerArithmeticPointerTag = "ptrAddPtr";
  const BinaryOperator *PA; // pointer arithmetic expression
  const Expr * Ptr;         // the pointer expression in `PA`

public:
    PointerArithmeticGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::PointerArithmetic),
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


/// A call of a function or method that performs unchecked buffer operations
/// over one of its pointer parameters.
class UnsafeBufferUsageAttrGadget : public WarningGadget {
    constexpr static const char *const OpTag = "call_expr";
    const CallExpr *Op;

public:
    UnsafeBufferUsageAttrGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::UnsafeBufferUsageAttr),
        Op(Result.Nodes.getNodeAs<CallExpr>(OpTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::UnsafeBufferUsageAttr;
  }

  static Matcher matcher() {
    return stmt(callExpr(callee(functionDecl(hasAttr(attr::UnsafeBufferUsage))))
                          .bind(OpTag));
  }
  const Stmt *getBaseStmt() const override { return Op; }

  DeclUseList getClaimedVarUseSites() const override {
    return {};
  }
};
  

// Represents expressions of the form `DRE[*]` in the Unspecified Lvalue
// Context (see `isInUnspecifiedLvalueContext`).
// Note here `[]` is the built-in subscript operator.
class ULCArraySubscriptGadget : public FixableGadget {
private:
  static constexpr const char *const ULCArraySubscriptTag = "ArraySubscriptUnderULC";
  const ArraySubscriptExpr *Node;

public:
  ULCArraySubscriptGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::ULCArraySubscript),
        Node(Result.Nodes.getNodeAs<ArraySubscriptExpr>(ULCArraySubscriptTag)) {
    assert(Node != nullptr && "Expecting a non-null matching result");
  }

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::ULCArraySubscript;
  }

  static Matcher matcher() {
    auto ArrayOrPtr = anyOf(hasPointerType(), hasArrayType());
    auto BaseIsArrayOrPtrDRE =
        hasBase(ignoringParenImpCasts(declRefExpr(ArrayOrPtr)));
    auto Target =
        arraySubscriptExpr(BaseIsArrayOrPtrDRE).bind(ULCArraySubscriptTag);

    return expr(isInUnspecifiedLvalueContext(Target));
  }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;

  virtual const Stmt *getBaseStmt() const override { return Node; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Node->getBase()->IgnoreImpCasts())) {
      return {DRE};
    }
    return {};
  }
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
  DeclUseTracker &operator=(DeclUseTracker &&) = default;

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
static std::tuple<FixableGadgetList, WarningGadgetList, DeclUseTracker>
findGadgets(const Decl *D, const UnsafeBufferUsageHandler &Handler) {

  struct GadgetFinderCallback : MatchFinder::MatchCallback {
    FixableGadgetList FixableGadgets;
    WarningGadgetList WarningGadgets;
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
#define FIXABLE_GADGET(name)                                                           \
      if (Result.Nodes.getNodeAs<Stmt>(#name)) {                               \
        FixableGadgets.push_back(std::make_unique<name ## Gadget>(Result));           \
        NEXT;                                                                  \
      }
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
#define WARNING_GADGET(name)                                                           \
      if (Result.Nodes.getNodeAs<Stmt>(#name)) {                               \
        WarningGadgets.push_back(std::make_unique<name ## Gadget>(Result));           \
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
    stmt(forEveryDescendant(
      eachOf(
      // A `FixableGadget` matcher and a `WarningGadget` matcher should not disable
      // each other (they could if they were put in the same `anyOf` group).
      // We also should make sure no two `FixableGadget` (resp. `WarningGadget`) matchers
      // match for the same node, so that we can group them
      // in one `anyOf` group (for better performance via short-circuiting).
      stmt(anyOf(
#define FIXABLE_GADGET(x)                                                              \
        x ## Gadget::matcher().bind(#x),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
        // Also match DeclStmts because we'll need them when fixing
        // their underlying VarDecls that otherwise don't have
        // any backreferences to DeclStmts.
        declStmt().bind("any_ds")
      )),
      stmt(anyOf(
        // Add Gadget::matcher() for every gadget in the registry.
#define WARNING_GADGET(x)                                                              \
        allOf(x ## Gadget::matcher().bind(#x), notInSafeBufferOptOut(&Handler)),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
        // In parallel, match all DeclRefExprs so that to find out
        // whether there are any uncovered by gadgets.
        declRefExpr(anyOf(hasPointerType(), hasArrayType()), to(varDecl())).bind("any_dre")
      )))
    )),
    &CB
  );
  // clang-format on

  M.match(*D->getBody(), D->getASTContext());

  // Gadgets "claim" variables they're responsible for. Once this loop finishes,
  // the tracker will only track DREs that weren't claimed by any gadgets,
  // i.e. not understood by the analysis.
  for (const auto &G : CB.FixableGadgets) {
    for (const auto *DRE : G->getClaimedVarUseSites()) {
      CB.Tracker.claimUse(DRE);
    }
  }

  return {std::move(CB.FixableGadgets), std::move(CB.WarningGadgets), std::move(CB.Tracker)};
}

struct WarningGadgetSets {
  std::map<const VarDecl *, std::set<std::unique_ptr<WarningGadget>>> byVar;
  // These Gadgets are not related to pointer variables (e. g. temporaries).
  llvm::SmallVector<std::unique_ptr<WarningGadget>, 16> noVar;
};

static WarningGadgetSets
groupWarningGadgetsByVar(WarningGadgetList &&AllUnsafeOperations) {
  WarningGadgetSets result;
  // If some gadgets cover more than one
  // variable, they'll appear more than once in the map.
  for (auto &G : AllUnsafeOperations) {
    DeclUseList ClaimedVarUseSites = G->getClaimedVarUseSites();

    bool AssociatedWithVarDecl = false;
    for (const DeclRefExpr *DRE : ClaimedVarUseSites) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        result.byVar[VD].emplace(std::move(G));
        AssociatedWithVarDecl = true;
      }
    }

    if (!AssociatedWithVarDecl) {
      result.noVar.emplace_back(std::move(G));
      continue;
    }
  }
  return result;
}

struct FixableGadgetSets {
  std::map<const VarDecl *, std::set<std::unique_ptr<FixableGadget>>> byVar;
};

static FixableGadgetSets
groupFixablesByVar(FixableGadgetList &&AllFixableOperations) {
  FixableGadgetSets FixablesForUnsafeVars;
  for (auto &F : AllFixableOperations) {
    DeclUseList DREs = F->getClaimedVarUseSites();

    for (const DeclRefExpr *DRE : DREs) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        FixablesForUnsafeVars.byVar[VD].emplace(std::move(F));
      }
    }
  }
  return FixablesForUnsafeVars;
}

bool clang::internal::anyConflict(
    const SmallVectorImpl<FixItHint> &FixIts, const SourceManager &SM) {
  // A simple interval overlap detection algorithm.  Sorts all ranges by their
  // begin location then finds the first overlap in one pass.
  std::vector<const FixItHint *> All; // a copy of `FixIts`

  for (const FixItHint &H : FixIts)
    All.push_back(&H);
  std::sort(All.begin(), All.end(),
            [&SM](const FixItHint *H1, const FixItHint *H2) {
              return SM.isBeforeInTranslationUnit(H1->RemoveRange.getBegin(),
                                                  H2->RemoveRange.getBegin());
            });

  const FixItHint *CurrHint = nullptr;

  for (const FixItHint *Hint : All) {
    if (!CurrHint ||
        SM.isBeforeInTranslationUnit(CurrHint->RemoveRange.getEnd(),
                                     Hint->RemoveRange.getBegin())) {
      // Either to initialize `CurrHint` or `CurrHint` does not
      // overlap with `Hint`:
      CurrHint = Hint;
    } else
      // In case `Hint` overlaps the `CurrHint`, we found at least one
      // conflict:
      return true;
  }
  return false;
}

std::optional<FixItList>
ULCArraySubscriptGadget::getFixits(const Strategy &S) const {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Node->getBase()->IgnoreImpCasts()))
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      switch (S.lookup(VD)) {
      case Strategy::Kind::Span: {
        // If the index has a negative constant value, we give up as no valid
        // fix-it can be generated:
        const ASTContext &Ctx = // FIXME: we need ASTContext to be passed in!
            VD->getASTContext();
        if (auto ConstVal = Node->getIdx()->getIntegerConstantExpr(Ctx)) {
          if (ConstVal->isNegative())
            return std::nullopt;
        } else if (!Node->getIdx()->getType()->isUnsignedIntegerType())
          return std::nullopt;
        // no-op is a good fix-it, otherwise
        return FixItList{};
      }
      case Strategy::Kind::Wontfix:
      case Strategy::Kind::Iterator:
      case Strategy::Kind::Array:
      case Strategy::Kind::Vector:
        llvm_unreachable("unsupported strategies for FixableGadgets");
      }
    }
  return std::nullopt;
}

// Return the text representation of the given `APInt Val`:
static std::string getAPIntText(APInt Val) {
  SmallVector<char> Txt;
  Val.toString(Txt, 10, true);
  // APInt::toString does not add '\0' to the end of the string for us:
  Txt.push_back('\0');
  return Txt.data();
}

// Return the source location of the last character of the AST `Node`.
template <typename NodeTy>
static SourceLocation getEndCharLoc(const NodeTy *Node, const SourceManager &SM,
                                    const LangOptions &LangOpts) {
  return Lexer::getLocForEndOfToken(Node->getEndLoc(), 1, SM, LangOpts);
}

// Return the source location just past the last character of the AST `Node`.
template <typename NodeTy>
static SourceLocation getPastLoc(const NodeTy *Node, const SourceManager &SM,
                                 const LangOptions &LangOpts) {
  return Lexer::getLocForEndOfToken(Node->getEndLoc(), 0, SM, LangOpts);
}

// Return text representation of an `Expr`.
static StringRef getExprText(const Expr *E, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  SourceLocation LastCharLoc = getPastLoc(E, SM, LangOpts);

  return Lexer::getSourceText(
      CharSourceRange::getCharRange(E->getBeginLoc(), LastCharLoc), SM,
      LangOpts);
}

// For a non-null initializer `Init` of `T *` type, this function returns
// `FixItHint`s producing a list initializer `{Init,  S}` as a part of a fix-it
// to output stream.
// In many cases, this function cannot figure out the actual extent `S`.  It
// then will use a place holder to replace `S` to ask users to fill `S` in.  The
// initializer shall be used to initialize a variable of type `std::span<T>`.
//
// FIXME: Support multi-level pointers
//
// Parameters:
//   `Init` a pointer to the initializer expression
//   `Ctx` a reference to the ASTContext
static FixItList
populateInitializerFixItWithSpan(const Expr *Init, const ASTContext &Ctx,
                                 const StringRef UserFillPlaceHolder) {
  FixItList FixIts{};
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  std::string ExtentText = UserFillPlaceHolder.data();
  StringRef One = "1";

  // Insert `{` before `Init`:
  FixIts.push_back(FixItHint::CreateInsertion(Init->getBeginLoc(), "{"));
  // Try to get the data extent. Break into different cases:
  if (auto CxxNew = dyn_cast<CXXNewExpr>(Init->IgnoreImpCasts())) {
    // In cases `Init` is `new T[n]` and there is no explicit cast over
    // `Init`, we know that `Init` must evaluates to a pointer to `n` objects
    // of `T`. So the extent is `n` unless `n` has side effects.  Similar but
    // simpler for the case where `Init` is `new T`.
    if (const Expr *Ext = CxxNew->getArraySize().value_or(nullptr)) {
      if (!Ext->HasSideEffects(Ctx))
        ExtentText = getExprText(Ext, SM, LangOpts);
    } else if (!CxxNew->isArray())
      // Although the initializer is not allocating a buffer, the pointer
      // variable could still be used in buffer access operations.
      ExtentText = One;
  } else if (const auto *CArrTy = Ctx.getAsConstantArrayType(
                 Init->IgnoreImpCasts()->getType())) {
    // In cases `Init` is of an array type after stripping off implicit casts,
    // the extent is the array size.  Note that if the array size is not a
    // constant, we cannot use it as the extent.
    ExtentText = getAPIntText(CArrTy->getSize());
  } else {
    // In cases `Init` is of the form `&Var` after stripping of implicit
    // casts, where `&` is the built-in operator, the extent is 1.
    if (auto AddrOfExpr = dyn_cast<UnaryOperator>(Init->IgnoreImpCasts()))
      if (AddrOfExpr->getOpcode() == UnaryOperatorKind::UO_AddrOf &&
          isa_and_present<DeclRefExpr>(AddrOfExpr->getSubExpr()))
        ExtentText = One;
    // TODO: we can handle more cases, e.g., `&a[0]`, `&a`, `std::addressof`, and explicit casting, etc.
    // etc.
  }

  SmallString<32> StrBuffer{};
  SourceLocation LocPassInit = getPastLoc(Init, SM, LangOpts);

  StrBuffer.append(", ");
  StrBuffer.append(ExtentText);
  StrBuffer.append("}");
  FixIts.push_back(FixItHint::CreateInsertion(LocPassInit, StrBuffer.str()));
  return FixIts;
}

// For a `VarDecl` of the form `T  * var (= Init)?`, this
// function generates a fix-it for the declaration, which re-declares `var` to
// be of `span<T>` type and transforms the initializer, if present, to a span
// constructor---`span<T> var {Init, Extent}`, where `Extent` may need the user
// to fill in.
//
// FIXME: support Multi-level pointers
//
// Parameters:
//   `D` a pointer the variable declaration node
//   `Ctx` a reference to the ASTContext
// Returns:
//    the generated fix-it
static FixItList fixVarDeclWithSpan(const VarDecl *D, const ASTContext &Ctx,
                                    const StringRef UserFillPlaceHolder) {
  const QualType &SpanEltT = D->getType()->getPointeeType();
  assert(!SpanEltT.isNull() && "Trying to fix a non-pointer type variable!");

  FixItList FixIts{};
  SourceLocation
      ReplacementLastLoc; // the inclusive end location of the replacement
  const SourceManager &SM = Ctx.getSourceManager();

  if (const Expr *Init = D->getInit()) {
    FixItList InitFixIts =
        populateInitializerFixItWithSpan(Init, Ctx, UserFillPlaceHolder);

    if (InitFixIts.empty())
      return {}; // Something wrong with fixing initializer, give up
    // The loc right before the initializer:
    ReplacementLastLoc = Init->getBeginLoc().getLocWithOffset(-1);
    FixIts.insert(FixIts.end(), std::make_move_iterator(InitFixIts.begin()),
                  std::make_move_iterator(InitFixIts.end()));
  } else
    ReplacementLastLoc = getEndCharLoc(D, SM, Ctx.getLangOpts());

  // Producing fix-it for variable declaration---make `D` to be of span type:
  SmallString<32> Replacement;
  raw_svector_ostream OS(Replacement);

  OS << "std::span<" << SpanEltT.getAsString() << "> " << D->getName();
  FixIts.push_back(FixItHint::CreateReplacement(
      SourceRange{D->getBeginLoc(), ReplacementLastLoc}, OS.str()));
  return FixIts;
}

static FixItList fixVariableWithSpan(const VarDecl *VD,
                                     const DeclUseTracker &Tracker,
                                     const ASTContext &Ctx,
                                     UnsafeBufferUsageHandler &Handler) {
  const DeclStmt *DS = Tracker.lookupDecl(VD);
  assert(DS && "Fixing non-local variables not implemented yet!");
  if (!DS->isSingleDecl()) {
    // FIXME: to support handling multiple `VarDecl`s in a single `DeclStmt`
    return{};
  }
  // Currently DS is an unused variable but we'll need it when
  // non-single decls are implemented, where the pointee type name
  // and the '*' are spread around the place.
  (void)DS;

  // FIXME: handle cases where DS has multiple declarations
  return fixVarDeclWithSpan(VD, Ctx, Handler.getUserFillPlaceHolder());
}

static FixItList fixVariable(const VarDecl *VD, Strategy::Kind K,
                             const DeclUseTracker &Tracker,
                             const ASTContext &Ctx,
                             UnsafeBufferUsageHandler &Handler) {
  switch (K) {
  case Strategy::Kind::Span: {
    if (VD->getType()->isPointerType() && VD->isLocalVarDecl())
      return fixVariableWithSpan(VD, Tracker, Ctx, Handler);
    return {};
  }
  case Strategy::Kind::Iterator:
  case Strategy::Kind::Array:
  case Strategy::Kind::Vector:
    llvm_unreachable("Strategy not implemented yet!");
  case Strategy::Kind::Wontfix:
    llvm_unreachable("Invalid strategy!");
  }
  llvm_unreachable("Unknown strategy!");
}

// Returns true iff there exists a `FixItHint` 'h' in `FixIts` such that the
// `RemoveRange` of 'h' overlaps with a macro use.
static bool overlapWithMacro(const FixItList &FixIts) {
  // FIXME: For now we only check if the range (or the first token) is (part of)
  // a macro expansion.  Ideally, we want to check for all tokens in the range.
  return llvm::any_of(FixIts, [](const FixItHint &Hint) {
    auto BeginLoc = Hint.RemoveRange.getBegin();
    if (BeginLoc.isMacroID())
      // If the range (or the first token) is (part of) a macro expansion:
      return true;
    return false;
  });
}

static std::map<const VarDecl *, FixItList>
getFixIts(FixableGadgetSets &FixablesForUnsafeVars, const Strategy &S,
          const DeclUseTracker &Tracker, const ASTContext &Ctx,
          UnsafeBufferUsageHandler &Handler) {
  std::map<const VarDecl *, FixItList> FixItsForVariable;
  for (const auto &[VD, Fixables] : FixablesForUnsafeVars.byVar) {
    FixItsForVariable[VD] = fixVariable(VD, S.lookup(VD), Tracker, Ctx, Handler);
    // If we fail to produce Fix-It for the declaration we have to skip the
    // variable entirely.
    if (FixItsForVariable[VD].empty()) {
      FixItsForVariable.erase(VD);
      continue;
    }
    bool ImpossibleToFix = false;
    llvm::SmallVector<FixItHint, 16> FixItsForVD;
    for (const auto &F : Fixables) {
      std::optional<FixItList> Fixits = F->getFixits(S);
      if (!Fixits) {
        ImpossibleToFix = true;
        break;
      } else {
        const FixItList CorrectFixes = Fixits.value();
        FixItsForVD.insert(FixItsForVD.end(), CorrectFixes.begin(),
                           CorrectFixes.end());
      }
    }
    if (ImpossibleToFix)
      FixItsForVariable.erase(VD);
    else
      FixItsForVariable[VD].insert(FixItsForVariable[VD].end(),
                                   FixItsForVD.begin(), FixItsForVD.end());
    // We conservatively discard fix-its of a variable if
    // a fix-it overlaps with macros; or
    // a fix-it conflicts with another one
    if (overlapWithMacro(FixItsForVariable[VD]) ||
        clang::internal::anyConflict(FixItsForVariable[VD],
                                     Ctx.getSourceManager())) {      
      FixItsForVariable.erase(VD);
    }
  }
  return FixItsForVariable;
}

static Strategy
getNaiveStrategy(const llvm::SmallVectorImpl<const VarDecl *> &UnsafeVars) {
  Strategy S;
  for (const VarDecl *VD : UnsafeVars) {
    S.set(VD, Strategy::Kind::Span);
  }
  return S;
}

void clang::checkUnsafeBufferUsage(const Decl *D,
                                   UnsafeBufferUsageHandler &Handler) {
  assert(D && D->getBody());

  WarningGadgetSets UnsafeOps;
  FixableGadgetSets FixablesForUnsafeVars;
  DeclUseTracker Tracker;

  {
    auto [FixableGadgets, WarningGadgets, TrackerRes] = findGadgets(D, Handler);
    UnsafeOps = groupWarningGadgetsByVar(std::move(WarningGadgets));
    FixablesForUnsafeVars = groupFixablesByVar(std::move(FixableGadgets));
    Tracker = std::move(TrackerRes);
  }

  // Filter out non-local vars and vars with unclaimed DeclRefExpr-s.
  for (auto it = FixablesForUnsafeVars.byVar.cbegin();
       it != FixablesForUnsafeVars.byVar.cend();) {
    // FIXME: Support ParmVarDecl as well.
    if (!it->first->isLocalVarDecl() || Tracker.hasUnclaimedUses(it->first)) {
      it = FixablesForUnsafeVars.byVar.erase(it);
    } else {
      ++it;
    }
  }

  llvm::SmallVector<const VarDecl *, 16> UnsafeVars;
  for (const auto &[VD, ignore] : FixablesForUnsafeVars.byVar)
    UnsafeVars.push_back(VD);

  Strategy NaiveStrategy = getNaiveStrategy(UnsafeVars);
  std::map<const VarDecl *, FixItList> FixItsForVariable =
      getFixIts(FixablesForUnsafeVars, NaiveStrategy, Tracker,
                D->getASTContext(), Handler);

  // FIXME Detect overlapping FixIts.

  for (const auto &G : UnsafeOps.noVar) {
    Handler.handleUnsafeOperation(G->getBaseStmt(), /*IsRelatedToDecl=*/false);
  }

  for (const auto &[VD, WarningGadgets] : UnsafeOps.byVar) {
    auto FixItsIt = FixItsForVariable.find(VD);
    Handler.handleFixableVariable(VD, FixItsIt != FixItsForVariable.end()
                                          ? std::move(FixItsIt->second)
                                          : FixItList{});
    for (const auto &G : WarningGadgets) {
      Handler.handleUnsafeOperation(G->getBaseStmt(), /*IsRelatedToDecl=*/true);
    }
  }
}

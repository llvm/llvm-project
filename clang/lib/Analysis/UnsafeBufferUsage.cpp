//===- UnsafeBufferUsage.cpp - Replace pointers with modern C++ -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <sstream>
#include <queue>

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
                         internal::ASTMatchFinder::BindKind Bind,
                         const bool ignoreUnevaluatedContext)
      : Matcher(Matcher), Finder(Finder), Builder(Builder), Bind(Bind),
        Matches(false), ignoreUnevaluatedContext(ignoreUnevaluatedContext) {}

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

  bool TraverseGenericSelectionExpr(GenericSelectionExpr *Node) {
    // These are unevaluated, except the result expression.
    if(ignoreUnevaluatedContext)
      return TraverseStmt(Node->getResultExpr());
    return VisitorBase::TraverseGenericSelectionExpr(Node);
  }

  bool TraverseUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
    // Unevaluated context.
    if(ignoreUnevaluatedContext)
      return true;
    return VisitorBase::TraverseUnaryExprOrTypeTraitExpr(Node);
  }

  bool TraverseTypeOfExprTypeLoc(TypeOfExprTypeLoc Node) {
    // Unevaluated context.
    if(ignoreUnevaluatedContext)
      return true;
    return VisitorBase::TraverseTypeOfExprTypeLoc(Node);
  }

  bool TraverseDecltypeTypeLoc(DecltypeTypeLoc Node) {
    // Unevaluated context.
    if(ignoreUnevaluatedContext)
      return true;
    return VisitorBase::TraverseDecltypeTypeLoc(Node);
  }

  bool TraverseCXXNoexceptExpr(CXXNoexceptExpr *Node) {
    // Unevaluated context.
    if(ignoreUnevaluatedContext)
      return true;
    return VisitorBase::TraverseCXXNoexceptExpr(Node);
  }

  bool TraverseCXXTypeidExpr(CXXTypeidExpr *Node) {
    // Unevaluated context.
    if(ignoreUnevaluatedContext)
      return true;
    return VisitorBase::TraverseCXXTypeidExpr(Node);
  }

  bool TraverseStmt(Stmt *Node, DataRecursionQueue *Queue = nullptr) {
    if (!Node)
      return true;
    if (!match(*Node))
      return false;
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
  bool ignoreUnevaluatedContext;
};

// Because we're dealing with raw pointers, let's define what we mean by that.
static auto hasPointerType() {
    return hasType(hasCanonicalType(pointerType()));
}

static auto hasArrayType() {
    return hasType(hasCanonicalType(arrayType()));
}

AST_MATCHER_P(Stmt, forEachDescendantEvaluatedStmt, internal::Matcher<Stmt>, innerMatcher) {
  const DynTypedMatcher &DTM = static_cast<DynTypedMatcher>(innerMatcher);

  MatchDescendantVisitor Visitor(&DTM, Finder, Builder, ASTMatchFinder::BK_All, true);
  return Visitor.findMatch(DynTypedNode::create(Node));
}

AST_MATCHER_P(Stmt, forEachDescendantStmt, internal::Matcher<Stmt>, innerMatcher) {
  const DynTypedMatcher &DTM = static_cast<DynTypedMatcher>(innerMatcher);

  MatchDescendantVisitor Visitor(&DTM, Finder, Builder, ASTMatchFinder::BK_All, false);
  return Visitor.findMatch(DynTypedNode::create(Node));
}

// Matches a `Stmt` node iff the node is in a safe-buffer opt-out region
AST_MATCHER_P(Stmt, notInSafeBufferOptOut, const UnsafeBufferUsageHandler *,
              Handler) {
  return !Handler->isSafeBufferOptOut(Node.getBeginLoc());
}

AST_MATCHER_P(CastExpr, castSubExpr, internal::Matcher<Expr>, innerMatcher) {
  return innerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
}

// Matches a `UnaryOperator` whose operator is pre-increment:
AST_MATCHER(UnaryOperator, isPreInc) {
  return Node.getOpcode() == UnaryOperator::Opcode::UO_PreInc;
}

// Returns a matcher that matches any expression 'e' such that `innerMatcher`
// matches 'e' and 'e' is in an Unspecified Lvalue Context.
static auto isInUnspecifiedLvalueContext(internal::Matcher<Expr> innerMatcher) {
  // clang-format off
  return
    expr(anyOf(
      implicitCastExpr(
        hasCastKind(CastKind::CK_LValueToRValue),
        castSubExpr(innerMatcher)),
      binaryOperator(
        hasAnyOperatorName("="),
        hasLHS(innerMatcher)
      )
    ));
// clang-format on
}


// Returns a matcher that matches any expression `e` such that `InnerMatcher`
// matches `e` and `e` is in an Unspecified Pointer Context (UPC).
static internal::Matcher<Stmt>
isInUnspecifiedPointerContext(internal::Matcher<Stmt> InnerMatcher) {
  // A UPC can be
  // 1. an argument of a function call (except the callee has [[unsafe_...]]
  //    attribute), or
  // 2. the operand of a pointer-to-(integer or bool) cast operation; or
  // 3. the operand of a comparator operation; or
  // 4. the operand of a pointer subtraction operation
  //    (i.e., computing the distance between two pointers); or ...

  auto CallArgMatcher =
      callExpr(forEachArgumentWithParam(InnerMatcher,
                  hasPointerType() /* array also decays to pointer type*/),
          unless(callee(functionDecl(hasAttr(attr::UnsafeBufferUsage)))));

  auto CastOperandMatcher =
      castExpr(anyOf(hasCastKind(CastKind::CK_PointerToIntegral),
		     hasCastKind(CastKind::CK_PointerToBoolean)),
	       castSubExpr(allOf(hasPointerType(), InnerMatcher)));

  auto CompOperandMatcher =
      binaryOperator(hasAnyOperatorName("!=", "==", "<", "<=", ">", ">="),
                     eachOf(hasLHS(allOf(hasPointerType(), InnerMatcher)),
                            hasRHS(allOf(hasPointerType(), InnerMatcher))));

  // A matcher that matches pointer subtractions:
  auto PtrSubtractionMatcher =
      binaryOperator(hasOperatorName("-"),
		     // Note that here we need both LHS and RHS to be
		     // pointer. Then the inner matcher can match any of
		     // them:
		     allOf(hasLHS(hasPointerType()),
			   hasRHS(hasPointerType())),
		     eachOf(hasLHS(InnerMatcher),
			    hasRHS(InnerMatcher)));

  return stmt(anyOf(CallArgMatcher, CastOperandMatcher, CompOperandMatcher,
		    PtrSubtractionMatcher));
  // FIXME: any more cases? (UPC excludes the RHS of an assignment.  For now we
  // don't have to check that.)
}

// Returns a matcher that matches any expression 'e' such that `innerMatcher`
// matches 'e' and 'e' is in an unspecified untyped context (i.e the expression
// 'e' isn't evaluated to an RValue). For example, consider the following code:
//    int *p = new int[4];
//    int *q = new int[4];
//    if ((p = q)) {}
//    p = q;
// The expression `p = q` in the conditional of the `if` statement
// `if ((p = q))` is evaluated as an RValue, whereas the expression `p = q;`
// in the assignment statement is in an untyped context.
static internal::Matcher<Stmt>
isInUnspecifiedUntypedContext(internal::Matcher<Stmt> InnerMatcher) {
  // An unspecified context can be
  // 1. A compound statement,
  // 2. The body of an if statement
  // 3. Body of a loop
  auto CompStmt = compoundStmt(forEach(InnerMatcher));
  auto IfStmtThen = ifStmt(hasThen(InnerMatcher));
  auto IfStmtElse = ifStmt(hasElse(InnerMatcher));
  // FIXME: Handle loop bodies.
  return stmt(anyOf(CompStmt, IfStmtThen, IfStmtElse));
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
  /// the current strategy. Returns std::nullopt if the fix cannot be produced;
  /// returns an empty list if no fixes are necessary.
  virtual std::optional<FixItList> getFixits(const Strategy &) const {
    return std::nullopt;
  }

  /// Returns a list of two elements where the first element is the LHS of a pointer assignment
  /// statement and the second element is the RHS. This two-element list represents the fact that
  /// the LHS buffer gets its bounds information from the RHS buffer. This information will be used
  /// later to group all those variables whose types must be modified together to prevent type
  /// mismatches.
  virtual std::optional<std::pair<const VarDecl *, const VarDecl *>>
  getStrategyImplications() const {
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
            unless(hasIndex(
                anyOf(integerLiteral(equals(0)), arrayInitIndexExpr())
             )))
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
  const Expr *Ptr;          // the pointer expression in `PA`

public:
  PointerArithmeticGadget(const MatchFinder::MatchResult &Result)
      : WarningGadget(Kind::PointerArithmetic),
        PA(Result.Nodes.getNodeAs<BinaryOperator>(PointerArithmeticTag)),
        Ptr(Result.Nodes.getNodeAs<Expr>(PointerArithmeticPointerTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::PointerArithmetic;
  }

  static Matcher matcher() {
    auto HasIntegerType = anyOf(hasType(isInteger()), hasType(enumType()));
    auto PtrAtRight =
        allOf(hasOperatorName("+"),
              hasRHS(expr(hasPointerType()).bind(PointerArithmeticPointerTag)),
              hasLHS(HasIntegerType));
    auto PtrAtLeft =
        allOf(anyOf(hasOperatorName("+"), hasOperatorName("-"),
                    hasOperatorName("+="), hasOperatorName("-=")),
              hasLHS(expr(hasPointerType()).bind(PointerArithmeticPointerTag)),
              hasRHS(HasIntegerType));

    return stmt(binaryOperator(anyOf(PtrAtLeft, PtrAtRight))
                    .bind(PointerArithmeticTag));
  }

  const Stmt *getBaseStmt() const override { return PA; }

  DeclUseList getClaimedVarUseSites() const override {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Ptr->IgnoreParenImpCasts())) {
      return {DRE};
    }

    return {};
  }
  // FIXME: pointer adding zero should be fine
  // FIXME: this gadge will need a fix-it
};

/// A pointer initialization expression of the form:
///  \code
///  int *p = q;
///  \endcode
class PointerInitGadget : public FixableGadget {
private:
  static constexpr const char *const PointerInitLHSTag = "ptrInitLHS";
  static constexpr const char *const PointerInitRHSTag = "ptrInitRHS";
  const VarDecl * PtrInitLHS;         // the LHS pointer expression in `PI`
  const DeclRefExpr * PtrInitRHS;         // the RHS pointer expression in `PI`

public:
  PointerInitGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::PointerInit),
    PtrInitLHS(Result.Nodes.getNodeAs<VarDecl>(PointerInitLHSTag)),
    PtrInitRHS(Result.Nodes.getNodeAs<DeclRefExpr>(PointerInitRHSTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::PointerInit;
  }

  static Matcher matcher() {
    auto PtrInitStmt = declStmt(hasSingleDecl(varDecl(
                                 hasInitializer(ignoringImpCasts(declRefExpr(
                                                  hasPointerType()).
                                                  bind(PointerInitRHSTag)))).
                                              bind(PointerInitLHSTag)));

    return stmt(PtrInitStmt);
  }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;

  virtual const Stmt *getBaseStmt() const override { return nullptr; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    return DeclUseList{PtrInitRHS};
  }

  virtual std::optional<std::pair<const VarDecl *, const VarDecl *>>
  getStrategyImplications() const override {
      return std::make_pair(PtrInitLHS,
                            cast<VarDecl>(PtrInitRHS->getDecl()));
  }
};

/// A pointer assignment expression of the form:
///  \code
///  p = q;
///  \endcode
class PointerAssignmentGadget : public FixableGadget {
private:
  static constexpr const char *const PointerAssignLHSTag = "ptrLHS";
  static constexpr const char *const PointerAssignRHSTag = "ptrRHS";
  const DeclRefExpr * PtrLHS;         // the LHS pointer expression in `PA`
  const DeclRefExpr * PtrRHS;         // the RHS pointer expression in `PA`

public:
  PointerAssignmentGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::PointerAssignment),
    PtrLHS(Result.Nodes.getNodeAs<DeclRefExpr>(PointerAssignLHSTag)),
    PtrRHS(Result.Nodes.getNodeAs<DeclRefExpr>(PointerAssignRHSTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::PointerAssignment;
  }

  static Matcher matcher() {
    auto PtrAssignExpr = binaryOperator(allOf(hasOperatorName("="),
      hasRHS(ignoringParenImpCasts(declRefExpr(hasPointerType(),
                                               to(varDecl())).
                                   bind(PointerAssignRHSTag))),
                                   hasLHS(declRefExpr(hasPointerType(),
                                                      to(varDecl())).
                                          bind(PointerAssignLHSTag))));

    return stmt(isInUnspecifiedUntypedContext(PtrAssignExpr));
  }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;

  virtual const Stmt *getBaseStmt() const override { return nullptr; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    return DeclUseList{PtrLHS, PtrRHS};
  }

  virtual std::optional<std::pair<const VarDecl *, const VarDecl *>>
  getStrategyImplications() const override {
    return std::make_pair(cast<VarDecl>(PtrLHS->getDecl()),
                          cast<VarDecl>(PtrRHS->getDecl()));
  }
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

  DeclUseList getClaimedVarUseSites() const override { return {}; }
};

// Represents expressions of the form `DRE[*]` in the Unspecified Lvalue
// Context (see `isInUnspecifiedLvalueContext`).
// Note here `[]` is the built-in subscript operator.
class ULCArraySubscriptGadget : public FixableGadget {
private:
  static constexpr const char *const ULCArraySubscriptTag =
      "ArraySubscriptUnderULC";
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
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(Node->getBase()->IgnoreImpCasts())) {
      return {DRE};
    }
    return {};
  }
};

// Fixable gadget to handle stand alone pointers of the form `UPC(DRE)` in the
// unspecified pointer context (isInUnspecifiedPointerContext). The gadget emits
// fixit of the form `UPC(DRE.data())`.
class UPCStandalonePointerGadget : public FixableGadget {
private:
  static constexpr const char *const DeclRefExprTag = "StandalonePointer";
  const DeclRefExpr *Node;

public:
  UPCStandalonePointerGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::UPCStandalonePointer),
        Node(Result.Nodes.getNodeAs<DeclRefExpr>(DeclRefExprTag)) {
    assert(Node != nullptr && "Expecting a non-null matching result");
  }

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::UPCStandalonePointer;
  }

  static Matcher matcher() {
    auto ArrayOrPtr = anyOf(hasPointerType(), hasArrayType());
    auto target = expr(
        ignoringParenImpCasts(declRefExpr(allOf(ArrayOrPtr, to(varDecl()))).bind(DeclRefExprTag)));
    return stmt(isInUnspecifiedPointerContext(target));
  }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;

  virtual const Stmt *getBaseStmt() const override { return Node; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    return {Node};
  }
};

class PointerDereferenceGadget : public FixableGadget {
  static constexpr const char *const BaseDeclRefExprTag = "BaseDRE";
  static constexpr const char *const OperatorTag = "op";

  const DeclRefExpr *BaseDeclRefExpr = nullptr;
  const UnaryOperator *Op = nullptr;

public:
  PointerDereferenceGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::PointerDereference),
        BaseDeclRefExpr(
            Result.Nodes.getNodeAs<DeclRefExpr>(BaseDeclRefExprTag)),
        Op(Result.Nodes.getNodeAs<UnaryOperator>(OperatorTag)) {}

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::PointerDereference;
  }

  static Matcher matcher() {
    auto Target =
        unaryOperator(
            hasOperatorName("*"),
            has(expr(ignoringParenImpCasts(
                declRefExpr(to(varDecl())).bind(BaseDeclRefExprTag)))))
            .bind(OperatorTag);

    return expr(isInUnspecifiedLvalueContext(Target));
  }

  DeclUseList getClaimedVarUseSites() const override {
    return {BaseDeclRefExpr};
  }

  virtual const Stmt *getBaseStmt() const final { return Op; }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;
};

// Represents expressions of the form `&DRE[any]` in the Unspecified Pointer
// Context (see `isInUnspecifiedPointerContext`).
// Note here `[]` is the built-in subscript operator.
class UPCAddressofArraySubscriptGadget : public FixableGadget {
private:
  static constexpr const char *const UPCAddressofArraySubscriptTag =
      "AddressofArraySubscriptUnderUPC";
  const UnaryOperator *Node; // the `&DRE[any]` node

public:
  UPCAddressofArraySubscriptGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::ULCArraySubscript),
        Node(Result.Nodes.getNodeAs<UnaryOperator>(
            UPCAddressofArraySubscriptTag)) {
    assert(Node != nullptr && "Expecting a non-null matching result");
  }

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::UPCAddressofArraySubscript;
  }

  static Matcher matcher() {
    return expr(isInUnspecifiedPointerContext(expr(ignoringImpCasts(
        unaryOperator(hasOperatorName("&"),
                      hasUnaryOperand(arraySubscriptExpr(
                          hasBase(ignoringParenImpCasts(declRefExpr())))))
            .bind(UPCAddressofArraySubscriptTag)))));
  }

  virtual std::optional<FixItList> getFixits(const Strategy &) const override;

  virtual const Stmt *getBaseStmt() const override { return Node; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    const auto *ArraySubst = cast<ArraySubscriptExpr>(Node->getSubExpr());
    const auto *DRE =
        cast<DeclRefExpr>(ArraySubst->getBase()->IgnoreImpCasts());
    return {DRE};
  }
};
} // namespace

namespace {
// An auxiliary tracking facility for the fixit analysis. It helps connect
// declarations to its uses and make sure we've covered all uses with our
// analysis before we try to fix the declaration.
class DeclUseTracker {
  using UseSetTy = SmallSet<const DeclRefExpr *, 16>;
  using DefMapTy = DenseMap<const VarDecl *, const DeclStmt *>;

  // Allocate on the heap for easier move.
  std::unique_ptr<UseSetTy> Uses{std::make_unique<UseSetTy>()};
  DefMapTy Defs{};

public:
  DeclUseTracker() = default;
  DeclUseTracker(const DeclUseTracker &) = delete; // Let's avoid copies.
  DeclUseTracker &operator=(const DeclUseTracker &) = delete;
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
    Wontfix,  // We don't plan to emit a fixit for this variable.
    Span,     // We recommend replacing the variable with std::span.
    Iterator, // We recommend replacing the variable with std::span::iterator.
    Array,    // We recommend replacing the variable with std::array.
    Vector    // We recommend replacing the variable with std::vector.
  };

private:
  using MapTy = llvm::DenseMap<const VarDecl *, Kind>;

  MapTy Map;

public:
  Strategy() = default;
  Strategy(const Strategy &) = delete; // Let's avoid copies.
  Strategy &operator=(const Strategy &) = delete;
  Strategy(Strategy &&) = default;
  Strategy &operator=(Strategy &&) = default;

  void set(const VarDecl *VD, Kind K) { Map[VD] = K; }

  Kind lookup(const VarDecl *VD) const {
    auto I = Map.find(VD);
    if (I == Map.end())
      return Kind::Wontfix;

    return I->second;
  }
};
} // namespace


// Representing a pointer type expression of the form `++Ptr` in an Unspecified
// Pointer Context (UPC):
class UPCPreIncrementGadget : public FixableGadget {
private:
  static constexpr const char *const UPCPreIncrementTag =
    "PointerPreIncrementUnderUPC";
  const UnaryOperator *Node; // the `++Ptr` node

public:
  UPCPreIncrementGadget(const MatchFinder::MatchResult &Result)
    : FixableGadget(Kind::UPCPreIncrement),
      Node(Result.Nodes.getNodeAs<UnaryOperator>(UPCPreIncrementTag)) {
    assert(Node != nullptr && "Expecting a non-null matching result");
  }

  static bool classof(const Gadget *G) {
    return G->getKind() == Kind::UPCPreIncrement;
  }

  static Matcher matcher() {
    // Note here we match `++Ptr` for any expression `Ptr` of pointer type.
    // Although currently we can only provide fix-its when `Ptr` is a DRE, we
    // can have the matcher be general, so long as `getClaimedVarUseSites` does
    // things right.
    return stmt(isInUnspecifiedPointerContext(expr(ignoringImpCasts(
								    unaryOperator(isPreInc(),
										  hasUnaryOperand(declRefExpr())
										  ).bind(UPCPreIncrementTag)))));
  }

  virtual std::optional<FixItList> getFixits(const Strategy &S) const override;

  virtual const Stmt *getBaseStmt() const override { return Node; }

  virtual DeclUseList getClaimedVarUseSites() const override {
    return {dyn_cast<DeclRefExpr>(Node->getSubExpr())};
  }
};

// Representing a fixable expression of the form `*(ptr + 123)` or `*(123 +
// ptr)`:
class DerefSimplePtrArithFixableGadget : public FixableGadget {
  static constexpr const char *const BaseDeclRefExprTag = "BaseDRE";
  static constexpr const char *const DerefOpTag = "DerefOp";
  static constexpr const char *const AddOpTag = "AddOp";
  static constexpr const char *const OffsetTag = "Offset";

  const DeclRefExpr *BaseDeclRefExpr = nullptr;
  const UnaryOperator *DerefOp = nullptr;
  const BinaryOperator *AddOp = nullptr;
  const IntegerLiteral *Offset = nullptr;

public:
  DerefSimplePtrArithFixableGadget(const MatchFinder::MatchResult &Result)
      : FixableGadget(Kind::DerefSimplePtrArithFixable),
        BaseDeclRefExpr(
            Result.Nodes.getNodeAs<DeclRefExpr>(BaseDeclRefExprTag)),
        DerefOp(Result.Nodes.getNodeAs<UnaryOperator>(DerefOpTag)),
        AddOp(Result.Nodes.getNodeAs<BinaryOperator>(AddOpTag)),
        Offset(Result.Nodes.getNodeAs<IntegerLiteral>(OffsetTag)) {}

  static Matcher matcher() {
    // clang-format off
    auto ThePtr = expr(hasPointerType(),
                       ignoringImpCasts(declRefExpr(to(varDecl())).bind(BaseDeclRefExprTag)));
    auto PlusOverPtrAndInteger = expr(anyOf(
          binaryOperator(hasOperatorName("+"), hasLHS(ThePtr),
                         hasRHS(integerLiteral().bind(OffsetTag)))
                         .bind(AddOpTag),
          binaryOperator(hasOperatorName("+"), hasRHS(ThePtr),
                         hasLHS(integerLiteral().bind(OffsetTag)))
                         .bind(AddOpTag)));
    return isInUnspecifiedLvalueContext(unaryOperator(
        hasOperatorName("*"),
        hasUnaryOperand(ignoringParens(PlusOverPtrAndInteger)))
        .bind(DerefOpTag));
    // clang-format on
  }

  virtual std::optional<FixItList> getFixits(const Strategy &s) const final;

  // TODO remove this method from FixableGadget interface
  virtual const Stmt *getBaseStmt() const final { return nullptr; }

  virtual DeclUseList getClaimedVarUseSites() const final {
    return {BaseDeclRefExpr};
  }
};

/// Scan the function and return a list of gadgets found with provided kits.
static std::tuple<FixableGadgetList, WarningGadgetList, DeclUseTracker>
findGadgets(const Decl *D, const UnsafeBufferUsageHandler &Handler,
            bool EmitSuggestions) {

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
#define FIXABLE_GADGET(name)                                                   \
  if (Result.Nodes.getNodeAs<Stmt>(#name)) {                                   \
    FixableGadgets.push_back(std::make_unique<name##Gadget>(Result));          \
    NEXT;                                                                      \
  }
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
#define WARNING_GADGET(name)                                                   \
  if (Result.Nodes.getNodeAs<Stmt>(#name)) {                                   \
    WarningGadgets.push_back(std::make_unique<name##Gadget>(Result));          \
    NEXT;                                                                      \
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
      stmt(
        forEachDescendantEvaluatedStmt(stmt(anyOf(
          // Add Gadget::matcher() for every gadget in the registry.
#define WARNING_GADGET(x)                                                      \
          allOf(x ## Gadget::matcher().bind(#x),                               \
                notInSafeBufferOptOut(&Handler)),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
            // Avoid a hanging comma.
            unless(stmt())
        )))
    ),
    &CB
  );
  // clang-format on

  if (EmitSuggestions) {
    // clang-format off
    M.addMatcher(
        stmt(
          forEachDescendantStmt(stmt(eachOf(
#define FIXABLE_GADGET(x)                                                      \
            x ## Gadget::matcher().bind(#x),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
            // In parallel, match all DeclRefExprs so that to find out
            // whether there are any uncovered by gadgets.
            declRefExpr(anyOf(hasPointerType(), hasArrayType()),
                        to(varDecl())).bind("any_dre"),
            // Also match DeclStmts because we'll need them when fixing
            // their underlying VarDecls that otherwise don't have
            // any backreferences to DeclStmts.
            declStmt().bind("any_ds")
          )))
      ),
      &CB
    );
    // clang-format on
  }

  M.match(*D->getBody(), D->getASTContext());

  // Gadgets "claim" variables they're responsible for. Once this loop finishes,
  // the tracker will only track DREs that weren't claimed by any gadgets,
  // i.e. not understood by the analysis.
  for (const auto &G : CB.FixableGadgets) {
    for (const auto *DRE : G->getClaimedVarUseSites()) {
      CB.Tracker.claimUse(DRE);
    }
  }

  return {std::move(CB.FixableGadgets), std::move(CB.WarningGadgets),
          std::move(CB.Tracker)};
}

// Compares AST nodes by source locations.
template <typename NodeTy> struct CompareNode {
  bool operator()(const NodeTy *N1, const NodeTy *N2) const {
    return N1->getBeginLoc().getRawEncoding() <
           N2->getBeginLoc().getRawEncoding();
  }
};

struct WarningGadgetSets {
  std::map<const VarDecl *, std::set<const WarningGadget *>,
           // To keep keys sorted by their locations in the map so that the
           // order is deterministic:
           CompareNode<VarDecl>>
      byVar;
  // These Gadgets are not related to pointer variables (e. g. temporaries).
  llvm::SmallVector<const WarningGadget *, 16> noVar;
};

static WarningGadgetSets
groupWarningGadgetsByVar(const WarningGadgetList &AllUnsafeOperations) {
  WarningGadgetSets result;
  // If some gadgets cover more than one
  // variable, they'll appear more than once in the map.
  for (auto &G : AllUnsafeOperations) {
    DeclUseList ClaimedVarUseSites = G->getClaimedVarUseSites();

    bool AssociatedWithVarDecl = false;
    for (const DeclRefExpr *DRE : ClaimedVarUseSites) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        result.byVar[VD].insert(G.get());
        AssociatedWithVarDecl = true;
      }
    }

    if (!AssociatedWithVarDecl) {
      result.noVar.push_back(G.get());
      continue;
    }
  }
  return result;
}

struct FixableGadgetSets {
  std::map<const VarDecl *, std::set<const FixableGadget *>> byVar;
};

static FixableGadgetSets
groupFixablesByVar(FixableGadgetList &&AllFixableOperations) {
  FixableGadgetSets FixablesForUnsafeVars;
  for (auto &F : AllFixableOperations) {
    DeclUseList DREs = F->getClaimedVarUseSites();

    for (const DeclRefExpr *DRE : DREs) {
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        FixablesForUnsafeVars.byVar[VD].insert(F.get());
      }
    }
  }
  return FixablesForUnsafeVars;
}

bool clang::internal::anyConflict(const SmallVectorImpl<FixItHint> &FixIts,
                                  const SourceManager &SM) {
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
PointerAssignmentGadget::getFixits(const Strategy &S) const {
  const auto *LeftVD = cast<VarDecl>(PtrLHS->getDecl());
  const auto *RightVD = cast<VarDecl>(PtrRHS->getDecl());
  switch (S.lookup(LeftVD)) {
    case Strategy::Kind::Span:
      if (S.lookup(RightVD) == Strategy::Kind::Span)
        return FixItList{};
      return std::nullopt;
    case Strategy::Kind::Wontfix:
      return std::nullopt;
    case Strategy::Kind::Iterator:
    case Strategy::Kind::Array:
    case Strategy::Kind::Vector:
      llvm_unreachable("unsupported strategies for FixableGadgets");
  }
  return std::nullopt;
}

std::optional<FixItList>
PointerInitGadget::getFixits(const Strategy &S) const {
  const auto *LeftVD = PtrInitLHS;
  const auto *RightVD = cast<VarDecl>(PtrInitRHS->getDecl());
  switch (S.lookup(LeftVD)) {
    case Strategy::Kind::Span:
      if (S.lookup(RightVD) == Strategy::Kind::Span)
        return FixItList{};
      return std::nullopt;
    case Strategy::Kind::Wontfix:
      return std::nullopt;
    case Strategy::Kind::Iterator:
    case Strategy::Kind::Array:
    case Strategy::Kind::Vector:
    llvm_unreachable("unsupported strategies for FixableGadgets");
  }
  return std::nullopt;
}

std::optional<FixItList>
ULCArraySubscriptGadget::getFixits(const Strategy &S) const {
  if (const auto *DRE =
          dyn_cast<DeclRefExpr>(Node->getBase()->IgnoreImpCasts()))
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

static std::optional<FixItList> // forward declaration
fixUPCAddressofArraySubscriptWithSpan(const UnaryOperator *Node);

std::optional<FixItList>
UPCAddressofArraySubscriptGadget::getFixits(const Strategy &S) const {
  auto DREs = getClaimedVarUseSites();
  const auto *VD = cast<VarDecl>(DREs.front()->getDecl());

  switch (S.lookup(VD)) {
  case Strategy::Kind::Span:
    return fixUPCAddressofArraySubscriptWithSpan(Node);
  case Strategy::Kind::Wontfix:
  case Strategy::Kind::Iterator:
  case Strategy::Kind::Array:
  case Strategy::Kind::Vector:
    llvm_unreachable("unsupported strategies for FixableGadgets");
  }
  return std::nullopt; // something went wrong, no fix-it
}

// FIXME: this function should be customizable through format
static StringRef getEndOfLine() {
  static const char *const EOL = "\n";
  return EOL;
}

// Returns the text indicating that the user needs to provide input there:
std::string getUserFillPlaceHolder(StringRef HintTextToUser = "placeholder") {
  std::string s = std::string("<# ");
  s += HintTextToUser;
  s += " #>";
  return s;
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
static std::optional<SourceLocation>
getEndCharLoc(const NodeTy *Node, const SourceManager &SM,
              const LangOptions &LangOpts) {
  unsigned TkLen = Lexer::MeasureTokenLength(Node->getEndLoc(), SM, LangOpts);
  SourceLocation Loc = Node->getEndLoc().getLocWithOffset(TkLen - 1);

  if (Loc.isValid())
    return Loc;

  return std::nullopt;
}

// Return the source location just past the last character of the AST `Node`.
template <typename NodeTy>
static std::optional<SourceLocation> getPastLoc(const NodeTy *Node,
                                                const SourceManager &SM,
                                                const LangOptions &LangOpts) {
  SourceLocation Loc =
      Lexer::getLocForEndOfToken(Node->getEndLoc(), 0, SM, LangOpts);

  if (Loc.isValid())
    return Loc;

  return std::nullopt;
}

// Return text representation of an `Expr`.
static std::optional<StringRef> getExprText(const Expr *E,
                                            const SourceManager &SM,
                                            const LangOptions &LangOpts) {
  std::optional<SourceLocation> LastCharLoc = getPastLoc(E, SM, LangOpts);

  if (LastCharLoc)
    return Lexer::getSourceText(
        CharSourceRange::getCharRange(E->getBeginLoc(), *LastCharLoc), SM,
        LangOpts);

  return std::nullopt;
}

// Returns the literal text in `SourceRange SR`, if `SR` is a valid range.
static std::optional<StringRef> getRangeText(SourceRange SR,
                                             const SourceManager &SM,
                                             const LangOptions &LangOpts) {
  bool Invalid = false;
  CharSourceRange CSR = CharSourceRange::getCharRange(SR.getBegin(), SR.getEnd());
  StringRef Text = Lexer::getSourceText(CSR, SM, LangOpts, &Invalid);

  if (!Invalid)
    return Text;
  return std::nullopt;
}

// Returns the text of the pointee type of `T` from a `VarDecl` of a pointer
// type. The text is obtained through from `TypeLoc`s.  Since `TypeLoc` does not
// have source ranges of qualifiers ( The `QualifiedTypeLoc` looks hacky too me
// :( ), `Qualifiers` of the pointee type is returned separately through the
// output parameter `QualifiersToAppend`.
static std::optional<std::string>
getPointeeTypeText(const ParmVarDecl *VD, const SourceManager &SM,
                   const LangOptions &LangOpts,
                   std::optional<Qualifiers> *QualifiersToAppend) {
  QualType Ty = VD->getType();
  QualType PteTy;

  assert(Ty->isPointerType() && !Ty->isFunctionPointerType() &&
         "Expecting a VarDecl of type of pointer to object type");
  PteTy = Ty->getPointeeType();
  if (PteTy->hasUnnamedOrLocalType())
    // If the pointee type is unnamed, we can't refer to it
    return std::nullopt;

  TypeLoc TyLoc = VD->getTypeSourceInfo()->getTypeLoc();
  TypeLoc PteTyLoc = TyLoc.getUnqualifiedLoc().getNextTypeLoc();
  SourceLocation VDNameStartLoc = VD->getLocation();

  if (!(VDNameStartLoc.isValid() && PteTyLoc.getSourceRange().isValid())) {
    // We are expecting these locations to be valid. But in some cases, they are
    // not all valid. It is a Clang bug to me and we are not responsible for
    // fixing it.  So we will just give up for now when it happens.
    return std::nullopt;
  }

  // Note that TypeLoc.getEndLoc() returns the begin location of the last token:
  SourceLocation PteEndOfTokenLoc =
      Lexer::getLocForEndOfToken(PteTyLoc.getEndLoc(), 0, SM, LangOpts);

  if (!SM.isBeforeInTranslationUnit(PteEndOfTokenLoc, VDNameStartLoc)) {
    // We only deal with the cases where the source text of the pointee type
    // appears on the left-hand side of the variable identifier completely,
    // including the following forms:
    // `T ident`,
    // `T ident[]`, where `T` is any type.
    // Examples of excluded cases are `T (*ident)[]` or `T ident[][n]`.
    return std::nullopt;
  }
  if (PteTy.hasQualifiers()) {
    // TypeLoc does not provide source ranges for qualifiers (it says it's
    // intentional but seems fishy to me), so we cannot get the full text
    // `PteTy` via source ranges.
    *QualifiersToAppend = PteTy.getQualifiers();
  }
  return getRangeText({PteTyLoc.getBeginLoc(), PteEndOfTokenLoc}, SM, LangOpts)
      ->str();
}

// Returns the text of the name (with qualifiers) of a `FunctionDecl`.
static std::optional<StringRef> getFunNameText(const FunctionDecl *FD,
                                               const SourceManager &SM,
                                               const LangOptions &LangOpts) {
  SourceLocation BeginLoc = FD->getQualifier()
                                ? FD->getQualifierLoc().getBeginLoc()
                                : FD->getNameInfo().getBeginLoc();
  // Note that `FD->getNameInfo().getEndLoc()` returns the begin location of the
  // last token:
  SourceLocation EndLoc = Lexer::getLocForEndOfToken(
      FD->getNameInfo().getEndLoc(), 0, SM, LangOpts);
  SourceRange NameRange{BeginLoc, EndLoc};

  return getRangeText(NameRange, SM, LangOpts);
}

std::optional<FixItList>
DerefSimplePtrArithFixableGadget::getFixits(const Strategy &s) const {
  const VarDecl *VD = dyn_cast<VarDecl>(BaseDeclRefExpr->getDecl());

  if (VD && s.lookup(VD) == Strategy::Kind::Span) {
    ASTContext &Ctx = VD->getASTContext();
    // std::span can't represent elements before its begin()
    if (auto ConstVal = Offset->getIntegerConstantExpr(Ctx))
      if (ConstVal->isNegative())
        return std::nullopt;

    // note that the expr may (oddly) has multiple layers of parens
    // example:
    //   *((..(pointer + 123)..))
    // goal:
    //   pointer[123]
    // Fix-It:
    //   remove '*('
    //   replace ' + ' with '['
    //   replace ')' with ']'

    // example:
    //   *((..(123 + pointer)..))
    // goal:
    //   123[pointer]
    // Fix-It:
    //   remove '*('
    //   replace ' + ' with '['
    //   replace ')' with ']'

    const Expr *LHS = AddOp->getLHS(), *RHS = AddOp->getRHS();
    const SourceManager &SM = Ctx.getSourceManager();
    const LangOptions &LangOpts = Ctx.getLangOpts();
    CharSourceRange StarWithTrailWhitespace =
        clang::CharSourceRange::getCharRange(DerefOp->getOperatorLoc(),
                                             LHS->getBeginLoc());

    std::optional<SourceLocation> LHSLocation = getPastLoc(LHS, SM, LangOpts);
    if (!LHSLocation)
      return std::nullopt;

    CharSourceRange PlusWithSurroundingWhitespace =
        clang::CharSourceRange::getCharRange(*LHSLocation, RHS->getBeginLoc());

    std::optional<SourceLocation> AddOpLocation =
        getPastLoc(AddOp, SM, LangOpts);
    std::optional<SourceLocation> DerefOpLocation =
        getPastLoc(DerefOp, SM, LangOpts);

    if (!AddOpLocation || !DerefOpLocation)
      return std::nullopt;

    CharSourceRange ClosingParenWithPrecWhitespace =
        clang::CharSourceRange::getCharRange(*AddOpLocation, *DerefOpLocation);

    return FixItList{
        {FixItHint::CreateRemoval(StarWithTrailWhitespace),
         FixItHint::CreateReplacement(PlusWithSurroundingWhitespace, "["),
         FixItHint::CreateReplacement(ClosingParenWithPrecWhitespace, "]")}};
  }
  return std::nullopt; // something wrong or unsupported, give up
}

std::optional<FixItList>
PointerDereferenceGadget::getFixits(const Strategy &S) const {
  const VarDecl *VD = cast<VarDecl>(BaseDeclRefExpr->getDecl());
  switch (S.lookup(VD)) {
  case Strategy::Kind::Span: {
    ASTContext &Ctx = VD->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();
    // Required changes: *(ptr); => (ptr[0]); and *ptr; => ptr[0]
    // Deletes the *operand
    CharSourceRange derefRange = clang::CharSourceRange::getCharRange(
        Op->getBeginLoc(), Op->getBeginLoc().getLocWithOffset(1));
    // Inserts the [0]
    std::optional<SourceLocation> EndOfOperand =
        getEndCharLoc(BaseDeclRefExpr, SM, Ctx.getLangOpts());
    if (EndOfOperand) {
      return FixItList{{FixItHint::CreateRemoval(derefRange),
                        FixItHint::CreateInsertion(
                            (*EndOfOperand).getLocWithOffset(1), "[0]")}};
    }
  }
    [[fallthrough]];
  case Strategy::Kind::Iterator:
  case Strategy::Kind::Array:
  case Strategy::Kind::Vector:
    llvm_unreachable("Strategy not implemented yet!");
  case Strategy::Kind::Wontfix:
    llvm_unreachable("Invalid strategy!");
  }

  return std::nullopt;
}

// Generates fix-its replacing an expression of the form UPC(DRE) with
// `DRE.data()`
std::optional<FixItList> UPCStandalonePointerGadget::getFixits(const Strategy &S)
      const {
  const auto VD = cast<VarDecl>(Node->getDecl());
  switch (S.lookup(VD)) {
    case Strategy::Kind::Span: {
      ASTContext &Ctx = VD->getASTContext();
      SourceManager &SM = Ctx.getSourceManager();
      // Inserts the .data() after the DRE
      std::optional<SourceLocation> EndOfOperand =
          getPastLoc(Node, SM, Ctx.getLangOpts());

      if (EndOfOperand)
        return FixItList{{FixItHint::CreateInsertion(
            *EndOfOperand, ".data()")}};
    }
      [[fallthrough]];
    case Strategy::Kind::Wontfix:
    case Strategy::Kind::Iterator:
    case Strategy::Kind::Array:
    case Strategy::Kind::Vector:
      llvm_unreachable("unsupported strategies for FixableGadgets");
  }

  return std::nullopt;
}

// Generates fix-its replacing an expression of the form `&DRE[e]` with
// `&DRE.data()[e]`:
static std::optional<FixItList>
fixUPCAddressofArraySubscriptWithSpan(const UnaryOperator *Node) {
  const auto *ArraySub = cast<ArraySubscriptExpr>(Node->getSubExpr());
  const auto *DRE = cast<DeclRefExpr>(ArraySub->getBase()->IgnoreImpCasts());
  // FIXME: this `getASTContext` call is costly, we should pass the
  // ASTContext in:
  const ASTContext &Ctx = DRE->getDecl()->getASTContext();
  const Expr *Idx = ArraySub->getIdx();
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  std::stringstream SS;
  bool IdxIsLitZero = false;

  if (auto ICE = Idx->getIntegerConstantExpr(Ctx))
    if ((*ICE).isZero())
      IdxIsLitZero = true;
  std::optional<StringRef> DreString = getExprText(DRE, SM, LangOpts);
  if (!DreString)
    return std::nullopt;

  if (IdxIsLitZero) {
    // If the index is literal zero, we produce the most concise fix-it:
    SS << (*DreString).str() << ".data()";
  } else {
    std::optional<StringRef> IndexString = getExprText(Idx, SM, LangOpts);
    if (!IndexString)
      return std::nullopt;

    SS << "&" << (*DreString).str() << ".data()"
       << "[" << (*IndexString).str() << "]";
  }
  return FixItList{
      FixItHint::CreateReplacement(Node->getSourceRange(), SS.str())};
}


std::optional<FixItList> UPCPreIncrementGadget::getFixits(const Strategy &S) const {
  DeclUseList DREs = getClaimedVarUseSites();

  if (DREs.size() != 1)
    return std::nullopt; // In cases of `++Ptr` where `Ptr` is not a DRE, we
                         // give up
  if (const VarDecl *VD = dyn_cast<VarDecl>(DREs.front()->getDecl())) {
    if (S.lookup(VD) == Strategy::Kind::Span) {
      FixItList Fixes;
      std::stringstream SS;
      const Stmt *PreIncNode = getBaseStmt();
      StringRef varName = VD->getName();
      const ASTContext &Ctx = VD->getASTContext();

      // To transform UPC(++p) to UPC((p = p.subspan(1)).data()):
      SS << "(" << varName.data() << " = " << varName.data()
         << ".subspan(1)).data()";
      std::optional<SourceLocation> PreIncLocation =
          getEndCharLoc(PreIncNode, Ctx.getSourceManager(), Ctx.getLangOpts());
      if (!PreIncLocation)
        return std::nullopt;

      Fixes.push_back(FixItHint::CreateReplacement(
          SourceRange(PreIncNode->getBeginLoc(), *PreIncLocation), SS.str()));
      return Fixes;
    }
  }
  return std::nullopt; // Not in the cases that we can handle for now, give up.
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
populateInitializerFixItWithSpan(const Expr *Init, ASTContext &Ctx,
                                 const StringRef UserFillPlaceHolder) {
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();

  // If `Init` has a constant value that is (or equivalent to) a
  // NULL pointer, we use the default constructor to initialize the span
  // object, i.e., a `std:span` variable declaration with no initializer.
  // So the fix-it is just to remove the initializer.
  if (Init->isNullPointerConstant(Ctx,
          // FIXME: Why does this function not ask for `const ASTContext
          // &`? It should. Maybe worth an NFC patch later.
          Expr::NullPointerConstantValueDependence::
              NPC_ValueDependentIsNotNull)) {
    std::optional<SourceLocation> InitLocation =
        getEndCharLoc(Init, SM, LangOpts);
    if (!InitLocation)
      return {};

    SourceRange SR(Init->getBeginLoc(), *InitLocation);

    return {FixItHint::CreateRemoval(SR)};
  }

  FixItList FixIts{};
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
      if (!Ext->HasSideEffects(Ctx)) {
        std::optional<StringRef> ExtentString = getExprText(Ext, SM, LangOpts);
        if (!ExtentString)
          return {};
        ExtentText = *ExtentString;
      }
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
    // TODO: we can handle more cases, e.g., `&a[0]`, `&a`, `std::addressof`,
    // and explicit casting, etc. etc.
  }

  SmallString<32> StrBuffer{};
  std::optional<SourceLocation> LocPassInit = getPastLoc(Init, SM, LangOpts);

  if (!LocPassInit)
    return {};

  StrBuffer.append(", ");
  StrBuffer.append(ExtentText);
  StrBuffer.append("}");
  FixIts.push_back(FixItHint::CreateInsertion(*LocPassInit, StrBuffer.str()));
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
static FixItList fixVarDeclWithSpan(const VarDecl *D, ASTContext &Ctx,
                                    const StringRef UserFillPlaceHolder) {
  const QualType &SpanEltT = D->getType()->getPointeeType();
  assert(!SpanEltT.isNull() && "Trying to fix a non-pointer type variable!");

  FixItList FixIts{};
  std::optional<SourceLocation>
      ReplacementLastLoc; // the inclusive end location of the replacement
  const SourceManager &SM = Ctx.getSourceManager();

  if (const Expr *Init = D->getInit()) {
    FixItList InitFixIts =
        populateInitializerFixItWithSpan(Init, Ctx, UserFillPlaceHolder);

    if (InitFixIts.empty())
      return {};

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

  if (!ReplacementLastLoc)
    return {};

  FixIts.push_back(FixItHint::CreateReplacement(
      SourceRange{D->getBeginLoc(), *ReplacementLastLoc}, OS.str()));
  return FixIts;
}

static bool hasConflictingOverload(const FunctionDecl *FD) {
  return !FD->getDeclContext()->lookup(FD->getDeclName()).isSingleResult();
}

// For a `FunDecl`, one of whose `ParmVarDecl`s is being changed to have a new
// type, this function produces fix-its to make the change self-contained.  Let
// 'F' be the entity defined by the original `FunDecl` and "NewF" be the entity
// defined by the `FunDecl` after the change to the parameter.  Fix-its produced
// by this function are
//   1. Add the `[[clang::unsafe_buffer_usage]]` attribute to each declaration
//   of 'F';
//   2. Create a declaration of "NewF" next to each declaration of `F`;
//   3. Create a definition of "F" (as its' original definition is now belongs
//      to "NewF") next to its original definition.  The body of the creating
//      definition calls to "NewF".
//
// Example:
//
// void f(int *p);  // original declaration
// void f(int *p) { // original definition
//    p[5];
// }
//
// To change the parameter `p` to be of `std::span<int>` type, we
// also add overloads:
//
// [[clang::unsafe_buffer_usage]] void f(int *p); // original decl
// void f(std::span<int> p);                      // added overload decl
// void f(std::span<int> p) {     // original def where param is changed
//    p[5];
// }
// [[clang::unsafe_buffer_usage]] void f(int *p) {  // added def
//   return f(std::span(p, <# size #>));
// }
//
// The actual fix-its may contain more details, e.g., the attribute may be guard
// by a macro
//   #if __has_cpp_attribute(clang::unsafe_buffer_usage)
//   [[clang::unsafe_buffer_usage]]
//   #endif
//
// `NewTypeText` is the string representation of the new type, to which the
// parameter indexed by `ParmIdx` is being changed.
static std::optional<FixItList>
createOverloadsForFixedParams(unsigned ParmIdx, StringRef NewTyText,
                              const FunctionDecl *FD, const ASTContext &Ctx,
                              UnsafeBufferUsageHandler &Handler) {
  // FIXME: need to make this conflict checking better:
  if (hasConflictingOverload(FD))
    return std::nullopt;

  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  // FIXME Respect indentation of the original code.

  // A lambda that creates the text representation of a function declaration
  // with the new type signature:
  const auto NewOverloadSignatureCreator =
      [&SM, &LangOpts](const FunctionDecl *FD, unsigned ParmIdx,
                       StringRef NewTypeText) -> std::optional<std::string> {
    std::stringstream SS;

    SS << ";";
    SS << getEndOfLine().str();
    // Append: ret-type func-name "("
    if (auto Prefix = getRangeText(
            SourceRange(FD->getBeginLoc(), (*FD->param_begin())->getBeginLoc()),
            SM, LangOpts))
      SS << Prefix->str();
    else
      return std::nullopt; // give up
    // Append: parameter-type-list
    const unsigned NumParms = FD->getNumParams();

    for (unsigned i = 0; i < NumParms; i++) {
      const ParmVarDecl *Parm = FD->getParamDecl(i);

      if (Parm->isImplicit())
        continue;
      if (i == ParmIdx) {
        SS << NewTypeText.str();
        // print parameter name if provided:
        if (IdentifierInfo * II = Parm->getIdentifier())
          SS << " " << II->getName().str();
      } else if (auto ParmTypeText =
                     getRangeText(Parm->getSourceRange(), SM, LangOpts)) {
        // print the whole `Parm` without modification:
        SS << ParmTypeText->str();
      } else
        return std::nullopt; // something wrong, give up
      if (i != NumParms - 1)
        SS << ", ";
    }
    SS << ")";
    return SS.str();
  };

  // A lambda that creates the text representation of a function definition with
  // the original signature:
  const auto OldOverloadDefCreator =
      [&SM, &Handler,
       &LangOpts](const FunctionDecl *FD, unsigned ParmIdx,
                  StringRef NewTypeText) -> std::optional<std::string> {
    std::stringstream SS;

    SS << getEndOfLine().str();
    // Append: attr-name ret-type func-name "(" param-list ")" "{"
    if (auto FDPrefix = getRangeText(
            SourceRange(FD->getBeginLoc(), FD->getBody()->getBeginLoc()), SM,
            LangOpts))
      SS << Handler.getUnsafeBufferUsageAttributeTextAt(FD->getBeginLoc(), " ")
         << FDPrefix->str() << "{";
    else
      return std::nullopt;
    // Append: "return" func-name "("
    if (auto FunQualName = getFunNameText(FD, SM, LangOpts))
      SS << "return " << FunQualName->str() << "(";
    else
      return std::nullopt;

    // Append: arg-list
    const unsigned NumParms = FD->getNumParams();
    for (unsigned i = 0; i < NumParms; i++) {
      const ParmVarDecl *Parm = FD->getParamDecl(i);

      if (Parm->isImplicit())
        continue;
      // FIXME: If a parameter has no name, it is unused in the
      // definition. So we could just leave it as it is.
      if (!Parm->getIdentifier())
        // If a parameter of a function definition has no name:
        return std::nullopt;
      if (i == ParmIdx)
        // This is our spanified paramter!
        SS << NewTypeText.str() << "(" << Parm->getIdentifier()->getName().str() << ", "
           << getUserFillPlaceHolder("size") << ")";
      else
        SS << Parm->getIdentifier()->getName().str();
      if (i != NumParms - 1)
        SS << ", ";
    }
    // finish call and the body
    SS << ");}" << getEndOfLine().str();
    // FIXME: 80-char line formatting?
    return SS.str();
  };

  FixItList FixIts{};
  for (FunctionDecl *FReDecl : FD->redecls()) {
    std::optional<SourceLocation> Loc = getPastLoc(FReDecl, SM, LangOpts);

    if (!Loc)
      return {};
    if (FReDecl->isThisDeclarationADefinition()) {
      assert(FReDecl == FD && "inconsistent function definition");
      // Inserts a definition with the old signature to the end of
      // `FReDecl`:
      if (auto OldOverloadDef =
              OldOverloadDefCreator(FReDecl, ParmIdx, NewTyText))
        FixIts.emplace_back(FixItHint::CreateInsertion(*Loc, *OldOverloadDef));
      else
        return {}; // give up
    } else {
      // Adds the unsafe-buffer attribute (if not already there) to `FReDecl`:
      if (!FReDecl->hasAttr<UnsafeBufferUsageAttr>()) {
        FixIts.emplace_back(FixItHint::CreateInsertion(
            FReDecl->getBeginLoc(), Handler.getUnsafeBufferUsageAttributeTextAt(
                                        FReDecl->getBeginLoc(), " ")));
      }
      // Inserts a declaration with the new signature to the end of `FReDecl`:
      if (auto NewOverloadDecl =
              NewOverloadSignatureCreator(FReDecl, ParmIdx, NewTyText))
        FixIts.emplace_back(FixItHint::CreateInsertion(*Loc, *NewOverloadDecl));
      else
        return {};
    }
  }
  return FixIts;
}

// To fix a `ParmVarDecl` to be of `std::span` type.  In addition, creating a
// new overload of the function so that the change is self-contained (see
// `createOverloadsForFixedParams`).
static FixItList fixParamWithSpan(const ParmVarDecl *PVD, const ASTContext &Ctx,
                                  UnsafeBufferUsageHandler &Handler) {
  if (PVD->hasDefaultArg())
    // FIXME: generate fix-its for default values:
    return {};
  assert(PVD->getType()->isPointerType());
  auto *FD = dyn_cast<FunctionDecl>(PVD->getDeclContext());

  if (!FD)
    return {};

  std::optional<Qualifiers> PteTyQualifiers = std::nullopt;
  std::optional<std::string> PteTyText = getPointeeTypeText(
      PVD, Ctx.getSourceManager(), Ctx.getLangOpts(), &PteTyQualifiers);

  if (!PteTyText)
    return {};

  std::optional<StringRef> PVDNameText = PVD->getIdentifier()->getName();

  if (!PVDNameText)
    return {};

  std::string SpanOpen = "std::span<";
  std::string SpanClose = ">";
  std::string SpanTyText;
  std::stringstream SS;

  SS << SpanOpen << *PteTyText;
  // Append qualifiers to span element type:
  if (PteTyQualifiers)
    SS << " " << PteTyQualifiers->getAsString();
  SS << SpanClose;
  // Save the Span type text:
  SpanTyText = SS.str();
  // Append qualifiers to the type of the parameter:
  if (PVD->getType().hasQualifiers())
    SS << " " << PVD->getType().getQualifiers().getAsString();
  // Append parameter's name:
  SS << " " << PVDNameText->str();

  FixItList Fixes;
  unsigned ParmIdx = 0;

  Fixes.push_back(
      FixItHint::CreateReplacement(PVD->getSourceRange(), SS.str()));
  for (auto *ParmIter : FD->parameters()) {
    if (PVD == ParmIter)
      break;
    ParmIdx++;
  }
  if (ParmIdx < FD->getNumParams())
    if (auto OverloadFix = createOverloadsForFixedParams(ParmIdx, SpanTyText,
                                                         FD, Ctx, Handler)) {
      Fixes.append(*OverloadFix);
      return Fixes;
    }
  return {};
}

static FixItList fixVariableWithSpan(const VarDecl *VD,
                                     const DeclUseTracker &Tracker,
                                     ASTContext &Ctx,
                                     UnsafeBufferUsageHandler &Handler) {
  const DeclStmt *DS = Tracker.lookupDecl(VD);
  assert(DS && "Fixing non-local variables not implemented yet!");
  if (!DS->isSingleDecl()) {
    // FIXME: to support handling multiple `VarDecl`s in a single `DeclStmt`
    return {};
  }
  // Currently DS is an unused variable but we'll need it when
  // non-single decls are implemented, where the pointee type name
  // and the '*' are spread around the place.
  (void)DS;

  // FIXME: handle cases where DS has multiple declarations
  return fixVarDeclWithSpan(VD, Ctx, getUserFillPlaceHolder());
}

// TODO: we should be consistent to use `std::nullopt` to represent no-fix due
// to any unexpected problem.
static FixItList
fixVariable(const VarDecl *VD, Strategy::Kind K,
            /* The function decl under analysis */ const Decl *D,
            const DeclUseTracker &Tracker, ASTContext &Ctx,
            UnsafeBufferUsageHandler &Handler) {
  if (const auto *PVD = dyn_cast<ParmVarDecl>(VD)) {
    auto *FD = dyn_cast<clang::FunctionDecl>(PVD->getDeclContext());
    if (!FD || FD != D)
      // `FD != D` means that `PVD` belongs to a function that is not being
      // analyzed currently.  Thus `FD` may not be complete.
      return {};

    // TODO If function has a try block we can't change params unless we check
    // also its catch block for their use.
    // FIXME We might support static class methods, some select methods,
    // operators and possibly lamdas.
    if (FD->isMain() || FD->isConstexpr() ||
        FD->getTemplatedKind() != FunctionDecl::TemplatedKind::TK_NonTemplate ||
        FD->isVariadic() ||
        // also covers call-operator of lamdas
        isa<CXXMethodDecl>(FD) ||
        // skip when the function body is a try-block
        (FD->hasBody() && isa<CXXTryStmt>(FD->getBody())) ||
        FD->isOverloadedOperator())
      return {}; // TODO test all these cases
  }

  switch (K) {
  case Strategy::Kind::Span: {
    if (VD->getType()->isPointerType()) {
      if (const auto *PVD = dyn_cast<ParmVarDecl>(VD))
        return fixParamWithSpan(PVD, Ctx, Handler);

      if (VD->isLocalVarDecl())
        return fixVariableWithSpan(VD, Tracker, Ctx, Handler);
    }
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
    auto Range = Hint.RemoveRange;
    if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
      // If the range (or the first token) is (part of) a macro expansion:
      return true;
    return false;
  });
}

static bool impossibleToFixForVar(const FixableGadgetSets &FixablesForAllVars,
                                  const Strategy &S,
                                  const VarDecl * Var) {
  for (const auto &F : FixablesForAllVars.byVar.find(Var)->second) {
    std::optional<FixItList> Fixits = F->getFixits(S);
    if (!Fixits) {
      return true;
    }
  }
  return false;
}

static std::map<const VarDecl *, FixItList>
getFixIts(FixableGadgetSets &FixablesForAllVars, const Strategy &S,
	  ASTContext &Ctx,
          /* The function decl under analysis */ const Decl *D,
          const DeclUseTracker &Tracker, UnsafeBufferUsageHandler &Handler,
          const DefMapTy &VarGrpMap) {
  std::map<const VarDecl *, FixItList> FixItsForVariable;
  for (const auto &[VD, Fixables] : FixablesForAllVars.byVar) {
    FixItsForVariable[VD] =
        fixVariable(VD, S.lookup(VD), D, Tracker, Ctx, Handler);
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

    if (ImpossibleToFix) {
      FixItsForVariable.erase(VD);
      continue;
    }

    const auto VarGroupForVD = VarGrpMap.find(VD);
    if (VarGroupForVD != VarGrpMap.end()) {
      for (const VarDecl * V : VarGroupForVD->second) {
        if (V == VD) {
          continue;
        }
        if (impossibleToFixForVar(FixablesForAllVars, S, V)) {
          ImpossibleToFix = true;
          break;
        }
      }

      if (ImpossibleToFix) {
        FixItsForVariable.erase(VD);
        for (const VarDecl * V : VarGroupForVD->second) {
          FixItsForVariable.erase(V);
        }
        continue;
      }
    }
    FixItsForVariable[VD].insert(FixItsForVariable[VD].end(),
                                 FixItsForVD.begin(), FixItsForVD.end());

    // Fix-it shall not overlap with macros or/and templates:
    if (overlapWithMacro(FixItsForVariable[VD]) ||
        clang::internal::anyConflict(FixItsForVariable[VD],
                                     Ctx.getSourceManager())) {
      FixItsForVariable.erase(VD);
      continue;
    }
  }

  for (auto VD : FixItsForVariable) {
    const auto VarGroupForVD = VarGrpMap.find(VD.first);
    const Strategy::Kind ReplacementTypeForVD = S.lookup(VD.first);
    if (VarGroupForVD != VarGrpMap.end()) {
      for (const VarDecl * Var : VarGroupForVD->second) {
        if (Var == VD.first) {
          continue;
        }

        FixItList GroupFix;
        if (FixItsForVariable.find(Var) == FixItsForVariable.end()) {
          GroupFix = fixVariable(Var, ReplacementTypeForVD, D, Tracker,
                                 Var->getASTContext(), Handler);
        } else {
          GroupFix = FixItsForVariable[Var];
        }

        for (auto Fix : GroupFix) {
          FixItsForVariable[VD.first].push_back(Fix);
        }
      }
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
                                   UnsafeBufferUsageHandler &Handler,
                                   bool EmitSuggestions) {
  assert(D && D->getBody());

  // We do not want to visit a Lambda expression defined inside a method independently.
  // Instead, it should be visited along with the outer method.
  if (const auto *fd = dyn_cast<CXXMethodDecl>(D)) {
    if (fd->getParent()->isLambda() && fd->getParent()->isLocalClass())
      return;
  }

  // Do not emit fixit suggestions for functions declared in an
  // extern "C" block.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
      for (FunctionDecl *FReDecl : FD->redecls()) {
      if (FReDecl->isExternC()) {
        EmitSuggestions = false;
        break;
      }
    }
  }

  WarningGadgetSets UnsafeOps;
  FixableGadgetSets FixablesForAllVars;

  auto [FixableGadgets, WarningGadgets, Tracker] =
    findGadgets(D, Handler, EmitSuggestions);

  if (!EmitSuggestions) {
    // Our job is very easy without suggestions. Just warn about
    // every problematic operation and consider it done. No need to deal
    // with fixable gadgets, no need to group operations by variable.
    for (const auto &G : WarningGadgets) {
      Handler.handleUnsafeOperation(G->getBaseStmt(),
                                    /*IsRelatedToDecl=*/false);
    }

    // This return guarantees that most of the machine doesn't run when
    // suggestions aren't requested.
    assert(FixableGadgets.size() == 0 &&
           "Fixable gadgets found but suggestions not requested!");
    return;
  }

  UnsafeOps = groupWarningGadgetsByVar(std::move(WarningGadgets));
  FixablesForAllVars = groupFixablesByVar(std::move(FixableGadgets));

  std::map<const VarDecl *, FixItList> FixItsForVariableGroup;
  DefMapTy VariableGroupsMap{};

  // Filter out non-local vars and vars with unclaimed DeclRefExpr-s.
  for (auto it = FixablesForAllVars.byVar.cbegin();
       it != FixablesForAllVars.byVar.cend();) {
      // FIXME: need to deal with global variables later
      if ((!it->first->isLocalVarDecl() && !isa<ParmVarDecl>(it->first)) ||
          Tracker.hasUnclaimedUses(it->first) || it->first->isInitCapture()) {
      it = FixablesForAllVars.byVar.erase(it);
    } else {
      ++it;
    }
  }

  llvm::SmallVector<const VarDecl *, 16> UnsafeVars;
  for (const auto &[VD, ignore] : FixablesForAllVars.byVar)
    UnsafeVars.push_back(VD);

  // Fixpoint iteration for pointer assignments
  using DepMapTy = DenseMap<const VarDecl *, std::set<const VarDecl *>>;
  DepMapTy DependenciesMap{};
  DepMapTy PtrAssignmentGraph{};

  for (auto it : FixablesForAllVars.byVar) {
    for (const FixableGadget *fixable : it.second) {
      std::optional<std::pair<const VarDecl *, const VarDecl *>> ImplPair =
                                  fixable->getStrategyImplications();
      if (ImplPair) {
        std::pair<const VarDecl *, const VarDecl *> Impl = ImplPair.value();
        PtrAssignmentGraph[Impl.first].insert(Impl.second);
      }
    }
  }

  /*
   The following code does a BFS traversal of the `PtrAssignmentGraph`
   considering all unsafe vars as starting nodes and constructs an undirected
   graph `DependenciesMap`. Constructing the `DependenciesMap` in this manner
   elimiates all variables that are unreachable from any unsafe var. In other
   words, this removes all dependencies that don't include any unsafe variable
   and consequently don't need any fixit generation.
   Note: A careful reader would observe that the code traverses
   `PtrAssignmentGraph` using `CurrentVar` but adds edges between `Var` and
   `Adj` and not between `CurrentVar` and `Adj`. Both approaches would
   achieve the same result but the one used here dramatically cuts the
   amount of hoops the second part of the algorithm needs to jump, given that
   a lot of these connections become "direct". The reader is advised not to
   imagine how the graph is transformed because of using `Var` instead of
   `CurrentVar`. The reader can continue reading as if `CurrentVar` was used,
   and think about why it's equivalent later.
   */
  std::set<const VarDecl *> VisitedVarsDirected{};
  for (const auto &[Var, ignore] : UnsafeOps.byVar) {
    if (VisitedVarsDirected.find(Var) == VisitedVarsDirected.end()) {

      std::queue<const VarDecl*> QueueDirected{};
      QueueDirected.push(Var);
      while(!QueueDirected.empty()) {
        const VarDecl* CurrentVar = QueueDirected.front();
        QueueDirected.pop();
        VisitedVarsDirected.insert(CurrentVar);
        auto AdjacentNodes = PtrAssignmentGraph[CurrentVar];
        for (const VarDecl *Adj : AdjacentNodes) {
          if (VisitedVarsDirected.find(Adj) == VisitedVarsDirected.end()) {
            QueueDirected.push(Adj);
          }
          DependenciesMap[Var].insert(Adj);
          DependenciesMap[Adj].insert(Var);
        }
      }
    }
  }

  // Group Connected Components for Unsafe Vars
  // (Dependencies based on pointer assignments)
  std::set<const VarDecl *> VisitedVars{};
  for (const auto &[Var, ignore] : UnsafeOps.byVar) {
    if (VisitedVars.find(Var) == VisitedVars.end()) {
      std::vector<const VarDecl *> VarGroup{};

      std::queue<const VarDecl*> Queue{};
      Queue.push(Var);
      while(!Queue.empty()) {
        const VarDecl* CurrentVar = Queue.front();
        Queue.pop();
        VisitedVars.insert(CurrentVar);
        VarGroup.push_back(CurrentVar);
        auto AdjacentNodes = DependenciesMap[CurrentVar];
        for (const VarDecl *Adj : AdjacentNodes) {
          if (VisitedVars.find(Adj) == VisitedVars.end()) {
            Queue.push(Adj);
          }
        }
      }
      for (const VarDecl * V : VarGroup) {
        if (UnsafeOps.byVar.find(V) != UnsafeOps.byVar.end()) {
          VariableGroupsMap[V] = VarGroup;
        }
      }
    }
  }

  Strategy NaiveStrategy = getNaiveStrategy(UnsafeVars);

  FixItsForVariableGroup =
      getFixIts(FixablesForAllVars, NaiveStrategy, D->getASTContext(), D,
                Tracker, Handler, VariableGroupsMap);

  // FIXME Detect overlapping FixIts.

  for (const auto &G : UnsafeOps.noVar) {
    Handler.handleUnsafeOperation(G->getBaseStmt(), /*IsRelatedToDecl=*/false);
  }

  for (const auto &[VD, WarningGadgets] : UnsafeOps.byVar) {
    auto FixItsIt = FixItsForVariableGroup.find(VD);
    Handler.handleUnsafeVariableGroup(VD, VariableGroupsMap,
                                      FixItsIt != FixItsForVariableGroup.end()
                                      ? std::move(FixItsIt->second)
                                      : FixItList{});
    for (const auto &G : WarningGadgets) {
      Handler.handleUnsafeOperation(G->getBaseStmt(), /*IsRelatedToDecl=*/true);
    }
  }
}

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

// Because we're dealing with raw pointers, let's define what we mean by that.
static auto hasPointerType() {
  return anyOf(hasType(pointerType()),
               hasType(autoType(hasDeducedType(
                   hasUnqualifiedDesugaredType(pointerType())))));
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
};
} // namespace

// Scan the function and return a list of gadgets found with provided kits.
static GadgetList findGadgets(const Decl *D) {

  class GadgetFinderCallback : public MatchFinder::MatchCallback {
    GadgetList &Output;

  public:
    GadgetFinderCallback(GadgetList &Output) : Output(Output) {}

    void run(const MatchFinder::MatchResult &Result) override {
      // In debug mode, assert that we've found exactly one gadget.
      // This helps us avoid conflicts in .bind() tags.
#if NDEBUG
#define NEXT return
#else
      int numFound = 0;
#define NEXT ++numFound
#endif

      // Figure out which matcher we've found, and call the appropriate
      // subclass constructor.
      // FIXME: Can we do this more logarithmically?
#define GADGET(name)                                                           \
      if (Result.Nodes.getNodeAs<Stmt>(#name)) {                               \
        Output.push_back(std::make_unique<name ## Gadget>(Result));            \
        NEXT;                                                                  \
      }
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"

      assert(numFound >= 1 && "Gadgets not found in match result!");
      assert(numFound <= 1 && "Conflicting bind tags in gadgets!");
      (void)numFound;
    }
  };

  GadgetList G;
  MatchFinder M;
  GadgetFinderCallback CB(G);

  // clang-format off
  M.addMatcher(
    stmt(forEachDescendant(
      stmt(anyOf(
        // Add Gadget::matcher() for every gadget in the registry.
#define GADGET(x)                                                              \
        x ## Gadget::matcher().bind(#x),
#include "clang/Analysis/Analyses/UnsafeBufferUsageGadgets.def"
        // FIXME: Is there a better way to avoid hanging comma?
        unless(stmt())
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

  return G; // NRVO!
}

void clang::checkUnsafeBufferUsage(const Decl *D,
                                   UnsafeBufferUsageHandler &Handler) {
  assert(D && D->getBody());

  GadgetList Gadgets = findGadgets(D);
  for (const auto &G : Gadgets) {
    Handler.handleUnsafeOperation(G->getBaseStmt());
  }
}

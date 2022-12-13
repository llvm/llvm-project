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
// TODO: Better abstractions over gadgets.
using GadgetList = std::vector<const Stmt *>;
}

// Scan the function and return a list of gadgets found with provided kits.
static GadgetList findGadgets(const Decl *D) {

  class GadgetFinderCallback : public MatchFinder::MatchCallback {
    GadgetList &Output;

  public:
    GadgetFinderCallback(GadgetList &Output) : Output(Output) {}

    void run(const MatchFinder::MatchResult &Result) override {
      Output.push_back(Result.Nodes.getNodeAs<Stmt>("root_node"));
    }
  };

  GadgetList G;
  MatchFinder M;

  auto IncrementMatcher = unaryOperator(
    hasOperatorName("++"),
    hasUnaryOperand(hasType(pointerType()))
  );
  auto DecrementMatcher = unaryOperator(
    hasOperatorName("--"),
    hasUnaryOperand(hasType(pointerType()))
  );

  GadgetFinderCallback CB(G);

  M.addMatcher(
      stmt(forEachDescendant(
        stmt(
          anyOf(
            IncrementMatcher,
            DecrementMatcher
            /* Fill me in! */
          )
          // FIXME: Idiomatically there should be a forCallable(equalsNode(D))
          // here, to make sure that the statement actually belongs to the
          // function and not to a nested function. However, forCallable uses
          // ParentMap which can't be used before the AST is fully constructed.
          // The original problem doesn't sound like it needs ParentMap though,
          // maybe there's a more direct solution?
        ).bind("root_node")
      )), &CB);

  M.match(*D->getBody(), D->getASTContext());

  return G; // NRVO!
}

void clang::checkUnsafeBufferUsage(const Decl *D,
                                   UnsafeBufferUsageHandler &Handler) {
  assert(D && D->getBody());

  GadgetList G = findGadgets(D);
  for (const Stmt *S : G) {
    Handler.handleUnsafeOperation(S);
  }
}

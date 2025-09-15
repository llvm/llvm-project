//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleWorkItemBarrierCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::altera {

void SingleWorkItemBarrierCheck::registerMatchers(MatchFinder *Finder) {
  // Find any function that calls barrier but does not call an ID function.
  // hasAttr(attr::Kind::DeviceKernel) restricts it to only kernel functions.
  // FIXME: Have it accept all functions but check for a parameter that gets an
  // ID from one of the four ID functions.
  Finder->addMatcher(
      // Find function declarations...
      functionDecl(
          // That are device kernels...
          hasAttr(attr::Kind::DeviceKernel),
          // And call a barrier function (either 1.x or 2.x version)...
          forEachDescendant(callExpr(callee(functionDecl(hasAnyName(
                                         "barrier", "work_group_barrier"))))
                                .bind("barrier")),
          // But do not call an ID function.
          unless(hasDescendant(callExpr(callee(functionDecl(
              hasAnyName("get_global_id", "get_local_id", "get_group_id",
                         "get_local_linear_id")))))))
          .bind("function"),
      this);
}

void SingleWorkItemBarrierCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("function");
  const auto *MatchedBarrier = Result.Nodes.getNodeAs<CallExpr>("barrier");
  if (AOCVersion < 1701) {
    // get_group_id and get_local_linear_id were added at/after v17.01
    diag(MatchedDecl->getLocation(),
         "kernel function %0 does not call 'get_global_id' or 'get_local_id' "
         "and will be treated as a single work-item")
        << MatchedDecl;
    diag(MatchedBarrier->getBeginLoc(),
         "barrier call is in a single work-item and may error out",
         DiagnosticIDs::Note);
  } else {
    // If reqd_work_group_size is anything other than (1,1,1), it will be
    // interpreted as an NDRange in AOC version >= 17.1.
    bool IsNDRange = false;
    if (MatchedDecl->hasAttr<ReqdWorkGroupSizeAttr>()) {
      const auto *Attribute = MatchedDecl->getAttr<ReqdWorkGroupSizeAttr>();
      auto Eval = [&](Expr *E) {
        return E->EvaluateKnownConstInt(MatchedDecl->getASTContext())
            .getExtValue();
      };
      if (Eval(Attribute->getXDim()) > 1 || Eval(Attribute->getYDim()) > 1 ||
          Eval(Attribute->getZDim()) > 1)
        IsNDRange = true;
    }
    if (IsNDRange) // No warning if kernel is treated as an NDRange.
      return;
    diag(MatchedDecl->getLocation(),
         "kernel function %0 does not call an ID function and may be a viable "
         "single work-item, but will be forced to execute as an NDRange")
        << MatchedDecl;
    diag(MatchedBarrier->getBeginLoc(),
         "barrier call will force NDRange execution; if single work-item "
         "semantics are desired a mem_fence may be more efficient",
         DiagnosticIDs::Note);
  }
}

void SingleWorkItemBarrierCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AOCVersion", AOCVersion);
}

} // namespace clang::tidy::altera

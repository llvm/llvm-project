//===--- ConvertTernaryIf.h ------------------------------------*- C++ -*-===//
//
// This file declares the refactoring logic that converts
// ternary operators (?:) into if/else statements and vice versa.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CONVERT_TERNARY_IF_H
#define LLVM_CLANG_CONVERT_TERNARY_IF_H

#include "clang/AST/AST.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace clang {
namespace convertternary {

class ConvertTernaryIfCallback
    : public ast_matchers::MatchFinder::MatchCallback {
public:
  ConvertTernaryIfCallback(Rewriter &R)
	  : TheRewriter(R), IsInitialized(false) {}

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  Rewriter &TheRewriter;
  bool IsInitialized;
    };

void setupMatchers(ast_matchers::MatchFinder &Finder,
                   ConvertTernaryIfCallback &Callback);

} // namespace convertternary
} // namespace clang

#endif


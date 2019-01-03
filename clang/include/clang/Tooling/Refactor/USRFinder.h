//===--- USRFinder.h - Clang refactoring library --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Methods for determining the USR of a symbol at a location in source
/// code.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_USR_FINDER_H
#define LLVM_CLANG_TOOLING_REFACTOR_USR_FINDER_H

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <string>
#include <vector>

namespace clang {

class ASTContext;
class Decl;
class SourceLocation;
class NamedDecl;

namespace tooling {
namespace rename {

using llvm::StringRef;
using namespace clang::ast_matchers;

// Given an AST context and a point, returns a NamedDecl identifying the symbol
// at the point. Returns null if nothing is found at the point.
const NamedDecl *getNamedDeclAt(const ASTContext &Context,
                                SourceLocation Point);

/// Returns a \c NamedDecl that corresponds to the given \p USR in the given
/// AST context. Returns null if there's no declaration that matches the given
/// \p USR.
const NamedDecl *getNamedDeclWithUSR(const ASTContext &Context, StringRef USR);

// Converts a Decl into a USR.
std::string getUSRForDecl(const Decl *Decl);

// FIXME: Implement RecursiveASTVisitor<T>::VisitNestedNameSpecifier instead.
class NestedNameSpecifierLocFinder : public MatchFinder::MatchCallback {
public:
  explicit NestedNameSpecifierLocFinder(ASTContext &Context)
      : Context(Context) {}

  ArrayRef<NestedNameSpecifierLoc> getNestedNameSpecifierLocations() {
    addMatchers();
    Finder.matchAST(Context);
    return Locations;
  }

private:
  void addMatchers() {
    const auto NestedNameSpecifierLocMatcher =
        nestedNameSpecifierLoc().bind("nestedNameSpecifierLoc");
    Finder.addMatcher(NestedNameSpecifierLocMatcher, this);
  }

  void run(const MatchFinder::MatchResult &Result) override {
    const auto *NNS = Result.Nodes.getNodeAs<NestedNameSpecifierLoc>(
        "nestedNameSpecifierLoc");
    Locations.push_back(*NNS);
  }

  ASTContext &Context;
  std::vector<NestedNameSpecifierLoc> Locations;
  MatchFinder Finder;
};

} // end namespace rename
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_USR_FINDER_H

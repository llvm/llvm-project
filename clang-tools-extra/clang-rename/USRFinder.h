//===--- tools/extra/clang-rename/USRFinder.h - Clang rename tool ---------===//
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
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDER_H

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <string>

using namespace llvm;
using namespace clang::ast_matchers;

namespace clang {
class ASTContext;
class Decl;
class SourceLocation;
class NamedDecl;

namespace rename {

// Given an AST context and a point, returns a NamedDecl identifying the symbol
// at the point. Returns null if nothing is found at the point.
const NamedDecl *getNamedDeclAt(const ASTContext &Context,
                                const SourceLocation Point);

// Given an AST context and a fully qualified name, returns a NamedDecl
// identifying the symbol with a matching name. Returns null if nothing is
// found for the name.
const NamedDecl *getNamedDeclFor(const ASTContext &Context,
                                 const std::string &Name);

// Converts a Decl into a USR.
std::string getUSRForDecl(const Decl *Decl);

// FIXME: Implement RecursiveASTVisitor<T>::VisitNestedNameSpecifier instead.
class NestedNameSpecifierLocFinder : public MatchFinder::MatchCallback {
public:
  explicit NestedNameSpecifierLocFinder(ASTContext &Context)
      : Context(Context) {}

  std::vector<NestedNameSpecifierLoc> getNestedNameSpecifierLocations() {
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

  virtual void run(const MatchFinder::MatchResult &Result) {
    const auto *NNS = Result.Nodes.getNodeAs<NestedNameSpecifierLoc>(
        "nestedNameSpecifierLoc");
    Locations.push_back(*NNS);
  }

  ASTContext &Context;
  std::vector<NestedNameSpecifierLoc> Locations;
  MatchFinder Finder;
};
}
}

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDER_H

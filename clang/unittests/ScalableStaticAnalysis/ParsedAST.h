//===- ParsedAST.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Owns an AST parsed from a code snippet and resolves declarations in it by
// qualified name. Held by value in a test fixture, so that fixtures which
// already have a base class of their own can still get AST lookups.
//
// Unlike findDeclByName in FindDecl.h, lookups here take a *qualified* name
// ("Base::foo") and can pick among overloads.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_PARSEDAST_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_PARSEDAST_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace clang::ssaf {

class ParsedAST {
  std::unique_ptr<ASTUnit> AST;

public:
  /// Parses \p Code as C++17. Returns false if the AST could not be built, in
  /// which case the lookups below all return nullptr.
  [[nodiscard]] bool parse(llvm::StringRef Code) {
    AST = tooling::buildASTFromCodeWithArgs(Code, {"-std=c++17"});
    return AST != nullptr;
  }

  explicit operator bool() const { return AST != nullptr; }

  ASTContext &getASTContext() const { return AST->getASTContext(); }

  /// Finds a function by qualified name, e.g. "foo" or "Base::foo". Methods
  /// are functions too, so this finds those as well.
  ///
  /// To pick among overloads, append the parameter list as clang prints it:
  /// "A::f(int *)" and "A::f(char *)" name the two overloads of A::f. A bare
  /// name that matches more than one function is an error, not a silent pick
  /// of the first: it reports a gtest failure and returns nullptr, as does a
  /// name that matches nothing.
  const FunctionDecl *fn(llvm::StringRef NameOrSignature) const {
    using namespace ast_matchers;
    if (!AST)
      return nullptr;

    auto [Name, Params] = NameOrSignature.split('(');
    auto Matches =
        match(functionDecl(hasName(Name)).bind("f"), AST->getASTContext());

    llvm::SmallVector<const FunctionDecl *> Candidates;
    for (const auto &M : Matches) {
      const auto *FD = M.getNodeAs<FunctionDecl>("f");
      if (Params.empty() || paramsOf(FD) == Params.rtrim(')'))
        Candidates.push_back(FD);
    }

    if (Candidates.size() == 1)
      return Candidates.front();
    if (Candidates.empty()) {
      ADD_FAILURE() << "no function named '" << NameOrSignature << "'";
    } else {
      ADD_FAILURE() << "'" << NameOrSignature << "' is ambiguous; it matches "
                    << Candidates.size()
                    << " overloads. Append the parameter list to select one, "
                       "e.g. '"
                    << Name << "(" << paramsOf(Candidates.front()) << ")'";
    }
    return nullptr;
  }

  /// Finds parameter \p Index of the function named \p NameOrSignature.
  /// Returns nullptr if there is no such function, or if it has too few
  /// parameters.
  const ParmVarDecl *findParam(llvm::StringRef NameOrSignature,
                               unsigned Index) const {
    const FunctionDecl *FD = fn(NameOrSignature);
    if (!FD)
      return nullptr;
    if (Index >= FD->getNumParams()) {
      ADD_FAILURE() << "'" << NameOrSignature << "' has no parameter " << Index;
      return nullptr;
    }
    return FD->getParamDecl(Index);
  }

  /// Every non-implicit function (methods included) in the parsed snippet, in
  /// the order the matcher walks the AST. Unlike fn(), a miss is not a test
  /// failure: this is a plain enumeration, meant for building diagnostics.
  llvm::SmallVector<const FunctionDecl *> functions() const {
    using namespace ast_matchers;
    llvm::SmallVector<const FunctionDecl *> Result;
    if (!AST)
      return Result;
    for (const auto &M :
         match(functionDecl().bind("f"), AST->getASTContext())) {
      const auto *FD = M.getNodeAs<FunctionDecl>("f");
      if (FD && !FD->isImplicit())
        Result.push_back(FD);
    }
    return Result;
  }

  /// \p FD spelled the way fn() takes it: "Base::foo(int *)".
  static std::string signatureOf(const FunctionDecl *FD) {
    if (!FD)
      return "<null>";
    return FD->getQualifiedNameAsString() + "(" + paramsOf(FD) + ")";
  }

private:
  /// The parameter list of \p FD as clang prints it, without the parentheses:
  /// "int *", or "unsigned long, A &".
  static std::string paramsOf(const FunctionDecl *FD) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    llvm::interleave(
        FD->parameters(), OS,
        [&](const ParmVarDecl *P) { OS << P->getType().getAsString(); }, ", ");
    return Result;
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSIS_PARSEDAST_H

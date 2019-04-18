//===--- TestTU.h - Scratch source files for testing -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Many tests for indexing, code completion etc are most naturally expressed
// using code examples.
// TestTU lets test define these examples in a common way without dealing with
// the mechanics of VFS and compiler interactions, and then easily grab the
// AST, particular symbols, etc.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTTU_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTTU_H

#include "ClangdUnit.h"
#include "index/Index.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

struct TestTU {
  static TestTU withCode(llvm::StringRef Code) {
    TestTU TU;
    TU.Code = Code;
    return TU;
  }

  static TestTU withHeaderCode(llvm::StringRef HeaderCode) {
    TestTU TU;
    TU.HeaderCode = HeaderCode;
    return TU;
  }

  // The code to be compiled.
  std::string Code;
  std::string Filename = "TestTU.cpp";

  // Define contents of a header which will be implicitly included by Code.
  std::string HeaderCode;
  std::string HeaderFilename = "TestTU.h";

  // Extra arguments for the compiler invocation.
  std::vector<const char *> ExtraArgs;

  llvm::Optional<std::string> ClangTidyChecks;
  // Index to use when building AST.
  const SymbolIndex *ExternalIndex = nullptr;

  // Simulate a header guard of the header (using an #import directive).
  bool ImplicitHeaderGuard = true;

  ParsedAST build() const;
  SymbolSlab headerSymbols() const;
  std::unique_ptr<SymbolIndex> index() const;
};

// Look up an index symbol by qualified name, which must be unique.
const Symbol &findSymbol(const SymbolSlab &, llvm::StringRef QName);
// Look up an AST symbol by qualified name, which must be unique and top-level.
const NamedDecl &findDecl(ParsedAST &AST, llvm::StringRef QName);
// Look up an AST symbol that satisfies \p Filter.
const NamedDecl &findDecl(ParsedAST &AST,
                          std::function<bool(const NamedDecl &)> Filter);
// Look up an AST symbol by unqualified name, which must be unique.
const NamedDecl &findUnqualifiedDecl(ParsedAST &AST, llvm::StringRef Name);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTTU_H

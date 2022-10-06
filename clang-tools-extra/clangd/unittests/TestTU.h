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

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TESTTU_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TESTTU_H

#include "../TidyProvider.h"
#include "Compiler.h"
#include "FeatureModule.h"
#include "ParsedAST.h"
#include "TestFS.h"
#include "index/Index.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {

struct TestTU {
  static TestTU withCode(llvm::StringRef Code) {
    TestTU TU;
    TU.Code = std::string(Code);
    return TU;
  }

  static TestTU withHeaderCode(llvm::StringRef HeaderCode) {
    TestTU TU;
    TU.HeaderCode = std::string(HeaderCode);
    return TU;
  }

  // The code to be compiled.
  std::string Code;
  std::string Filename = "TestTU.cpp";

  // Define contents of a header which will be implicitly included by Code.
  std::string HeaderCode;
  std::string HeaderFilename = "TestTU.h";

  // Name and contents of each file.
  llvm::StringMap<std::string> AdditionalFiles;

  // Extra arguments for the compiler invocation.
  std::vector<std::string> ExtraArgs;

  // Predefine macros such as __UINTPTR_TYPE__.
  bool PredefineMacros = false;

  TidyProvider ClangTidyProvider = {};
  // Index to use when building AST.
  const SymbolIndex *ExternalIndex = nullptr;

  // Simulate a header guard of the header (using an #import directive).
  bool ImplicitHeaderGuard = true;

  // Parse options pass on to the ParseInputs
  ParseOptions ParseOpts = {};

  // Whether to use overlay the TestFS over the real filesystem. This is
  // required for use of implicit modules.where the module file is written to
  // disk and later read back.
  // FIXME: Change the way reading/writing modules work to allow us to keep them
  // in memory across multiple clang invocations, at least in tests, to
  // eliminate the need for real file system here.
  // Please avoid using this for things other than implicit modules. The plan is
  // to eliminate this option some day.
  bool OverlayRealFileSystemForModules = false;

  FeatureModuleSet *FeatureModules = nullptr;

  // By default, build() will report Error diagnostics as GTest errors.
  // Suppress this behavior by adding an 'error-ok' comment to the code.
  // The result will always have getDiagnostics() populated.
  ParsedAST build() const;
  std::shared_ptr<const PreambleData>
  preamble(PreambleParsedCallback PreambleCallback = nullptr) const;
  ParseInputs inputs(MockFS &FS) const;
  SymbolSlab headerSymbols() const;
  RefSlab headerRefs() const;
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

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TESTTU_H

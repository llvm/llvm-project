//===- TreeTestBase.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the test infrastructure for syntax trees.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_TOOLING_SYNTAX_TREETESTBASE_H
#define LLVM_CLANG_UNITTESTS_TOOLING_SYNTAX_TREETESTBASE_H

#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Testing/TestClangConfig.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gtest/gtest.h"

namespace clang {
namespace syntax {
class SyntaxTreeTest : public ::testing::Test,
                       public ::testing::WithParamInterface<TestClangConfig> {
protected:
  // Build a syntax tree for the code.
  TranslationUnit *buildTree(StringRef Code,
                             const TestClangConfig &ClangConfig);

  /// Finds the deepest node in the tree that covers exactly \p R.
  /// FIXME: implement this efficiently and move to public syntax tree API.
  syntax::Node *nodeByRange(llvm::Annotations::Range R, syntax::Node *Root);

  // Data fields.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      new DiagnosticsEngine(new DiagnosticIDs, DiagOpts.get());
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      new llvm::vfs::InMemoryFileSystem;
  IntrusiveRefCntPtr<FileManager> FileMgr =
      new FileManager(FileSystemOptions(), FS);
  IntrusiveRefCntPtr<SourceManager> SourceMgr =
      new SourceManager(*Diags, *FileMgr);
  std::shared_ptr<CompilerInvocation> Invocation;
  // Set after calling buildTree().
  std::unique_ptr<syntax::TokenBuffer> TB;
  std::unique_ptr<syntax::Arena> Arena;
};

std::vector<TestClangConfig> allTestClangConfigs();
} // namespace syntax
} // namespace clang
#endif // LLVM_CLANG_UNITTESTS_TOOLING_SYNTAX_TREETESTBASE_H

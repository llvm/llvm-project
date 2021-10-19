//===--- TestWorkspace.h - Utility for writing multi-file tests --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TestWorkspace builds on TestTU to provide a way to write tests involving
// several related files with inclusion relationships between them.
//
// The tests can exercise both index and AST based operations.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTWORKSPACE_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTWORKSPACE_H

#include "TestFS.h"
#include "TestTU.h"
#include "index/FileIndex.h"
#include "index/Index.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {

class TestWorkspace {
public:
  // The difference between addSource() and addMainFile() is that only main
  // files will be indexed.
  void addSource(llvm::StringRef Filename, llvm::StringRef Code) {
    addInput(Filename.str(), {Code.str(), /*IsMainFile=*/false});
  }
  void addMainFile(llvm::StringRef Filename, llvm::StringRef Code) {
    addInput(Filename.str(), {Code.str(), /*IsMainFile=*/true});
  }

  std::unique_ptr<SymbolIndex> index();

  Optional<ParsedAST> openFile(llvm::StringRef Filename);

private:
  struct SourceFile {
    std::string Code;
    bool IsMainFile = false;
  };
  llvm::StringMap<SourceFile> Inputs;
  TestTU TU;

  void addInput(llvm::StringRef Filename, const SourceFile &Input);
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTWORKSPACE_H

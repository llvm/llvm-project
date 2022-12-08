//===--- TestWorkspace.cpp - Utility for writing multi-file tests -*- C++-*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestWorkspace.h"
#include "index/FileIndex.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

std::unique_ptr<SymbolIndex> TestWorkspace::index() {
  auto Index = std::make_unique<FileIndex>();
  for (const auto &Input : Inputs) {
    if (!Input.second.IsMainFile)
      continue;
    TU.Code = Input.second.Code;
    TU.Filename = Input.first().str();
    TU.preamble([&](ASTContext &Ctx, Preprocessor &PP,
                    const CanonicalIncludes &CanonIncludes) {
      Index->updatePreamble(testPath(Input.first()), "null", Ctx, PP,
                            CanonIncludes);
    });
    ParsedAST MainAST = TU.build();
    Index->updateMain(testPath(Input.first()), MainAST);
  }
  return Index;
}

Optional<ParsedAST> TestWorkspace::openFile(llvm::StringRef Filename) {
  auto It = Inputs.find(Filename);
  if (It == Inputs.end()) {
    ADD_FAILURE() << "Accessing non-existing file: " << Filename;
    return std::nullopt;
  }
  TU.Code = It->second.Code;
  TU.Filename = It->first().str();
  return TU.build();
}

void TestWorkspace::addInput(llvm::StringRef Filename,
                             const SourceFile &Input) {
  Inputs.insert(std::make_pair(Filename, Input));
  TU.AdditionalFiles.insert(std::make_pair(Filename, Input.Code));
}
} // namespace clangd
} // namespace clang

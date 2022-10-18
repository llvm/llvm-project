//===--- Record.h - Record compiler events ------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where Analysis.h analyzes AST nodes and recorded preprocessor events, this
// file defines ways to capture AST and preprocessor information from a parse.
//
// These are the simplest way to connect include-cleaner logic to the parser,
// but other ways are possible (for example clangd records includes separately).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_RECORD_H
#define CLANG_INCLUDE_CLEANER_RECORD_H

#include <memory>
#include <vector>

namespace clang {
class ASTConsumer;
class ASTContext;
class Decl;
namespace include_cleaner {

// Contains recorded parser events relevant to include-cleaner.
struct RecordedAST {
  // The consumer (when installed into clang) tracks declarations in this.
  std::unique_ptr<ASTConsumer> record();

  ASTContext *Ctx = nullptr;
  // The set of declarations written at file scope inside the main file.
  //
  // These are the roots of the subtrees that should be traversed to find uses.
  // (Traversing the TranslationUnitDecl would find uses inside headers!)
  std::vector<Decl *> Roots;
};

} // namespace include_cleaner
} // namespace clang

#endif
//===--- IncrementalParser.h - Incremental Compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code compilation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_INCREMENTALPARSER_H
#define LLVM_CLANG_LIB_INTERPRETER_INCREMENTALPARSER_H

#include "clang/Interpreter/PartialTranslationUnit.h"

#include "clang/AST/GlobalDecl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <list>
#include <memory>
namespace llvm {
class LLVMContext;
}

namespace clang {
class ASTConsumer;
class CompilerInstance;
class IncrementalAction;
class Parser;

/// Provides support for incremental compilation. Keeps track of the state
/// changes between the subsequent incremental input.
///
class IncrementalParser {
  /// Long-lived, incremental parsing action.
  std::unique_ptr<IncrementalAction> Act;

  /// Compiler instance performing the incremental compilation.
  std::unique_ptr<CompilerInstance> CI;

  /// Parser.
  std::unique_ptr<Parser> P;

  /// Consumer to process the produced top level decls. Owned by Act.
  ASTConsumer *Consumer = nullptr;

  /// Counts the number of direct user input lines that have been parsed.
  unsigned InputCount = 0;

  /// List containing every information about every incrementally parsed piece
  /// of code.
  std::list<PartialTranslationUnit> PTUs;

public:
  IncrementalParser(std::unique_ptr<CompilerInstance> Instance,
                    llvm::LLVMContext &LLVMCtx, llvm::Error &Err);
  ~IncrementalParser();

  const CompilerInstance *getCI() const { return CI.get(); }

  /// Parses incremental input by creating an in-memory file.
  ///\returns a \c PartialTranslationUnit which holds information about the
  /// \c TranslationUnitDecl and \c llvm::Module corresponding to the input.
  llvm::Expected<PartialTranslationUnit &> Parse(llvm::StringRef Input);

  /// Uses the CodeGenModule mangled name cache and avoids recomputing.
  ///\returns the mangled name of a \c GD.
  llvm::StringRef GetMangledName(GlobalDecl GD) const;

private:
  llvm::Expected<PartialTranslationUnit &> ParseOrWrapTopLevelDecl();
};
} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALPARSER_H

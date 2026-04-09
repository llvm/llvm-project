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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <list>
#include <memory>

namespace llvm {
class Module;
}

namespace clang {
class ASTConsumer;
class CompilerInstance;
class Parser;
class Sema;
class TranslationUnitDecl;
class IncrementalAction;
struct PartialTranslationUnit;

/// Provides support for incremental compilation. Keeps track of the state
/// changes between the subsequent incremental input.
///
class IncrementalParser {
protected:
  /// The Sema performing the incremental compilation.
  Sema &S;

  /// Parser.
  std::unique_ptr<Parser> P;

  /// Consumer to process the produced top level decls. Owned by Act.
  ASTConsumer *Consumer = nullptr;

  /// Counts the number of direct user input lines that have been parsed.
  unsigned InputCount = 0;

  /// The FrontendAction used during incremental parsing.
  IncrementalAction *Act = nullptr;

  std::list<PartialTranslationUnit> &PTUs;

public:
  IncrementalParser(CompilerInstance &Instance, IncrementalAction *Act,
                    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);
  virtual ~IncrementalParser();

  /// Parses incremental input by creating an in-memory file.
  ///\returns a \c PartialTranslationUnit which holds information about the
  /// \c TranslationUnitDecl.
  virtual llvm::Expected<TranslationUnitDecl *> Parse(llvm::StringRef Input);

  void CleanUpPTU(TranslationUnitDecl *MostRecentTU);

  /// Register a PTU produced by Parse.
  PartialTranslationUnit &RegisterPTU(TranslationUnitDecl *TU,
                                      std::unique_ptr<llvm::Module> M = {});

private:
  llvm::Expected<TranslationUnitDecl *> ParseOrWrapTopLevelDecl();
};
} // end namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_INCREMENTALPARSER_H

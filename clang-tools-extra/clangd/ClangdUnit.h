//===--- ClangdUnit.h -------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNIT_H

#include "Protocol.h"
#include "Path.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Core/Replacement.h"
#include <memory>

namespace clang {
class ASTUnit;
class PCHContainerOperations;

namespace tooling {
struct CompileCommand;
}

namespace clangd {

/// A diagnostic with its FixIts.
struct DiagWithFixIts {
  clangd::Diagnostic Diag;
  llvm::SmallVector<tooling::Replacement, 1> FixIts;
};

/// Stores parsed C++ AST and provides implementations of all operations clangd
/// would want to perform on parsed C++ files.
class ClangdUnit {
public:
  ClangdUnit(PathRef FileName, StringRef Contents,
             std::shared_ptr<PCHContainerOperations> PCHs,
             std::vector<tooling::CompileCommand> Commands);

  /// Reparse with new contents.
  void reparse(StringRef Contents);

  /// Get code completions at a specified \p Line and \p Column in \p File.
  ///
  /// This function is thread-safe and returns completion items that own the
  /// data they contain.
  std::vector<CompletionItem> codeComplete(StringRef Contents, Position Pos);
  /// Returns diagnostics and corresponding FixIts for each diagnostic that are
  /// located in the current file.
  std::vector<DiagWithFixIts> getLocalDiagnostics() const;

private:
  Path FileName;
  std::unique_ptr<ASTUnit> Unit;
  std::shared_ptr<PCHContainerOperations> PCHs;
};

} // namespace clangd
} // namespace clang
#endif

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

#include "Path.h"
#include "Protocol.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Core/Replacement.h"
#include <memory>

namespace llvm {
class raw_ostream;
}

namespace clang {
class ASTUnit;
class PCHContainerOperations;

namespace vfs {
class FileSystem;
}

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
  ClangdUnit(PathRef FileName, StringRef Contents, StringRef ResourceDir,
             std::shared_ptr<PCHContainerOperations> PCHs,
             std::vector<tooling::CompileCommand> Commands,
             IntrusiveRefCntPtr<vfs::FileSystem> VFS);

  /// Reparse with new contents.
  void reparse(StringRef Contents, IntrusiveRefCntPtr<vfs::FileSystem> VFS);

  /// Get code completions at a specified \p Line and \p Column in \p File.
  ///
  /// This function is thread-safe and returns completion items that own the
  /// data they contain.
  std::vector<CompletionItem>
  codeComplete(StringRef Contents, Position Pos,
               IntrusiveRefCntPtr<vfs::FileSystem> VFS);
  /// Get definition of symbol at a specified \p Line and \p Column in \p File.
  std::vector<Location> findDefinitions(Position Pos);
  /// Returns diagnostics and corresponding FixIts for each diagnostic that are
  /// located in the current file.
  std::vector<DiagWithFixIts> getLocalDiagnostics() const;

  /// For testing/debugging purposes. Note that this method deserializes all
  /// unserialized Decls, so use with care.
  void dumpAST(llvm::raw_ostream &OS) const;

private:
  Path FileName;
  std::unique_ptr<ASTUnit> Unit;
  std::shared_ptr<PCHContainerOperations> PCHs;

  SourceLocation getBeginningOfIdentifier(const Position& Pos, const FileEntry* FE) const;
};

} // namespace clangd
} // namespace clang
#endif

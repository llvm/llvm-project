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
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Tooling/CompilationDatabase.h"
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
  /// Stores and provides access to parsed AST.
  class ParsedAST {
  public:
    /// Attempts to run Clang and store parsed AST. If \p Preamble is non-null
    /// it is reused during parsing.
    static llvm::Optional<ParsedAST>
    Build(std::unique_ptr<clang::CompilerInvocation> CI,
          const PrecompiledPreamble *Preamble,
          ArrayRef<serialization::DeclID> PreambleDeclIDs,
          std::unique_ptr<llvm::MemoryBuffer> Buffer,
          std::shared_ptr<PCHContainerOperations> PCHs,
          IntrusiveRefCntPtr<vfs::FileSystem> VFS);

    ParsedAST(ParsedAST &&Other);
    ParsedAST &operator=(ParsedAST &&Other);

    ~ParsedAST();

    ASTContext &getASTContext();
    const ASTContext &getASTContext() const;

    Preprocessor &getPreprocessor();
    const Preprocessor &getPreprocessor() const;

    /// This function returns all top-level decls, including those that come
    /// from Preamble. Decls, coming from Preamble, have to be deserialized, so
    /// this call might be expensive.
    ArrayRef<const Decl *> getTopLevelDecls();

    const std::vector<DiagWithFixIts> &getDiagnostics() const;

  private:
    ParsedAST(std::unique_ptr<CompilerInstance> Clang,
              std::unique_ptr<FrontendAction> Action,
              std::vector<const Decl *> TopLevelDecls,
              std::vector<serialization::DeclID> PendingTopLevelDecls,
              std::vector<DiagWithFixIts> Diags);

  private:
    void ensurePreambleDeclsDeserialized();

    // We store an "incomplete" FrontendAction (i.e. no EndSourceFile was called
    // on it) and CompilerInstance used to run it. That way we don't have to do
    // complex memory management of all Clang structures on our own. (They are
    // stored in CompilerInstance and cleaned up by
    // FrontendAction.EndSourceFile).
    std::unique_ptr<CompilerInstance> Clang;
    std::unique_ptr<FrontendAction> Action;

    // Data, stored after parsing.
    std::vector<DiagWithFixIts> Diags;
    std::vector<const Decl *> TopLevelDecls;
    std::vector<serialization::DeclID> PendingTopLevelDecls;
  };

  // Store Preamble and all associated data
  struct PreambleData {
    PreambleData(PrecompiledPreamble Preamble,
                 std::vector<serialization::DeclID> TopLevelDeclIDs,
                 std::vector<DiagWithFixIts> Diags);

    PrecompiledPreamble Preamble;
    std::vector<serialization::DeclID> TopLevelDeclIDs;
    std::vector<DiagWithFixIts> Diags;
  };

  SourceLocation getBeginningOfIdentifier(const Position &Pos,
                                          const FileEntry *FE) const;

  Path FileName;
  tooling::CompileCommand Command;

  llvm::Optional<PreambleData> Preamble;
  llvm::Optional<ParsedAST> Unit;

  std::shared_ptr<PCHContainerOperations> PCHs;
};

} // namespace clangd
} // namespace clang
#endif

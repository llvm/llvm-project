//===--- ClangdUnit.h --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNIT_H

#include "Compiler.h"
#include "Diagnostics.h"
#include "FS.h"
#include "Function.h"
#include "Headers.h"
#include "Path.h"
#include "Protocol.h"
#include "index/Index.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;

namespace vfs {
class FileSystem;
} // namespace vfs
} // namespace llvm

namespace clang {
class PCHContainerOperations;

namespace tooling {
struct CompileCommand;
} // namespace tooling

namespace clangd {

// Stores Preamble and associated data.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble, std::vector<Diag> Diags,
               IncludeStructure Includes,
               std::unique_ptr<PreambleFileStatusCache> StatCache);

  tooling::CompileCommand CompileCommand;
  PrecompiledPreamble Preamble;
  std::vector<Diag> Diags;
  // Processes like code completions and go-to-definitions will need #include
  // information, and their compile action skips preamble range.
  IncludeStructure Includes;
  // Cache of FS operations performed when building the preamble.
  // When reusing a preamble, this cache can be consumed to save IO.
  std::unique_ptr<PreambleFileStatusCache> StatCache;
};

/// Stores and provides access to parsed AST.
class ParsedAST {
public:
  /// Attempts to run Clang and store parsed AST. If \p Preamble is non-null
  /// it is reused during parsing.
  static llvm::Optional<ParsedAST>
  build(std::unique_ptr<clang::CompilerInvocation> CI,
        std::shared_ptr<const PreambleData> Preamble,
        std::unique_ptr<llvm::MemoryBuffer> Buffer,
        std::shared_ptr<PCHContainerOperations> PCHs,
        IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS, const SymbolIndex *Index,
        const ParseOptions &Opts);

  ParsedAST(ParsedAST &&Other);
  ParsedAST &operator=(ParsedAST &&Other);

  ~ParsedAST();

  /// Note that the returned ast will not contain decls from the preamble that
  /// were not deserialized during parsing. Clients should expect only decls
  /// from the main file to be in the AST.
  ASTContext &getASTContext();
  const ASTContext &getASTContext() const;

  Preprocessor &getPreprocessor();
  std::shared_ptr<Preprocessor> getPreprocessorPtr();
  const Preprocessor &getPreprocessor() const;

  /// This function returns top-level decls present in the main file of the AST.
  /// The result does not include the decls that come from the preamble.
  /// (These should be const, but RecursiveASTVisitor requires Decl*).
  ArrayRef<Decl *> getLocalTopLevelDecls();

  const std::vector<Diag> &getDiagnostics() const;

  /// Returns the esitmated size of the AST and the accessory structures, in
  /// bytes. Does not include the size of the preamble.
  std::size_t getUsedBytes() const;
  const IncludeStructure &getIncludeStructure() const;

private:
  ParsedAST(std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action,
            std::vector<Decl *> LocalTopLevelDecls, std::vector<Diag> Diags,
            IncludeStructure Includes);

  // In-memory preambles must outlive the AST, it is important that this member
  // goes before Clang and Action.
  std::shared_ptr<const PreambleData> Preamble;
  // We store an "incomplete" FrontendAction (i.e. no EndSourceFile was called
  // on it) and CompilerInstance used to run it. That way we don't have to do
  // complex memory management of all Clang structures on our own. (They are
  // stored in CompilerInstance and cleaned up by
  // FrontendAction.EndSourceFile).
  std::unique_ptr<CompilerInstance> Clang;
  std::unique_ptr<FrontendAction> Action;

  // Data, stored after parsing.
  std::vector<Diag> Diags;
  // Top-level decls inside the current file. Not that this does not include
  // top-level decls from the preamble.
  std::vector<Decl *> LocalTopLevelDecls;
  IncludeStructure Includes;
};

using PreambleParsedCallback =
    std::function<void(ASTContext &, std::shared_ptr<clang::Preprocessor>)>;

/// Rebuild the preamble for the new inputs unless the old one can be reused.
/// If \p OldPreamble can be reused, it is returned unchanged.
/// If \p OldPreamble is null, always builds the preamble.
/// If \p PreambleCallback is set, it will be run on top of the AST while
/// building the preamble. Note that if the old preamble was reused, no AST is
/// built and, therefore, the callback will not be executed.
std::shared_ptr<const PreambleData>
buildPreamble(PathRef FileName, CompilerInvocation &CI,
              std::shared_ptr<const PreambleData> OldPreamble,
              const tooling::CompileCommand &OldCompileCommand,
              const ParseInputs &Inputs,
              std::shared_ptr<PCHContainerOperations> PCHs, bool StoreInMemory,
              PreambleParsedCallback PreambleCallback);

/// Build an AST from provided user inputs. This function does not check if
/// preamble can be reused, as this function expects that \p Preamble is the
/// result of calling buildPreamble.
llvm::Optional<ParsedAST>
buildAST(PathRef FileName, std::unique_ptr<CompilerInvocation> Invocation,
         const ParseInputs &Inputs,
         std::shared_ptr<const PreambleData> Preamble,
         std::shared_ptr<PCHContainerOperations> PCHs);

/// Get the beginning SourceLocation at a specified \p Pos.
/// May be invalid if Pos is, or if there's no identifier.
SourceLocation getBeginningOfIdentifier(ParsedAST &Unit, const Position &Pos,
                                        const FileID FID);

/// For testing/debugging purposes. Note that this method deserializes all
/// unserialized Decls, so use with care.
void dumpAST(ParsedAST &AST, llvm::raw_ostream &OS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNIT_H

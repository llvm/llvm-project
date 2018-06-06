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

#include "Diagnostics.h"
#include "Function.h"
#include "Headers.h"
#include "Path.h"
#include "Protocol.h"
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
}

namespace clang {
class PCHContainerOperations;

namespace vfs {
class FileSystem;
}

namespace tooling {
struct CompileCommand;
}

namespace clangd {

// Stores Preamble and associated data.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble, std::vector<Diag> Diags,
               std::vector<Inclusion> Inclusions);

  tooling::CompileCommand CompileCommand;
  PrecompiledPreamble Preamble;
  std::vector<Diag> Diags;
  // Processes like code completions and go-to-definitions will need #include
  // information, and their compile action skips preamble range.
  std::vector<Inclusion> Inclusions;
};

/// Information required to run clang, e.g. to parse AST or do code completion.
struct ParseInputs {
  tooling::CompileCommand CompileCommand;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  std::string Contents;
};

/// Stores and provides access to parsed AST.
class ParsedAST {
public:
  /// Attempts to run Clang and store parsed AST. If \p Preamble is non-null
  /// it is reused during parsing.
  static llvm::Optional<ParsedAST>
  Build(std::unique_ptr<clang::CompilerInvocation> CI,
        std::shared_ptr<const PreambleData> Preamble,
        std::unique_ptr<llvm::MemoryBuffer> Buffer,
        std::shared_ptr<PCHContainerOperations> PCHs,
        IntrusiveRefCntPtr<vfs::FileSystem> VFS);

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
  const std::vector<Inclusion> &getInclusions() const;

private:
  ParsedAST(std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action,
            std::vector<Decl *> LocalTopLevelDecls, std::vector<Diag> Diags,
            std::vector<Inclusion> Inclusions);

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
  std::vector<Inclusion> Inclusions;
};

using PreambleParsedCallback = std::function<void(
    PathRef Path, ASTContext &, std::shared_ptr<clang::Preprocessor>)>;

/// Builds compiler invocation that could be used to build AST or preamble.
std::unique_ptr<CompilerInvocation>
buildCompilerInvocation(const ParseInputs &Inputs);

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
#endif

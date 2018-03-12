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
#include "Path.h"
#include "Protocol.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/PrecompiledPreamble.h"
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

using InclusionLocations = std::vector<std::pair<Range, Path>>;

// Stores Preamble and associated data.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble,
               std::vector<serialization::DeclID> TopLevelDeclIDs,
               std::vector<Diag> Diags, InclusionLocations IncLocations);

  PrecompiledPreamble Preamble;
  std::vector<serialization::DeclID> TopLevelDeclIDs;
  std::vector<Diag> Diags;
  InclusionLocations IncLocations;
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

  ASTContext &getASTContext();
  const ASTContext &getASTContext() const;

  Preprocessor &getPreprocessor();
  std::shared_ptr<Preprocessor> getPreprocessorPtr();
  const Preprocessor &getPreprocessor() const;

  /// This function returns all top-level decls, including those that come
  /// from Preamble. Decls, coming from Preamble, have to be deserialized, so
  /// this call might be expensive.
  ArrayRef<const Decl *> getTopLevelDecls();

  const std::vector<Diag> &getDiagnostics() const;

  /// Returns the esitmated size of the AST and the accessory structures, in
  /// bytes. Does not include the size of the preamble.
  std::size_t getUsedBytes() const;
  const InclusionLocations &getInclusionLocations() const;

private:
  ParsedAST(std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action,
            std::vector<const Decl *> TopLevelDecls, std::vector<Diag> Diags,
            InclusionLocations IncLocations);

private:
  void ensurePreambleDeclsDeserialized();

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
  std::vector<const Decl *> TopLevelDecls;
  bool PreambleDeclsDeserialized;
  InclusionLocations IncLocations;
};

using ASTParsedCallback = std::function<void(PathRef Path, ParsedAST *)>;

/// Manages resources, required by clangd. Allows to rebuild file with new
/// contents, and provides AST and Preamble for it.
class CppFile {
public:
  CppFile(PathRef FileName, bool StorePreamblesInMemory,
          std::shared_ptr<PCHContainerOperations> PCHs,
          ASTParsedCallback ASTCallback);

  /// Rebuild the AST and the preamble.
  /// Returns a list of diagnostics or llvm::None, if an error occured.
  llvm::Optional<std::vector<Diag>> rebuild(ParseInputs &&Inputs);
  /// Returns the last built preamble.
  const std::shared_ptr<const PreambleData> &getPreamble() const;
  /// Returns the last built AST.
  ParsedAST *getAST() const;
  /// Returns an estimated size, in bytes, currently occupied by the AST and the
  /// Preamble.
  std::size_t getUsedBytes() const;

private:
  /// Build a new preamble for \p Inputs. If the current preamble can be reused,
  /// it is returned instead.
  /// This method is const to ensure we don't incidentally modify any fields.
  std::shared_ptr<const PreambleData>
  rebuildPreamble(CompilerInvocation &CI,
                  const tooling::CompileCommand &Command,
                  IntrusiveRefCntPtr<vfs::FileSystem> FS,
                  llvm::MemoryBuffer &ContentsBuffer) const;

  const Path FileName;
  const bool StorePreamblesInMemory;

  /// The last CompileCommand used to build AST and Preamble.
  tooling::CompileCommand Command;
  /// The last parsed AST.
  llvm::Optional<ParsedAST> AST;
  /// The last built Preamble.
  std::shared_ptr<const PreambleData> Preamble;
  /// Utility class required by clang
  std::shared_ptr<PCHContainerOperations> PCHs;
  /// This is called after the file is parsed. This can be nullptr if there is
  /// no callback.
  ASTParsedCallback ASTCallback;
};

/// Get the beginning SourceLocation at a specified \p Pos.
SourceLocation getBeginningOfIdentifier(ParsedAST &Unit, const Position &Pos,
                                        const FileEntry *FE);

/// For testing/debugging purposes. Note that this method deserializes all
/// unserialized Decls, so use with care.
void dumpAST(ParsedAST &AST, llvm::raw_ostream &OS);

} // namespace clangd
} // namespace clang
#endif

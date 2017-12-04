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

#include "Function.h"
#include "Path.h"
#include "Protocol.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Core/Replacement.h"
#include <atomic>
#include <future>
#include <memory>
#include <mutex>

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

class Logger;

/// A diagnostic with its FixIts.
struct DiagWithFixIts {
  clangd::Diagnostic Diag;
  llvm::SmallVector<tooling::Replacement, 1> FixIts;
};

// Stores Preamble and associated data.
struct PreambleData {
  PreambleData(PrecompiledPreamble Preamble,
               std::vector<serialization::DeclID> TopLevelDeclIDs,
               std::vector<DiagWithFixIts> Diags);

  PrecompiledPreamble Preamble;
  std::vector<serialization::DeclID> TopLevelDeclIDs;
  std::vector<DiagWithFixIts> Diags;
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
        IntrusiveRefCntPtr<vfs::FileSystem> VFS, clangd::Logger &Logger);

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
  ParsedAST(std::shared_ptr<const PreambleData> Preamble,
            std::unique_ptr<CompilerInstance> Clang,
            std::unique_ptr<FrontendAction> Action,
            std::vector<const Decl *> TopLevelDecls,
            std::vector<DiagWithFixIts> Diags);

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
  std::vector<DiagWithFixIts> Diags;
  std::vector<const Decl *> TopLevelDecls;
  bool PreambleDeclsDeserialized;
};

// Provides thread-safe access to ParsedAST.
class ParsedASTWrapper {
public:
  ParsedASTWrapper(ParsedASTWrapper &&Wrapper);
  ParsedASTWrapper(llvm::Optional<ParsedAST> AST);

  /// Runs \p F on wrapped ParsedAST under lock. Ensures it is not accessed
  /// concurrently.
  template <class Func> void runUnderLock(Func F) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    F(AST ? AST.getPointer() : nullptr);
  }

private:
  // This wrapper is used as an argument to std::shared_future (and it returns a
  // const ref in get()), but we need to have non-const ref in order to
  // implement some features.
  mutable std::mutex Mutex;
  mutable llvm::Optional<ParsedAST> AST;
};

/// Manages resources, required by clangd. Allows to rebuild file with new
/// contents, and provides AST and Preamble for it.
class CppFile : public std::enable_shared_from_this<CppFile> {
public:
  // We only allow to create CppFile as shared_ptr, because a future returned by
  // deferRebuild will hold references to it.
  static std::shared_ptr<CppFile>
  Create(PathRef FileName, tooling::CompileCommand Command,
         bool StorePreamblesInMemory,
         std::shared_ptr<PCHContainerOperations> PCHs, clangd::Logger &Logger);

private:
  CppFile(PathRef FileName, tooling::CompileCommand Command,
          bool StorePreamblesInMemory,
          std::shared_ptr<PCHContainerOperations> PCHs, clangd::Logger &Logger);

public:
  CppFile(CppFile const &) = delete;
  CppFile(CppFile &&) = delete;

  /// Cancels a scheduled rebuild, if any, and sets AST and Preamble to nulls.
  /// If a rebuild is in progress, will wait for it to finish.
  void cancelRebuild();

  /// Similar to deferRebuild, but sets both Preamble and AST to nulls instead
  /// of doing an actual parsing. Returned function is a deferred computation
  /// that will wait for any ongoing rebuilds to finish and actually set the AST
  /// and Preamble to nulls. It can be run on a different thread. This function
  /// is useful to cancel ongoing rebuilds, if any, before removing CppFile.
  UniqueFunction<void()> deferCancelRebuild();

  /// Rebuild AST and Preamble synchronously on the calling thread.
  /// Returns a list of diagnostics or a llvm::None, if another rebuild was
  /// requested in parallel (effectively cancelling this rebuild) before
  /// diagnostics were produced.
  llvm::Optional<std::vector<DiagWithFixIts>>
  rebuild(StringRef NewContents, IntrusiveRefCntPtr<vfs::FileSystem> VFS);

  /// Schedule a rebuild and return a deferred computation that will finish the
  /// rebuild, that can be called on a different thread.
  /// After calling this method, resources, available via futures returned by
  /// getPreamble() and getAST(), will be waiting for rebuild to finish. A
  /// continuation fininshing rebuild, returned by this function, must be
  /// computed(i.e., operator() must be called on it) in order to make those
  /// resources ready. If deferRebuild is called again before the rebuild is
  /// finished (either because returned future had not been called or because it
  /// had not returned yet), the previous rebuild request is cancelled and the
  /// resource futures (returned by getPreamble() or getAST()) that were not
  /// ready will be waiting for the last rebuild to finish instead.
  /// The future to finish rebuild returns a list of diagnostics built during
  /// reparse, or None, if another deferRebuild was called before this
  /// rebuild was finished.
  UniqueFunction<llvm::Optional<std::vector<DiagWithFixIts>>()>
  deferRebuild(StringRef NewContents, IntrusiveRefCntPtr<vfs::FileSystem> VFS);

  /// Returns a future to get the most fresh PreambleData for a file. The
  /// future will wait until the Preamble is rebuilt.
  std::shared_future<std::shared_ptr<const PreambleData>> getPreamble() const;
  /// Return some preamble for a file. It might be stale, but won't wait for
  /// rebuild to finish.
  std::shared_ptr<const PreambleData> getPossiblyStalePreamble() const;

  /// Returns a future to get the most fresh AST for a file. Returned AST is
  /// wrapped to prevent concurrent accesses.
  /// We use std::shared_ptr here because MVSC fails to compile non-copyable
  /// classes as template arguments of promise/future. It is guaranteed to
  /// always be non-null.
  std::shared_future<std::shared_ptr<ParsedASTWrapper>> getAST() const;

  /// Get CompileCommand used to build this CppFile.
  tooling::CompileCommand const &getCompileCommand() const;

private:
  /// A helper guard that manages the state of CppFile during rebuild.
  class RebuildGuard {
  public:
    RebuildGuard(CppFile &File, unsigned RequestRebuildCounter);
    ~RebuildGuard();

    bool wasCancelledBeforeConstruction() const;

  private:
    CppFile &File;
    unsigned RequestRebuildCounter;
    bool WasCancelledBeforeConstruction;
  };

  Path FileName;
  tooling::CompileCommand Command;
  bool StorePreamblesInMemory;

  /// Mutex protects all fields, declared below it, FileName and Command are not
  /// mutated.
  mutable std::mutex Mutex;
  /// A counter to cancel old rebuilds.
  unsigned RebuildCounter;
  /// Used to wait when rebuild is finished before starting another one.
  bool RebuildInProgress;
  /// Condition variable to indicate changes to RebuildInProgress.
  std::condition_variable RebuildCond;

  /// Promise and future for the latests AST. Fulfilled during rebuild.
  /// We use std::shared_ptr here because MVSC fails to compile non-copyable
  /// classes as template arguments of promise/future.
  std::promise<std::shared_ptr<ParsedASTWrapper>> ASTPromise;
  std::shared_future<std::shared_ptr<ParsedASTWrapper>> ASTFuture;

  /// Promise and future for the latests Preamble. Fulfilled during rebuild.
  std::promise<std::shared_ptr<const PreambleData>> PreamblePromise;
  std::shared_future<std::shared_ptr<const PreambleData>> PreambleFuture;
  /// Latest preamble that was built. May be stale, but always available without
  /// waiting for rebuild to finish.
  std::shared_ptr<const PreambleData> LatestAvailablePreamble;
  /// Utility class, required by clang.
  std::shared_ptr<PCHContainerOperations> PCHs;
  /// Used for logging various messages.
  clangd::Logger &Logger;
};


/// Get the beginning SourceLocation at a specified \p Pos.
SourceLocation getBeginningOfIdentifier(ParsedAST &Unit, const Position &Pos,
                                        const FileEntry *FE);

/// Get definition of symbol at a specified \p Pos.
std::vector<Location> findDefinitions(ParsedAST &AST, Position Pos,
                                      clangd::Logger &Logger);

/// For testing/debugging purposes. Note that this method deserializes all
/// unserialized Decls, so use with care.
void dumpAST(ParsedAST &AST, llvm::raw_ostream &OS);

} // namespace clangd
} // namespace clang
#endif

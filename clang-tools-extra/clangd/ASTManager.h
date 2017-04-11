//===--- ASTManager.h - Clang AST manager -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_ASTMANAGER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_ASTMANAGER_H

#include "DocumentStore.h"
#include "JSONRPCDispatcher.h"
#include "Protocol.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include <condition_variable>
#include <deque>
#include <thread>

namespace clang {
class ASTUnit;
class DiagnosticsEngine;
class PCHContainerOperations;
namespace tooling {
class CompilationDatabase;
} // namespace tooling

namespace clangd {

/// Using 'unsigned' here to avoid undefined behaviour on overflow.
typedef unsigned DocVersion;

/// Stores ASTUnit and FixIts map for an opened document
class DocData {
public:
  typedef std::map<clangd::Diagnostic, std::vector<clang::tooling::Replacement>>
      DiagnosticToReplacementMap;

public:
  void setAST(std::unique_ptr<ASTUnit> AST);
  ASTUnit *getAST() const;

  void cacheFixIts(DiagnosticToReplacementMap FixIts);
  std::vector<clang::tooling::Replacement>
  getFixIts(const clangd::Diagnostic &D) const;

private:
  std::unique_ptr<ASTUnit> AST;
  DiagnosticToReplacementMap FixIts;
};

enum class ASTManagerRequestType { ParseAndPublishDiagnostics, RemoveDocData };

/// A request to the worker thread
class ASTManagerRequest {
public:
  ASTManagerRequest() = default;
  ASTManagerRequest(ASTManagerRequestType Type, std::string File,
                    DocVersion Version);

  ASTManagerRequestType Type;
  std::string File;
  DocVersion Version;
};

class ASTManager : public DocumentStoreListener {
public:
  ASTManager(JSONOutput &Output, DocumentStore &Store, bool RunSynchronously);
  ~ASTManager() override;

  void onDocumentAdd(StringRef File) override;
  void onDocumentRemove(StringRef File) override;

  /// Get code completions at a specified \p Line and \p Column in \p File.
  ///
  /// This function is thread-safe and returns completion items that own the
  /// data they contain.
  std::vector<CompletionItem> codeComplete(StringRef File, unsigned Line,
                                           unsigned Column);

  /// Get the fixes associated with a certain diagnostic in a specified file as
  /// replacements.
  ///
  /// This function is thread-safe. It returns a copy to avoid handing out
  /// references to unguarded data.
  std::vector<clang::tooling::Replacement>
  getFixIts(StringRef File, const clangd::Diagnostic &D);

  DocumentStore &getStore() const { return Store; }

private:
  JSONOutput &Output;
  DocumentStore &Store;

  // Set to true if requests should block instead of being processed
  // asynchronously.
  bool RunSynchronously;

  /// Loads a compilation database for File. May return nullptr if it fails. The
  /// database is cached for subsequent accesses.
  clang::tooling::CompilationDatabase *
  getOrCreateCompilationDatabaseForFile(StringRef File);
  // Creates a new ASTUnit for the document at File.
  // FIXME: This calls chdir internally, which is thread unsafe.
  std::unique_ptr<clang::ASTUnit>
  createASTUnitForFile(StringRef File, const DocumentStore &Docs);

  /// If RunSynchronously is false, queues the request to be run on the worker
  /// thread.
  /// If RunSynchronously is true, runs the request handler immediately on the
  /// main thread.
  void queueOrRun(ASTManagerRequestType RequestType, StringRef File);

  void runWorker();
  void handleRequest(ASTManagerRequestType RequestType, StringRef File);

  /// Parses files and publishes diagnostics.
  /// This function is called on the worker thread in asynchronous mode and
  /// on the main thread in synchronous mode.
  void parseFileAndPublishDiagnostics(StringRef File);

  /// Caches compilation databases loaded from directories(keys are directories).
  llvm::StringMap<std::unique_ptr<clang::tooling::CompilationDatabase>>
      CompilationDatabases;

  /// Clang objects.
  /// A map from filenames to DocData structures that store ASTUnit and Fixits for
  /// the files. The ASTUnits are used for generating diagnostics and fix-it-s
  /// asynchronously by the worker thread and synchronously for code completion.
  llvm::StringMap<DocData> DocDatas;
  std::shared_ptr<clang::PCHContainerOperations> PCHs;
  /// A lock for access to the DocDatas, CompilationDatabases and PCHs.
  std::mutex ClangObjectLock;

  /// Stores latest versions of the tracked documents to discard outdated requests.
  /// Guarded by RequestLock.
  /// TODO(ibiryukov): the entries are neved deleted from this map.
  llvm::StringMap<DocVersion> DocVersions;

  /// A LIFO queue of requests. Note that requests are discarded if the `version`
  /// field is not equal to the one stored inside DocVersions.
  /// TODO(krasimir): code completion should always have priority over parsing
  /// for diagnostics.
  std::deque<ASTManagerRequest> RequestQueue;
  /// Setting Done to true will make the worker thread terminate.
  bool Done = false;
  /// Condition variable to wake up the worker thread.
  std::condition_variable ClangRequestCV;
  /// Lock for accesses to RequestQueue, DocVersions and Done.
  std::mutex RequestLock;

  /// We run parsing on a separate thread. This thread looks into RequestQueue to
  /// find requests to handle and terminates when Done is set to true.
  std::thread ClangWorker;
};

} // namespace clangd
} // namespace clang

#endif

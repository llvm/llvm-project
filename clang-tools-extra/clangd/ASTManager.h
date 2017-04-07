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

class ASTManager : public DocumentStoreListener {
public:
  ASTManager(JSONOutput &Output, DocumentStore &Store, bool RunSynchronously);
  ~ASTManager() override;

  void onDocumentAdd(StringRef File) override;
  // FIXME: Implement onDocumentRemove

  /// Get code completions at a specified \p Line and \p Column in \p File.
  ///
  /// This function is thread-safe and returns completion items that own the
  /// data they contain.
  std::vector<CompletionItem> codeComplete(StringRef File, unsigned Line,
                                           unsigned Column);

  /// Get the fixes associated with a certain diagnostic as replacements.
  ///
  /// This function is thread-safe. It returns a copy to avoid handing out
  /// references to unguarded data.
  std::vector<clang::tooling::Replacement>
  getFixIts(const clangd::Diagnostic &D);

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

  void runWorker();
  void parseFileAndPublishDiagnostics(StringRef File);

  /// Clang objects.

  /// A map from File-s to ASTUnit-s. Guarded by \c ASTLock. ASTUnit-s are used
  /// for generating diagnostics and fix-it-s asynchronously by the worker
  /// thread and synchronously for code completion.
  ///
  /// TODO(krasimir): code completion should always have priority over parsing
  /// for diagnostics.
  llvm::StringMap<std::unique_ptr<clang::ASTUnit>> ASTs;
  /// A lock for access to the map \c ASTs.
  std::mutex ASTLock;

  llvm::StringMap<std::unique_ptr<clang::tooling::CompilationDatabase>>
      CompilationDatabases;
  std::shared_ptr<clang::PCHContainerOperations> PCHs;

  typedef std::map<clangd::Diagnostic, std::vector<clang::tooling::Replacement>>
      DiagnosticToReplacementMap;
  DiagnosticToReplacementMap FixIts;
  std::mutex FixItLock;

  /// Queue of requests.
  std::deque<std::string> RequestQueue;
  /// Setting Done to true will make the worker thread terminate.
  bool Done = false;
  /// Condition variable to wake up the worker thread.
  std::condition_variable ClangRequestCV;
  /// Lock for accesses to RequestQueue and Done.
  std::mutex RequestLock;

  /// We run parsing on a separate thread. This thread looks into PendingRequest
  /// as a 'one element work queue' as the queue is non-empty.
  std::thread ClangWorker;
};

} // namespace clangd
} // namespace clang

#endif

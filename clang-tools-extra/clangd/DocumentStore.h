//===--- DocumentStore.h - File contents container --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DOCUMENTSTORE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DOCUMENTSTORE_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
class DocumentStore;

struct DocumentStoreListener {
  virtual ~DocumentStoreListener() = default;
  virtual void onDocumentAdd(StringRef Uri) {}
  virtual void onDocumentRemove(StringRef Uri) {}
};

/// A container for files opened in a workspace, addressed by URI. The contents
/// are owned by the DocumentStore.
class DocumentStore {
public:
  /// Add a document to the store. Overwrites existing contents.
  void addDocument(StringRef Uri, StringRef Text) {
    {
      std::lock_guard<std::mutex> Guard(DocsMutex);
      Docs[Uri] = Text;
    }
    for (const auto &Listener : Listeners)
      Listener->onDocumentAdd(Uri);
  }
  /// Delete a document from the store.
  void removeDocument(StringRef Uri) {
    {
      std::lock_guard<std::mutex> Guard(DocsMutex);
      Docs.erase(Uri);
    }
    for (const auto &Listener : Listeners)
      Listener->onDocumentRemove(Uri);
  }
  /// Retrieve a document from the store. Empty string if it's unknown.
  std::string getDocument(StringRef Uri) const {
    // FIXME: This could be a reader lock.
    std::lock_guard<std::mutex> Guard(DocsMutex);
    return Docs.lookup(Uri);
  }

  /// Add a listener. Does not take ownership.
  void addListener(DocumentStoreListener *DSL) { Listeners.push_back(DSL); }

  /// Get name and constents of all documents in this store.
  std::vector<std::pair<std::string, std::string>> getAllDocuments() const {
    std::vector<std::pair<std::string, std::string>> AllDocs;
    std::lock_guard<std::mutex> Guard(DocsMutex);
    for (const auto &P : Docs)
      AllDocs.emplace_back(P.first(), P.second);
    return AllDocs;
  }

private:
  llvm::StringMap<std::string> Docs;
  std::vector<DocumentStoreListener *> Listeners;

  mutable std::mutex DocsMutex;
};

} // namespace clangd
} // namespace clang

#endif

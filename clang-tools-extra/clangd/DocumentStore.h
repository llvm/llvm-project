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
#include <string>

namespace clang {
namespace clangd {

/// A container for files opened in a workspace, addressed by URI. The contents
/// are owned by the DocumentStore.
class DocumentStore {
public:
  /// Add a document to the store. Overwrites existing contents.
  void addDocument(StringRef Uri, StringRef Text) { Docs[Uri] = Text; }
  /// Delete a document from the store.
  void removeDocument(StringRef Uri) { Docs.erase(Uri); }
  /// Retrieve a document from the store. Empty string if it's unknown.
  StringRef getDocument(StringRef Uri) const {
    auto I = Docs.find(Uri);
    return I == Docs.end() ? StringRef("") : StringRef(I->second);
  }

private:
  llvm::StringMap<std::string> Docs;
};

} // namespace clangd
} // namespace clang

#endif

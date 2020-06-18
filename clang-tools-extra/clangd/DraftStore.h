//===--- DraftStore.h - File contents container -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DRAFTSTORE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DRAFTSTORE_H

#include "Protocol.h"
#include "support/Path.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// A thread-safe container for files opened in a workspace, addressed by
/// filenames. The contents are owned by the DraftStore. This class supports
/// both whole and incremental updates of the documents.
/// Each time a draft is updated, it is assigned a version number. This can be
/// specified by the caller or incremented from the previous version.
class DraftStore {
public:
  struct Draft {
    std::string Contents;
    int64_t Version = -1;
  };

  /// \return Contents of the stored document.
  /// For untracked files, a llvm::None is returned.
  llvm::Optional<Draft> getDraft(PathRef File) const;

  /// \return List of names of the drafts in this store.
  std::vector<Path> getActiveFiles() const;

  /// Replace contents of the draft for \p File with \p Contents.
  /// If no version is specified, one will be automatically assigned.
  /// Returns the version.
  int64_t addDraft(PathRef File, llvm::Optional<int64_t> Version,
                   StringRef Contents);

  /// Update the contents of the draft for \p File based on \p Changes.
  /// If a position in \p Changes is invalid (e.g. out-of-range), the
  /// draft is not modified.
  /// If no version is specified, one will be automatically assigned.
  ///
  /// \return The new version of the draft for \p File, or an error if the
  /// changes couldn't be applied.
  llvm::Expected<Draft>
  updateDraft(PathRef File, llvm::Optional<int64_t> Version,
              llvm::ArrayRef<TextDocumentContentChangeEvent> Changes);

  /// Remove the draft from the store.
  void removeDraft(PathRef File);

private:
  mutable std::mutex Mutex;
  llvm::StringMap<Draft> Drafts;
};

} // namespace clangd
} // namespace clang

#endif

//===--- DraftStore.h - File contents container -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DRAFTSTORE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DRAFTSTORE_H

#include "Path.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// A thread-safe container for files opened in a workspace, addressed by
/// filenames. The contents are owned by the DraftStore.
class DraftStore {
public:
  /// \return Contents of the stored document.
  /// For untracked files, a llvm::None is returned.
  llvm::Optional<std::string> getDraft(PathRef File) const;

  /// \return List of names of the drafts in this store.
  std::vector<Path> getActiveFiles() const;

  /// Replace contents of the draft for \p File with \p Contents.
  void updateDraft(PathRef File, StringRef Contents);

  /// Remove the draft from the store.
  void removeDraft(PathRef File);

private:
  mutable std::mutex Mutex;
  llvm::StringMap<std::string> Drafts;
};

} // namespace clangd
} // namespace clang

#endif

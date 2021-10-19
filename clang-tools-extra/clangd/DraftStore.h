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
#include "llvm/Support/VirtualFileSystem.h"
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// A thread-safe container for files opened in a workspace, addressed by
/// filenames. The contents are owned by the DraftStore.
/// Each time a draft is updated, it is assigned a version. This can be
/// specified by the caller or incremented from the previous version.
class DraftStore {
public:
  struct Draft {
    std::shared_ptr<const std::string> Contents;
    std::string Version;
  };

  /// \return Contents of the stored document.
  /// For untracked files, a llvm::None is returned.
  llvm::Optional<Draft> getDraft(PathRef File) const;

  /// \return List of names of the drafts in this store.
  std::vector<Path> getActiveFiles() const;

  /// Replace contents of the draft for \p File with \p Contents.
  /// If version is empty, one will be automatically assigned.
  /// Returns the version.
  std::string addDraft(PathRef File, llvm::StringRef Version,
                       StringRef Contents);

  /// Remove the draft from the store.
  void removeDraft(PathRef File);

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> asVFS() const;

private:
  struct DraftAndTime {
    Draft D;
    std::time_t MTime;
  };
  mutable std::mutex Mutex;
  llvm::StringMap<DraftAndTime> Drafts;
};

} // namespace clangd
} // namespace clang

#endif

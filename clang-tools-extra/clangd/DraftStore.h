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
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// Using unsigned int type here to avoid undefined behaviour on overflow.
typedef uint64_t DocVersion;

/// Document draft with a version of this draft.
struct VersionedDraft {
  DocVersion Version;
  /// If the value of the field is None, draft is now deleted
  llvm::Optional<std::string> Draft;
};

/// A thread-safe container for files opened in a workspace, addressed by
/// filenames. The contents are owned by the DraftStore. Versions are mantained
/// for the all added documents, including removed ones. The document version is
/// incremented on each update and removal of the document.
class DraftStore {
public:
  /// \return version and contents of the stored document.
  /// For untracked files, a (0, None) pair is returned.
  VersionedDraft getDraft(PathRef File) const;
  /// \return version of the tracked document.
  /// For untracked files, 0 is returned.
  DocVersion getVersion(PathRef File) const;

  /// Replace contents of the draft for \p File with \p Contents.
  /// \return The new version of the draft for \p File.
  DocVersion updateDraft(PathRef File, StringRef Contents);
  /// Remove the contents of the draft
  /// \return The new version of the draft for \p File.
  DocVersion removeDraft(PathRef File);

private:
  mutable std::mutex Mutex;
  llvm::StringMap<VersionedDraft> Drafts;
};

} // namespace clangd
} // namespace clang

#endif

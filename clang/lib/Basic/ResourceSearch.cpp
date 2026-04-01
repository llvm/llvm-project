//===--- ResourceSearch.h - Searching for Resources -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided filters include/exclude profile instrumentation in certain
// functions or files.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/ResourceSearch.h"
#include "clang/Basic/FileManager.h"

namespace clang {

template <typename Strings>
OptionalFileEntryRef
LookupFileWithImpl(StringRef Filename, bool isAngled, bool OpenFile,
                   FileManager &FM, const Strings &SearchEntries,
                   OptionalFileEntryRef LookupFromFile, StringRef *FoundEntry) {
  if (llvm::sys::path::is_absolute(Filename)) {
    // lookup path or immediately fail
    return FM.getOptionalFileRef(Filename, OpenFile, /*CacheFailure=*/true,
                                 /*IsText=*/false);
  }

  auto SeparateComponents = [](SmallVectorImpl<char> &LookupPath,
                               StringRef StartingFrom, StringRef FileName,
                               bool RemoveInitialFileComponentFromLookupPath) {
    llvm::sys::path::native(StartingFrom, LookupPath);
    if (RemoveInitialFileComponentFromLookupPath)
      llvm::sys::path::remove_filename(LookupPath);
    if (!LookupPath.empty() &&
        !llvm::sys::path::is_separator(LookupPath.back())) {
      LookupPath.push_back(llvm::sys::path::get_separator().front());
    }
    LookupPath.append(FileName.begin(), FileName.end());
  };

  // Otherwise, it's search time!
  SmallString<512> LookupPath;
  // Non-angled lookup
  if (!isAngled) {
    if (LookupFromFile) {
      // Use file-based local lookup.
      SmallString<1024> TmpDir;
      TmpDir = LookupFromFile->getDir().getName();
      llvm::sys::path::append(TmpDir, Filename);
      if (!TmpDir.empty()) {
        OptionalFileEntryRef ShouldBeEntry = FM.getOptionalFileRef(
            TmpDir, OpenFile, /*CacheFailure=*/true, /*IsText=*/false);
        if (ShouldBeEntry) {
          if (FoundEntry)
            *FoundEntry = LookupFromFile->getDir().getName();
          return ShouldBeEntry;
        }
      }
    } else {
      // Otherwise, do working directory lookup.
      LookupPath.clear();
      auto MaybeWorkingDirEntry = FM.getOptionalDirectoryRef(".");
      if (MaybeWorkingDirEntry) {
        DirectoryEntryRef WorkingDirEntry = *MaybeWorkingDirEntry;
        StringRef WorkingDir = WorkingDirEntry.getName();
        if (!WorkingDir.empty()) {
          SeparateComponents(LookupPath, WorkingDir, Filename, false);
          OptionalFileEntryRef ShouldBeEntry = FM.getOptionalFileRef(
              LookupPath, OpenFile, /*CacheFailure=*/true, /*IsText=*/false);
          if (ShouldBeEntry) {
            if (FoundEntry)
              *FoundEntry = WorkingDir;
            return ShouldBeEntry;
          }
        }
      }
    }
  }

  for (const auto &Entry : SearchEntries) {
    LookupPath.clear();
    SeparateComponents(LookupPath, Entry, Filename, false);
    OptionalFileEntryRef ShouldBeEntry = FM.getOptionalFileRef(
        LookupPath, OpenFile, /*CacheFailure=*/true, /*IsText=*/false);
    if (ShouldBeEntry) {
      if (FoundEntry)
        *FoundEntry = Entry;
      return ShouldBeEntry;
    }
  }
  return std::nullopt;
}

OptionalFileEntryRef LookupFileWithStdVec(
    StringRef Filename, bool isAngled, bool OpenFile, FileManager &FM,
    const std::vector<std::string> &SearchEntries,
    OptionalFileEntryRef LookupFromFile, StringRef *FoundEntry) {
  return LookupFileWithImpl(Filename, isAngled, OpenFile, FM, SearchEntries,
                            LookupFromFile, FoundEntry);
}

OptionalFileEntryRef LookupFileWith(StringRef Filename, bool isAngled,
                                    bool OpenFile, FileManager &FM,
                                    ArrayRef<StringRef> SearchEntries,
                                    OptionalFileEntryRef LookupFromFile,
                                    StringRef *FoundEntry) {
  return LookupFileWithImpl(Filename, isAngled, OpenFile, FM, SearchEntries,
                            LookupFromFile, FoundEntry);
}
} // namespace clang

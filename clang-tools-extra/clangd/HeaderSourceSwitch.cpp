//===--- HeaderSourceSwitch.cpp - --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderSourceSwitch.h"

namespace clang {
namespace clangd {

llvm::Optional<Path> getCorrespondingHeaderOrSource(
    const Path &OriginalFile,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  llvm::StringRef SourceExtensions[] = {".cpp", ".c", ".cc", ".cxx",
                                        ".c++", ".m", ".mm"};
  llvm::StringRef HeaderExtensions[] = {".h", ".hh", ".hpp", ".hxx", ".inc"};

  llvm::StringRef PathExt = llvm::sys::path::extension(OriginalFile);

  // Lookup in a list of known extensions.
  auto SourceIter =
      llvm::find_if(SourceExtensions, [&PathExt](PathRef SourceExt) {
        return SourceExt.equals_lower(PathExt);
      });
  bool IsSource = SourceIter != std::end(SourceExtensions);

  auto HeaderIter =
      llvm::find_if(HeaderExtensions, [&PathExt](PathRef HeaderExt) {
        return HeaderExt.equals_lower(PathExt);
      });
  bool IsHeader = HeaderIter != std::end(HeaderExtensions);

  // We can only switch between the known extensions.
  if (!IsSource && !IsHeader)
    return None;

  // Array to lookup extensions for the switch. An opposite of where original
  // extension was found.
  llvm::ArrayRef<llvm::StringRef> NewExts;
  if (IsSource)
    NewExts = HeaderExtensions;
  else
    NewExts = SourceExtensions;

  // Storage for the new path.
  llvm::SmallString<128> NewPath = llvm::StringRef(OriginalFile);

  // Loop through switched extension candidates.
  for (llvm::StringRef NewExt : NewExts) {
    llvm::sys::path::replace_extension(NewPath, NewExt);
    if (VFS->exists(NewPath))
      return NewPath.str().str(); // First str() to convert from SmallString to
                                  // StringRef, second to convert from StringRef
                                  // to std::string

    // Also check NewExt in upper-case, just in case.
    llvm::sys::path::replace_extension(NewPath, NewExt.upper());
    if (VFS->exists(NewPath))
      return NewPath.str().str();
  }
  return None;
}

} // namespace clangd
} // namespace clang

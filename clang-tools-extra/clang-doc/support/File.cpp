//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "File.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace doc {

llvm::Error copyFile(llvm::StringRef FilePath, llvm::StringRef OutDirectory) {
  llvm::SmallString<128> PathWrite;
  llvm::sys::path::native(OutDirectory, PathWrite);
  llvm::sys::path::append(PathWrite, llvm::sys::path::filename(FilePath));
  llvm::SmallString<128> PathRead;
  llvm::sys::path::native(FilePath, PathRead);
  std::error_code FileErr = llvm::sys::fs::copy_file(PathRead, PathWrite);
  if (FileErr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error creating file " +
                                       llvm::sys::path::filename(FilePath) +
                                       ": " + FileErr.message() + "\n");
  }
  return llvm::Error::success();
}

llvm::SmallString<128> computeRelativePath(llvm::StringRef Destination,
                                           llvm::StringRef Origin) {
  // If Origin is empty, the relative path to the Destination is its complete
  // path.
  if (Origin.empty())
    return Destination;

  // The relative path is an empty path if both directories are the same.
  if (Destination == Origin)
    return {};

  // These iterators iterate through each of their parent directories
  llvm::sys::path::const_iterator FileI = llvm::sys::path::begin(Destination);
  llvm::sys::path::const_iterator FileE = llvm::sys::path::end(Destination);
  llvm::sys::path::const_iterator DirI = llvm::sys::path::begin(Origin);
  llvm::sys::path::const_iterator DirE = llvm::sys::path::end(Origin);
  // Advance both iterators until the paths differ. Example:
  //    Destination = A/B/C/D
  //    Origin      = A/B/E/F
  // FileI will point to C and DirI to E. The directories behind them is the
  // directory they share (A/B).
  while (FileI != FileE && DirI != DirE && *FileI == *DirI) {
    ++FileI;
    ++DirI;
  }
  llvm::SmallString<128> Result; // This will hold the resulting path.
  // Result has to go up one directory for each of the remaining directories in
  // Origin
  while (DirI != DirE) {
    llvm::sys::path::append(Result, "..");
    ++DirI;
  }
  // Result has to append each of the remaining directories in Destination
  while (FileI != FileE) {
    llvm::sys::path::append(Result, *FileI);
    ++FileI;
  }
  return Result;
}

} // namespace doc
} // namespace clang

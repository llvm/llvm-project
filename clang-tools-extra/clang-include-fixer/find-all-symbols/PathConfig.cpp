//===-- PathConfig.cpp - Process paths of symbols ---------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PathConfig.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace find_all_symbols {

std::string getIncludePath(const SourceManager &SM, SourceLocation Loc,
                           const HeaderMapCollector *Collector) {
  llvm::StringRef FilePath;
  // Walk up the include stack to skip .inc files.
  while (true) {
    if (!Loc.isValid() || SM.isInMainFile(Loc))
      return "";
    FilePath = SM.getFilename(Loc);
    if (FilePath.empty())
      return "";
    if (!FilePath.endswith(".inc"))
      break;
    FileID ID = SM.getFileID(Loc);
    Loc = SM.getIncludeLoc(ID);
  }

  if (Collector)
    FilePath = Collector->getMappedHeader(FilePath);
  SmallString<256> CleanedFilePath = FilePath;
  llvm::sys::path::remove_dots(CleanedFilePath, /*remove_dot_dot=*/false);

  return CleanedFilePath.str();
}

} // namespace find_all_symbols
} // namespace clang

//===-- PathConfig.cpp - Process paths of symbols ---------------*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

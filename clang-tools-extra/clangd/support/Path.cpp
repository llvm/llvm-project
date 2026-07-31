//===--- Path.cpp -------------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Path.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include <optional>
namespace clang {
namespace clangd {

#ifdef CLANGD_PATH_CASE_INSENSITIVE
std::string maybeCaseFoldPath(PathRef Path) { return Path.lower(); }
bool pathEqual(PathRef A, PathRef B) { return A.equals_insensitive(B); }
#else  // NOT CLANGD_PATH_CASE_INSENSITIVE
std::string maybeCaseFoldPath(PathRef Path) { return Path.str(); }
bool pathEqual(PathRef A, PathRef B) { return A == B; }
#endif // CLANGD_PATH_CASE_INSENSITIVE

PathRef absoluteParent(PathRef Path) {
  assert(llvm::sys::path::is_absolute(Path));
#if defined(_WIN32)
  // llvm::sys says "C:\" is absolute, and its parent is "C:" which is relative.
  // This unhelpful behavior seems to have been inherited from boost.
  if (llvm::sys::path::relative_path(Path).empty()) {
    return PathRef();
  }
#endif
  PathRef Result = llvm::sys::path::parent_path(Path);
  assert(Result.empty() || llvm::sys::path::is_absolute(Result));
  return Result;
}

bool pathStartsWith(PathRef Ancestor, PathRef Path,
                    llvm::sys::path::Style Style) {
  assert(llvm::sys::path::is_absolute(Ancestor) &&
         llvm::sys::path::is_absolute(Path));
  // If ancestor ends with a separator drop that, so that we can match /foo/ as
  // a parent of /foo.
  if (llvm::sys::path::is_separator(Ancestor.back(), Style))
    Ancestor = Ancestor.drop_back();
  // Ensure Path starts with Ancestor.
  if (!pathEqual(Ancestor, Path.take_front(Ancestor.size())))
    return false;
  Path = Path.drop_front(Ancestor.size());
  // Then make sure either two paths are equal or Path has a separator
  // afterwards.
  return Path.empty() || llvm::sys::path::is_separator(Path.front(), Style);
}

llvm::Expected<Path>
mapPathAfterRenames(PathRef Original,
                    llvm::ArrayRef<std::pair<Path, Path>> Renames) {
  llvm::SmallString<256> NormalizedOriginal(Original);
  llvm::sys::path::remove_dots(NormalizedOriginal, /*remove_dot_dot=*/true);
  std::optional<Path> Result;
  for (const auto &[Old, New] : Renames) {
    llvm::SmallString<256> NormalizedOld(Old);
    llvm::sys::path::remove_dots(NormalizedOld, /*remove_dot_dot=*/true);
    llvm::SmallString<256> NormalizedNew(New);
    llvm::sys::path::remove_dots(NormalizedNew, /*remove_dot_dot=*/true);
    bool IsDescendant = llvm::sys::path::is_absolute(NormalizedOld) &&
                        llvm::sys::path::is_absolute(NormalizedOriginal) &&
                        pathStartsWith(NormalizedOld, NormalizedOriginal);
    if (!pathEqual(NormalizedOriginal, NormalizedOld) && !IsDescendant)
      continue;
    llvm::SmallString<256> Rewritten(NormalizedNew);
    llvm::StringRef Suffix =
        llvm::StringRef(NormalizedOriginal).drop_front(NormalizedOld.size());
    if (!Suffix.empty())
      llvm::sys::path::append(Rewritten,
                              llvm::sys::path::relative_path(Suffix));
    llvm::sys::path::remove_dots(Rewritten, /*remove_dot_dot=*/true);
    Path Candidate = Rewritten.str().str();
    if (Result && !pathEqual(*Result, Candidate))
      return llvm::createStringError(
          llvm::formatv("overlapping file renames map {0} to both {1} and {2}",
                        Original, *Result, Candidate)
              .str());
    Result = std::move(Candidate);
  }
  return Result.value_or(Original.str());
}
} // namespace clangd
} // namespace clang

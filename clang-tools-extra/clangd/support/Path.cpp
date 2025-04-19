//===--- Path.cpp -------------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Path.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <ostream>

namespace clang {
namespace clangd {

PathRef PathRef::absoluteParent() const {
  assert(llvm::sys::path::is_absolute(Data));
#if defined(_WIN32)
  // llvm::sys says "C:\" is absolute, and its parent is "C:" which is relative.
  // This unhelpful behavior seems to have been inherited from boost.
  if (llvm::sys::path::relative_path(Data).empty()) {
    return PathRef();
  }
#endif
  llvm::StringRef Result = llvm::sys::path::parent_path(Data);
  assert(Result.empty() || llvm::sys::path::is_absolute(Result));
  return Result;
}

PathRef PathRef::parentPath(Style Style) const {
  return llvm::sys::path::parent_path(Data, Style);
}

bool PathRef::startsWith(PathRef Path, Style Style) const {
  assert(isAbsolute() && Path.isAbsolute()); // XXX: this doesn't pass the Style

  // If ancestor ends with a separator drop that, so that we can match /foo/ as
  // a parent of /foo.
  PathRef Ancestor = withoutTrailingSeparator(Style);
  // Ensure Path starts with Ancestor.
  if (Ancestor != PathRef(Path.raw().take_front(Ancestor.size())))
    return false;
  Path = Path.Data.drop_front(Ancestor.size());
  // Then make sure either two paths are equal or Path has a separator
  // afterwards.
  return Path.empty() ||
         llvm::sys::path::is_separator(Path.raw().front(), Style);
}

llvm::StringRef PathRef::filename(Style Style) const {
  return llvm::sys::path::filename(raw(), Style);
}

llvm::StringRef PathRef::extension(Style Style) const {
  return llvm::sys::path::extension(raw(), Style);
}

PathRef PathRef::stem(Style Style) const {
  return llvm::sys::path::stem(raw(), Style);
}

PathRef PathRef::withoutTrailingSeparator(Style Style) const {
  if (llvm::sys::path::is_separator(Data.back(), Style))
    return Data.drop_back();
  return Data;
}

Path PathRef::removeDots() const {
  llvm::SmallString<128> CanonPath(Data);
  llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
  return CanonPath.str().str();
}

Path PathRef::caseFolded() const {
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  return Data.lower();
#else
  return Data.str();
#endif
}

bool PathRef::isAbsolute(Style Style) const {
  return llvm::sys::path::is_absolute(Data, Style);
}

bool PathRef::isRelative(Style Style) const {
  return llvm::sys::path::is_relative(Data, Style);
}

bool PathRef::exists() const { return llvm::sys::fs::exists(Data); }

std::ostream &operator<<(std::ostream &OS, PathRef Path) {
  OS << std::string_view(Path.raw());
  return OS;
}

bool operator==(PathRef LHS, PathRef RHS) {
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  return LHS.raw().equals_insensitive(RHS.raw());
#else
  return LHS.raw() == RHS.raw();
#endif
}

llvm::hash_code hash_value(PathRef P) {
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  return hash_combine_range(llvm::map_iterator(P.raw().begin(), llvm::toLower),
                            llvm::map_iterator(P.raw().end(), llvm::toLower));
#else
  return hash_value(P.raw());
#endif
}

} // namespace clangd
} // namespace clang

namespace llvm {

unsigned DenseMapInfo<clang::clangd::PathRef, void>::getHashValue(
    clang::clangd::PathRef Val) {
  assert(Val.Data.data() != getEmptyKey().Data.data() &&
         "Cannot hash the empty key!");
  assert(Val.Data.data() != getTombstoneKey().Data.data() &&
         "Cannot hash the tombstone key!");
  return (unsigned)(hash_value(Val));
}

namespace cl {

void parser<clang::clangd::Path>::printOptionDiff(const Option &O,
                                                  clang::clangd::PathRef V,
                                                  const OptVal &Default,
                                                  size_t GlobalWidth) const {
  constexpr static const size_t MaxOptWidth = 8;

  printOptionName(O, GlobalWidth);
  outs() << "= " << V.raw();
  size_t NumSpaces = MaxOptWidth > V.size() ? MaxOptWidth - V.size() : 0;
  outs().indent(NumSpaces) << " (default: ";
  if (Default.hasValue())
    outs() << Default.getValue().raw();
  else
    outs() << "*no default*";
  outs() << ")\n";
}

void parser<clang::clangd::Path>::anchor() {}

} // namespace cl

} // namespace llvm

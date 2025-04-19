//===--- Path.h - Helper typedefs --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_PATH_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_PATH_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <string>

/// Whether current platform treats paths case insensitively.
#if defined(_WIN32) || defined(__APPLE__)
#define CLANGD_PATH_CASE_INSENSITIVE
#endif

namespace clang {
namespace clangd {

class PathRef;
class Path {
public:
  Path() = default;
  Path(std::string Data) : Data(std::move(Data)) {}
  Path(const char *Data) : Data(Data) {}

  explicit Path(PathRef Ref);

  operator PathRef() const;
  PathRef ref() const;

  [[nodiscard]] const std::string &raw() const & { return Data; }
  [[nodiscard]] std::string &&raw() && { return std::move(Data); }

  [[nodiscard]] size_t size() const { return Data.size(); }
  [[nodiscard]] bool empty() const { return Data.empty(); }

private:
  std::string Data;

  friend llvm::json::Value toJSON(const Path &Path) { return Path.Data; }

  friend bool fromJSON(const llvm::json::Value &Value, Path &Path,
                       llvm::json::Path Cursor) {
    return fromJSON(Value, Path.Data, Cursor);
  }
};

class LLVM_GSL_POINTER PathRef {
public:
  using Style = llvm::sys::path::Style;

  PathRef() = default;
  PathRef(llvm::StringRef Ref) : Data(Ref) {}
  PathRef(const std::string &Str) : Data(Str) {}
  PathRef(const char *Str) : Data(Str) {}
  template <unsigned int N>
  PathRef(const llvm::SmallString<N> &Str) : Data(Str) {}

  /// Variant of parent_path that operates only on absolute paths.
  /// Unlike parent_path doesn't consider C: a parent of C:\.
  [[nodiscard]] PathRef absoluteParent() const;

  [[nodiscard]] PathRef parentPath(Style Style = Style::native) const;

  /// Checks if \p this is a proper ancestor of \p Path. This is just a
  /// smarter lexical prefix match, e.g: foo/bar/baz doesn't start with
  /// foo/./bar. Both \p this and \p Path must be absolute.
  [[nodiscard]] bool startsWith(PathRef Path,
                                Style Style = Style::native) const;

  [[nodiscard]] llvm::StringRef filename(Style Style = Style::native) const;

  [[nodiscard]] llvm::StringRef extension(Style Style = Style::native) const;

  [[nodiscard]] PathRef stem(Style Style = Style::native) const;

  /// Returns a version of \p File that doesn't contain dots and dot dots.
  /// e.g /a/b/../c -> /a/c
  ///     /a/b/./c -> /a/b/c
  /// FIXME: We should avoid encountering such paths in clangd internals by
  /// filtering everything we get over LSP, CDB, etc.
  [[nodiscard]] Path removeDots() const;

  [[nodiscard]] Path caseFolded() const;

  [[nodiscard]] Path owned() const { return Path(*this); }
  [[nodiscard]] llvm::StringRef raw() const { return Data; }

  [[nodiscard]] PathRef
  withoutTrailingSeparator(Style Style = Style::native) const;

  [[nodiscard]] size_t size() const { return Data.size(); }

  [[nodiscard]] bool empty() const { return Data.empty(); }

  [[nodiscard]] bool isAbsolute(Style Style = Style::native) const;
  [[nodiscard]] bool isRelative(Style Style = Style::native) const;

  [[nodiscard]] bool exists() const;

private:
  llvm::StringRef Data;

  friend llvm::DenseMapInfo<clang::clangd::PathRef, void>;
};

inline Path::Path(PathRef Ref) : Data(Ref.raw().str()) {}

// for gtest
std::ostream &operator<<(std::ostream &OS, PathRef Path);

inline Path::operator PathRef() const { return PathRef(Data); }
inline PathRef Path::ref() const { return PathRef(Data); }

bool operator==(PathRef LHS, PathRef RHS);
inline bool operator!=(PathRef LHS, PathRef RHS) { return !(LHS == RHS); }

inline bool operator==(Path LHS, Path RHS) {
  return clang::clangd::PathRef(LHS) == clang::clangd::PathRef(RHS);
}
inline bool operator!=(Path LHS, Path RHS) { return !(LHS == RHS); }

[[nodiscard]] llvm::hash_code hash_value(PathRef P);
[[nodiscard]] inline llvm::hash_code hash_value(const Path &P) {
  return hash_value(PathRef(P));
}

inline llvm::json::Value toJson(PathRef Path) { return Path.raw(); }

} // namespace clangd
} // namespace clang

namespace llvm {

template <> struct format_provider<clang::clangd::PathRef> {
  static void format(const clang::clangd::PathRef &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    format_provider<llvm::StringRef>::format(V.raw(), Stream, Style);
  }
};

template <> struct format_provider<clang::clangd::Path> {
  static void format(const clang::clangd::Path &V, llvm::raw_ostream &Stream,
                     StringRef Style) {
    format_provider<clang::clangd::PathRef>::format(V, Stream, Style);
  }
};

template <> struct DenseMapInfo<clang::clangd::PathRef, void> {
  static inline clang::clangd::PathRef getEmptyKey() {
    return DenseMapInfo<llvm::StringRef>::getEmptyKey();
  }

  static inline clang::clangd::PathRef getTombstoneKey() {
    return DenseMapInfo<llvm::StringRef>::getTombstoneKey();
  }

  static unsigned getHashValue(clang::clangd::PathRef Val);

  static bool isEqual(clang::clangd::PathRef LHS, clang::clangd::PathRef RHS) {
    if (RHS.Data.data() == getEmptyKey().Data.data())
      return LHS.Data.data() == getEmptyKey().Data.data();
    if (RHS.Data.data() == getTombstoneKey().Data.data())
      return LHS.Data.data() == getTombstoneKey().Data.data();
    return LHS == RHS;
  }
};

namespace cl {

template <>
struct parser<clang::clangd::Path> : public basic_parser<clang::clangd::Path> {
public:
  parser(Option &O) : basic_parser(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef ArgValue,
             clang::clangd::Path &Val) {
    Val = ArgValue.str();
    return false;
  }

  StringRef getValueName() const override { return "path"; }

  void printOptionDiff(const Option &O, clang::clangd::PathRef V,
                       const OptVal &Default, size_t GlobalWidth) const;

  void anchor() override;
};

} // namespace cl

} // namespace llvm

#endif

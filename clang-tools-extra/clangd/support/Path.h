//===--- Path.h - Path identity for clangd -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Identity (equality, hashing, map keys) preserves filename case: even Windows
// and macOS can have case-sensitive directories. Only absolute Windows drive
// letters are folded, and Windows path separators are interchangeable. This
// handles the common CMake vs LSP mismatch (C: vs c:) without merging distinct
// files. Operations that explicitly require legacy host-default case folding
// use pathEqualLegacyCaseFold() or maybeCaseFoldPath() instead.
//
// Path/PathRef deliberately do not convert to StringRef implicitly. Callers
// must use raw() when crossing into string-based APIs, making the loss of path
// identity visible at the call site.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_PATH_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_PATH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

/// Whether legacy comparisons assume case-insensitive paths on this platform.
#if defined(_WIN32) || defined(__APPLE__)
#define CLANGD_PATH_CASE_INSENSITIVE
#endif

namespace clang {
namespace clangd {

class PathRef;

/// Whether the path starts with an ASCII drive letter and ':'.
/// This includes drive-relative paths such as C:foo, not just C:/foo.
bool hasWindowsDrive(llvm::StringRef Path);

/// Owned filesystem path with conservative lexical identity.
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
  friend struct llvm::DenseMapInfo<Path, void>;
};

/// Non-owning filesystem path with the same identity as Path.
class LLVM_GSL_POINTER PathRef {
public:
  using Style = llvm::sys::path::Style;

  PathRef() = default;
  PathRef(llvm::StringRef Ref) : Data(Ref) {}
  PathRef(const std::string &Str) : Data(Str) {}
  PathRef(const char *Str) : Data(Str) {}
  template <unsigned N> PathRef(const llvm::SmallString<N> &Str) : Data(Str) {}

  /// Variant of parent_path that operates only on absolute paths.
  /// Unlike parent_path doesn't consider C: a parent of C:\.
  [[nodiscard]] PathRef absoluteParent() const;

  [[nodiscard]] PathRef parentPath(Style Style = Style::native) const {
    return llvm::sys::path::parent_path(Data, Style);
  }

  /// True if this is a proper ancestor of \p Other, or the same path.
  /// Lexical only: foo/bar/baz does not start with foo/./bar.
  /// Both paths must be absolute.
  /// Retains legacy host-default case folding, unlike path identity equality.
  [[nodiscard]] bool isAncestorOf(PathRef Other,
                                  Style Style = Style::native) const;

  [[nodiscard]] llvm::StringRef filename(Style Style = Style::native) const {
    return llvm::sys::path::filename(Data, Style);
  }
  [[nodiscard]] llvm::StringRef extension(Style Style = Style::native) const {
    return llvm::sys::path::extension(Data, Style);
  }
  [[nodiscard]] PathRef stem(Style Style = Style::native) const {
    return llvm::sys::path::stem(Data, Style);
  }

  [[nodiscard]] Path removeDots() const;
  /// Canonical spelling for lexical identity, not a filesystem realpath.
  [[nodiscard]] Path identityNormalized() const;
  /// Explicit legacy host-default case folding, not the identity of Path.
  [[nodiscard]] Path caseFolded() const;
  [[nodiscard]] Path owned() const { return Path(*this); }
  [[nodiscard]] llvm::StringRef raw() const { return Data; }

  [[nodiscard]] PathRef
  withoutTrailingSeparator(Style Style = Style::native) const {
    if (!Data.empty() && llvm::sys::path::is_separator(Data.back(), Style))
      return Data.drop_back();
    return Data;
  }

  [[nodiscard]] size_t size() const { return Data.size(); }
  [[nodiscard]] bool empty() const { return Data.empty(); }
  [[nodiscard]] bool isAbsolute(Style Style = Style::native) const {
    return llvm::sys::path::is_absolute(Data, Style);
  }
  [[nodiscard]] bool isRelative(Style Style = Style::native) const {
    return llvm::sys::path::is_relative(Data, Style);
  }
  [[nodiscard]] bool exists() const;

private:
  llvm::StringRef Data;

  friend struct llvm::DenseMapInfo<PathRef, void>;
};

inline Path::Path(PathRef Ref) : Data(Ref.raw().str()) {}
inline Path::operator PathRef() const { return PathRef(Data); }
inline PathRef Path::ref() const { return PathRef(Data); }

// For gtest diagnostics.
std::ostream &operator<<(std::ostream &OS, PathRef Path);

/// Conservative lexical path identity (see Path.cpp).
bool pathEquals(llvm::StringRef LHS, llvm::StringRef RHS);
unsigned pathHash(llvm::StringRef P);
/// Orders normalized identities without allocating temporary strings.
int pathCompare(llvm::StringRef LHS, llvm::StringRef RHS);

inline bool operator==(PathRef LHS, PathRef RHS) {
  return pathEquals(LHS.raw(), RHS.raw());
}
inline bool operator!=(PathRef LHS, PathRef RHS) { return !(LHS == RHS); }
inline bool operator==(const Path &LHS, const Path &RHS) {
  return PathRef(LHS) == PathRef(RHS);
}
inline bool operator!=(const Path &LHS, const Path &RHS) {
  return !(LHS == RHS);
}

inline llvm::hash_code hash_value(PathRef P) { return pathHash(P.raw()); }
inline llvm::hash_code hash_value(const Path &P) { return hash_value(P.ref()); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, PathRef P);
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Path &P) {
  return OS << P.ref();
}

inline llvm::json::Value toJSON(PathRef P) { return P.raw(); }

// Explicit legacy comparisons retain their host-default case folding.
inline Path maybeCaseFoldPath(PathRef P) { return P.caseFolded(); }
bool pathEqualLegacyCaseFold(PathRef A, PathRef B);

/// Map keyed by Path. Lookups normalize drive letters and Windows separators,
/// preserving filename case and the first-inserted spelling.
template <typename ValueT> class PathMap {
  llvm::DenseMap<Path, ValueT> Impl;

public:
  using MapType = llvm::DenseMap<Path, ValueT>;
  using iterator = typename MapType::iterator;
  using const_iterator = typename MapType::const_iterator;
  using value_type = typename MapType::value_type;
  using mapped_type = ValueT;
  using key_type = Path;

  PathMap() = default;
  PathMap(std::initializer_list<std::pair<Path, ValueT>> Init) {
    for (auto &E : Init)
      try_emplace(std::move(E.first), std::move(E.second));
  }

  iterator begin() { return Impl.begin(); }
  const_iterator begin() const { return Impl.begin(); }
  iterator end() { return Impl.end(); }
  const_iterator end() const { return Impl.end(); }

  bool empty() const { return Impl.empty(); }
  size_t size() const { return Impl.size(); }
  void clear() { Impl.clear(); }

  iterator find(PathRef Key) { return Impl.find_as(Key); }
  const_iterator find(PathRef Key) const { return Impl.find_as(Key); }

  bool contains(PathRef Key) const { return find(Key) != end(); }

  ValueT lookup(PathRef Key) const {
    auto It = find(Key);
    return It == end() ? ValueT() : It->second;
  }

  ValueT &operator[](PathRef Key) {
    if (auto It = find(Key); It != end())
      return It->second;
    return Impl[Path(Key)];
  }

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(PathRef Key, Args &&...Rest) {
    if (auto It = find(Key); It != end())
      return {It, false};
    return Impl.try_emplace(Path(Key), std::forward<Args>(Rest)...);
  }

  // When the caller already owns the key, DenseMap can both check and insert
  // it with one hash, and only copies/moves the string when insertion occurs.
  template <
      typename KeyT, typename... Args,
      std::enable_if_t<
          std::is_same_v<std::remove_cv_t<std::remove_reference_t<KeyT>>, Path>,
          int> = 0>
  std::pair<iterator, bool> try_emplace(KeyT &&Key, Args &&...Rest) {
    return Impl.try_emplace(std::forward<KeyT>(Key),
                            std::forward<Args>(Rest)...);
  }

  std::pair<iterator, bool> insert(const std::pair<Path, ValueT> &KV) {
    return try_emplace(KV.first, KV.second);
  }
  std::pair<iterator, bool> insert(std::pair<Path, ValueT> &&KV) {
    return try_emplace(std::move(KV.first), std::move(KV.second));
  }

  bool erase(PathRef Key) {
    auto It = find(Key);
    if (It == end())
      return false;
    Impl.erase(It);
    return true;
  }
  void erase(iterator It) { Impl.erase(It); }

  size_t getMemorySize() const { return Impl.getMemorySize(); }

  llvm::SmallVector<PathRef, 8> keys() const {
    llvm::SmallVector<PathRef, 8> Result;
    Result.reserve(Impl.size());
    for (const auto &E : Impl)
      Result.push_back(E.first);
    return Result;
  }
};

using PathSet = llvm::DenseSet<Path>;

} // namespace clangd
} // namespace clang

namespace llvm {

template <> struct format_provider<clang::clangd::PathRef> {
  static void format(const clang::clangd::PathRef &V, raw_ostream &Stream,
                     StringRef Style) {
    format_provider<StringRef>::format(V.raw(), Stream, Style);
  }
};

template <> struct format_provider<clang::clangd::Path> {
  static void format(const clang::clangd::Path &V, raw_ostream &Stream,
                     StringRef Style) {
    format_provider<clang::clangd::PathRef>::format(V, Stream, Style);
  }
};

template <> struct DenseMapInfo<clang::clangd::PathRef, void> {
  static unsigned getHashValue(clang::clangd::PathRef Val) {
    return (unsigned)hash_value(Val);
  }
  static bool isEqual(clang::clangd::PathRef LHS, clang::clangd::PathRef RHS) {
    return LHS == RHS;
  }
};

template <> struct DenseMapInfo<clang::clangd::Path, void> {
  static unsigned getHashValue(const clang::clangd::Path &Val) {
    return (unsigned)hash_value(Val);
  }
  static unsigned getHashValue(clang::clangd::PathRef Val) {
    return (unsigned)hash_value(Val);
  }
  static bool isEqual(const clang::clangd::Path &LHS,
                      const clang::clangd::Path &RHS) {
    return LHS == RHS;
  }
  static bool isEqual(clang::clangd::PathRef LHS,
                      const clang::clangd::Path &RHS) {
    return LHS == clang::clangd::PathRef(RHS);
  }
};

namespace cl {

template <>
struct parser<clang::clangd::Path> : public basic_parser<clang::clangd::Path> {
public:
  parser(Option &O) : basic_parser(O) {}

  bool parse(Option &, StringRef, StringRef ArgValue,
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

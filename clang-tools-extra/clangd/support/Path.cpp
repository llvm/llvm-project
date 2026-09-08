//===--- Path.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <ostream>
#include <string_view>

namespace clang {
namespace clangd {

bool hasWindowsDrive(llvm::StringRef P) {
  return P.size() >= 2 && llvm::isAlpha(P[0]) && P[1] == ':';
}

namespace {

llvm::StringRef dropWindowsDrive(llvm::StringRef P) {
  return hasWindowsDrive(P) ? P.drop_front(2) : P;
}

bool isPathSep(char C) { return C == '/' || C == '\\'; }

bool isAbsoluteWindowsDrivePath(llvm::StringRef P) {
  return P.size() >= 3 && hasWindowsDrive(P) && isPathSep(P[2]);
}

bool isWindowsUNCPath(llvm::StringRef P) {
  return P.size() >= 2 && isPathSep(P[0]) && isPathSep(P[1]);
}

bool usesWindowsSeparators(llvm::StringRef P) {
#ifdef _WIN32
  return true;
#else
  return isAbsoluteWindowsDrivePath(P) || isWindowsUNCPath(P);
#endif
}

llvm::sys::path::Style pathStyle(llvm::StringRef P) {
  return usesWindowsSeparators(P) ? llvm::sys::path::Style::windows
                                  : llvm::sys::path::Style::native;
}

// Compare the tail of two Windows paths. Drive letter is already stripped.
// '/' and '\\' are the same separator. Full case folding is opt-in.
bool pathRestEqual(llvm::StringRef LHS, llvm::StringRef RHS,
                   bool NormalizeSeparators, bool IgnoreCase) {
  if (LHS.size() != RHS.size())
    return false;
  for (size_t I = 0; I < LHS.size(); ++I) {
    unsigned char A = LHS[I], B = RHS[I];
    if (NormalizeSeparators && isPathSep(A) && isPathSep(B))
      continue;
    if (IgnoreCase) {
      A = llvm::toLower(A);
      B = llvm::toLower(B);
    }
    if (A != B)
      return false;
  }
  return true;
}

llvm::StringRef slashNormalized(llvm::StringRef P,
                                llvm::SmallVectorImpl<char> &Buf) {
  if (!P.contains('\\'))
    return P;
  Buf.assign(P.begin(), P.end());
  for (char &C : Buf)
    if (C == '\\')
      C = '/';
  return llvm::StringRef(Buf.data(), Buf.size());
}

bool pathEqualsImpl(llvm::StringRef LHS, llvm::StringRef RHS, bool IgnoreCase) {
  if (LHS == RHS)
    return true;
  const bool NormalizeSeparators =
      usesWindowsSeparators(LHS) && usesWindowsSeparators(RHS);
  const bool BothDrivePaths =
      isAbsoluteWindowsDrivePath(LHS) && isAbsoluteWindowsDrivePath(RHS);
  if (BothDrivePaths) {
    if (llvm::toLower(LHS[0]) != llvm::toLower(RHS[0]))
      return false;
    LHS = dropWindowsDrive(LHS);
    RHS = dropWindowsDrive(RHS);
  }
  if (NormalizeSeparators)
    return pathRestEqual(LHS, RHS, /*NormalizeSeparators=*/true, IgnoreCase);
  return IgnoreCase ? LHS.equals_insensitive(RHS) : LHS == RHS;
}

Path normalizedIdentity(llvm::StringRef Data, bool FoldCase) {
  std::string Result = FoldCase ? Data.lower() : Data.str();
  if (isAbsoluteWindowsDrivePath(Data))
    Result[0] = llvm::toLower(Result[0]);
  if (usesWindowsSeparators(Data)) {
    for (char &C : Result)
      if (C == '\\')
        C = '/';
  }
  return Path(std::move(Result));
}

} // namespace

bool pathEquals(llvm::StringRef LHS, llvm::StringRef RHS) {
  return pathEqualsImpl(LHS, RHS, /*IgnoreCase=*/false);
}

bool pathEqual(PathRef LHS, PathRef RHS) {
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  return pathEqualsImpl(LHS.raw(), RHS.raw(), /*IgnoreCase=*/true);
#else
  return LHS == RHS;
#endif
}

unsigned pathHash(llvm::StringRef P) {
  llvm::SmallString<256> Norm;
  const bool HasWindowsDrive = isAbsoluteWindowsDrivePath(P);
  const bool NormalizeSeparators = usesWindowsSeparators(P);
  unsigned char Drive = 0;
  if (HasWindowsDrive) {
    Drive = llvm::toLower(P[0]);
    P = dropWindowsDrive(P);
  }
  if (NormalizeSeparators)
    P = slashNormalized(P, Norm);
  unsigned Result = static_cast<unsigned>(llvm::xxh3_64bits(P));
  // The normalized tail is the expensive part. Mix the drive into its hash
  // directly rather than running another general-purpose hash-combine step.
  if (HasWindowsDrive)
    Result ^= static_cast<unsigned>(Drive) * 0x9e3779b9U;
  return Result;
}

int pathCompare(llvm::StringRef LHS, llvm::StringRef RHS) {
  const bool LDrive = isAbsoluteWindowsDrivePath(LHS);
  const bool RDrive = isAbsoluteWindowsDrivePath(RHS);
  const bool LSeparators = usesWindowsSeparators(LHS);
  const bool RSeparators = usesWindowsSeparators(RHS);
  for (size_t I = 0, E = std::min(LHS.size(), RHS.size()); I != E; ++I) {
    unsigned char L = LHS[I], R = RHS[I];
    if (I == 0) {
      if (LDrive)
        L = llvm::toLower(L);
      if (RDrive)
        R = llvm::toLower(R);
    }
    if (LSeparators && L == '\\')
      L = '/';
    if (RSeparators && R == '\\')
      R = '/';
    if (L != R)
      return int(L) - int(R);
  }
  return LHS.size() < RHS.size() ? -1 : LHS.size() > RHS.size() ? 1 : 0;
}

PathRef PathRef::absoluteParent() const {
  assert(llvm::sys::path::is_absolute(Data));
#if defined(_WIN32)
  // llvm::sys says "C:\" is absolute, and its parent is "C:" which is relative.
  // This unhelpful behavior seems to have been inherited from boost.
  if (llvm::sys::path::relative_path(Data).empty())
    return PathRef();
#endif
  llvm::StringRef Result = llvm::sys::path::parent_path(Data);
  assert(Result.empty() || llvm::sys::path::is_absolute(Result));
  return Result;
}

bool PathRef::startsWith(PathRef Other, Style Style) const {
  // Style describes separators, not necessarily the host path's root syntax.
  // Config paths on Windows use POSIX separators but still have drive roots.
  assert((isAbsolute() || isAbsolute(Style)) &&
         (Other.isAbsolute() || Other.isAbsolute(Style)));
  PathRef Ancestor = withoutTrailingSeparator(Style);
  // Keep the root separator when comparing drive roots: C: alone is relative
  // and does not have case-insensitive drive-letter identity on POSIX hosts.
  if (Ancestor.size() == 2 && isAbsoluteWindowsDrivePath(Data))
    return pathEqual(Data, Other.raw().take_front(3));
  if (Ancestor.size() > Other.size())
    return false;
  if (!pathEqual(Ancestor.raw(), Other.raw().take_front(Ancestor.size())))
    return false;
  llvm::StringRef Rest = Other.raw().drop_front(Ancestor.size());
  // Windows paths treat both slashes as separators even when Style is native
  // on a POSIX host (Linux tests of C: vs c:).
  return Rest.empty() || llvm::sys::path::is_separator(Rest.front(), Style) ||
         (usesWindowsSeparators(Other.raw()) && isPathSep(Rest.front()));
}

Path PathRef::removeDots() const {
  llvm::SmallString<128> CanonPath(Data);
  llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true,
                               pathStyle(Data));
  return Path(CanonPath.str());
}

Path PathRef::identityNormalized() const {
  return normalizedIdentity(Data, /*FoldCase=*/false);
}

Path PathRef::caseFolded() const {
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  return normalizedIdentity(Data, /*FoldCase=*/true);
#else
  return identityNormalized();
#endif
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, PathRef P) {
  return OS << P.raw();
}

std::ostream &operator<<(std::ostream &OS, PathRef P) {
  return OS << std::string_view(P.raw().data(), P.raw().size());
}

bool PathRef::exists() const { return llvm::sys::fs::exists(Data); }

} // namespace clangd
} // namespace clang

namespace llvm {
namespace cl {

void parser<clang::clangd::Path>::printOptionDiff(const Option &O,
                                                  clang::clangd::PathRef V,
                                                  const OptVal &Default,
                                                  size_t GlobalWidth) const {
  constexpr size_t MaxOptWidth = 8;
  printOptionName(O, GlobalWidth);
  outs() << "= " << V.raw();
  outs().indent(MaxOptWidth > V.size() ? MaxOptWidth - V.size() : 0)
      << " (default: ";
  if (Default.hasValue())
    outs() << Default.getValue().raw();
  else
    outs() << "*no default*";
  outs() << ")\n";
}

void parser<clang::clangd::Path>::anchor() {}

} // namespace cl
} // namespace llvm

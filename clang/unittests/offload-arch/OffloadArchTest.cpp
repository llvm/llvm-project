//===-- OffloadArchTest.cpp - Tests for offload-arch helpers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// Defined in AMDGPUArchByHIP.cpp (non-static, compiled into this test).
#ifdef _WIN32
using LoadLibraryExWFn = decltype(&::LoadLibraryExW);

bool compareVersions(llvm::StringRef A, llvm::StringRef B);
llvm::SmallVector<std::string, 8> getCandidateBinPaths(llvm::StringRef ExeDir);
void primeLibraryLoad(llvm::StringRef Path, LoadLibraryExWFn Loader);
#endif

using namespace llvm;

cl::opt<bool> Verbose("offload-arch-test-verbose", cl::Hidden, cl::init(false));

#ifdef _WIN32

// --- primeLibraryLoad ---

namespace {
std::wstring CapturedPath;
HANDLE CapturedFile;
DWORD CapturedFlags;

HMODULE WINAPI mockLoadLibraryExW(LPCWSTR Path, HANDLE File, DWORD Flags) {
  CapturedPath = Path;
  CapturedFile = File;
  CapturedFlags = Flags;
  return reinterpret_cast<HMODULE>(1);
}
} // namespace

TEST(PrimeLibraryLoad, UsesWindowsBackslashes) {
  CapturedPath.clear();
  CapturedFile = reinterpret_cast<HANDLE>(1);
  CapturedFlags = 0;

  primeLibraryLoad("C:/rocm\\bin/amdhip64_7.dll", mockLoadLibraryExW);

  // The underlying LoadLibraryExW function should be called with backslashes,
  // not forward slashes to ensure that it can discover and load dependent
  // DLLs in the same directory.
  EXPECT_EQ(CapturedPath, L"C:\\rocm\\bin\\amdhip64_7.dll");
  EXPECT_EQ(CapturedPath.find(L'/'), std::wstring::npos);
  EXPECT_EQ(CapturedFile, nullptr);
  EXPECT_EQ(CapturedFlags, LOAD_WITH_ALTERED_SEARCH_PATH);
}

// --- compareVersions ---

TEST(CompareVersions, HigherVersionWins) {
  EXPECT_TRUE(
      compareVersions("C:/bin/amdhip64_7.dll", "C:/bin/amdhip64_6.dll"));
  EXPECT_FALSE(
      compareVersions("C:/bin/amdhip64_6.dll", "C:/bin/amdhip64_7.dll"));
}

TEST(CompareVersions, EqualVersionsReturnFalse) {
  EXPECT_FALSE(compareVersions("C:/a/amdhip64_7.dll", "C:/b/amdhip64_7.dll"));
}

TEST(CompareVersions, MultiDigitVersions) {
  EXPECT_TRUE(compareVersions("amdhip64_12.dll", "amdhip64_6.dll"));
}

TEST(CompareVersions, StableSortPreservesInsertionOrder) {
  std::vector<std::string> DLLs = {"C:/rocm/bin/amdhip64_7.dll",
                                   "C:/Windows/System32/amdhip64_7.dll"};
  llvm::stable_sort(DLLs, compareVersions);
  EXPECT_EQ(DLLs[0], "C:/rocm/bin/amdhip64_7.dll");
}

// --- getCandidateBinPaths ---

TEST(CandidateBinPaths, FindsParentBin) {
  auto Paths = getCandidateBinPaths("C:/root/lib/llvm/bin");
  bool Found = false;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/root/bin"))
      Found = true;
  EXPECT_TRUE(Found);
}

TEST(CandidateBinPaths, NoDuplicatesWhenExeInBin) {
  auto Paths = getCandidateBinPaths("C:/root/bin");
  int Count = 0;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/root/bin"))
      Count++;
  EXPECT_EQ(Count, 1);
}

TEST(CandidateBinPaths, CaseInsensitiveDedup) {
  // Paths differing only in case should not both appear.
  auto Paths = getCandidateBinPaths("C:/Root/Lib/Bin");
  int Count = 0;
  for (const auto &P : Paths)
    if (StringRef(P).equals_insensitive("C:/Root/bin"))
      Count++;
  EXPECT_LE(Count, 1);
}

TEST(CandidateBinPaths, StopsWithinBound) {
  auto Paths = getCandidateBinPaths("C:/a/b/c/d/e/f/g/h");
  // MaxParentLevels=6 + self = 7 max entries.
  EXPECT_LE(Paths.size(), 7u);
}

TEST(CandidateBinPaths, RootInput) {
  auto Paths = getCandidateBinPaths("C:/");
  // Should produce at least 1 entry (self) and not crash.
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, NonAsciiPath) {
  // Paths with non-ASCII characters should not crash.
  auto Paths = getCandidateBinPaths("C:/\xC3\xBCser/\xC3\xA4pp/bin");
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, UnicodePathDedup) {
  auto Paths =
      getCandidateBinPaths("C:/\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E/lib/bin");
  // Should produce entries without crashing on CJK characters.
  EXPECT_GE(Paths.size(), 1u);
}

TEST(CandidateBinPaths, NoDriveRootBin) {
  auto Paths = getCandidateBinPaths("C:\\Program Files\\AMD\\HIP\\bin");
  for (const auto &P : Paths)
    EXPECT_FALSE(StringRef(P).equals_insensitive("C:/bin"))
        << "Drive-root bin/ must not appear (DLL planting risk)";
}

#endif // _WIN32

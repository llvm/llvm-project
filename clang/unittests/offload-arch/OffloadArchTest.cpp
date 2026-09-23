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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

// Defined in AMDGPUArchByHIP.cpp (non-static, compiled into this test).
#ifdef _WIN32
bool compareVersions(llvm::StringRef A, llvm::StringRef B);
llvm::SmallVector<std::string, 8> getCandidateBinPaths(llvm::StringRef ExeDir);
#endif

// Defined in AMDGPUArchByKFD.cpp (non-static, compiled into this test).
int printGPUsByKFD(llvm::StringRef NodePath);

using namespace llvm;

cl::opt<bool> Verbose("offload-arch-test-verbose", cl::Hidden, cl::init(false));

#ifdef _WIN32

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

// --- printGPUsByKFD ---

namespace {
// Write <Dir>/<Node>/properties containing the given lines.
void addNode(StringRef Dir, unsigned Node, StringRef Properties) {
  SmallString<128> NodeDir(Dir);
  sys::path::append(NodeDir, Twine(Node));
  ASSERT_FALSE(sys::fs::create_directories(NodeDir));

  SmallString<128> PropertiesPath(NodeDir);
  sys::path::append(PropertiesPath, "properties");
  std::error_code EC;
  raw_fd_ostream OS(PropertiesPath, EC);
  ASSERT_FALSE(EC);
  OS << Properties;
}

// Write a node describing a GPU with the given gfx_target_version.
void addGPUNode(StringRef Dir, unsigned Node, StringRef GFXVersion) {
  addNode(Dir, Node, ("gfx_target_version " + GFXVersion + "\n").str());
}

// Write a node describing a GPU with the given gfx_target_version and
// capabilities. Write capability2 before and after to catch accidental
// reads of something other than "capability"
void addGPUNodeWithCapability(StringRef Dir, unsigned Node,
                              StringRef GFXVersion, uint64_t Capability,
                              uint64_t Capability2) {
  addNode(Dir, Node,
          ("gfx_target_version " + GFXVersion + "\n" + "capability2 " +
           Twine(Capability2) + "\n" + "capability " + Twine(Capability) +
           "\n" + "capability2 " + Twine(Capability2) + "\n")
              .str());
}

// Run printGPUsByKFD, collecting what it writes to stdout.
int printGPUsByKFDCapturingStdout(StringRef NodePath, std::string &Output) {
  testing::internal::CaptureStdout();
  int Result = printGPUsByKFD(NodePath);
  outs().flush();
  Output = testing::internal::GetCapturedStdout();
  return Result;
}

// RAII helper to set an environment variable for the duration of a test
// Based on the class of the same name in llvm's Jobserver unit tests
class ScopedEnvironment {
  std::string Name;
  std::optional<std::string> OldValue;

  static void setEnv(const std::string &Name, std::optional<StringRef> Value) {
#if defined(_WIN32)
    // On Windows, setting an environment variable to the empty string
    // unsets it, so getenv() returns NULL
    _putenv_s(Name.c_str(), Value ? Value->str().c_str() : "");
#else
    if (Value)
      setenv(Name.c_str(), Value->str().c_str(), 1);
    else
      unsetenv(Name.c_str());
#endif
  }

public:
  ScopedEnvironment(StringRef Name, std::optional<StringRef> Value)
      : Name(Name.str()), OldValue(sys::Process::GetEnv(Name)) {
    setEnv(this->Name, Value);
  }

  ~ScopedEnvironment() {
    setEnv(Name, OldValue ? std::optional<StringRef>(*OldValue) : std::nullopt);
  }

  ScopedEnvironment(const ScopedEnvironment &) = delete;
  ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;
};
} // namespace

// A topology directory that cannot be opened must be reported as a failure, so
// that the caller falls back to enumerating with the HIP runtime.
TEST(KFDTopology, MissingDirectoryFails) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path("does-not-exist"), Output),
            1);
  EXPECT_EQ(Output, "");
}

// A readable topology describing no GPUs is not an error, and prints nothing.
TEST(KFDTopology, CPUOnlyTopologySucceeds) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNode(Dir.path(), 0, "0");
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "");
}

// A node whose properties do not mention gfx_target_version is a CPU too.
TEST(KFDTopology, NodeWithoutGFXVersionSucceeds) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addNode(Dir.path(), 0, "cpu_cores_count 16\n");
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "");
}

TEST(KFDTopology, EmptyTopologySucceeds) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "");
}

TEST(KFDTopology, GPUNodeIsPrinted) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNode(Dir.path(), 0, "0");      // CPU
  addGPUNode(Dir.path(), 1, "110001"); // gfx1101
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1101\n");
}

// Devices are printed in node order, and the step is printed in hex so that
// e.g. gfx90a renders correctly.
TEST(KFDTopology, MultipleGPUsArePrintedInNodeOrder) {
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNode(Dir.path(), 0, "0");      // CPU
  addGPUNode(Dir.path(), 2, "90010");  // gfx90a
  addGPUNode(Dir.path(), 1, "110001"); // gfx1101
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1101\ngfx90a\n");
}

// Make sure that A0 of gfx1250 is printed as gfx1250-strict. Happens when
// ASIC revision is 0. Also tests to make sure other properties that look like
// capability (like capability2) are not read instead.
TEST(KFDTopology, GFX1250A0IsPrintedAsStrict) {
  // A temporary patch in ROCr requires this env var to be 0
  ScopedEnvironment Env("HSA_DISABLE_GFX12_STRICT", "0");
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNodeWithCapability(Dir.path(), 0, "120500", /*Capability=*/0xF837A280,
                           /*Capability2=*/0xFFFFFFFF);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1250-strict\n");
}

// HSA_DISABLE_GFX12_STRICT=1 suppresses the suffix on A0.
TEST(KFDTopology, GFX1250A0StrictSuppressedByEnvVar) {
  ScopedEnvironment Env("HSA_DISABLE_GFX12_STRICT", "1");
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNodeWithCapability(Dir.path(), 0, "120500", /*Capability=*/0xF837A280,
                           /*Capability2=*/0xFFFFFFFF);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1250\n");
}

// Temporary test: Only the exact value "0" enables the suffix.
TEST(KFDTopology, GFX1250A0StrictIgnoresOtherEnvValues) {
  ScopedEnvironment Env("HSA_DISABLE_GFX12_STRICT", std::nullopt);
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNodeWithCapability(Dir.path(), 0, "120500", /*Capability=*/0xF837A280,
                           /*Capability2=*/0xFFFFFFFF);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1250\n");
}

// Make sure any other version of gfx1250 is printed as gfx1250.
TEST(KFDTopology, GFX1250NonA0IsPrintedPlain) {
  ScopedEnvironment Env("HSA_DISABLE_GFX12_STRICT", "0");
  unittest::TempDir Dir("kfd-topology", /*Unique=*/true);
  addGPUNodeWithCapability(Dir.path(), 0, "120500", /*Capability=*/0xF877A280,
                           /*Capability2=*/0x00000000);
  std::string Output;
  EXPECT_EQ(printGPUsByKFDCapturingStdout(Dir.path(), Output), 0);
  EXPECT_EQ(Output, "gfx1250\n");
}

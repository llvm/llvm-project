//===- unittests/Driver/GCCVersionTest.cpp --- GCCVersion parser tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Generic_GCC::GCCVersion
//
//===----------------------------------------------------------------------===//

#include "../../lib/Driver/ToolChains/Gnu.h"
#include "gtest/gtest.h"

// The Generic_GCC class is hidden in dylib/shared library builds, so
// this test can only be built if neither of those configurations are
// enabled.
#if !defined(LLVM_BUILD_LLVM_DYLIB) && !defined(LLVM_BUILD_SHARED_LIBS)

using namespace clang;
using namespace clang::driver;

namespace {

struct VersionParseTest {
  std::string Text;

  int Major, Minor, Patch;
  std::string MajorStr, MinorStr, PatchSuffix;
};

const VersionParseTest TestCases[] = {
    {"5", 5, -1, -1, "5", "", ""},
    {"4.4", 4, 4, -1, "4", "4", ""},
    {"4.4-patched", 4, 4, -1, "4", "4", "-patched"},
    {"4.4.0", 4, 4, 0, "4", "4", ""},
    {"4.4.x", 4, 4, -1, "4", "4", ""},
    {"4.4.2-rc4", 4, 4, 2, "4", "4", "-rc4"},
    {"4.4.x-patched", 4, 4, -1, "4", "4", ""},
    {"not-a-version", -1, -1, -1, "", "", ""},
    {"10-win32", 10, -1, -1, "10", "", "-win32"},
};

TEST(GCCVersionTest, Parse) {
  for (const auto &TC : TestCases) {
    auto V = toolchains::Generic_GCC::GCCVersion::Parse(TC.Text);
    EXPECT_EQ(V.Text, TC.Text);
    EXPECT_EQ(V.Major, TC.Major);
    EXPECT_EQ(V.Minor, TC.Minor);
    EXPECT_EQ(V.Patch, TC.Patch);
    EXPECT_EQ(V.MajorStr, TC.MajorStr);
    EXPECT_EQ(V.MinorStr, TC.MinorStr);
    EXPECT_EQ(V.PatchSuffix, TC.PatchSuffix);
  }
}

} // end anonymous namespace

#endif

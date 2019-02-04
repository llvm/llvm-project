//===- unittests/Driver/ModuleCacheTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the LLDB module cache API.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"
#include "gtest/gtest.h"
using namespace clang;
using namespace clang::driver;

namespace {

TEST(ModuleCacheTest, GetTargetAndMode) {
  SmallString<128> Buf;
  Driver::getDefaultModuleCachePath(Buf);
  StringRef Path = Buf;
  EXPECT_TRUE(Path.find("org.llvm.clang") != Path.npos);
  EXPECT_TRUE(Path.endswith("ModuleCache"));  
}
} // end anonymous namespace.

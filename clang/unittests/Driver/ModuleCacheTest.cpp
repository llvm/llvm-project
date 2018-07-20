//===- unittests/Driver/ModuleCacheTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

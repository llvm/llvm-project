//===- lld/unittest/UniversalDriverTest.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Universal driver tests that depend on the value of argv[0].
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace lld;

TEST(UniversalDriver, flavor) {
  const char *args[] = {"ld", "-flavor", "old-gnu"};

  std::string diags;
  raw_string_ostream os(diags);
  UniversalDriver::link(args, os);
  EXPECT_EQ(os.str().find("failed to determine driver flavor"),
            std::string::npos);
  EXPECT_NE(os.str().find("No input files"),
            std::string::npos);
}

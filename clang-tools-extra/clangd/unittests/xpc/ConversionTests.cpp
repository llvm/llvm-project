//===-- ConversionTests.cpp  --------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xpc/Conversion.h"
#include "gtest/gtest.h"

#include <limits>

namespace clang {
namespace clangd {
namespace {

using namespace llvm;

TEST(JsonXpcConversionTest, JsonToXpcToJson) {

  for (auto &testcase :
       {json::Value(false), json::Value(3.14), json::Value(42),
        json::Value(-100), json::Value("foo"), json::Value(""),
        json::Value("123"), json::Value(" "),
        json::Value{true, "foo", nullptr, 42},
        json::Value(json::Object{
            {"a", true}, {"b", "foo"}, {"c", nullptr}, {"d", 42}})}) {
    EXPECT_TRUE(testcase == xpcToJson(jsonToXpc(testcase))) << testcase;
  }
}

} // namespace
} // namespace clangd
} // namespace clang

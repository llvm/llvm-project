//===-- LoggerTests.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "support/Logger.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

TEST(ErrorTest, Overloads) {
  EXPECT_EQ("foo", llvm::toString(error("foo")));
  // Inconvertible to error code when none is specified.
  // Don't actually try to convert, it'll crash.
  handleAllErrors(error("foo"), [&](const llvm::ErrorInfoBase &EI) {
    EXPECT_EQ(llvm::inconvertibleErrorCode(), EI.convertToErrorCode());
  });

  EXPECT_EQ("foo 42", llvm::toString(error("foo {0}", 42)));
  handleAllErrors(error("foo {0}", 42), [&](const llvm::ErrorInfoBase &EI) {
    EXPECT_EQ(llvm::inconvertibleErrorCode(), EI.convertToErrorCode());
  });

  EXPECT_EQ("foo", llvm::toString(error(llvm::errc::invalid_argument, "foo")));
  EXPECT_EQ(llvm::errc::invalid_argument,
            llvm::errorToErrorCode(error(llvm::errc::invalid_argument, "foo")));

  EXPECT_EQ("foo 42",
            llvm::toString(error(llvm::errc::invalid_argument, "foo {0}", 42)));
  EXPECT_EQ(llvm::errc::invalid_argument,
            llvm::errorToErrorCode(
                error(llvm::errc::invalid_argument, "foo {0}", 42)));
}

TEST(ErrorTest, Lifetimes) {
  std::optional<llvm::Error> Err;
  {
    // Check the error contains the value when error() was called.
    std::string S = "hello, world";
    Err = error("S={0}", llvm::StringRef(S));
    S = "garbage";
  }
  EXPECT_EQ("S=hello, world", llvm::toString(std::move(*Err)));
}

TEST(ErrorTest, ConsumeError) {
  llvm::Error Foo = error("foo");
  llvm::Error Bar = error("bar: {0}", std::move(Foo));
  EXPECT_EQ("bar: foo", llvm::toString(std::move(Bar)));
  // No assert for unchecked Foo.
}

} // namespace
} // namespace clangd
} // namespace clang

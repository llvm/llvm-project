//===- unittests/Analysis/Scalable/ErrorBuilderTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Support/ErrorBuilder.h"
#include "gtest/gtest.h"
#include <system_error>

using namespace llvm;

namespace clang::ssaf {

namespace {

class ErrorBuilderTest : public ::testing::Test {
protected:
  struct ErrorInfo {
    std::error_code Code;
    std::string Message;
  };

  ErrorInfo extractErrorInfo(Error Err) {
    ErrorInfo Info;

    handleAllErrors(std::move(Err), [&](const StringError &SE) {
      Info.Code = SE.convertToErrorCode();
      Info.Message = SE.getMessage();
    });

    return Info;
  }
};

TEST_F(ErrorBuilderTest, CreatesSimpleError) {
  auto Err =
      ErrorBuilder::create(std::errc::invalid_argument, "test error").build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "test error");
}

TEST_F(ErrorBuilderTest, CreatesErrorWithErrorCode) {
  auto EC = std::make_error_code(std::errc::no_such_file_or_directory);
  auto Err = ErrorBuilder::create(EC, "file not found").build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::no_such_file_or_directory);
  EXPECT_EQ(Info.Message, "file not found");
}

TEST_F(ErrorBuilderTest, CreatesErrorWithFormattedMessage) {
  auto Err = ErrorBuilder::create(std::errc::invalid_argument,
                                  "field '{0}' has value {1}", "age", 150)
                 .build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "field 'age' has value 150");
}

TEST_F(ErrorBuilderTest, AddsPlainContext) {
  auto Err = ErrorBuilder::create(std::errc::invalid_argument, "inner error")
                 .context("outer context")
                 .build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "outer context\ninner error");
}

TEST_F(ErrorBuilderTest, AddsMultipleContextLayers) {
  auto Err = ErrorBuilder::create(std::errc::invalid_argument,
                                  "expected {0} but got {1} in field '{2}'",
                                  "string", "number", "value")
                 .context("parsing line {0}", 42)
                 .context("reading file '{0}'", "config.json")
                 .build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "reading file 'config.json'\n"
                          "parsing line 42\n"
                          "expected string but got number in field 'value'");
}

TEST_F(ErrorBuilderTest, HandlesSpecialCharactersInContext) {
  auto Err = ErrorBuilder::create(std::errc::invalid_argument,
                                  "special chars: {0}", "test\nwith\nnewlines")
                 .context("tab\tseparated\tvalues")
                 .build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "tab\tseparated\tvalues\n"
                          "special chars: test\n"
                          "with\n"
                          "newlines");
}

TEST_F(ErrorBuilderTest, WrapsExistingError) {
  auto OriginalErr =
      createStringError(std::errc::invalid_argument, "original error message");

  auto WrappedErr = ErrorBuilder::wrap(std::move(OriginalErr))
                        .context("additional context")
                        .build();

  auto Info = extractErrorInfo(std::move(WrappedErr));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "additional context\noriginal error message");
}

TEST_F(ErrorBuilderTest, WrapsMultipleJoinedErrors) {
  auto Err1 = createStringError(std::errc::invalid_argument, "first");
  auto Err2 = createStringError(std::errc::argument_list_too_long, "second");
  auto Err3 = createStringError(std::errc::filename_too_long, "third");

  auto JoinedErr =
      joinErrors(std::move(Err1), joinErrors(std::move(Err2), std::move(Err3)));

  auto WrappedErr = ErrorBuilder::wrap(std::move(JoinedErr))
                        .context("wrapping three joined errors")
                        .build();

  auto Info = extractErrorInfo(std::move(WrappedErr));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  // All three messages combined with " + " in the order they were handled
  EXPECT_EQ(Info.Message,
            "wrapping three joined errors\nfirst + second + third");
}

TEST_F(ErrorBuilderTest, WrapsErrorWithEmptyMessage) {
  auto EmptyErr = createStringError(std::errc::invalid_argument, "");

  auto WrappedErr = ErrorBuilder::wrap(std::move(EmptyErr))
                        .context("")
                        .context("wrapping error with empty message")
                        .context("")
                        .build();

  auto Info = extractErrorInfo(std::move(WrappedErr));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "wrapping error with empty message");
}

TEST_F(ErrorBuilderTest, CreatesErrorWithEmptyMessage) {
  auto Err = ErrorBuilder::create(std::errc::invalid_argument, "")
                 .context("")
                 .context("creating error with empty message")
                 .context("")
                 .build();

  auto Info = extractErrorInfo(std::move(Err));

  EXPECT_EQ(Info.Code, std::errc::invalid_argument);
  EXPECT_EQ(Info.Message, "creating error with empty message");
}

#ifndef NDEBUG
// Death test only works in debug builds where assertions are enabled
TEST_F(ErrorBuilderTest, TriggersAssertionOnWrappingSuccessError) {
  EXPECT_DEATH(
      {
        auto SuccessErr = Error::success();
        ErrorBuilder::wrap(std::move(SuccessErr));
      },
      "Cannot wrap a success error - check for success before calling wrap");
}
#endif // !NDEBUG

TEST_F(ErrorBuilderTest, FatalTerminatesExecution) {
  EXPECT_DEATH(
      {
        ErrorBuilder::fatal("Entity {0} with {1} linkage already exists", 42,
                            "Internal");
      },
      "Entity 42 with Internal linkage already exists");
}

} // namespace

} // namespace clang::ssaf

//===- unittest/Format/FormatTestBase.h - Formatting test base classs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the base class for format tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTBASE_H
#define LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTBASE_H

#include "FormatTestUtils.h"

#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace test {

#define DEBUG_TYPE "format-test-base"

class FormatTestBase : public ::testing::Test {
protected:
  enum StatusCheck { SC_ExpectComplete, SC_ExpectIncomplete, SC_DoNotCheck };

  virtual FormatStyle getDefaultStyle() const { return getLLVMStyle(); }

  virtual std::string messUp(llvm::StringRef Code) const {
    return test::messUp(Code);
  }

  std::string format(llvm::StringRef Code,
                     const std::optional<FormatStyle> &Style = {},
                     StatusCheck CheckComplete = SC_ExpectComplete,
                     const std::vector<tooling::Range> &Ranges = {}) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    auto NonEmptyRanges =
        !Ranges.empty()
            ? Ranges
            : std::vector<tooling::Range>{1, tooling::Range(0, Code.size())};
    auto UsedStyle = Style ? Style.value() : getDefaultStyle();
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(UsedStyle, Code, NonEmptyRanges, "<stdin>", &Status);
    if (CheckComplete != SC_DoNotCheck) {
      bool ExpectedCompleteFormat = CheckComplete == SC_ExpectComplete;
      EXPECT_EQ(ExpectedCompleteFormat, Status.FormatComplete)
          << Code << "\n\n";
    }
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  FormatStyle getStyleWithColumns(FormatStyle Style, unsigned ColumnLimit) {
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    return getStyleWithColumns(getLLVMStyle(), ColumnLimit);
  }

  FormatStyle getGoogleStyleWithColumns(unsigned ColumnLimit) {
    return getStyleWithColumns(getGoogleStyle(), ColumnLimit);
  }

  FormatStyle getTextProtoStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::FormatStyle::LK_TextProto);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Expected,
                     llvm::StringRef Code,
                     const std::optional<FormatStyle> &Style = {},
                     const std::vector<tooling::Range> &Ranges = {}) {
    testing::ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    if (Expected != Code) {
      EXPECT_EQ(Expected.str(),
                format(Expected, Style, SC_ExpectComplete, Ranges))
          << "Expected code is not stable";
    }
    EXPECT_EQ(Expected.str(), format(Code, Style, SC_ExpectComplete, Ranges));
    auto UsedStyle = Style ? Style.value() : getDefaultStyle();
    if (UsedStyle.Language == FormatStyle::LK_Cpp) {
      // Objective-C++ is a superset of C++, so everything checked for C++
      // needs to be checked for Objective-C++ as well.
      FormatStyle ObjCStyle = UsedStyle;
      ObjCStyle.Language = FormatStyle::LK_ObjC;
      // FIXME: Additional messUp is superfluous.
      EXPECT_EQ(Expected.str(),
                format(Code, ObjCStyle, SC_ExpectComplete, Ranges));
    }
  }

  void _verifyFormat(const char *File, int Line, llvm::StringRef Code,
                     const std::optional<FormatStyle> &Style = {}) {
    _verifyFormat(File, Line, Code, test::messUp(Code), Style);
  }

  void _verifyIncompleteFormat(const char *File, int Line, llvm::StringRef Code,
                               const std::optional<FormatStyle> &Style = {}) {
    testing::ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    EXPECT_EQ(Code.str(), format(messUp(Code), Style, SC_ExpectIncomplete));
  }

  void
  _verifyIndependentOfContext(const char *File, int Line, llvm::StringRef Text,
                              const std::optional<FormatStyle> &Style = {}) {
    _verifyFormat(File, Line, Text, Style);
    _verifyFormat(File, Line, llvm::Twine("void f() { " + Text + " }").str(),
                  Style);
  }

  void _verifyNoChange(const char *File, int Line, llvm::StringRef Code,
                       const std::optional<FormatStyle> &Style = {}) {
    _verifyFormat(File, Line, Code, Code, Style);
  }

  /// \brief Verify that clang-format does not crash on the given input.
  void verifyNoCrash(llvm::StringRef Code,
                     const std::optional<FormatStyle> &Style = {}) {
    format(Code, Style, SC_DoNotCheck);
  }

  int ReplacementCount;
};

#undef DEBUG_TYPE

#define verifyIndependentOfContext(...)                                        \
  _verifyIndependentOfContext(__FILE__, __LINE__, __VA_ARGS__)
#define verifyIncompleteFormat(...)                                            \
  _verifyIncompleteFormat(__FILE__, __LINE__, __VA_ARGS__)
#define verifyNoChange(...) _verifyNoChange(__FILE__, __LINE__, __VA_ARGS__)
#define verifyFormat(...) _verifyFormat(__FILE__, __LINE__, __VA_ARGS__)
#define verifyGoogleFormat(Code) verifyFormat(Code, getGoogleStyle())

} // namespace test
} // namespace format
} // namespace clang

#endif

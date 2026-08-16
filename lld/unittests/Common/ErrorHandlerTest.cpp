//===- ErrorHandlerTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace lld;

namespace {

struct ErrorResult {
  std::string output;
  uint64_t count;
};

static std::string getWarning(StringRef message) {
  SmallString<0> stdoutBuffer;
  SmallString<256> stderrBuffer;
  raw_svector_ostream stdoutOS(stdoutBuffer);
  raw_svector_ostream stderrOS(stderrBuffer);
  ErrorHandler handler;
  handler.initialize(stdoutOS, stderrOS, /*exitEarly=*/false,
                     /*disableOutput=*/false);
  handler.logName = "lld";
  handler.vsDiagnostics = true;
  handler.warn(message);
  return stderrBuffer.str().str();
}

static ErrorResult getError(StringRef message) {
  SmallString<0> stdoutBuffer;
  SmallString<256> stderrBuffer;
  raw_svector_ostream stdoutOS(stdoutBuffer);
  raw_svector_ostream stderrOS(stderrBuffer);
  ErrorHandler handler;
  handler.initialize(stdoutOS, stderrOS, /*exitEarly=*/false,
                     /*disableOutput=*/false);
  handler.logName = "lld";
  handler.vsDiagnostics = true;
  handler.error(message);
  return {stderrBuffer.str().str(), handler.errorCount};
}

TEST(ErrorHandlerTest, VsDiagnosticLocations) {
  struct {
    StringRef message;
    StringRef location;
  } cases[] = {
      {"undefined symbol: foo\n"
       ">>> referenced by foo.cc:12 (/tmp/foo.cc:12)",
       "/tmp/foo.cc(12)"},
      {"undefined hidden symbol: foo\n>>> referenced by foo.cc:13",
       "foo.cc(13)"},
      {"undefined protected symbol: foo\n"
       ">>> referenced by foo.cc:14 (/tmp/foo.cc:14)",
       "/tmp/foo.cc(14)"},
      {"undefined symbol: foo\n"
       ">>> referenced by C:\\dir name\\foo.o:(.text+0x1)",
       "C:\\dir name\\foo.o"},
      {"undefined symbol: foo\n>>> referenced by :section", ""},
      {"undefined hidden symbol: foo\n>>> referenced by file.cc:1:2",
       "file.cc:1(2)"},
      {"duplicate symbol: foo\n"
       ">>> defined in first.o\n"
       ">>> defined in second.o",
       "first.o"},
      {"duplicate symbol: foo\n"
       ">>> defined at foo.cc:21 (/tmp/foo.cc:21)\n"
       ">>> first.o",
       "/tmp/foo.cc(21)"},
      {"duplicate symbol: foo\n>>> defined at foo.cc:22\n>>> first.o",
       "foo.cc(22)"},
      {"relocation failed\n"
       ">>> defined in first.o\n"
       ">>> referenced by foo.cc:31 (/tmp/foo.cc:31)",
       "/tmp/foo.cc(31)"},
      {"relocation failed\n"
       ">>> defined in first.o\n"
       ">>> referenced by foo.cc:32",
       "foo.cc(32)"},
      {"/tmp/version.script:41: unclosed quote", "/tmp/version.script(41)"},
      {"while parsing path/to/version.script:42: unclosed quote",
       "path/to/version.script(42)"},
      {"first diagnostic\n"
       ">>> defined in first.o\n"
       ">>> referenced by first.cc:51\n"
       "second diagnostic\n"
       ">>> defined in second.o\n"
       ">>> referenced by second.cc:52 (/tmp/second.cc:52)",
       "/tmp/second.cc(52)"},
      {"undefined two word symbol: foo\n>>> referenced by foo.cc:61", "lld"},
      {"undefined symbol: foo\r\n>>> referenced by foo.cc:61", "lld"},
      {"undefined hidden symbol: foo\n"
       ">>> referenced by foo.cc:not-a-line",
       "lld"},
      {"duplicate symbol: foo\n"
       ">>> defined in first.o with-details\n"
       ">>> defined in second.o",
       "lld"},
      {"relocation failed\n"
       ">>> defined infirst.o\n"
       ">>> referenced by foo.cc:62",
       "lld"},
      {"version.script:not-a-line: unclosed quote", "lld"},
  };

  for (const auto &test : cases) {
    SCOPED_TRACE(test.message);
    EXPECT_EQ(
        (Twine(test.location) + ": warning: " + test.message + "\n").str(),
        getWarning(test.message));
  }
}

TEST(ErrorHandlerTest, SplitVsDiagnosticDuplicateSymbol) {
  StringRef message = "duplicate symbol: foo\n"
                      ">>> defined at first.cc:1\n"
                      ">>> first.o\n"
                      ">>> defined at second.cc:2\n"
                      ">>> second.o";
  ErrorResult result = getError(message);

  EXPECT_EQ(2u, result.count);
  EXPECT_EQ("first.cc(1): error: duplicate symbol: foo\n"
            ">>> defined at first.cc:1\n"
            ">>> first.o\n"
            "\n"
            "second.cc(2): error: duplicate symbol: foo\n"
            ">>> defined at second.cc:2\n"
            ">>> second.o\n",
            result.output);
}

TEST(ErrorHandlerTest, DoNotSplitOtherDuplicateSymbolForms) {
  StringRef messages[] = {
      "duplicate symbol: foo\n"
      ">>> defined at first.cc:1\n"
      ">>> first.o\n"
      ">>> defined at second.cc:2\n"
      ">>> second.o\n"
      ">>> extra line",
      "duplicate symbol: foo\n"
      ">>> defined at source file.cc:1\n"
      ">>> first.o\n"
      ">>> defined at second.cc:2\n"
      ">>> second.o",
      "duplicate symbol: foo\n"
      ">>> defined in first.o\n"
      ">>> first.o\n"
      ">>> defined at second.cc:2\n"
      ">>> second.o",
      "duplicate symbol: foo\r\n"
      ">>> defined at first.cc:1\r\n"
      ">>> first.o\r\n"
      ">>> defined at second.cc:2\r\n"
      ">>> second.o\r",
  };

  for (StringRef message : messages) {
    SCOPED_TRACE(message);
    EXPECT_EQ(1u, getError(message).count);
  }
}

} // namespace

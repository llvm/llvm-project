//===- unittest/Format/FormatTestRawStrings.cpp - Formatting unit tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "../Tooling/ReplacementTest.h"
#include "FormatTestUtils.h"

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

using clang::tooling::ReplacementTest;
using clang::tooling::toReplacements;

namespace clang {
namespace format {
namespace {

class FormatTestRawStrings : public ::testing::Test {
protected:
  enum StatusCheck { SC_ExpectComplete, SC_ExpectIncomplete, SC_DoNotCheck };

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle(),
                     StatusCheck CheckComplete = SC_ExpectComplete) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &Status);
    if (CheckComplete != SC_DoNotCheck) {
      bool ExpectedCompleteFormat = CheckComplete == SC_ExpectComplete;
      EXPECT_EQ(ExpectedCompleteFormat, Status.FormatComplete)
          << Code << "\n\n";
    }
    ReplacementCount = Replaces.size();
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  FormatStyle getStyleWithColumns(FormatStyle Style, unsigned ColumnLimit) {
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    return getStyleWithColumns(getLLVMStyle(), ColumnLimit);
  }

  int ReplacementCount;

  FormatStyle getRawStringPbStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getLLVMStyle();
    Style.ColumnLimit = ColumnLimit;
    Style.RawStringFormats = {
        {
            /*Language=*/FormatStyle::LK_TextProto,
            /*Delimiters=*/{"pb"},
            /*EnclosingFunctions=*/{},
            /*CanonicalDelimiter=*/"",
            /*BasedOnStyle=*/"google",
        },
    };
    return Style;
  }

  FormatStyle getRawStringLLVMCppStyleBasedOn(std::string BasedOnStyle) {
    FormatStyle Style = getLLVMStyle();
    Style.RawStringFormats = {
        {
            /*Language=*/FormatStyle::LK_Cpp,
            /*Delimiters=*/{"cpp"},
            /*EnclosingFunctions=*/{},
            /*CanonicalDelimiter=*/"",
            BasedOnStyle,
        },
    };
    return Style;
  }

  FormatStyle getRawStringGoogleCppStyleBasedOn(std::string BasedOnStyle) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_Cpp);
    Style.RawStringFormats = {
        {
            /*Language=*/FormatStyle::LK_Cpp,
            /*Delimiters=*/{"cpp"},
            /*EnclosingFunctions=*/{},
            /*CanonicalDelimiter=*/"",
            BasedOnStyle,
        },
    };
    return Style;
  }

  // Gcc 4.8 doesn't support raw string literals in macros, which breaks some
  // build bots. We use this function instead.
  void expect_eq(const std::string Expected, const std::string Actual) {
    EXPECT_EQ(Expected, Actual);
  }
};

TEST_F(FormatTestRawStrings, ReformatsAccordingToBaseStyle) {
  // llvm style puts '*' on the right.
  // google style puts '*' on the left.

  // Use the llvm style if the raw string style has no BasedOnStyle.
  expect_eq(R"test(int *i = R"cpp(int *p = nullptr;)cpp")test",
            format(R"test(int * i = R"cpp(int * p = nullptr;)cpp")test",
                   getRawStringLLVMCppStyleBasedOn("")));

  // Use the google style if the raw string style has BasedOnStyle=google.
  expect_eq(R"test(int *i = R"cpp(int* p = nullptr;)cpp")test",
            format(R"test(int * i = R"cpp(int * p = nullptr;)cpp")test",
                   getRawStringLLVMCppStyleBasedOn("google")));

  // Use the llvm style if the raw string style has no BasedOnStyle=llvm.
  expect_eq(R"test(int* i = R"cpp(int *p = nullptr;)cpp")test",
            format(R"test(int * i = R"cpp(int * p = nullptr;)cpp")test",
                   getRawStringGoogleCppStyleBasedOn("llvm")));
}

TEST_F(FormatTestRawStrings, UsesConfigurationOverBaseStyle) {
  // llvm style puts '*' on the right.
  // google style puts '*' on the left.

  // Uses the configured google style inside raw strings even if BasedOnStyle in
  // the raw string format is llvm.
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_Cpp);
  EXPECT_EQ(0, parseConfiguration("---\n"
                                  "Language: Cpp\n"
                                  "BasedOnStyle: Google", &Style).value());
  Style.RawStringFormats = {{
      FormatStyle::LK_Cpp,
      {"cpp"},
      {},
      /*CanonicalDelimiter=*/"",
      /*BasedOnStyle=*/"llvm",
  }};
  expect_eq(R"test(int* i = R"cpp(int* j = 0;)cpp";)test",
            format(R"test(int * i = R"cpp(int * j = 0;)cpp";)test", Style));
}

TEST_F(FormatTestRawStrings, MatchesDelimitersCaseSensitively) {
  // Don't touch the 'PB' raw string, format the 'pb' raw string.
  expect_eq(R"test(
s = R"PB(item:1)PB";
t = R"pb(item: 1)pb";)test",
            format(R"test(
s = R"PB(item:1)PB";
t = R"pb(item:1)pb";)test",
                   getRawStringPbStyleWithColumns(40)));
}

TEST_F(FormatTestRawStrings, ReformatsShortRawStringsOnSingleLine) {
  expect_eq(
      R"test(P p = TP(R"pb()pb");)test",
      format(
          R"test(P p = TP(R"pb( )pb");)test",
          getRawStringPbStyleWithColumns(40)));
  expect_eq(
      R"test(P p = TP(R"pb(item_1: 1)pb");)test",
      format(
          R"test(P p = TP(R"pb(item_1:1)pb");)test",
          getRawStringPbStyleWithColumns(40)));
  expect_eq(
      R"test(P p = TP(R"pb(item_1: 1)pb");)test",
      format(
          R"test(P p = TP(R"pb(  item_1 :  1   )pb");)test",
          getRawStringPbStyleWithColumns(40)));
  expect_eq(
      R"test(P p = TP(R"pb(item_1: 1 item_2: 2)pb");)test",
      format(
          R"test(P p = TP(R"pb(item_1:1 item_2:2)pb");)test",
          getRawStringPbStyleWithColumns(40)));
  expect_eq(
      R"test(P p = TP(R"pb(item_1 <1> item_2: { 2 })pb");)test",
      format(
          R"test(P p = TP(R"pb(item_1<1> item_2:{2})pb");)test",
          getRawStringPbStyleWithColumns(40)));

  // Merge two short lines into one.
  expect_eq(R"test(
std::string s = R"pb(
  item_1: 1 item_2: 2
)pb";
)test",
            format(R"test(
std::string s = R"pb(
  item_1:1
  item_2:2
)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));
}

TEST_F(FormatTestRawStrings, BreaksRawStringsExceedingColumnLimit) {
  expect_eq(R"test(
P p = TPPPPPPPPPPPPPPP(
    R"pb(item_1: 1, item_2: 2)pb");)test",
            format(R"test(
P p = TPPPPPPPPPPPPPPP(R"pb(item_1: 1, item_2: 2)pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
P p =
    TPPPPPPPPPPPPPPP(
        R"pb(item_1: 1,
             item_2: 2,
             item_3: 3)pb");)test",
            format(R"test(
P p = TPPPPPPPPPPPPPPP(R"pb(item_1: 1, item_2: 2, item_3: 3)pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
P p = TP(R"pb(item_1 <1>
              item_2: <2>
              item_3 {})pb");)test",
      format(R"test(
P p = TP(R"pb(item_1<1> item_2:<2> item_3{ })pb");)test",
          getRawStringPbStyleWithColumns(40)));

  expect_eq(
      R"test(
P p = TP(R"pb(item_1: 1,
              item_2: 2,
              item_3: 3,
              item_4: 4)pb");)test",
      format(
          R"test(
P p = TP(R"pb(item_1: 1, item_2: 2, item_3: 3, item_4: 4)pb");)test",
          getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
P p = TPPPPPPPPPPPPPPP(
    R"pb(item_1 <1>,
         item_2: { 2 },
         item_3: <3>,
         item_4: { 4 })pb");)test",
            format(R"test(
P p = TPPPPPPPPPPPPPPP(R"pb(item_1<1>, item_2: {2}, item_3: <3>, item_4:{4})pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  // Breaks before a short raw string exceeding the column limit.
  expect_eq(R"test(
FFFFFFFFFFFFFFFFFFFFFFFFFFF(
    R"pb(key: 1)pb");
P p = TPPPPPPPPPPPPPPPPPPPP(
    R"pb(key: 2)pb");
auto TPPPPPPPPPPPPPPPPPPPP =
    R"pb(key: 3)pb";
P p = TPPPPPPPPPPPPPPPPPPPP(
    R"pb(i: 1, j: 2)pb");

int f(string s) {
  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF(
      R"pb(key: 1)pb");
  P p = TPPPPPPPPPPPPPPPPPPPP(
      R"pb(key: 2)pb");
  auto TPPPPPPPPPPPPPPPPPPPP =
      R"pb(key: 3)pb";
  if (s.empty())
    P p = TPPPPPPPPPPPPPPPPPPPP(
        R"pb(i: 1, j: 2)pb");
}
)test",
            format(R"test(
FFFFFFFFFFFFFFFFFFFFFFFFFFF(R"pb(key:1)pb");
P p = TPPPPPPPPPPPPPPPPPPPP(R"pb(key:2)pb");
auto TPPPPPPPPPPPPPPPPPPPP = R"pb(key:3)pb";
P p = TPPPPPPPPPPPPPPPPPPPP(R"pb(i: 1, j:2)pb");

int f(string s) {
  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF(R"pb(key:1)pb");
  P p = TPPPPPPPPPPPPPPPPPPPP(R"pb(key:2)pb");
  auto TPPPPPPPPPPPPPPPPPPPP = R"pb(key:3)pb";
  if (s.empty())
    P p = TPPPPPPPPPPPPPPPPPPPP(R"pb(i: 1, j:2)pb");
}
)test",
                   getRawStringPbStyleWithColumns(40)));
}

TEST_F(FormatTestRawStrings, FormatsRawStringArguments) {
  expect_eq(R"test(
P p = TP(R"pb(key { 1 })pb", param_2);)test",
            format(R"test(
P p = TP(R"pb(key{1})pb",param_2);)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
PPPPPPPPPPPPP(R"pb(keykeyk)pb",
              param_2);)test",
            format(R"test(
PPPPPPPPPPPPP(R"pb(keykeyk)pb", param_2);)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
P p = TP(
    R"pb(item: { i: 1, s: 's' }
         item: { i: 2, s: 't' })pb");)test",
            format(R"test(
P p = TP(R"pb(item: {i: 1, s: 's'} item: {i: 2, s: 't'})pb");)test",
                   getRawStringPbStyleWithColumns(40)));
  expect_eq(R"test(
FFFFFFFFFFFFFFFFFFF(
    R"pb(key: "value")pb",
    R"pb(key2: "value")pb");)test",
            format(R"test(
FFFFFFFFFFFFFFFFFFF(R"pb(key: "value")pb", R"pb(key2: "value")pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats the first out of two arguments.
  expect_eq(R"test(
FFFFFFFF(R"pb(key: 1)pb", argument2);
struct S {
  const s =
      f(R"pb(key: 1)pb", argument2);
  void f() {
    if (gol)
      return g(R"pb(key: 1)pb",
               132789237);
    return g(R"pb(key: 1)pb", "172893");
  }
};)test",
            format(R"test(
FFFFFFFF(R"pb(key:1)pb", argument2);
struct S {
const s = f(R"pb(key:1)pb", argument2);
void f() {
  if (gol)
    return g(R"pb(key:1)pb", 132789237);
  return g(R"pb(key:1)pb", "172893");
}
};)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats the second out of two arguments.
  expect_eq(R"test(
FFFFFFFF(argument1, R"pb(key: 2)pb");
struct S {
  const s =
      f(argument1, R"pb(key: 2)pb");
  void f() {
    if (gol)
      return g(12784137,
               R"pb(key: 2)pb");
    return g(17283122, R"pb(key: 2)pb");
  }
};)test",
            format(R"test(
FFFFFFFF(argument1, R"pb(key:2)pb");
struct S {
const s = f(argument1, R"pb(key:2)pb");
void f() {
  if (gol)
    return g(12784137, R"pb(key:2)pb");
  return g(17283122, R"pb(key:2)pb");
}
};)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats two short raw string arguments.
  expect_eq(R"test(
FFFFF(R"pb(key: 1)pb", R"pb(key: 2)pb");)test",
            format(R"test(
FFFFF(R"pb(key:1)pb", R"pb(key:2)pb");)test",
                   getRawStringPbStyleWithColumns(40)));
  // TODO(krasimir): The original source code fits on one line, so the
  // non-optimizing formatter is chosen. But after the formatting in protos is
  // made, the code doesn't fit on one line anymore and further formatting
  // splits it.
  //
  // Should we disable raw string formatting for the non-optimizing formatter?
  expect_eq(R"test(
FFFFFFF(R"pb(key: 1)pb", R"pb(key: 2)pb");)test",
            format(R"test(
FFFFFFF(R"pb(key:1)pb", R"pb(key:2)pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats two short raw string arguments, puts second on newline.
  expect_eq(R"test(
FFFFFFFF(R"pb(key: 1)pb",
         R"pb(key: 2)pb");)test",
            format(R"test(
FFFFFFFF(R"pb(key:1)pb", R"pb(key:2)pb");)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats both arguments.
  expect_eq(R"test(
FFFFFFFF(R"pb(key: 1)pb",
         R"pb(key: 2)pb");
struct S {
  const s = f(R"pb(key: 1)pb",
              R"pb(key: 2)pb");
  void f() {
    if (gol)
      return g(R"pb(key: 1)pb",
               R"pb(key: 2)pb");
    return g(R"pb(k1)pb", R"pb(k2)pb");
  }
};)test",
            format(R"test(
FFFFFFFF(R"pb(key:1)pb", R"pb(key:2)pb");
struct S {
const s = f(R"pb(key:1)pb", R"pb(key:2)pb");
void f() {
  if (gol)
    return g(R"pb(key:1)pb", R"pb(key:2)pb");
  return g(R"pb( k1 )pb", R"pb( k2 )pb");
}
};)test",
                   getRawStringPbStyleWithColumns(40)));
}

TEST_F(FormatTestRawStrings, RawStringStartingWithNewlines) {
  expect_eq(R"test(
std::string s = R"pb(
  item_1: 1
)pb";
)test",
            format(R"test(
std::string s = R"pb(
    item_1:1
)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
std::string s = R"pb(

  item_1: 1
)pb";
)test",
            format(R"test(
std::string s = R"pb(

    item_1:1
)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
std::string s = R"pb(
  item_1: 1
)pb";
)test",
            format(R"test(
std::string s = R"pb(
    item_1:1

)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
std::string s = R"pb(
  item_1: 1,
  item_2: 2
)pb";
)test",
            format(R"test(
std::string s = R"pb(
  item_1:1, item_2:2
)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
std::string s = R"pb(
  book {
    title: "Alice's Adventures"
    author: "Lewis Caroll"
  }
  book {
    title: "Peter Pan"
    author: "J. M. Barrie"
  }
)pb";
)test",
            format(R"test(
std::string s = R"pb(
    book { title: "Alice's Adventures" author: "Lewis Caroll" }
    book { title: "Peter Pan" author: "J. M. Barrie" }
)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));
}

TEST_F(FormatTestRawStrings, BreaksBeforeRawStrings) {
  expect_eq(R"test(
ASSERT_TRUE(
    ParseFromString(R"pb(item_1: 1)pb"),
    ptr);)test",
            format(R"test(
ASSERT_TRUE(ParseFromString(R"pb(item_1: 1)pb"), ptr);)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
ASSERT_TRUE(toolong::ParseFromString(
                R"pb(item_1: 1)pb"),
            ptr);)test",
            format(R"test(
ASSERT_TRUE(toolong::ParseFromString(R"pb(item_1: 1)pb"), ptr);)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
ASSERT_TRUE(ParseFromString(
                R"pb(item_1: 1,
                     item_2: 2)pb"),
            ptr);)test",
            format(R"test(
ASSERT_TRUE(ParseFromString(R"pb(item_1: 1, item_2: 2)pb"), ptr);)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
ASSERT_TRUE(
    ParseFromString(
        R"pb(item_1: 1 item_2: 2)pb"),
    ptr);)test",
            format(R"test(
ASSERT_TRUE(ParseFromString(R"pb(item_1: 1 item_2: 2)pb"), ptr);)test",
                   getRawStringPbStyleWithColumns(40)));

}

TEST_F(FormatTestRawStrings, RawStringsInOperands) {
  // Formats the raw string first operand of a binary operator expression.
  expect_eq(R"test(auto S = R"pb(item_1: 1)pb" + rest;)test",
            format(R"test(auto S = R"pb(item_1:1)pb" + rest;)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1, item_2: 2)pb" +
         rest;)test",
            format(R"test(
auto S = R"pb(item_1:1,item_2:2)pb"+rest;)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S =
    R"pb(item_1: 1 item_2: 2)pb" + rest;)test",
            format(R"test(
auto S = R"pb(item_1:1 item_2:2)pb"+rest;)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1,
              item_2: 2,
              item_3: 3)pb" + rest;)test",
            format(R"test(
auto S = R"pb(item_1:1,item_2:2,item_3:3)pb"+rest;)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1,
              item_2: 2,
              item_3: 3)pb" +
         longlongrest;)test",
            format(R"test(
auto S = R"pb(item_1:1,item_2:2,item_3:3)pb"+longlongrest;)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats the raw string second operand of a binary operator expression.
  expect_eq(R"test(auto S = first + R"pb(item_1: 1)pb";)test",
            format(R"test(auto S = first + R"pb(item_1:1)pb";)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = first + R"pb(item_1: 1,
                      item_2: 2)pb";)test",
            format(R"test(
auto S = first+R"pb(item_1:1,item_2:2)pb";)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = first + R"pb(item_1: 1
                      item_2: 2)pb";)test",
            format(R"test(
auto S = first+R"pb(item_1:1 item_2:2)pb";)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1,
              item_2: 2,
              item_3: 3)pb" + rest;)test",
            format(R"test(
auto S = R"pb(item_1:1,item_2:2,item_3:3)pb"+rest;)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1,
              item_2: 2,
              item_3: 3)pb" +
         longlongrest;)test",
            format(R"test(
auto S = R"pb(item_1:1,item_2:2,item_3:3)pb"+longlongrest;)test",
                   getRawStringPbStyleWithColumns(40)));

  // Formats the raw string operands in expressions.
  expect_eq(R"test(
auto S = R"pb(item_1: 1)pb" +
         R"pb(item_2: 2)pb";
)test",
            format(R"test(
auto S=R"pb(item_1:1)pb"+R"pb(item_2:2)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = R"pb(item_1: 1)pb" +
         R"pb(item_2: 2)pb" +
         R"pb(item_3: 3)pb";
)test",
            format(R"test(
auto S=R"pb(item_1:1)pb"+R"pb(item_2:2)pb"+R"pb(item_3:3)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S = (count < 3)
             ? R"pb(item_1: 1)pb"
             : R"pb(item_2: 2)pb";
)test",
            format(R"test(
auto S=(count<3)?R"pb(item_1:1)pb":R"pb(item_2:2)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S =
    (count < 3)
        ? R"pb(item_1: 1, item_2: 2)pb"
        : R"pb(item_3: 3)pb";
)test",
            format(R"test(
auto S=(count<3)?R"pb(item_1:1,item_2:2)pb":R"pb(item_3:3)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

  expect_eq(R"test(
auto S =
    (count < 3)
        ? R"pb(item_1: 1)pb"
        : R"pb(item_2: 2, item_3: 3)pb";
)test",
            format(R"test(
auto S=(count<3)?R"pb(item_1:1)pb":R"pb(item_2:2,item_3:3)pb";
)test",
                   getRawStringPbStyleWithColumns(40)));

}

TEST_F(FormatTestRawStrings, PrefixAndSuffixAlignment) {
  // Keep the suffix at the end of line if not on newline.
  expect_eq(R"test(
int s() {
  auto S = PTP(
      R"pb(
        item_1: 1,
        item_2: 2)pb");
})test",
            format(R"test(
int s() {
  auto S = PTP(
      R"pb(
      item_1: 1,
      item_2: 2)pb");
})test",
                   getRawStringPbStyleWithColumns(20)));

  // Align the suffix with the surrounding FirstIndent if the prefix is not on
  // a line of its own.
  expect_eq(R"test(
int s() {
  auto S = PTP(
      R"pb(
        item_1: 1,
        item_2: 2
      )pb");
})test",
            format(R"test(
int s() {
  auto S = PTP(R"pb(
      item_1: 1,
      item_2: 2
      )pb");
})test",
                   getRawStringPbStyleWithColumns(20)));

  // Align the prefix with the suffix if both the prefix and suffix are on a
  // line of their own.
  expect_eq(R"test(
int s() {
  auto S = PTP(
      R"pb(
        item_1: 1,
        item_2: 2,
      )pb");
})test",
            format(R"test(
int s() {
  auto S = PTP(
      R"pb(
      item_1: 1,
      item_2: 2,
      )pb");
})test",
                   getRawStringPbStyleWithColumns(20)));
}

TEST_F(FormatTestRawStrings, EstimatesPenalty) {
  // The penalty for characters exceeding the column limit in the raw string
  // forces 'hh' to be put on a newline.
  expect_eq(R"test(
ff(gggggg,
   hh(R"pb(key {
             i1: k1
             i2: k2
           })pb"));
)test",
            format(R"test(
ff(gggggg, hh(R"pb(key {
    i1: k1
    i2: k2
    })pb"));
)test",
                   getRawStringPbStyleWithColumns(20)));
}

TEST_F(FormatTestRawStrings, DontFormatNonRawStrings) {
  expect_eq(R"test(a = R"pb(key:value)";)test",
            format(R"test(a = R"pb(key:value)";)test",
                   getRawStringPbStyleWithColumns(20)));
}

TEST_F(FormatTestRawStrings, FormatsRawStringsWithEnclosingFunctionName) {
  FormatStyle Style = getRawStringPbStyleWithColumns(40);
  Style.RawStringFormats[0].EnclosingFunctions.push_back(
      "PARSE_TEXT_PROTO");
  Style.RawStringFormats[0].EnclosingFunctions.push_back("ParseTextProto");
  expect_eq(R"test(a = PARSE_TEXT_PROTO(R"(key: value)");)test",
            format(R"test(a = PARSE_TEXT_PROTO(R"(key:value)");)test", Style));

  expect_eq(R"test(
a = PARSE_TEXT_PROTO /**/ (
    /**/ R"(key: value)");)test",
            format(R"test(
a = PARSE_TEXT_PROTO/**/(/**/R"(key:value)");)test",
                   Style));

  expect_eq(R"test(
a = ParseTextProto<ProtoType>(
    R"(key: value)");)test",
            format(R"test(
a = ParseTextProto<ProtoType>(R"(key:value)");)test",
                   Style));
}

TEST_F(FormatTestRawStrings, UpdatesToCanonicalDelimiters) {
  FormatStyle Style = getRawStringPbStyleWithColumns(25);
  Style.RawStringFormats[0].CanonicalDelimiter = "proto";
  expect_eq(R"test(a = R"proto(key: value)proto";)test",
            format(R"test(a = R"pb(key:value)pb";)test", Style));

  // Don't update to canonical delimiter if it occurs as a raw string suffix in
  // the raw string content.
  expect_eq(R"test(a = R"pb(key: ")proto")pb";)test",
            format(R"test(a = R"pb(key:")proto")pb";)test", Style));
}

} // end namespace
} // end namespace format
} // end namespace clang

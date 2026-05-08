//===- AnalyzerFormattingTest.cpp - SA Formatting test --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for printFormattedEntry function, which is used for
// printing available analyzers and their descriptions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang::ento;

static std::string runPrintFormattedEntry(
    std::pair<llvm::StringRef, llvm::StringRef> EntryDescPair,
    size_t InitialPad, size_t EntryWidth, size_t MinLineWidth) {

  std::string OutBuffer;
  llvm::raw_string_ostream Out(OutBuffer);
  clang::AnalyzerOptions::printFormattedEntry(Out, EntryDescPair, InitialPad,
                                              EntryWidth, MinLineWidth);
  return OutBuffer;
}

// No wrapping after checker's name.
// No initial pad.
TEST(PrintFormattedEntryTest, SimplePrint) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                             /*InitialPad=*/0,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "Checker             A description");
}

// With wrapping after checker's name.
// No initial pad.
TEST(PrintFormattedEntryTest, EntryLongerThanWidth) {
  std::string Buffer = runPrintFormattedEntry(
      {/*EntryDescPair=*/"VeryLongCheckerName", "A description"},
      /*InitialPad=*/0,
      /*EntryWidth=*/10,
      /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "VeryLongCheckerName\n"
                    "          A description");
}

// With wrapping after checker's name.
// No initial pad.
// Corner case, when checker's name length equal to EntryWidth.
TEST(PrintFormattedEntryTest, ExactFillWidth) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                             /*InitialPad=*/0,
                             /*EntryWidth=*/7,
                             /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "Checker\n"
                    "       A description");
}

// With wrapping after checker's name.
// With initial pad.
// Corner case, when checker's name length equal to EntryWidth.
// This test matches how printFormattedEntry was called
// in -analyzer-checker-help, which led to a formatting bug.
TEST(PrintFormattedEntryTest, ExactFillWidthWithInitialPad) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                             /*InitialPad=*/2,
                             /*EntryWidth=*/7,
                             /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "  Checker\n"
                    "         A description");
}

// No wrapping after checker's name.
// With initial pad.
TEST(PrintFormattedEntryTest, WithInitialPadding) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                             /*InitialPad=*/2,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "  Checker             A description");
}

// No wrapping after checker's name.
// With initial pad.
// With wrapping in checker's description (MinLineWidth > 0).
TEST(PrintFormattedEntryTest, WrapDescription) {
  std::string Desc =
      "This is a long description that should be wrapped into multiple lines.";

  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", Desc},
                             /*InitialPad=*/2,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/40);

  std::string Expected = "  Checker             This is a long description\n"
                         "                      that should be wrapped\n"
                         "                      into multiple lines.";
  EXPECT_EQ(Buffer, Expected);
}

// No wrapping after checker's name.
// With initial pad.
// No wrapping in checker's descriptions (MinLineWidth = 0).
TEST(PrintFormattedEntryTest, NoWrap) {
  std::string Desc = "This is a very long description that will not be "
                     "wrapped because MinLineWidth=0.";

  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", Desc},
                             /*InitialPad=*/2,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/0);

  std::string Expected = "  Checker             " + Desc;
  EXPECT_EQ(Buffer, Expected);
}

// No wrapping after checker's name.
// No initial pad.
// Corner case with empty descriptions.
TEST(PrintFormattedEntryTest, EmptyDescription) {
  std::string Buffer = runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", ""},
                                              /*InitialPad=*/0,
                                              /*EntryWidth=*/20,
                                              /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "Checker             ");
}

// No wrapping after checker's name.
// No initial pad.
// Corner case with empty checker's name.
TEST(PrintFormattedEntryTest, EmptyEntry) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"", "Some description"},
                             /*InitialPad=*/0,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/0);

  EXPECT_EQ(Buffer, "                    Some description");
}

// No wrapping after checker's name.
// With initial pad.
// Narrow MinLineWidth with a word without spaces (no break).
TEST(PrintFormattedEntryTest, NarrowMinLineWidthNoSpaces) {
  std::string Buffer = runPrintFormattedEntry(
      /*EntryDescPair=*/{"Checker", "This_is_a_long_word_without_spaces"},
      /*InitialPad=*/2,
      /*EntryWidth=*/20,
      /*MinLineWidth=*/10);

  EXPECT_EQ(Buffer, "  Checker             This_is_a_long_word_without_spaces");
}

// No wrapping after checker's name.
// With initial pad.
// With wrapping in checker's description.
TEST(PrintFormattedEntryTest, MinLineWidthLessThanPad) {
  std::string Buffer =
      runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "short phrase"},
                             /*InitialPad=*/2,
                             /*EntryWidth=*/20,
                             /*MinLineWidth=*/15);

  llvm::StringRef Expected = "  Checker             short\n"
                             "                      phrase";
  EXPECT_EQ(Buffer, Expected);
}

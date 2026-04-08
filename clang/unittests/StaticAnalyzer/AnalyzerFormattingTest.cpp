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

using namespace clang::ento;

static void runPrintFormattedEntry(
    std::pair<llvm::StringRef, llvm::StringRef> EntryDescPair,
    size_t InitialPad, size_t EntryWidth, size_t MinLineWidth,
    llvm::SmallString<256> &OutBuffer) {
  llvm::raw_svector_ostream Out(OutBuffer);
  clang::AnalyzerOptions::printFormattedEntry(Out, EntryDescPair, InitialPad,
                                              EntryWidth, MinLineWidth);
}

// No wrapping after checker's name.
// No initial pad.
TEST(PrintFormattedEntryTest, SimplePrint) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                         /*InitialPad=*/0,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "Checker             A description");
}

// With wrapping after checker's name.
// No initial pad.
TEST(PrintFormattedEntryTest, EntryLongerThanWidth) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(
      {/*EntryDescPair=*/"VeryLongCheckerName", "A description"},
      /*InitialPad=*/0,
      /*EntryWidth=*/10,
      /*MinLineWidth=*/0,
      /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "VeryLongCheckerName\n"
                    "          A description");
}

// With wrapping after checker's name.
// No initial pad.
// Corner case, when checker's name length equal to EntryWidth.
TEST(PrintFormattedEntryTest, ExactFillWidth) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                         /*InitialPad=*/0,
                         /*EntryWidth=*/7,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "Checker\n"
                    "       A description");
}

// With wrapping after checker's name.
// With initial pad.
// Corner case, when checker's name length equal to EntryWidth.
// This test matches how printFormattedEntry was called
// in -analyzer-checker-help, which led to a formatting bug.
TEST(PrintFormattedEntryTest, ExactFillWidthWithInitialPad) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                         /*InitialPad=*/2,
                         /*EntryWidth=*/7,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "  Checker\n"
                    "         A description");
}

// No wrapping after checker's name.
// With initial pad.
TEST(PrintFormattedEntryTest, WithInitialPadding) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "A description"},
                         /*InitialPad=*/2,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "  Checker             A description");
}

// No wrapping after checker's name.
// With initial pad.
// With wrapping in checker's description (MinLineWidth > 0).
TEST(PrintFormattedEntryTest, WrapDescription) {
  llvm::SmallString<256> Buffer;
  llvm::StringRef Desc =
      "This is a long description that should be wrapped into multiple lines.";
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", Desc},
                         /*InitialPad=*/2,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/40,
                         /*OutBuffer=*/Buffer);

  llvm::StringRef Expected =
      "  Checker             This is a long description\n"
      "                      that should be wrapped\n"
      "                      into multiple lines.";
  EXPECT_EQ(Buffer, Expected);
}

// No wrapping after checker's name.
// With initial pad.
// No wrapping in checker's descriptions (MinLineWidth = 0).
TEST(PrintFormattedEntryTest, NoWrap) {
  llvm::SmallString<256> Buffer;
  llvm::StringRef Desc = "This is a very long description that will not be "
                         "wrapped because MinLineWidth=0.";
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", Desc},
                         /*InitialPad=*/2,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  std::string Expected = "  Checker             " + Desc.str();
  EXPECT_EQ(Buffer, Expected);
}

// No wrapping after checker's name.
// No initial pad.
// Corner case with empty descriptions.
TEST(PrintFormattedEntryTest, EmptyDescription) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", ""},
                         /*InitialPad=*/0,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "Checker             ");
}

// No wrapping after checker's name.
// No initial pad.
// Corner case with empty checker's name.
TEST(PrintFormattedEntryTest, EmptyEntry) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"", "Some description"},
                         /*InitialPad=*/0,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/0,
                         /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "                    Some description");
}

// No wrapping after checker's name.
// With initial pad.
// Narrow MinLineWidth with a word without spaces (no break).
TEST(PrintFormattedEntryTest, NarrowMinLineWidthNoSpaces) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(
      /*EntryDescPair=*/{"Checker", "This_is_a_long_word_without_spaces"},
      /*InitialPad=*/2,
      /*EntryWidth=*/20,
      /*MinLineWidth=*/10,
      /*OutBuffer=*/Buffer);

  EXPECT_EQ(Buffer, "  Checker             This_is_a_long_word_without_spaces");
}

// No wrapping after checker's name.
// With initial pad.
// With wrapping in checker's description.
TEST(PrintFormattedEntryTest, MinLineWidthLessThanPad) {
  llvm::SmallString<256> Buffer;
  runPrintFormattedEntry(/*EntryDescPair=*/{"Checker", "short phrase"},
                         /*InitialPad=*/2,
                         /*EntryWidth=*/20,
                         /*MinLineWidth=*/15,
                         /*OutBuffer=*/Buffer);

  llvm::StringRef Expected = "  Checker             short\n"
                             "                      phrase";
  EXPECT_EQ(Buffer, Expected);
}

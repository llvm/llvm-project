//===- unittest/Format/FormatReplacementTest.cpp - Formatting unit test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Tooling/ReplacementTest.h"
#include "clang/Format/Format.h"

namespace clang {
namespace tooling {
namespace {

using format::FormatStyle;
using format::getLLVMStyle;

TEST_F(ReplacementTest, FormatCodeAfterReplacements) {
  // Column limit is 20.
  std::string Code = "Type *a =\n"
                     "    new Type();\n"
                     "g(iiiii, 0, jjjjj,\n"
                     "  0, kkkkk, 0, mm);\n"
                     "int  bad     = format   ;";
  std::string Expected = "auto a = new Type();\n"
                         "g(iiiii, nullptr,\n"
                         "  jjjjj, nullptr,\n"
                         "  kkkkk, nullptr,\n"
                         "  mm);\n"
                         "int  bad     = format   ;";
  FileID ID = Context.createInMemoryFile("format.cpp", Code);
  tooling::Replacements Replaces = toReplacements(
      {tooling::Replacement(Context.Sources, Context.getLocation(ID, 1, 1), 6,
                            "auto "),
       tooling::Replacement(Context.Sources, Context.getLocation(ID, 3, 10), 1,
                            "nullptr"),
       tooling::Replacement(Context.Sources, Context.getLocation(ID, 4, 3), 1,
                            "nullptr"),
       tooling::Replacement(Context.Sources, Context.getLocation(ID, 4, 13), 1,
                            "nullptr")});

  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 20; // Set column limit to 20 to increase readibility.
  auto FormattedReplaces = formatReplacements(Code, Replaces, Style);
  EXPECT_TRUE(static_cast<bool>(FormattedReplaces))
      << llvm::toString(FormattedReplaces.takeError()) << "\n";
  auto Result = applyAllReplacements(Code, *FormattedReplaces);
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(Expected, *Result);
}

TEST_F(ReplacementTest, SortIncludesAfterReplacement) {
  std::string Code = "#include \"a.h\"\n"
                     "#include \"c.h\"\n"
                     "\n"
                     "int main() {\n"
                     "  return 0;\n"
                     "}";
  std::string Expected = "#include \"a.h\"\n"
                         "#include \"b.h\"\n"
                         "#include \"c.h\"\n"
                         "\n"
                         "int main() {\n"
                         "  return 0;\n"
                         "}";
  FileID ID = Context.createInMemoryFile("fix.cpp", Code);
  tooling::Replacements Replaces = toReplacements(
      {tooling::Replacement(Context.Sources, Context.getLocation(ID, 1, 1), 0,
                            "#include \"b.h\"\n")});

  FormatStyle Style = getLLVMStyle();
  Style.SortIncludes = FormatStyle::SI_CaseSensitive;
  auto FormattedReplaces = formatReplacements(Code, Replaces, Style);
  EXPECT_TRUE(static_cast<bool>(FormattedReplaces))
      << llvm::toString(FormattedReplaces.takeError()) << "\n";
  auto Result = applyAllReplacements(Code, *FormattedReplaces);
  EXPECT_TRUE(static_cast<bool>(Result));
  EXPECT_EQ(Expected, *Result);
}

} // namespace
} // namespace tooling
} // namespace clang

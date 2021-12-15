//===- unittest/Format/FormatTestJson.cpp - Formatting tests for Json     -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test-json"

namespace clang {
namespace format {

class FormatTestJson : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");

    tooling::Replacements Replaces;

    // Mock up what ClangFormat.cpp will do for JSON by adding a variable
    // to trick JSON into being JavaScript
    if (Style.isJson() && !Style.DisableFormat) {
      auto Err = Replaces.add(
          tooling::Replacement(tooling::Replacement("", 0, 0, "x = ")));
      if (Err) {
        llvm::errs() << "Bad Json variable insertion\n";
      }
    }
    auto ChangedCode = applyAllReplacements(Code, Replaces);
    if (!ChangedCode) {
      llvm::errs() << "Bad Json varibale replacement\n";
    }
    StringRef NewCode = *ChangedCode;

    std::vector<tooling::Range> Ranges(1, tooling::Range(0, NewCode.size()));
    Replaces = reformat(Style, NewCode, Ranges);
    auto Result = applyAllReplacements(NewCode, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string
  format(llvm::StringRef Code,
         const FormatStyle &Style = getLLVMStyle(FormatStyle::LK_Json)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getLLVMStyle(FormatStyle::LK_Json);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormatStable(llvm::StringRef Code,
                                 const FormatStyle &Style) {
    EXPECT_EQ(Code.str(), format(Code, Style)) << "Expected code is not stable";
  }

  static void
  verifyFormat(llvm::StringRef Code,
               const FormatStyle &Style = getLLVMStyle(FormatStyle::LK_Json)) {
    verifyFormatStable(Code, Style);
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestJson, JsonRecord) {
  verifyFormat("{}");
  verifyFormat("{\n"
               "  \"name\": 1\n"
               "}");
  verifyFormat("{\n"
               "  \"name\": \"Foo\"\n"
               "}");
  verifyFormat("{\n"
               "  \"name\": {\n"
               "    \"value\": 1\n"
               "  }\n"
               "}");
  verifyFormat("{\n"
               "  \"name\": {\n"
               "    \"value\": 1\n"
               "  },\n"
               "  \"name\": {\n"
               "    \"value\": 2\n"
               "  }\n"
               "}");
  verifyFormat("{\n"
               "  \"name\": {\n"
               "    \"value\": [\n"
               "      1,\n"
               "      2,\n"
               "    ]\n"
               "  }\n"
               "}");
  verifyFormat("{\n"
               "  \"name\": {\n"
               "    \"value\": [\n"
               "      \"name\": {\n"
               "        \"value\": 1\n"
               "      },\n"
               "      \"name\": {\n"
               "        \"value\": 2\n"
               "      }\n"
               "    ]\n"
               "  }\n"
               "}");
  verifyFormat(R"({
  "firstName": "John",
  "lastName": "Smith",
  "isAlive": true,
  "age": 27,
  "address": {
    "streetAddress": "21 2nd Street",
    "city": "New York",
    "state": "NY",
    "postalCode": "10021-3100"
  },
  "phoneNumbers": [
    {
      "type": "home",
      "number": "212 555-1234"
    },
    {
      "type": "office",
      "number": "646 555-4567"
    }
  ],
  "children": [],
  "spouse": null
})");
}

TEST_F(FormatTestJson, JsonArray) {
  verifyFormat("[]");
  verifyFormat("[\n"
               "  1\n"
               "]");
  verifyFormat("[\n"
               "  1,\n"
               "  2\n"
               "]");
  verifyFormat("[\n"
               "  {},\n"
               "  {}\n"
               "]");
  verifyFormat("[\n"
               "  {\n"
               "    \"name\": 1\n"
               "  },\n"
               "  {}\n"
               "]");
}

TEST_F(FormatTestJson, JsonNoStringSplit) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Json);
  Style.IndentWidth = 4;
  verifyFormat(
      "[\n"
      "    {\n"
      "        "
      "\"naaaaaaaa\": \"foooooooooooooooooooooo oooooooooooooooooooooo\"\n"
      "    },\n"
      "    {}\n"
      "]",
      Style);
  verifyFormat("[\n"
               "    {\n"
               "        "
               "\"naaaaaaaa\": "
               "\"foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
               "oooooooooooooooooooooooooo\"\n"
               "    },\n"
               "    {}\n"
               "]",
               Style);

  Style.ColumnLimit = 80;
  verifyFormat("[\n"
               "    {\n"
               "        "
               "\"naaaaaaaa\":\n"
               "            "
               "\"foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
               "oooooooooooooooooooooooooo\"\n"
               "    },\n"
               "    {}\n"
               "]",
               Style);
}

TEST_F(FormatTestJson, DisableJsonFormat) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Json);
  verifyFormatStable("{}", Style);
  verifyFormatStable("{\n"
                     "  \"name\": 1\n"
                     "}",
                     Style);

  // Since we have to disable formatting to run this test, we shall refrain from
  // calling test::messUp lest we change the unformatted code and cannot format
  // it back to how it started.
  Style.DisableFormat = true;
  verifyFormatStable("{}", Style);
  verifyFormatStable("{\n"
                     "  \"name\": 1\n"
                     "}",
                     Style);
}

} // namespace format
} // end namespace clang

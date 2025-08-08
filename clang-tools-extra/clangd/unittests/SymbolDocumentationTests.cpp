//===-- SymbolDocumentationTests.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SymbolDocumentation.h"

#include "support/Markup.h"
#include "clang/Basic/CommentOptions.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

TEST(SymbolDocumentation, Parse) {

  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {
          "foo bar",
          "foo bar",
          "foo bar",
          "foo bar",
      },
      {
          "foo\nbar\n",
          "foo\nbar",
          "foo\nbar",
          "foo bar",
      },
      {
          "foo\n\nbar\n",
          "foo\n\nbar",
          "foo\n\nbar",
          "foo\n\nbar",
      },
      {
          "foo \\p bar baz",
          "foo `bar` baz",
          "foo `bar` baz",
          "foo bar baz",
      },
      {
          "foo \\e bar baz",
          "foo \\*bar\\* baz",
          "foo *bar* baz",
          "foo *bar* baz",
      },
      {
          "foo \\b bar baz",
          "foo \\*\\*bar\\*\\* baz",
          "foo **bar** baz",
          "foo **bar** baz",
      },
      {
          "foo \\ref bar baz",
          "foo \\*\\*\\\\ref\\*\\* \\*bar\\* baz",
          "foo **\\ref** *bar* baz",
          "foo **\\ref** *bar* baz",
      },
      {
          "foo @ref bar baz",
          "foo \\*\\*@ref\\*\\* \\*bar\\* baz",
          "foo **@ref** *bar* baz",
          "foo **@ref** *bar* baz",
      },
      {
          "\\brief this is a \\n\nbrief description",
          "\\*\\*\\\\brief\\*\\* this is a   \nbrief description",
          "**\\brief** this is a   \nbrief description",
          "**\\brief** this is a\nbrief description",
      },
      {
          "\\throw exception foo",
          "\\*\\*\\\\throw\\*\\* \\*exception\\* foo",
          "**\\throw** *exception* foo",
          "**\\throw** *exception* foo",
      },
      {
          "\\brief this is a brief description\n\n\\li item 1\n\\li item "
          "2\n\\arg item 3",
          "\\*\\*\\\\brief\\*\\* this is a brief description\n\n- item 1\n\n- "
          "item "
          "2\n\n- "
          "item 3",
          "**\\brief** this is a brief description\n\n- item 1\n\n- item "
          "2\n\n- "
          "item 3",
          "**\\brief** this is a brief description\n\n- item 1\n\n- item "
          "2\n\n- "
          "item 3",
      },
      {
          "\\defgroup mygroup this is a group\nthis is not a group description",
          "\\*\\*@defgroup\\*\\* `mygroup this is a group`\n\nthis is not a "
          "group "
          "description",
          "**@defgroup** `mygroup this is a group`\n\nthis is not a group "
          "description",
          "**@defgroup** `mygroup this is a group`\n\nthis is not a group "
          "description",
      },
      {
          "\\verbatim\nthis is a\nverbatim block containing\nsome verbatim "
          "text\n\\endverbatim",
          "\\*\\*@verbatim\\*\\*\n\n```\nthis is a\nverbatim block "
          "containing\nsome "
          "verbatim text\n```\n\n\\*\\*@endverbatim\\*\\*",
          "**@verbatim**\n\n```\nthis is a\nverbatim block containing\nsome "
          "verbatim text\n```\n\n**@endverbatim**",
          "**@verbatim**\n\nthis is a\nverbatim block containing\nsome "
          "verbatim text\n\n**@endverbatim**",
      },
      {
          "@param foo this is a parameter\n@param bar this is another "
          "parameter",
          "",
          "",
          "",
      },
      {
          "@brief brief docs\n\n@param foo this is a parameter\n\nMore "
          "description\ndocumentation",
          "\\*\\*@brief\\*\\* brief docs\n\nMore description\ndocumentation",
          "**@brief** brief docs\n\nMore description\ndocumentation",
          "**@brief** brief docs\n\nMore description documentation",
      },
      {
          "<b>this is a bold text</b>\nnormal text\n<i>this is an italic "
          "text</i>\n<code>this is a code block</code>",
          "\\<b>this is a bold text\\</b>\nnormal text\n\\<i>this is an italic "
          "text\\</i>\n\\<code>this is a code block\\</code>",
          "\\<b>this is a bold text\\</b>\nnormal text\n\\<i>this is an italic "
          "text\\</i>\n\\<code>this is a code block\\</code>",
          "<b>this is a bold text</b> normal text <i>this is an italic "
          "text</i> <code>this is a code block</code>",
      },
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.docToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

} // namespace clangd
} // namespace clang

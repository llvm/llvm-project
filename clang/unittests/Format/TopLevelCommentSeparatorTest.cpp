//===- unittest/Format/TopLevelCommentSeparatorTest.cpp - Formatting unit tests
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test-comments"

namespace clang {
namespace format {
namespace test {
namespace {

class TopLevelCommentSeparatorTest : public FormatTestBase {};

TEST_F(TopLevelCommentSeparatorTest, CheckEmptyLines) {
  FormatStyle Style = getDefaultStyle();
  Style.EmptyLinesAfterTopLevelComment = 2;
  Style.MaxEmptyLinesToKeep = 2;
  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n\n"
               "class Test {};",
               Style);

  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n\n"
               "static int test = 10;",
               Style);

  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n\n"
               "#include <iostream>",
               Style);

  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license",
               Style);

  verifyFormat("/* top level comment */\n\n\n"
               "#include <iostream>\n"
               "class Test {\n"
               "public:\n"
               "  void test() {}\n"
               "};\n"
               "int main() {\n"
               "  Test test;\n"
               "  test.test();\n"
               "  return 0;\n"
               "}",
               Style);

  Style.EmptyLinesAfterTopLevelComment = 1;
  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n"
               "class Test {};",
               Style);

  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n"
               "#include <iostream>",
               Style);

  verifyFormat("/* top level comment */\n\n"
               "#include <iostream>\n"
               "class Test {};",
               Style);
}

TEST_F(TopLevelCommentSeparatorTest, LimitedByMaxEmptyLinesToKeep) {
  FormatStyle Style = getDefaultStyle();
  Style.EmptyLinesAfterTopLevelComment = 2;
  Style.MaxEmptyLinesToKeep = 1;
  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n"
               "class Test {};",
               Style);

  verifyFormat("// start license\n"
               "// license text\n"
               "// more license text\n"
               "// end license\n\n"
               "#include <iostream>",
               Style);

  verifyFormat("/* top level comment */\n\n"
               "#include <iostream>\n"
               "class Test {};",
               Style);
}
} // namespace
} // namespace test
} // namespace format
} // namespace clang

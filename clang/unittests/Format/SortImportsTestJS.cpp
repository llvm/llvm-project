//===- unittest/Format/SortImportsTestJS.cpp - JS import sort unit tests --===//
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

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace {

class SortImportsTestJS : public ::testing::Test {
protected:
  std::string sort(StringRef Code, unsigned Offset = 0, unsigned Length = 0) {
    StringRef FileName = "input.js";
    if (Length == 0U)
      Length = Code.size() - Offset;
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    auto Sorted =
        applyAllReplacements(Code, sortIncludes(Style, Code, Ranges, FileName));
    EXPECT_TRUE(static_cast<bool>(Sorted));
    auto Formatted = applyAllReplacements(
        *Sorted, reformat(Style, *Sorted, Ranges, FileName));
    EXPECT_TRUE(static_cast<bool>(Formatted));
    return *Formatted;
  }

  void verifySort(llvm::StringRef Expected, llvm::StringRef Code,
                  unsigned Offset = 0, unsigned Length = 0) {
    std::string Result = sort(Code, Offset, Length);
    EXPECT_EQ(Expected.str(), Result) << "Expected:\n"
                                      << Expected << "\nActual:\n"
                                      << Result;
  }

  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
};

TEST_F(SortImportsTestJS, AlreadySorted) {
  verifySort("import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'c';\n"
             "\n"
             "let x = 1;",
             "import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'c';\n"
             "\n"
             "let x = 1;");
}

TEST_F(SortImportsTestJS, BasicSorting) {
  verifySort("import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'c';\n"
             "\n"
             "let x = 1;",
             "import {sym} from 'a';\n"
             "import {sym} from 'c';\n"
             "import {sym} from 'b';\n"
             "let x = 1;");
}

TEST_F(SortImportsTestJS, DefaultBinding) {
  verifySort("import A from 'a';\n"
             "import B from 'b';\n"
             "\n"
             "let x = 1;",
             "import B from 'b';\n"
             "import A from 'a';\n"
             "let x = 1;");
}

TEST_F(SortImportsTestJS, DefaultAndNamedBinding) {
  verifySort("import A, {a} from 'a';\n"
             "import B, {b} from 'b';\n"
             "\n"
             "let x = 1;",
             "import B, {b} from 'b';\n"
             "import A, {a} from 'a';\n"
             "let x = 1;");
}

TEST_F(SortImportsTestJS, WrappedImportStatements) {
  verifySort("import {sym1, sym2} from 'a';\n"
             "import {sym} from 'b';\n"
             "\n"
             "1;",
             "import\n"
             "  {sym}\n"
             "  from 'b';\n"
             "import {\n"
             "  sym1,\n"
             "  sym2\n"
             "} from 'a';\n"
             "1;");
}

TEST_F(SortImportsTestJS, SeparateMainCodeBody) {
  verifySort("import {sym} from 'a';"
             "\n"
             "let x = 1;\n",
             "import {sym} from 'a'; let x = 1;\n");
}

TEST_F(SortImportsTestJS, Comments) {
  verifySort("/** @fileoverview This is a great file. */\n"
             "// A very important import follows.\n"
             "import {sym} from 'a';  /* more comments */\n"
             "import {sym} from 'b';  // from //foo:bar\n",
             "/** @fileoverview This is a great file. */\n"
             "import {sym} from 'b';  // from //foo:bar\n"
             "// A very important import follows.\n"
             "import {sym} from 'a';  /* more comments */\n");
  verifySort("import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "\n"
             "/** Comment on variable. */\n"
             "const x = 1;\n",
             "import {sym} from 'b';\n"
             "import {sym} from 'a';\n"
             "\n"
             "/** Comment on variable. */\n"
             "const x = 1;\n");
}

TEST_F(SortImportsTestJS, SortStar) {
  verifySort("import * as foo from 'a';\n"
             "import {sym} from 'a';\n"
             "import * as bar from 'b';\n",
             "import {sym} from 'a';\n"
             "import * as foo from 'a';\n"
             "import * as bar from 'b';\n");
}

TEST_F(SortImportsTestJS, AliasesSymbols) {
  verifySort("import {sym1 as alias1} from 'b';\n"
             "import {sym2 as alias2, sym3 as alias3} from 'c';\n",
             "import {sym2 as alias2, sym3 as alias3} from 'c';\n"
             "import {sym1 as alias1} from 'b';\n");
}

TEST_F(SortImportsTestJS, SortSymbols) {
  verifySort("import {sym1, sym2 as a, sym3} from 'b';\n",
             "import {sym2 as a, sym1, sym3} from 'b';\n");
  verifySort("import {sym1 /* important! */, /*!*/ sym2 as a} from 'b';\n",
             "import {/*!*/ sym2 as a, sym1 /* important! */} from 'b';\n");
  verifySort("import {sym1, sym2} from 'b';\n", "import {\n"
                                                "  sym2 \n"
                                                ",\n"
                                                " sym1 \n"
                                                "} from 'b';\n");
}

TEST_F(SortImportsTestJS, GroupImports) {
  verifySort("import {a} from 'absolute';\n"
             "\n"
             "import {b} from '../parent';\n"
             "import {b} from '../parent/nested';\n"
             "\n"
             "import {b} from './relative/path';\n"
             "import {b} from './relative/path/nested';\n"
             "\n"
             "let x = 1;\n",
             "import {b} from './relative/path/nested';\n"
             "import {b} from './relative/path';\n"
             "import {b} from '../parent/nested';\n"
             "import {b} from '../parent';\n"
             "import {a} from 'absolute';\n"
             "let x = 1;\n");
}

TEST_F(SortImportsTestJS, Exports) {
  verifySort("import {S} from 'bpath';\n"
             "\n"
             "import {T} from './cpath';\n"
             "\n"
             "export {A, B} from 'apath';\n"
             "export {P} from '../parent';\n"
             "export {R} from './relative';\n"
             "export {S};\n"
             "\n"
             "let x = 1;\n"
             "export y = 1;\n",
             "export {R} from './relative';\n"
             "import {T} from './cpath';\n"
             "export {S};\n"
             "export {A, B} from 'apath';\n"
             "import {S} from 'bpath';\n"
             "export {P} from '../parent';\n"
             "let x = 1;\n"
             "export y = 1;\n");
  verifySort("import {S} from 'bpath';\n"
             "\n"
             "export {T} from 'epath';\n",
             "export {T} from 'epath';\n"
             "import {S} from 'bpath';\n");
}

TEST_F(SortImportsTestJS, SideEffectImports) {
  verifySort("import 'ZZside-effect';\n"
             "import 'AAside-effect';\n"
             "\n"
             "import {A} from 'absolute';\n"
             "\n"
             "import {R} from './relative';\n",
             "import {R} from './relative';\n"
             "import 'ZZside-effect';\n"
             "import {A} from 'absolute';\n"
             "import 'AAside-effect';\n");
}

TEST_F(SortImportsTestJS, AffectedRange) {
  // Affected range inside of import statements.
  verifySort("import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'c';\n"
             "\n"
             "let x = 1;",
             "import {sym} from 'c';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'a';\n"
             "let x = 1;",
             0, 30);
  // Affected range outside of import statements.
  verifySort("import {sym} from 'c';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'a';\n"
             "\n"
             "let x = 1;",
             "import {sym} from 'c';\n"
             "import {sym} from 'b';\n"
             "import {sym} from 'a';\n"
             "\n"
             "let x = 1;",
             70, 1);
}

TEST_F(SortImportsTestJS, SortingCanShrink) {
  // Sort excluding a suffix.
  verifySort("import {B} from 'a';\n"
             "import {A} from 'b';\n"
             "\n"
             "1;",
             "import {A} from 'b';\n"
             "\n"
             "import {B} from 'a';\n"
             "\n"
             "1;");
}

TEST_F(SortImportsTestJS, TrailingComma) {
  verifySort("import {A, B,} from 'aa';\n", "import {B, A,} from 'aa';\n");
}

TEST_F(SortImportsTestJS, SortCaseInsensitive) {
  verifySort("import {A} from 'aa';\n"
             "import {A} from 'Ab';\n"
             "import {A} from 'b';\n"
             "import {A} from 'Bc';\n"
             "\n"
             "1;",
             "import {A} from 'b';\n"
             "import {A} from 'Bc';\n"
             "import {A} from 'Ab';\n"
             "import {A} from 'aa';\n"
             "\n"
             "1;");
  verifySort("import {aa, Ab, b, Bc} from 'x';\n"
             "\n"
             "1;",
             "import {b, Bc, Ab, aa} from 'x';\n"
             "\n"
             "1;");
}

TEST_F(SortImportsTestJS, SortMultiLine) {
  // Reproduces issue where multi-line import was not parsed correctly.
  verifySort("import {A} from 'a';\n"
             "import {A} from 'b';\n"
             "\n"
             "1;",
             "import\n"
             "{\n"
             "A\n"
             "}\n"
             "from\n"
             "'b';\n"
             "import {A} from 'a';\n"
             "\n"
             "1;");
}

TEST_F(SortImportsTestJS, SortDefaultImports) {
  // Reproduces issue where multi-line import was not parsed correctly.
  verifySort("import {A} from 'a';\n"
             "import {default as B} from 'b';\n",
             "import {default as B} from 'b';\n"
             "import {A} from 'a';\n");
}

TEST_F(SortImportsTestJS, MergeImports) {
  // basic operation
  verifySort("import {X, Y} from 'a';\n"
             "import {Z} from 'z';\n"
             "\n"
             "X + Y + Z;\n",
             "import {X} from 'a';\n"
             "import {Z} from 'z';\n"
             "import {Y} from 'a';\n"
             "\n"
             "X + Y + Z;\n");

  // merge only, no resorting.
  verifySort("import {A, B} from 'foo';\n", "import {A} from 'foo';\n"
                                            "import {B} from 'foo';");

  // empty imports
  verifySort("import {A} from 'foo';\n", "import {} from 'foo';\n"
                                         "import {A} from 'foo';");

  // ignores import *
  verifySort("import * as foo from 'foo';\n"
             "import {A} from 'foo';\n",
             "import   * as foo from 'foo';\n"
             "import {A} from 'foo';\n");

  // ignores default import
  verifySort("import X from 'foo';\n"
             "import {A} from 'foo';\n",
             "import    X from 'foo';\n"
             "import {A} from 'foo';\n");

  // keeps comments
  // known issue: loses the 'also a' comment.
  verifySort("// a\n"
             "import {/* x */ X, /* y */ Y} from 'a';\n"
             "// z\n"
             "import {Z} from 'z';\n"
             "\n"
             "X + Y + Z;\n",
             "// a\n"
             "import {/* y */ Y} from 'a';\n"
             "// z\n"
             "import {Z} from 'z';\n"
             "// also a\n"
             "import {/* x */ X} from 'a';\n"
             "\n"
             "X + Y + Z;\n");

  // do not merge imports and exports
  verifySort("import {A} from 'foo';\n"
             "\n"
             "export {B} from 'foo';\n",
             "import {A} from 'foo';\n"
             "export   {B} from 'foo';");
  // do merge exports
  verifySort("export {A, B} from 'foo';\n", "export {A} from 'foo';\n"
                                            "export   {B} from 'foo';");

  // do not merge side effect imports with named ones
  verifySort("import './a';\n"
             "\n"
             "import {bar} from './a';\n",
             "import {bar} from './a';\n"
             "import './a';\n");
}

TEST_F(SortImportsTestJS, RespectsClangFormatOff) {
  verifySort("// clang-format off\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format on\n",
             "// clang-format off\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format on\n");

  verifySort("import {A} from './sorted1_a';\n"
             "import {B} from './sorted1_b';\n"
             "// clang-format off\n"
             "import {B} from './unsorted_b';\n"
             "import {A} from './unsorted_a';\n"
             "// clang-format on\n"
             "import {A} from './sorted2_a';\n"
             "import {B} from './sorted2_b';\n",
             "import {B} from './sorted1_b';\n"
             "import {A} from './sorted1_a';\n"
             "// clang-format off\n"
             "import {B} from './unsorted_b';\n"
             "import {A} from './unsorted_a';\n"
             "// clang-format on\n"
             "import {B} from './sorted2_b';\n"
             "import {A} from './sorted2_a';\n");

  // Boundary cases
  verifySort("// clang-format on\n", "// clang-format on\n");
  verifySort("// clang-format off\n", "// clang-format off\n");
  verifySort("// clang-format on\n"
             "// clang-format off\n",
             "// clang-format on\n"
             "// clang-format off\n");
  verifySort("// clang-format off\n"
             "// clang-format on\n"
             "import {A} from './a';\n"
             "import {B} from './b';\n",
             "// clang-format off\n"
             "// clang-format on\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n");
  // section ends with comment
  verifySort("// clang-format on\n"
             "import {A} from './a';\n"
             "import {B} from './b';\n"
             "import {C} from './c';\n"
             "\n" // inserted empty line is working as intended: splits imports
                  // section from main code body
             "// clang-format off\n",
             "// clang-format on\n"
             "import {C} from './c';\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format off\n");
}

TEST_F(SortImportsTestJS, RespectsClangFormatOffInNamedImports) {
  verifySort("// clang-format off\n"
             "import {B, A} from './b';\n"
             "// clang-format on\n"
             "const x = 1;",
             "// clang-format off\n"
             "import {B, A} from './b';\n"
             "// clang-format on\n"
             "const x =   1;");
}

} // end namespace
} // end namespace format
} // end namespace clang

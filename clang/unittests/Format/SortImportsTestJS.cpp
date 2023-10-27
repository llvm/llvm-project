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

  void _verifySort(const char *File, int Line, llvm::StringRef Expected,
                   llvm::StringRef Code, unsigned Offset = 0,
                   unsigned Length = 0) {
    ::testing::ScopedTrace t(File, Line, ::testing::Message() << Code.str());
    std::string Result = sort(Code, Offset, Length);
    EXPECT_EQ(Expected.str(), Result) << "Expected:\n"
                                      << Expected << "\nActual:\n"
                                      << Result;
  }

  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
};

#define verifySort(...) _verifySort(__FILE__, __LINE__, __VA_ARGS__)

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
             "let x = 1;",
             "import {sym} from 'a'; let x = 1;");
}

TEST_F(SortImportsTestJS, Comments) {
  verifySort("/** @fileoverview This is a great file. */\n"
             "// A very important import follows.\n"
             "import {sym} from 'a';  /* more comments */\n"
             "import {sym} from 'b';  // from //foo:bar\n",
             "/** @fileoverview This is a great file. */\n"
             "import {sym} from 'b';  // from //foo:bar\n"
             "// A very important import follows.\n"
             "import {sym} from 'a';  /* more comments */");
  verifySort("import {sym} from 'a';\n"
             "import {sym} from 'b';\n"
             "\n"
             "/** Comment on variable. */\n"
             "const x = 1;",
             "import {sym} from 'b';\n"
             "import {sym} from 'a';\n"
             "\n"
             "/** Comment on variable. */\n"
             "const x = 1;");
}

TEST_F(SortImportsTestJS, SortStar) {
  verifySort("import * as foo from 'a';\n"
             "import {sym} from 'a';\n"
             "import * as bar from 'b';\n",
             "import {sym} from 'a';\n"
             "import * as foo from 'a';\n"
             "import * as bar from 'b';");
}

TEST_F(SortImportsTestJS, AliasesSymbols) {
  verifySort("import {sym1 as alias1} from 'b';\n"
             "import {sym2 as alias2, sym3 as alias3} from 'c';\n",
             "import {sym2 as alias2, sym3 as alias3} from 'c';\n"
             "import {sym1 as alias1} from 'b';");
}

TEST_F(SortImportsTestJS, SortSymbols) {
  verifySort("import {sym1, sym2 as a, sym3} from 'b';\n",
             "import {sym2 as a, sym1, sym3} from 'b';");
  verifySort("import {sym1 /* important! */, /*!*/ sym2 as a} from 'b';\n",
             "import {/*!*/ sym2 as a, sym1 /* important! */} from 'b';");
  verifySort("import {sym1, sym2} from 'b';\n", "import {\n"
                                                "  sym2 \n"
                                                ",\n"
                                                " sym1 \n"
                                                "} from 'b';");
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
             "let x = 1;",
             "import {b} from './relative/path/nested';\n"
             "import {b} from './relative/path';\n"
             "import {b} from '../parent/nested';\n"
             "import {b} from '../parent';\n"
             "import {a} from 'absolute';\n"
             "let x = 1;");
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
             "export y = 1;",
             "export {R} from './relative';\n"
             "import {T} from './cpath';\n"
             "export {S};\n"
             "export {A, B} from 'apath';\n"
             "import {S} from 'bpath';\n"
             "export {P} from '../parent';\n"
             "let x = 1;\n"
             "export y = 1;");
  verifySort("import {S} from 'bpath';\n"
             "\n"
             "export {T} from 'epath';\n",
             "export {T} from 'epath';\n"
             "import {S} from 'bpath';");
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
             "import 'AAside-effect';");
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
  verifySort("import {A, B,} from 'aa';\n", "import {B, A,} from 'aa';");
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
             "import {A} from 'a';");
}

TEST_F(SortImportsTestJS, MergeImports) {
  // basic operation
  verifySort("import {X, Y} from 'a';\n"
             "import {Z} from 'z';\n"
             "\n"
             "X + Y + Z;",
             "import {X} from 'a';\n"
             "import {Z} from 'z';\n"
             "import {Y} from 'a';\n"
             "\n"
             "X + Y + Z;");

  // merge only, no resorting.
  verifySort("import {A, B} from 'foo';\n", "import {A} from 'foo';\n"
                                            "import {B} from 'foo';");

  // empty imports
  verifySort("import {A} from 'foo';\n", "import {} from 'foo';\n"
                                         "import {A} from 'foo';");

  // ignores import *
  verifySort("import * as foo from 'foo';\n"
             "import {A} from 'foo';",
             "import   * as foo from 'foo';\n"
             "import {A} from 'foo';");

  // ignores default import
  verifySort("import X from 'foo';\n"
             "import {A} from 'foo';",
             "import    X from 'foo';\n"
             "import {A} from 'foo';");

  // keeps comments
  // known issue: loses the 'also a' comment.
  verifySort("// a\n"
             "import {/* x */ X, /* y */ Y} from 'a';\n"
             "// z\n"
             "import {Z} from 'z';\n"
             "\n"
             "X + Y + Z;",
             "// a\n"
             "import {/* y */ Y} from 'a';\n"
             "// z\n"
             "import {Z} from 'z';\n"
             "// also a\n"
             "import {/* x */ X} from 'a';\n"
             "\n"
             "X + Y + Z;");

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
             "import './a';");
}

TEST_F(SortImportsTestJS, RespectsClangFormatOff) {
  verifySort("// clang-format off\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format on",
             "// clang-format off\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format on");

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
             "import {A} from './sorted2_a';");

  // Boundary cases
  verifySort("// clang-format on", "// clang-format on");
  verifySort("// clang-format off", "// clang-format off");
  verifySort("// clang-format on\n"
             "// clang-format off",
             "// clang-format on\n"
             "// clang-format off");
  verifySort("// clang-format off\n"
             "// clang-format on\n"
             "import {A} from './a';\n"
             "import {B} from './b';\n",
             "// clang-format off\n"
             "// clang-format on\n"
             "import {B} from './b';\n"
             "import {A} from './a';");
  // section ends with comment
  verifySort("// clang-format on\n"
             "import {A} from './a';\n"
             "import {B} from './b';\n"
             "import {C} from './c';\n"
             "\n" // inserted empty line is working as intended: splits imports
                  // section from main code body
             "// clang-format off",
             "// clang-format on\n"
             "import {C} from './c';\n"
             "import {B} from './b';\n"
             "import {A} from './a';\n"
             "// clang-format off");
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

TEST_F(SortImportsTestJS, ImportEqAliases) {
  verifySort("import {B} from 'bar';\n"
             "import {A} from 'foo';\n"
             "\n"
             "import Z = A.C;\n"
             "import Y = B.C.Z;\n"
             "\n"
             "export {Z};\n"
             "\n"
             "console.log(Z);",
             "import {A} from 'foo';\n"
             "import Z = A.C;\n"
             "export {Z};\n"
             "import {B} from 'bar';\n"
             "import Y = B.C.Z;\n"
             "\n"
             "console.log(Z);");
}

TEST_F(SortImportsTestJS, ImportExportType) {
  verifySort("import type {sym} from 'a';\n"
             "import {type sym} from 'b';\n"
             "import {sym} from 'c';\n"
             "import type sym from 'd';\n"
             "import type * as sym from 'e';\n"
             "\n"
             "let x = 1;",
             "import {sym} from 'c';\n"
             "import type {sym} from 'a';\n"
             "import type * as sym from 'e';\n"
             "import type sym from 'd';\n"
             "import {type sym} from 'b';\n"
             "let x = 1;");

  // Symbols within import statement
  verifySort("import {type sym1, type sym2 as a, sym3} from 'b';\n",
             "import {type sym2 as a, type sym1, sym3} from 'b';");

  // Merging
  verifySort("import {X, type Z} from 'a';\n"
             "import type {Y} from 'a';\n"
             "\n"
             "X + Y + Z;",
             "import {X} from 'a';\n"
             "import {type Z} from 'a';\n"
             "import type {Y} from 'a';\n"
             "\n"
             "X + Y + Z;");

  // Merging: empty imports
  verifySort("import type {A} from 'foo';\n", "import type {} from 'foo';\n"
                                              "import type {A} from 'foo';");

  // Merging: exports
  verifySort("export {A, type B} from 'foo';\n",
             "export {A} from 'foo';\n"
             "export   {type B} from 'foo';");

  // `export type X = Y;` should terminate import sorting. The following export
  // statements should therefore not merge.
  verifySort("export type A = B;\n"
             "export {X};\n"
             "export {Y};",
             "export type A = B;\n"
             "export {X};\n"
             "export {Y};");
}

TEST_F(SortImportsTestJS, TemplateKeyword) {
  // Reproduces issue where importing "template" disables imports sorting.
  verifySort("import {template} from './a';\n"
             "import {b} from './b';\n",
             "import {b} from './b';\n"
             "import {template} from './a';");
}

} // end namespace
} // end namespace format
} // end namespace clang

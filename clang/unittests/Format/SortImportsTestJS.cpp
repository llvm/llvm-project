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

} // end namespace
} // end namespace format
} // end namespace clang

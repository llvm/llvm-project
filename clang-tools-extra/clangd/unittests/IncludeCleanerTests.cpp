//===--- IncludeCleanerTests.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "IncludeCleaner.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

std::string guard(llvm::StringRef Code) {
  return "#pragma once\n" + Code.str();
}

TEST(IncludeCleaner, ReferencedLocations) {
  struct TestCase {
    std::string HeaderCode;
    std::string MainCode;
  };
  TestCase Cases[] = {
      // DeclRefExpr
      {
          "int ^x();",
          "int y = x();",
      },
      // RecordDecl
      {
          "class ^X;",
          "X *y;",
      },
      // TypedefType and UsingDecls
      {
          "using ^Integer = int;",
          "Integer x;",
      },
      {
          "namespace ns { struct ^X; struct ^X {}; }",
          "using ns::X;",
      },
      {
          "namespace ns { struct X; struct X {}; }",
          "using namespace ns;",
      },
      {
          "struct ^A {}; using B = A; using ^C = B;",
          "C a;",
      },
      {
          "typedef bool ^Y; template <typename T> struct ^X {};",
          "X<Y> x;",
      },
      {
          "struct Foo; struct ^Foo{}; typedef Foo ^Bar;",
          "Bar b;",
      },
      // MemberExpr
      {
          "struct ^X{int ^a;}; X ^foo();",
          "int y = foo().a;",
      },
      // Expr (type is traversed)
      {
          "class ^X{}; X ^foo();",
          "auto bar() { return foo(); }",
      },
      // Redecls
      {
          "class ^X; class ^X{}; class ^X;",
          "X *y;",
      },
      // Constructor
      {
          "struct ^X { ^X(int) {} int ^foo(); };",
          "auto x = X(42); auto y = x.foo();",
      },
      // Function
      {
          "void ^foo();",
          "void foo() {}",
      },
      {
          "void foo() {}",
          "void foo();",
      },
      {
          "inline void ^foo() {}",
          "void bar() { foo(); }",
      },
      // Static function
      {
          "struct ^X { static bool ^foo(); }; bool X::^foo() {}",
          "auto b = X::foo();",
      },
      // TemplateRecordDecl
      {
          "template <typename> class ^X;",
          "X<int> *y;",
      },
      // Type name not spelled out in code
      {
          "class ^X{}; X ^getX();",
          "auto x = getX();",
      },
      // Enums
      {
          "enum ^Color { ^Red = 42, Green = 9000};",
          "int MyColor = Red;",
      },
      {
          "struct ^X { enum ^Language { ^CXX = 42, Python = 9000}; };",
          "int Lang = X::CXX;",
      },
      // Macros
      {
          "#define ^CONSTANT 42",
          "int Foo = CONSTANT;",
      },
      {
          "#define ^FOO x",
          "#define BAR FOO",
      },
      {
          "#define INNER 42\n"
          "#define ^OUTER INNER",
          "int answer = OUTER;",
      },
      {
          "#define ^ANSWER 42\n"
          "#define ^SQUARE(X) X * X",
          "int sq = SQUARE(ANSWER);",
      },
      {
          "#define ^FOO\n"
          "#define ^BAR",
          "#if 0\n"
          "#if FOO\n"
          "BAR\n"
          "#endif\n"
          "#endif",
      },
      // Misc
      {
          "enum class ^Color : int;",
          "enum class Color : int {};",
      },
      {
          "enum class Color : int {};",
          "enum class Color : int;",
      },
      {
          "enum class ^Color;",
          "Color c;",
      },
      {
          "enum class ^Color : int;",
          "Color c;",
      },
      {
          "enum class ^Color : char;",
          "Color *c;",
      },
      {
          "enum class ^Color : char {};",
          "Color *c;",
      },
      {
          // When a type is resolved via a using declaration, the
          // UsingShadowDecl is not referenced in the AST.
          // Compare to TypedefType, or DeclRefExpr::getFoundDecl().
          //                                 ^
          "namespace ns { class ^X; }; using ns::X;",
          "X *y;",
      }};
  for (const TestCase &T : Cases) {
    TestTU TU;
    TU.Code = T.MainCode;
    Annotations Header(T.HeaderCode);
    TU.HeaderCode = Header.code().str();
    auto AST = TU.build();

    std::vector<Position> Points;
    for (const auto &Loc : findReferencedLocations(AST)) {
      if (AST.getSourceManager().getBufferName(Loc).endswith(
              TU.HeaderFilename)) {
        Points.push_back(offsetToPosition(
            TU.HeaderCode, AST.getSourceManager().getFileOffset(Loc)));
      }
    }
    llvm::sort(Points);

    EXPECT_EQ(Points, Header.points()) << T.HeaderCode << "\n---\n"
                                       << T.MainCode;
  }
}

TEST(IncludeCleaner, GetUnusedHeaders) {
  llvm::StringLiteral MainFile = R"cpp(
    #include "a.h"
    #include "b.h"
    #include "dir/c.h"
    #include "dir/unused.h"
    #include "unguarded.h"
    #include "unused.h"
    #include <system_header.h>
    void foo() {
      a();
      b();
      c();
    })cpp";
  // Build expected ast with symbols coming from headers.
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["foo.h"] = guard("void foo();");
  TU.AdditionalFiles["a.h"] = guard("void a();");
  TU.AdditionalFiles["b.h"] = guard("void b();");
  TU.AdditionalFiles["dir/c.h"] = guard("void c();");
  TU.AdditionalFiles["unused.h"] = guard("void unused();");
  TU.AdditionalFiles["dir/unused.h"] = guard("void dirUnused();");
  TU.AdditionalFiles["system/system_header.h"] = guard("");
  TU.AdditionalFiles["unguarded.h"] = "";
  TU.ExtraArgs.push_back("-I" + testPath("dir"));
  TU.ExtraArgs.push_back("-isystem" + testPath("system"));
  TU.Code = MainFile.str();
  ParsedAST AST = TU.build();
  std::vector<std::string> UnusedIncludes;
  for (const auto &Include : computeUnusedIncludes(AST))
    UnusedIncludes.push_back(Include->Written);
  EXPECT_THAT(UnusedIncludes,
              UnorderedElementsAre("\"unused.h\"", "\"dir/unused.h\""));
}

TEST(IncludeCleaner, VirtualBuffers) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "macros.h"

    using flags::FLAGS_FOO;

    // CLI will come from a define, __llvm__ is a built-in. In both cases, they
    // come from non-existent files.
    int y = CLI + __llvm__;

    int concat(a, b) = 42;
    )cpp";
  // The pasting operator in combination with DEFINE_FLAG will create
  // ScratchBuffer with `flags::FLAGS_FOO` that will have FileID but not
  // FileEntry.
  TU.AdditionalFiles["macros.h"] = R"cpp(
    #ifndef MACROS_H
    #define MACROS_H

    #define DEFINE_FLAG(X) \
    namespace flags { \
    int FLAGS_##X; \
    } \

    DEFINE_FLAG(FOO)

    #define ab x
    #define concat(x, y) x##y

    #endif // MACROS_H
    )cpp";
  TU.ExtraArgs = {"-DCLI=42"};
  ParsedAST AST = TU.build();
  auto &SM = AST.getSourceManager();
  auto &Includes = AST.getIncludeStructure();

  auto ReferencedFiles = findReferencedFiles(findReferencedLocations(AST), SM);
  llvm::StringSet<> ReferencedFileNames;
  for (FileID FID : ReferencedFiles)
    ReferencedFileNames.insert(
        SM.getPresumedLoc(SM.getLocForStartOfFile(FID)).getFilename());
  // Note we deduped the names as _number_ of <built-in>s is uninteresting.
  EXPECT_THAT(ReferencedFileNames.keys(),
              UnorderedElementsAre("<built-in>", "<scratch space>",
                                   testPath("macros.h")));

  // Should not crash due to FileIDs that are not headers.
  auto ReferencedHeaders = translateToHeaderIDs(ReferencedFiles, Includes, SM);
  std::vector<llvm::StringRef> ReferencedHeaderNames;
  for (IncludeStructure::HeaderID HID : ReferencedHeaders)
    ReferencedHeaderNames.push_back(Includes.getRealPath(HID));
  // Non-header files are gone at this point.
  EXPECT_THAT(ReferencedHeaderNames, ElementsAre(testPath("macros.h")));

  // Sanity check.
  EXPECT_THAT(getUnused(AST, ReferencedHeaders), ::testing::IsEmpty());
}

} // namespace
} // namespace clangd
} // namespace clang

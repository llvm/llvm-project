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

using ::testing::UnorderedElementsAre;

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
    #include "unused.h"
    void foo() {
      a();
      b();
      c();
    })cpp";
  // Build expected ast with symbols coming from headers.
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["foo.h"] = "void foo();";
  TU.AdditionalFiles["a.h"] = "void a();";
  TU.AdditionalFiles["b.h"] = "void b();";
  TU.AdditionalFiles["dir/c.h"] = "void c();";
  TU.AdditionalFiles["unused.h"] = "void unused();";
  TU.AdditionalFiles["dir/unused.h"] = "void dirUnused();";
  TU.AdditionalFiles["not_included.h"] = "void notIncluded();";
  TU.ExtraArgs = {"-I" + testPath("dir")};
  TU.Code = MainFile.str();
  ParsedAST AST = TU.build();
  auto UnusedIncludes = computeUnusedIncludes(AST);
  std::vector<std::string> UnusedHeaders;
  UnusedHeaders.reserve(UnusedIncludes.size());
  for (const auto &Include : UnusedIncludes)
    UnusedHeaders.push_back(Include->Written);
  EXPECT_THAT(UnusedHeaders,
              UnorderedElementsAre("\"unused.h\"", "\"dir/unused.h\""));
}

TEST(IncludeCleaner, ScratchBuffer) {
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = R"cpp(
    #include "macro_spelling_in_scratch_buffer.h"

    using flags::FLAGS_FOO;

    int concat(a, b) = 42;
    )cpp";
  // The pasting operator in combination with DEFINE_FLAG will create
  // ScratchBuffer with `flags::FLAGS_FOO` that will have FileID but not
  // FileEntry.
  TU.AdditionalFiles["macro_spelling_in_scratch_buffer.h"] = R"cpp(
    #define DEFINE_FLAG(X) \
    namespace flags { \
    int FLAGS_##X; \
    } \

    DEFINE_FLAG(FOO)

    #define ab x
    #define concat(x, y) x##y
    )cpp";
  ParsedAST AST = TU.build();
  auto &SM = AST.getSourceManager();
  auto &Includes = AST.getIncludeStructure();
  auto ReferencedFiles = findReferencedFiles(findReferencedLocations(AST), SM);
  auto Entry = SM.getFileManager().getFile(
      testPath("macro_spelling_in_scratch_buffer.h"));
  ASSERT_TRUE(Entry);
  auto FID = SM.translateFile(*Entry);
  // No "<scratch space>" FID.
  EXPECT_THAT(ReferencedFiles, UnorderedElementsAre(FID));
  // Should not crash due to <scratch space> "files" missing from include
  // structure.
  EXPECT_THAT(
      getUnused(Includes, translateToHeaderIDs(ReferencedFiles, Includes, SM)),
      ::testing::IsEmpty());
}

} // namespace
} // namespace clangd
} // namespace clang

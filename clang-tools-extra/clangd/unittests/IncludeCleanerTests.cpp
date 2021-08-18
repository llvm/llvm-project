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

} // namespace
} // namespace clangd
} // namespace clang

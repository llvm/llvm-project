//===--- IncludeCleanerTests.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "IncludeCleaner.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::Pointee;
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
      // When definition is available, we don't need to mark forward
      // declarations as used.
      {
          "class ^X {}; class X;",
          "X *y;",
      },
      // We already have forward declaration in the main file, the other
      // non-definition declarations are not needed.
      {
          "class ^X {}; class X;",
          "class X; X *y;",
      },
      // Nested class definition can occur outside of the parent class
      // definition. Bar declaration should be visible to its definition but
      // it will always be because we will mark Foo definition as used.
      {
          "class ^Foo { class Bar; };",
          "class Foo::Bar {};",
      },
      // TypedefType and UsingDecls
      {
          "using ^Integer = int;",
          "Integer x;",
      },
      {
          "namespace ns { void ^foo(); void ^foo() {} }",
          "using ns::foo;",
      },
      {
          "namespace ns { void foo(); void foo() {}; }",
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
      {
          "namespace ns { class X; }; using ns::^X;",
          "X *y;",
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
          "void ^foo(); void ^foo() {} void ^foo();",
          "void bar() { foo(); }",
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
      {
          "int ^foo(char); int ^foo(float);",
          "template<class T> int x = foo(T{});",
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
      }};
  for (const TestCase &T : Cases) {
    TestTU TU;
    TU.Code = T.MainCode;
    Annotations Header(T.HeaderCode);
    TU.HeaderCode = Header.code().str();
    auto AST = TU.build();

    std::vector<Position> Points;
    for (const auto &Loc : findReferencedLocations(AST).User) {
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

TEST(IncludeCleaner, Stdlib) {
  // Smoke tests only for finding used symbols/headers.
  // Details of Decl -> stdlib::Symbol -> stdlib::Headers mapping tested there.
  auto TU = TestTU::withHeaderCode(R"cpp(
    namespace std { class error_code {}; }
    class error_code {};
    namespace nonstd { class error_code {}; }
  )cpp");
  struct {
    llvm::StringRef Code;
    std::vector<llvm::StringRef> Symbols;
    std::vector<llvm::StringRef> Headers;
  } Tests[] = {
      {"std::error_code x;", {"std::error_code"}, {"<system_error>"}},
      {"error_code x;", {}, {}},
      {"nonstd::error_code x;", {}, {}},
  };

  for (const auto &Test : Tests) {
    TU.Code = Test.Code.str();
    ParsedAST AST = TU.build();
    std::vector<tooling::stdlib::Symbol> WantSyms;
    for (const auto &SymName : Test.Symbols) {
      auto QName = splitQualifiedName(SymName);
      auto Sym = tooling::stdlib::Symbol::named(QName.first, QName.second);
      EXPECT_TRUE(Sym) << SymName;
      WantSyms.push_back(*Sym);
    }
    std::vector<tooling::stdlib::Header> WantHeaders;
    for (const auto &HeaderName : Test.Headers) {
      auto Header = tooling::stdlib::Header::named(HeaderName);
      EXPECT_TRUE(Header) << HeaderName;
      WantHeaders.push_back(*Header);
    }

    ReferencedLocations Locs = findReferencedLocations(AST);
    EXPECT_THAT(Locs.Stdlib, ElementsAreArray(WantSyms));
    ReferencedFiles Files = findReferencedFiles(Locs, AST.getIncludeStructure(),
                                                AST.getSourceManager());
    EXPECT_THAT(Files.Stdlib, ElementsAreArray(WantHeaders));
  }
}

MATCHER_P(writtenInclusion, Written, "") {
  if (arg.Written != Written)
    *result_listener << arg.Written;
  return arg.Written == Written;
}

TEST(IncludeCleaner, StdlibUnused) {
  setIncludeCleanerAnalyzesStdlib(true);
  auto Cleanup =
      llvm::make_scope_exit([] { setIncludeCleanerAnalyzesStdlib(false); });

  auto TU = TestTU::withCode(R"cpp(
    #include <list>
    #include <queue>
    std::list<int> x;
  )cpp");
  // Layout of std library impl is not relevant.
  TU.AdditionalFiles["bits"] = R"cpp(
    #pragma once
    namespace std {
      template <typename> class list {};
      template <typename> class queue {};
    }
  )cpp";
  TU.AdditionalFiles["list"] = "#include <bits>";
  TU.AdditionalFiles["queue"] = "#include <bits>";
  TU.ExtraArgs = {"-isystem", testRoot()};
  auto AST = TU.build();

  auto Unused = computeUnusedIncludes(AST);
  EXPECT_THAT(Unused, ElementsAre(Pointee(writtenInclusion("<queue>"))));
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

  auto ReferencedFiles =
      findReferencedFiles(findReferencedLocations(AST), Includes, SM);
  llvm::StringSet<> ReferencedFileNames;
  for (FileID FID : ReferencedFiles.User)
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
  EXPECT_THAT(getUnused(AST, ReferencedHeaders), IsEmpty());
}

TEST(IncludeCleaner, DistinctUnguardedInclusions) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "bar.h"
    #include "foo.h"

    int LocalFoo = foo::Variable;
    )cpp";
  TU.AdditionalFiles["foo.h"] = R"cpp(
    #pragma once
    namespace foo {
    #include "unguarded.h"
    }
    )cpp";
  TU.AdditionalFiles["bar.h"] = R"cpp(
    #pragma once
    namespace bar {
    #include "unguarded.h"
    }
    )cpp";
  TU.AdditionalFiles["unguarded.h"] = R"cpp(
    constexpr int Variable = 42;
    )cpp";

  ParsedAST AST = TU.build();

  auto ReferencedFiles =
      findReferencedFiles(findReferencedLocations(AST),
                          AST.getIncludeStructure(), AST.getSourceManager());
  llvm::StringSet<> ReferencedFileNames;
  auto &SM = AST.getSourceManager();
  for (FileID FID : ReferencedFiles.User)
    ReferencedFileNames.insert(
        SM.getPresumedLoc(SM.getLocForStartOfFile(FID)).getFilename());
  // Note that we have uplifted the referenced files from non self-contained
  // headers to header-guarded ones.
  EXPECT_THAT(ReferencedFileNames.keys(),
              UnorderedElementsAre(testPath("foo.h")));
}

TEST(IncludeCleaner, NonSelfContainedHeaders) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "foo.h"

    int LocalFoo = Variable;
    )cpp";
  TU.AdditionalFiles["foo.h"] = R"cpp(
    #pragma once
    #include "indirection.h"
    )cpp";
  TU.AdditionalFiles["indirection.h"] = R"cpp(
    #include "unguarded.h"
    )cpp";
  TU.AdditionalFiles["unguarded.h"] = R"cpp(
    constexpr int Variable = 42;
    )cpp";

  ParsedAST AST = TU.build();

  auto ReferencedFiles =
      findReferencedFiles(findReferencedLocations(AST),
                          AST.getIncludeStructure(), AST.getSourceManager());
  llvm::StringSet<> ReferencedFileNames;
  auto &SM = AST.getSourceManager();
  for (FileID FID : ReferencedFiles.User)
    ReferencedFileNames.insert(
        SM.getPresumedLoc(SM.getLocForStartOfFile(FID)).getFilename());
  // Note that we have uplifted the referenced files from non self-contained
  // headers to header-guarded ones.
  EXPECT_THAT(ReferencedFileNames.keys(),
              UnorderedElementsAre(testPath("foo.h")));
}

TEST(IncludeCleaner, IWYUPragmas) {
  TestTU TU;
  TU.Code = R"cpp(
    #include "behind_keep.h" // IWYU pragma: keep
    )cpp";
  TU.AdditionalFiles["behind_keep.h"] = guard("");
  ParsedAST AST = TU.build();

  auto ReferencedFiles =
      findReferencedFiles(findReferencedLocations(AST),
                          AST.getIncludeStructure(), AST.getSourceManager());
  EXPECT_TRUE(ReferencedFiles.User.empty());
  EXPECT_THAT(AST.getDiagnostics(), llvm::ValueIs(IsEmpty()));
}

} // namespace
} // namespace clangd
} // namespace clang

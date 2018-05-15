//===-- XRefsTests.cpp  ---------------------------*- C++ -*--------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ClangdUnit.h"
#include "Compiler.h"
#include "Matchers.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "XRefs.h"
#include "index/FileIndex.h"
#include "index/SymbolCollector.h"
#include "clang/Index/IndexingAction.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
using namespace llvm;

namespace {
using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::Matcher;
using testing::UnorderedElementsAreArray;

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(PathRef File,
                          std::vector<Diag> Diagnostics) override {}
};

// Extracts ranges from an annotated example, and constructs a matcher for a
// highlight set. Ranges should be named $read/$write as appropriate.
Matcher<const std::vector<DocumentHighlight> &>
HighlightsFrom(const Annotations &Test) {
  std::vector<DocumentHighlight> Expected;
  auto Add = [&](const Range &R, DocumentHighlightKind K) {
    Expected.emplace_back();
    Expected.back().range = R;
    Expected.back().kind = K;
  };
  for (const auto &Range : Test.ranges())
    Add(Range, DocumentHighlightKind::Text);
  for (const auto &Range : Test.ranges("read"))
    Add(Range, DocumentHighlightKind::Read);
  for (const auto &Range : Test.ranges("write"))
    Add(Range, DocumentHighlightKind::Write);
  return UnorderedElementsAreArray(Expected);
}

TEST(HighlightsTest, All) {
  const char *Tests[] = {
      R"cpp(// Local variable
        int main() {
          int [[bonjour]];
          $write[[^bonjour]] = 2;
          int test1 = $read[[bonjour]];
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        struct [[MyClass]] {
          static void foo([[MyClass]]*) {}
        };
        } // namespace ns1
        int main() {
          ns1::[[My^Class]]* Params;
        }
      )cpp",

      R"cpp(// Function
        int [[^foo]](int) {}
        int main() {
          [[foo]]([[foo]](42));
          auto *X = &[[foo]];
        }
      )cpp",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto AST = TestTU::withCode(T.code()).build();
    EXPECT_THAT(findDocumentHighlights(AST, T.point()), HighlightsFrom(T))
        << Test;
  }
}

MATCHER_P(RangeIs, R, "") { return arg.range == R; }

TEST(GoToDefinition, WithIndex) {
  Annotations SymbolHeader(R"cpp(
        class $forward[[Forward]];
        class $foo[[Foo]] {};

        void $f1[[f1]]();

        inline void $f2[[f2]]() {}
      )cpp");
  Annotations SymbolCpp(R"cpp(
      class $forward[[forward]] {};
      void  $f1[[f1]]() {}
    )cpp");

  TestTU TU;
  TU.Code = SymbolCpp.code();
  TU.HeaderCode = SymbolHeader.code();
  auto Index = TU.index();
  auto runFindDefinitionsWithIndex = [&Index](const Annotations &Main) {
    auto AST = TestTU::withCode(Main.code()).build();
    return clangd::findDefinitions(AST, Main.point(), Index.get());
  };

  Annotations Test(R"cpp(// only declaration in AST.
        void [[f1]]();
        int main() {
          ^f1();
        }
      )cpp");
  EXPECT_THAT(runFindDefinitionsWithIndex(Test),
              testing::ElementsAreArray(
                  {RangeIs(SymbolCpp.range("f1")), RangeIs(Test.range())}));

  Test = Annotations(R"cpp(// definition in AST.
        void [[f1]]() {}
        int main() {
          ^f1();
        }
      )cpp");
  EXPECT_THAT(runFindDefinitionsWithIndex(Test),
              testing::ElementsAreArray(
                  {RangeIs(Test.range()), RangeIs(SymbolHeader.range("f1"))}));

  Test = Annotations(R"cpp(// forward declaration in AST.
        class [[Foo]];
        F^oo* create();
      )cpp");
  EXPECT_THAT(runFindDefinitionsWithIndex(Test),
              testing::ElementsAreArray(
                  {RangeIs(SymbolHeader.range("foo")), RangeIs(Test.range())}));

  Test = Annotations(R"cpp(// defintion in AST.
        class [[Forward]] {};
        F^orward create();
      )cpp");
  EXPECT_THAT(runFindDefinitionsWithIndex(Test),
              testing::ElementsAreArray({
                  RangeIs(Test.range()), RangeIs(SymbolHeader.range("forward")),
              }));
}

TEST(GoToDefinition, All) {
  const char *Tests[] = {
      R"cpp(// Local variable
        int main() {
          int [[bonjour]];
          ^bonjour = 2;
          int test1 = bonjour;
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        struct [[MyClass]] {};
        } // namespace ns1
        int main() {
          ns1::My^Class* Params;
        }
      )cpp",

      R"cpp(// Function definition via pointer
        int [[foo]](int) {}
        int main() {
          auto *X = &^foo;
        }
      )cpp",

      R"cpp(// Function declaration via call
        int [[foo]](int);
        int main() {
          return ^foo(42);
        }
      )cpp",

      R"cpp(// Field
        struct Foo { int [[x]]; };
        int main() {
          Foo bar;
          bar.^x;
        }
      )cpp",

      R"cpp(// Field, member initializer
        struct Foo {
          int [[x]];
          Foo() : ^x(0) {}
        };
      )cpp",

      R"cpp(// Field, GNU old-style field designator
        struct Foo { int [[x]]; };
        int main() {
          Foo bar = { ^x : 1 };
        }
      )cpp",

      R"cpp(// Field, field designator
        struct Foo { int [[x]]; };
        int main() {
          Foo bar = { .^x = 2 };
        }
      )cpp",

      R"cpp(// Method call
        struct Foo { int [[x]](); };
        int main() {
          Foo bar;
          bar.^x();
        }
      )cpp",

      R"cpp(// Typedef
        typedef int [[Foo]];
        int main() {
          ^Foo bar;
        }
      )cpp",

      /* FIXME: clangIndex doesn't handle template type parameters
      R"cpp(// Template type parameter
        template <[[typename T]]>
        void foo() { ^T t; }
      )cpp", */

      R"cpp(// Namespace
        namespace [[ns]] {
        struct Foo { static void bar(); }
        } // namespace ns
        int main() { ^ns::Foo::bar(); }
      )cpp",

      R"cpp(// Macro
        #define MACRO 0
        #define [[MACRO]] 1
        int main() { return ^MACRO; }
        #define MACRO 2
        #undef macro
      )cpp",

      R"cpp(// Macro
       class TTT { public: int a; };
       #define [[FF]](S) if (int b = S.a) {}
       void f() {
         TTT t;
         F^F(t);
       }
      )cpp",

      R"cpp(// Macro argument
       int [[i]];
       #define ADDRESSOF(X) &X;
       int *j = ADDRESSOF(^i);
      )cpp",

      R"cpp(// Symbol concatenated inside macro (not supported)
       int *pi;
       #define POINTER(X) p # X;
       int i = *POINTER(^i);
      )cpp",

      R"cpp(// Forward class declaration
        class Foo;
        class [[Foo]] {};
        F^oo* foo();
      )cpp",

      R"cpp(// Function declaration
        void foo();
        void g() { f^oo(); }
        void [[foo]]() {}
      )cpp",

      R"cpp(
        #define FF(name) class name##_Test {};
        [[FF]](my);
        void f() { my^_Test a; }
      )cpp",

      R"cpp(
         #define FF() class [[Test]] {};
         FF();
         void f() { T^est a; }
      )cpp",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto AST = TestTU::withCode(T.code()).build();
    std::vector<Matcher<Location>> ExpectedLocations;
    for (const auto &R : T.ranges())
      ExpectedLocations.push_back(RangeIs(R));
    EXPECT_THAT(findDefinitions(AST, T.point()),
                ElementsAreArray(ExpectedLocations))
        << Test;
  }
}

TEST(GoToDefinition, RelPathsInCompileCommand) {
  Annotations SourceAnnotations(R"cpp(
int [[foo]];
int baz = f^oo;
)cpp");

  IgnoreDiagnostics DiagConsumer;
  MockCompilationDatabase CDB(/*UseRelPaths=*/true);
  MockFSProvider FS;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  auto FooCpp = testPath("foo.cpp");
  FS.Files[FooCpp] = "";

  Server.addDocument(FooCpp, SourceAnnotations.code());
  runAddDocument(Server, FooCpp, SourceAnnotations.code());
  auto Locations =
      runFindDefinitions(Server, FooCpp, SourceAnnotations.point());
  EXPECT_TRUE(bool(Locations)) << "findDefinitions returned an error";

  EXPECT_THAT(*Locations, ElementsAre(Location{URIForFile{FooCpp},
                                               SourceAnnotations.range()}));
}

TEST(Hover, All) {
  struct OneTest {
    StringRef Input;
    StringRef ExpectedHover;
  };

  OneTest Tests[] = {
      {
          R"cpp(// Local variable
            int main() {
              int bonjour;
              ^bonjour = 2;
              int test1 = bonjour;
            }
          )cpp",
          "Declared in function main\n\nint bonjour",
      },
      {
          R"cpp(// Local variable in method
            struct s {
              void method() {
                int bonjour;
                ^bonjour = 2;
              }
            };
          )cpp",
          "Declared in function s::method\n\nint bonjour",
      },
      {
          R"cpp(// Struct
            namespace ns1 {
              struct MyClass {};
            } // namespace ns1
            int main() {
              ns1::My^Class* Params;
            }
          )cpp",
          "Declared in namespace ns1\n\nstruct MyClass {}",
      },
      {
          R"cpp(// Class
            namespace ns1 {
              class MyClass {};
            } // namespace ns1
            int main() {
              ns1::My^Class* Params;
            }
          )cpp",
          "Declared in namespace ns1\n\nclass MyClass {}",
      },
      {
          R"cpp(// Union
            namespace ns1 {
              union MyUnion { int x; int y; };
            } // namespace ns1
            int main() {
              ns1::My^Union Params;
            }
          )cpp",
          "Declared in namespace ns1\n\nunion MyUnion {}",
      },
      {
          R"cpp(// Function definition via pointer
            int foo(int) {}
            int main() {
              auto *X = &^foo;
            }
          )cpp",
          "Declared in global namespace\n\nint foo(int)",
      },
      {
          R"cpp(// Function declaration via call
            int foo(int);
            int main() {
              return ^foo(42);
            }
          )cpp",
          "Declared in global namespace\n\nint foo(int)",
      },
      {
          R"cpp(// Field
            struct Foo { int x; };
            int main() {
              Foo bar;
              bar.^x;
            }
          )cpp",
          "Declared in struct Foo\n\nint x",
      },
      {
          R"cpp(// Field with initialization
            struct Foo { int x = 5; };
            int main() {
              Foo bar;
              bar.^x;
            }
          )cpp",
          "Declared in struct Foo\n\nint x = 5",
      },
      {
          R"cpp(// Static field
            struct Foo { static int x; };
            int main() {
              Foo::^x;
            }
          )cpp",
          "Declared in struct Foo\n\nstatic int x",
      },
      {
          R"cpp(// Field, member initializer
            struct Foo {
              int x;
              Foo() : ^x(0) {}
            };
          )cpp",
          "Declared in struct Foo\n\nint x",
      },
      {
          R"cpp(// Field, GNU old-style field designator
            struct Foo { int x; };
            int main() {
              Foo bar = { ^x : 1 };
            }
          )cpp",
          "Declared in struct Foo\n\nint x",
      },
      {
          R"cpp(// Field, field designator
            struct Foo { int x; };
            int main() {
              Foo bar = { .^x = 2 };
            }
          )cpp",
          "Declared in struct Foo\n\nint x",
      },
      {
          R"cpp(// Method call
            struct Foo { int x(); };
            int main() {
              Foo bar;
              bar.^x();
            }
          )cpp",
          "Declared in struct Foo\n\nint x()",
      },
      {
          R"cpp(// Static method call
            struct Foo { static int x(); };
            int main() {
              Foo::^x();
            }
          )cpp",
          "Declared in struct Foo\n\nstatic int x()",
      },
      {
          R"cpp(// Typedef
            typedef int Foo;
            int main() {
              ^Foo bar;
            }
          )cpp",
          "Declared in global namespace\n\ntypedef int Foo",
      },
      {
          R"cpp(// Namespace
            namespace ns {
            struct Foo { static void bar(); }
            } // namespace ns
            int main() { ^ns::Foo::bar(); }
          )cpp",
          "Declared in global namespace\n\nnamespace ns {\n}",
      },
      {
          R"cpp(// Anonymous namespace
            namespace ns {
              namespace {
                int foo;
              } // anonymous namespace
            } // namespace ns
            int main() { ns::f^oo++; }
          )cpp",
          "Declared in namespace ns::(anonymous)\n\nint foo",
      },
      {
          R"cpp(// Macro
            #define MACRO 0
            #define MACRO 1
            int main() { return ^MACRO; }
            #define MACRO 2
            #undef macro
          )cpp",
          "#define MACRO",
      },
      {
          R"cpp(// Forward class declaration
            class Foo;
            class Foo {};
            F^oo* foo();
          )cpp",
          "Declared in global namespace\n\nclass Foo {}",
      },
      {
          R"cpp(// Function declaration
            void foo();
            void g() { f^oo(); }
            void foo() {}
          )cpp",
          "Declared in global namespace\n\nvoid foo()",
      },
      {
          R"cpp(// Enum declaration
            enum Hello {
              ONE, TWO, THREE,
            };
            void foo() {
              Hel^lo hello = ONE;
            }
          )cpp",
          "Declared in global namespace\n\nenum Hello {\n}",
      },
      {
          R"cpp(// Enumerator
            enum Hello {
              ONE, TWO, THREE,
            };
            void foo() {
              Hello hello = O^NE;
            }
          )cpp",
          "Declared in enum Hello\n\nONE",
      },
      {
          R"cpp(// Enumerator in anonymous enum
            enum {
              ONE, TWO, THREE,
            };
            void foo() {
              int hello = O^NE;
            }
          )cpp",
          "Declared in enum (anonymous)\n\nONE",
      },
      {
          R"cpp(// Global variable
            static int hey = 10;
            void foo() {
              he^y++;
            }
          )cpp",
          "Declared in global namespace\n\nstatic int hey = 10",
      },
      {
          R"cpp(// Global variable in namespace
            namespace ns1 {
              static int hey = 10;
            }
            void foo() {
              ns1::he^y++;
            }
          )cpp",
          "Declared in namespace ns1\n\nstatic int hey = 10",
      },
      {
          R"cpp(// Field in anonymous struct
            static struct {
              int hello;
            } s;
            void foo() {
              s.he^llo++;
            }
          )cpp",
          "Declared in struct (anonymous)\n\nint hello",
      },
      {
          R"cpp(// Templated function
            template <typename T>
            T foo() {
              return 17;
            }
            void g() { auto x = f^oo<int>(); }
          )cpp",
          "Declared in global namespace\n\ntemplate <typename T> T foo()",
      },
      {
          R"cpp(// Anonymous union
            struct outer {
              union {
                int abc, def;
              } v;
            };
            void g() { struct outer o; o.v.d^ef++; }
          )cpp",
          "Declared in union outer::(anonymous)\n\nint def",
      },
  };

  for (const OneTest &Test : Tests) {
    Annotations T(Test.Input);
    auto AST = TestTU::withCode(T.code()).build();
    Hover H = getHover(AST, T.point());

    EXPECT_EQ(H.contents.value, Test.ExpectedHover) << Test.Input;
  }
}

TEST(GoToInclude, All) {
  MockFSProvider FS;
  IgnoreDiagnostics DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, DiagConsumer, ClangdServer::optsForTest());

  auto FooCpp = testPath("foo.cpp");
  const char *SourceContents = R"cpp(
  #include ^"$2^foo.h$3^"
  #include "$4^invalid.h"
  int b = a;
  // test
  int foo;
  #in$5^clude "$6^foo.h"$7^
  )cpp";
  Annotations SourceAnnotations(SourceContents);
  FS.Files[FooCpp] = SourceAnnotations.code();
  auto FooH = testPath("foo.h");
  auto FooHUri = URIForFile{FooH};

  const char *HeaderContents = R"cpp([[]]#pragma once
                                     int a;
                                     )cpp";
  Annotations HeaderAnnotations(HeaderContents);
  FS.Files[FooH] = HeaderAnnotations.code();

  Server.addDocument(FooH, HeaderAnnotations.code());
  Server.addDocument(FooCpp, SourceAnnotations.code());

  // Test include in preamble.
  auto Locations =
      runFindDefinitions(Server, FooCpp, SourceAnnotations.point());
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations,
              ElementsAre(Location{FooHUri, HeaderAnnotations.range()}));

  // Test include in preamble, last char.
  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("2"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations,
              ElementsAre(Location{FooHUri, HeaderAnnotations.range()}));

  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("3"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations,
              ElementsAre(Location{FooHUri, HeaderAnnotations.range()}));

  // Test include outside of preamble.
  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("6"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations,
              ElementsAre(Location{FooHUri, HeaderAnnotations.range()}));

  // Test a few positions that do not result in Locations.
  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("4"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations, IsEmpty());

  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("5"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations, IsEmpty());

  Locations = runFindDefinitions(Server, FooCpp, SourceAnnotations.point("7"));
  ASSERT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations, IsEmpty());
}

} // namespace
} // namespace clangd
} // namespace clang

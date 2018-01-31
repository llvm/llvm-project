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
#include "Matchers.h"
#include "TestFS.h"
#include "XRefs.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Frontend/Utils.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
using namespace llvm;

void PrintTo(const DocumentHighlight &V, std::ostream *O) {
  llvm::raw_os_ostream OS(*O);
  OS << V.range;
  if (V.kind == DocumentHighlightKind::Read)
    OS << "(r)";
  if (V.kind == DocumentHighlightKind::Write)
    OS << "(w)";
}

namespace {
using testing::ElementsAre;
using testing::Field;
using testing::Matcher;
using testing::UnorderedElementsAreArray;

// FIXME: this is duplicated with FileIndexTests. Share it.
ParsedAST build(StringRef Code) {
  auto TestFile = getVirtualTestFilePath("Foo.cpp");
  auto CI =
      createInvocationFromCommandLine({"clang", "-xc++", TestFile.c_str()});
  auto Buf = MemoryBuffer::getMemBuffer(Code);
  auto AST = ParsedAST::Build(std::move(CI), nullptr, std::move(Buf),
                              std::make_shared<PCHContainerOperations>(),
                              vfs::getRealFileSystem());
  assert(AST.hasValue());
  return std::move(*AST);
}

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
    auto AST = build(T.code());
    EXPECT_THAT(findDocumentHighlights(AST, T.point()), HighlightsFrom(T))
        << Test;
  }
}

MATCHER_P(RangeIs, R, "") { return arg.range == R; }

TEST(GoToDefinition, All) {
  const char *Tests[] = {
      R"cpp(// Local variable
        int main() {
          [[int bonjour]];
          ^bonjour = 2;
          int test1 = bonjour;
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        [[struct MyClass {}]];
        } // namespace ns1
        int main() {
          ns1::My^Class* Params;
        }
      )cpp",

      R"cpp(// Function definition via pointer
        [[int foo(int) {}]]
        int main() {
          auto *X = &^foo;
        }
      )cpp",

      R"cpp(// Function declaration via call
        [[int foo(int)]];
        int main() {
          return ^foo(42);
        }
      )cpp",

      R"cpp(// Field
        struct Foo { [[int x]]; };
        int main() {
          Foo bar;
          bar.^x;
        }
      )cpp",

      R"cpp(// Field, member initializer
        struct Foo {
          [[int x]];
          Foo() : ^x(0) {}
        };
      )cpp",

      R"cpp(// Field, GNU old-style field designator
        struct Foo { [[int x]]; };
        int main() {
          Foo bar = { ^x : 1 };
        }
      )cpp",

      R"cpp(// Field, field designator
        struct Foo { [[int x]]; };
        int main() {
          Foo bar = { .^x = 2 };
        }
      )cpp",

      R"cpp(// Method call
        struct Foo { [[int x()]]; };
        int main() {
          Foo bar;
          bar.^x();
        }
      )cpp",

      R"cpp(// Typedef
        [[typedef int Foo]];
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
        [[namespace ns {
        struct Foo { static void bar(); }
        }]] // namespace ns
        int main() { ^ns::Foo::bar(); }
      )cpp",

      R"cpp(// Macro
        #define MACRO 0
        #define [[MACRO 1]]
        int main() { return ^MACRO; }
        #define MACRO 2
        #undef macro
      )cpp",

      R"cpp(// Forward class declaration
        class Foo;
        [[class Foo {}]];
        F^oo* foo();
      )cpp",

      R"cpp(// Function declaration
        void foo();
        void g() { f^oo(); }
        [[void foo() {}]]
      )cpp",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto AST = build(T.code());
    EXPECT_THAT(findDefinitions(AST, T.point()),
                ElementsAre(RangeIs(T.range())))
        << Test;
  }
}

} // namespace
} // namespace clangd
} // namespace clang

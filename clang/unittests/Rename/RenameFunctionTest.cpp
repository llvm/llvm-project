//===-- RenameFunctionTest.cpp - unit tests for renaming functions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangRenameTest.h"

namespace clang {
namespace clang_rename {
namespace test {
namespace {

class RenameFunctionTest : public ClangRenameTest {
public:
  RenameFunctionTest() {
    AppendToHeader(R"(
      struct A {
        static bool Foo();
        static bool Spam();
      };
      struct B {
        static void Same();
        static bool Foo();
        static int Eric(int x);
      };
      void Same(int x);
      int Eric(int x);
      namespace base {
        void Same();
        void ToNanoSeconds();
        void ToInt64NanoSeconds();
      })");
  }
};

TEST_F(RenameFunctionTest, RefactorsAFoo) {
  std::string Before = R"(
      void f() {
        A::Foo();
        ::A::Foo();
      })";
  std::string Expected = R"(
      void f() {
        A::Bar();
        ::A::Bar();
      })";

  std::string After = runClangRenameOnCode(Before, "A::Foo", "A::Bar");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RefactorsNonCallingAFoo) {
  std::string Before = R"(
      bool g(bool (*func)()) {
        return func();
      }
      void f() {
        auto *ref1 = A::Foo;
        auto *ref2 = ::A::Foo;
        g(A::Foo);
      })";
  std::string Expected = R"(
      bool g(bool (*func)()) {
        return func();
      }
      void f() {
        auto *ref1 = A::Bar;
        auto *ref2 = ::A::Bar;
        g(A::Bar);
      })";
  std::string After = runClangRenameOnCode(Before, "A::Foo", "A::Bar");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RefactorsEric) {
  std::string Before = R"(
      void f() {
        if (Eric(3)==4) ::Eric(2);
      })";
  std::string Expected = R"(
      void f() {
        if (Larry(3)==4) ::Larry(2);
      })";
  std::string After = runClangRenameOnCode(Before, "Eric", "Larry");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RefactorsNonCallingEric) {
  std::string Before = R"(
        int g(int (*func)(int)) {
          return func(1);
        }
        void f() {
          auto *ref = ::Eric;
          g(Eric);
        })";
  std::string Expected = R"(
        int g(int (*func)(int)) {
          return func(1);
        }
        void f() {
          auto *ref = ::Larry;
          g(Larry);
        })";
  std::string After = runClangRenameOnCode(Before, "Eric", "Larry");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, DoesNotRefactorBFoo) {
  std::string Before = R"(
      void f() {
        B::Foo();
      })";
  std::string After = runClangRenameOnCode(Before, "A::Foo", "A::Bar");
  CompareSnippets(Before, After);
}

TEST_F(RenameFunctionTest, DoesNotRefactorBEric) {
  std::string Before = R"(
      void f() {
        B::Eric(2);
      })";
  std::string After = runClangRenameOnCode(Before, "Eric", "Larry");
  CompareSnippets(Before, After);
}

TEST_F(RenameFunctionTest, DoesNotRefactorCEric) {
  std::string Before = R"(
      namespace C { int Eric(int x); }
      void f() {
        if (C::Eric(3)==4) ::C::Eric(2);
      })";
  std::string Expected = R"(
      namespace C { int Eric(int x); }
      void f() {
        if (C::Eric(3)==4) ::C::Eric(2);
      })";
  std::string After = runClangRenameOnCode(Before, "Eric", "Larry");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, DoesNotRefactorEricInNamespaceC) {
  std::string Before = R"(
       namespace C {
       int Eric(int x);
       void f() {
         if (Eric(3)==4) Eric(2);
       }
       }  // namespace C)";
  std::string After = runClangRenameOnCode(Before, "Eric", "Larry");
  CompareSnippets(Before, After);
}

TEST_F(RenameFunctionTest, NamespaceQualified) {
  std::string Before = R"(
      void f() {
        base::ToNanoSeconds();
        ::base::ToNanoSeconds();
      }
      void g() {
        using base::ToNanoSeconds;
        base::ToNanoSeconds();
        ::base::ToNanoSeconds();
        ToNanoSeconds();
      }
      namespace foo {
        namespace base {
          void ToNanoSeconds();
          void f() {
            base::ToNanoSeconds();
          }
        }
        void f() {
          ::base::ToNanoSeconds();
        }
      })";
  std::string Expected = R"(
      void f() {
        base::ToInt64NanoSeconds();
        ::base::ToInt64NanoSeconds();
      }
      void g() {
        using base::ToInt64NanoSeconds;
        base::ToInt64NanoSeconds();
        ::base::ToInt64NanoSeconds();
        base::ToInt64NanoSeconds();
      }
      namespace foo {
        namespace base {
          void ToNanoSeconds();
          void f() {
            base::ToNanoSeconds();
          }
        }
        void f() {
          ::base::ToInt64NanoSeconds();
        }
      })";
  std::string After = runClangRenameOnCode(Before, "base::ToNanoSeconds",
                                           "base::ToInt64NanoSeconds");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RenameFunctionDecls) {
  std::string Before = R"(
      namespace na {
        void X();
        void X() {}
      })";
  std::string Expected = R"(
      namespace na {
        void Y();
        void Y() {}
      })";
  std::string After = runClangRenameOnCode(Before, "na::X", "na::Y");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RenameOutOfLineFunctionDecls) {
  std::string Before = R"(
      namespace na {
        void X();
      }
      void na::X() {}
      )";
  std::string Expected = R"(
      namespace na {
        void Y();
      }
      void na::Y() {}
      )";
  std::string After = runClangRenameOnCode(Before, "na::X", "na::Y");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, NewNamespaceWithoutLeadingDotDot) {
  std::string Before = R"(
      namespace old_ns {
        void X();
        void X() {}
      }
      // Assume that the reference is in another file.
      void f() { old_ns::X(); }
      namespace old_ns { void g() { X(); } }
      namespace new_ns { void h() { ::old_ns::X(); } }
      )";
  std::string Expected = R"(
      namespace old_ns {
        void Y();
        void Y() {}
      }
      // Assume that the reference is in another file.
      void f() { new_ns::Y(); }
      namespace old_ns { void g() { new_ns::Y(); } }
      namespace new_ns { void h() { Y(); } }
      )";
  std::string After = runClangRenameOnCode(Before, "::old_ns::X", "new_ns::Y");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, NewNamespaceWithLeadingDotDot) {
  std::string Before = R"(
      namespace old_ns {
        void X();
        void X() {}
      }
      // Assume that the reference is in another file.
      void f() { old_ns::X(); }
      namespace old_ns { void g() { X(); } }
      namespace new_ns { void h() { ::old_ns::X(); } }
      )";
  std::string Expected = R"(
      namespace old_ns {
        void Y();
        void Y() {}
      }
      // Assume that the reference is in another file.
      void f() { ::new_ns::Y(); }
      namespace old_ns { void g() { ::new_ns::Y(); } }
      namespace new_ns { void h() { Y(); } }
      )";
  std::string After =
      runClangRenameOnCode(Before, "::old_ns::X", "::new_ns::Y");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, DontRenameSymbolsDefinedInAnonymousNamespace) {
  std::string Before = R"(
      namespace old_ns {
      class X {};
      namespace {
        void X();
        void X() {}
        void f() { X(); }
      }
      }
      )";
  std::string Expected = R"(
      namespace old_ns {
      class Y {};
      namespace {
        void X();
        void X() {}
        void f() { X(); }
      }
      }
      )";
  std::string After =
      runClangRenameOnCode(Before, "::old_ns::X", "::old_ns::Y");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, NewNestedNamespace) {
  std::string Before = R"(
      namespace old_ns {
        void X();
        void X() {}
      }
      // Assume that the reference is in another file.
      namespace old_ns {
      void f() { X(); }
      }
      )";
  std::string Expected = R"(
      namespace old_ns {
        void X();
        void X() {}
      }
      // Assume that the reference is in another file.
      namespace old_ns {
      void f() { older_ns::X(); }
      }
      )";
  std::string After =
      runClangRenameOnCode(Before, "::old_ns::X", "::old_ns::older_ns::X");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, MoveFromGlobalToNamespaceWithoutLeadingDotDot) {
  std::string Before = R"(
      void X();
      void X() {}

      // Assume that the reference is in another file.
      namespace some_ns {
      void f() { X(); }
      }
      )";
  std::string Expected = R"(
      void X();
      void X() {}

      // Assume that the reference is in another file.
      namespace some_ns {
      void f() { ns::X(); }
      }
      )";
  std::string After =
      runClangRenameOnCode(Before, "::X", "ns::X");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, MoveFromGlobalToNamespaceWithLeadingDotDot) {
  std::string Before = R"(
      void Y() {}

      // Assume that the reference is in another file.
      namespace some_ns {
      void f() { Y(); }
      }
      )";
  std::string Expected = R"(
      void Y() {}

      // Assume that the reference is in another file.
      namespace some_ns {
      void f() { ::ns::Y(); }
      }
      )";
  std::string After =
      runClangRenameOnCode(Before, "::Y", "::ns::Y");
  CompareSnippets(Expected, After);
}

// FIXME: the rename of overloaded operator is not fully supported yet.
TEST_F(RenameFunctionTest, DISABLED_DoNotRenameOverloadedOperatorCalls) {
  std::string Before = R"(
      namespace old_ns {
      class T { public: int x; };
      bool operator==(const T& lhs, const T& rhs) {
        return lhs.x == rhs.x;
      }
      }  // namespace old_ns

      // Assume that the reference is in another file.
      bool f() {
        auto eq = old_ns::operator==;
        old_ns::T t1, t2;
        old_ns::operator==(t1, t2);
        return t1 == t2;
      }
      )";
  std::string Expected = R"(
      namespace old_ns {
      class T { public: int x; };
      bool operator==(const T& lhs, const T& rhs) {
        return lhs.x == rhs.x;
      }
      }  // namespace old_ns

      // Assume that the reference is in another file.
      bool f() {
        auto eq = new_ns::operator==;
        old_ns::T t1, t2;
        new_ns::operator==(t1, t2);
        return t1 == t2;
      }
      )";
  std::string After =
      runClangRenameOnCode(Before, "old_ns::operator==", "new_ns::operator==");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, FunctionRefAsTemplate) {
  std::string Before = R"(
      void X();

      // Assume that the reference is in another file.
      namespace some_ns {
      template <void (*Func)(void)>
      class TIterator {};

      template <void (*Func)(void)>
      class T {
      public:
        typedef TIterator<Func> IterType;
        using TI = TIterator<Func>;
        void g() {
          Func();
          auto func = Func;
          TIterator<Func> iter;
        }
      };


      void f() { T<X> tx; tx.g(); }
      }  // namespace some_ns
      )";
  std::string Expected = R"(
      void X();

      // Assume that the reference is in another file.
      namespace some_ns {
      template <void (*Func)(void)>
      class TIterator {};

      template <void (*Func)(void)>
      class T {
      public:
        typedef TIterator<Func> IterType;
        using TI = TIterator<Func>;
        void g() {
          Func();
          auto func = Func;
          TIterator<Func> iter;
        }
      };


      void f() { T<ns::X> tx; tx.g(); }
      }  // namespace some_ns
      )";
  std::string After = runClangRenameOnCode(Before, "::X", "ns::X");
  CompareSnippets(Expected, After);
}

TEST_F(RenameFunctionTest, RenameFunctionInUsingDecl) {
  std::string Before = R"(
      using base::ToNanoSeconds;
      namespace old_ns {
      using base::ToNanoSeconds;
      void f() {
        using base::ToNanoSeconds;
      }
      }
      )";
  std::string Expected = R"(
      using base::ToInt64NanoSeconds;
      namespace old_ns {
      using base::ToInt64NanoSeconds;
      void f() {
        using base::ToInt64NanoSeconds;
      }
      }
      )";
  std::string After = runClangRenameOnCode(Before, "base::ToNanoSeconds",
                                           "base::ToInt64NanoSeconds");
  CompareSnippets(Expected, After);
}

// FIXME: Fix the complex the case where the symbol being renamed is located in
// `std::function<decltype<renamed_symbol>>`.
TEST_F(ClangRenameTest, DISABLED_ReferencesInLambdaFunctionParameters) {
  std::string Before = R"(
      template <class T>
      class function;
      template <class R, class... ArgTypes>
      class function<R(ArgTypes...)> {
      public:
        template <typename Functor>
        function(Functor f) {}

        function() {}

        R operator()(ArgTypes...) const {}
      };

      namespace ns {
      void Old() {}
      void f() {
        function<decltype(Old)> func;
      }
      }  // namespace ns)";
  std::string Expected = R"(
      template <class T>
      class function;
      template <class R, class... ArgTypes>
      class function<R(ArgTypes...)> {
      public:
        template <typename Functor>
        function(Functor f) {}

        function() {}

        R operator()(ArgTypes...) const {}
      };

      namespace ns {
      void New() {}
      void f() {
        function<decltype(::new_ns::New)> func;
      }
      }  // namespace ns)";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "::new_ns::New");
  CompareSnippets(Expected, After);
}

} // anonymous namespace
} // namespace test
} // namespace clang_rename
} // namesdpace clang

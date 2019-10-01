//===-- ClangMemberTests.cpp - unit tests for renaming class members ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangRenameTest.h"

namespace clang {
namespace clang_rename {
namespace test {
namespace {

class RenameMemberTest : public ClangRenameTest {
public:
  RenameMemberTest() {
    AppendToHeader(R"(
        struct NA {
          void Foo();
          void NotFoo();
          static void SFoo();
          static void SNotFoo();
          int Moo;
        };
        struct A {
          virtual void Foo();
          void NotFoo();
          static void SFoo();
          static void SNotFoo();
          int Moo;
          int NotMoo;
          static int SMoo;
        };
        struct B : public A {
          void Foo() override;
        };
        template <typename T> struct TA {
          T* Foo();
          T* NotFoo();
          static T* SFoo();
          static T* NotSFoo();
        };
        template <typename T> struct TB : public TA<T> {};
        namespace ns {
          template <typename T> struct TA {
            T* Foo();
            T* NotFoo();
            static T* SFoo();
            static T* NotSFoo();
            static int SMoo;
          };
          template <typename T> struct TB : public TA<T> {};
          struct A {
            void Foo();
            void NotFoo();
            static void SFoo();
            static void SNotFoo();
          };
          struct B : public A {};
          struct C {
            template <class T>
            void SFoo(const T& t) {}
            template <class T>
            void Foo() {}
          };
        })");
  }
};

INSTANTIATE_TEST_CASE_P(
    DISABLED_RenameTemplatedClassStaticVariableTest, RenameMemberTest,
    testing::ValuesIn(std::vector<Case>({
        // FIXME: support renaming static variables for template classes.
        {"void f() { ns::TA<int>::SMoo; }",
         "void f() { ns::TA<int>::SMeh; }", "ns::TA::SMoo", "ns::TA::SMeh"},
    })), );

INSTANTIATE_TEST_CASE_P(
    RenameMemberTest, RenameMemberTest,
    testing::ValuesIn(std::vector<Case>({
        // Normal methods and fields.
        {"void f() { A a; a.Foo(); }", "void f() { A a; a.Bar(); }", "A::Foo",
         "A::Bar"},
        {"void f() { ns::A a; a.Foo(); }", "void f() { ns::A a; a.Bar(); }",
         "ns::A::Foo", "ns::A::Bar"},
        {"void f() { A a; int x = a.Moo; }", "void f() { A a; int x = a.Meh; }",
         "A::Moo", "A::Meh"},
        {"void f() { B b; b.Foo(); }", "void f() { B b; b.Bar(); }", "B::Foo",
         "B::Bar"},
        {"void f() { ns::B b; b.Foo(); }", "void f() { ns::B b; b.Bar(); }",
         "ns::A::Foo", "ns::A::Bar"},
        {"void f() { B b; int x = b.Moo; }", "void f() { B b; int x = b.Meh; }",
         "A::Moo", "A::Meh"},

        // Static methods.
        {"void f() { A::SFoo(); }", "void f() { A::SBar(); }", "A::SFoo",
         "A::SBar"},
        {"void f() { ns::A::SFoo(); }", "void f() { ns::A::SBar(); }",
         "ns::A::SFoo", "ns::A::SBar"},
        {"void f() { TA<int>::SFoo(); }", "void f() { TA<int>::SBar(); }",
         "TA::SFoo", "TA::SBar"},
        {"void f() { ns::TA<int>::SFoo(); }",
         "void f() { ns::TA<int>::SBar(); }", "ns::TA::SFoo", "ns::TA::SBar"},

        // Static variables.
        {"void f() { A::SMoo; }",
         "void f() { A::SMeh; }", "A::SMoo", "A::SMeh"},

        // Templated methods.
        {"void f() { TA<int> a; a.Foo(); }", "void f() { TA<int> a; a.Bar(); }",
         "TA::Foo", "TA::Bar"},
        {"void f() { ns::TA<int> a; a.Foo(); }",
         "void f() { ns::TA<int> a; a.Bar(); }", "ns::TA::Foo", "ns::TA::Bar"},
        {"void f() { TB<int> b; b.Foo(); }", "void f() { TB<int> b; b.Bar(); }",
         "TA::Foo", "TA::Bar"},
        {"void f() { ns::TB<int> b; b.Foo(); }",
         "void f() { ns::TB<int> b; b.Bar(); }", "ns::TA::Foo", "ns::TA::Bar"},
        {"void f() { ns::C c; int x; c.SFoo(x); }",
         "void f() { ns::C c; int x; c.SBar(x); }", "ns::C::SFoo",
         "ns::C::SBar"},
        {"void f() { ns::C c; c.Foo<int>(); }",
         "void f() { ns::C c; c.Bar<int>(); }", "ns::C::Foo", "ns::C::Bar"},

        // Pointers to methods.
        {"void f() { auto p = &A::Foo; }", "void f() { auto p = &A::Bar; }",
         "A::Foo", "A::Bar"},
        {"void f() { auto p = &A::SFoo; }", "void f() { auto p = &A::SBar; }",
         "A::SFoo", "A::SBar"},
        {"void f() { auto p = &B::Foo; }", "void f() { auto p = &B::Bar; }",
         "B::Foo", "B::Bar"},
        {"void f() { auto p = &ns::A::Foo; }",
         "void f() { auto p = &ns::A::Bar; }", "ns::A::Foo", "ns::A::Bar"},
        {"void f() { auto p = &ns::A::SFoo; }",
         "void f() { auto p = &ns::A::SBar; }", "ns::A::SFoo", "ns::A::SBar"},
        {"void f() { auto p = &ns::C::SFoo<int>; }",
         "void f() { auto p = &ns::C::SBar<int>; }", "ns::C::SFoo",
         "ns::C::SBar"},

        // These methods are not declared or overridden in the subclass B, we
        // have to use the qualified name with parent class A to identify them.
        {"void f() { auto p = &ns::B::Foo; }",
         "void f() { auto p = &ns::B::Bar; }", "ns::A::Foo", "ns::B::Bar"},
        {"void f() { B::SFoo(); }", "void f() { B::SBar(); }", "A::SFoo",
         "B::SBar"},
        {"void f() { ns::B::SFoo(); }", "void f() { ns::B::SBar(); }",
         "ns::A::SFoo", "ns::B::SBar"},
        {"void f() { auto p = &B::SFoo; }", "void f() { auto p = &B::SBar; }",
         "A::SFoo", "B::SBar"},
        {"void f() { auto p = &ns::B::SFoo; }",
         "void f() { auto p = &ns::B::SBar; }", "ns::A::SFoo", "ns::B::SBar"},
        {"void f() { TB<int>::SFoo(); }", "void f() { TB<int>::SBar(); }",
         "TA::SFoo", "TB::SBar"},
        {"void f() { ns::TB<int>::SFoo(); }",
         "void f() { ns::TB<int>::SBar(); }", "ns::TA::SFoo", "ns::TB::SBar"},
    })), );

TEST_P(RenameMemberTest, RenameMembers) {
  auto Param = GetParam();
  assert(!Param.OldName.empty());
  assert(!Param.NewName.empty());
  std::string Actual =
      runClangRenameOnCode(Param.Before, Param.OldName, Param.NewName);
  CompareSnippets(Param.After, Actual);
}

TEST_F(RenameMemberTest, RenameMemberInsideClassMethods) {
  std::string Before = R"(
      struct X {
        int Moo;
        void Baz() { Moo = 1; }
      };)";
  std::string Expected = R"(
      struct X {
        int Meh;
        void Baz() { Meh = 1; }
      };)";
  std::string After = runClangRenameOnCode(Before, "X::Moo", "Y::Meh");
  CompareSnippets(Expected, After);
}

TEST_F(RenameMemberTest, RenameMethodInsideClassMethods) {
  std::string Before = R"(
      struct X {
        void Foo() {}
        void Baz() { Foo(); }
      };)";
  std::string Expected = R"(
      struct X {
        void Bar() {}
        void Baz() { Bar(); }
      };)";
  std::string After = runClangRenameOnCode(Before, "X::Foo", "X::Bar");
  CompareSnippets(Expected, After);
}

TEST_F(RenameMemberTest, RenameCtorInitializer) {
  std::string Before = R"(
      class X {
      public:
       X();
       A a;
       A a2;
       B b;
      };

      X::X():a(), b() {}
      )";
  std::string Expected = R"(
      class X {
      public:
       X();
       A bar;
       A a2;
       B b;
      };

      X::X():bar(), b() {}
      )";
  std::string After = runClangRenameOnCode(Before, "X::a", "X::bar");
  CompareSnippets(Expected, After);
}

} // anonymous namespace
} // namespace test
} // namespace clang_rename
} // namesdpace clang

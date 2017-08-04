//===-- RenameClassTest.cpp - unit tests for renaming classes -------------===//
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

class RenameClassTest : public ClangRenameTest {
public:
  RenameClassTest() {
    AppendToHeader(R"(
      namespace a {
        class Foo {
          public:
            struct Nested {
              enum NestedEnum {E1, E2};
            };
            void func() {}
          static int Constant;
        };
        class Goo {
          public:
            struct Nested {
              enum NestedEnum {E1, E2};
            };
        };
        int Foo::Constant = 1;
      } // namespace a
      namespace b {
      class Foo {};
      } // namespace b

      #define MACRO(x) x

      template<typename T> class ptr {};
    )");
  }
};

INSTANTIATE_TEST_CASE_P(
    RenameClassTests, RenameClassTest,
    testing::ValuesIn(std::vector<Case>({
        // basic classes
        {"a::Foo f;", "b::Bar f;", "", ""},
        {"void f(a::Foo f) {}", "void f(b::Bar f) {}", "", ""},
        {"void f(a::Foo *f) {}", "void f(b::Bar *f) {}", "", ""},
        {"a::Foo f() { return a::Foo(); }", "b::Bar f() { return b::Bar(); }",
         "", ""},
        {"namespace a {a::Foo f() { return Foo(); }}",
         "namespace a {b::Bar f() { return b::Bar(); }}", "", ""},
        {"void f(const a::Foo& a1) {}", "void f(const b::Bar& a1) {}", "", ""},
        {"void f(const a::Foo* a1) {}", "void f(const b::Bar* a1) {}", "", ""},
        {"namespace a { void f(Foo a1) {} }",
         "namespace a { void f(b::Bar a1) {} }", "", ""},
        {"void f(MACRO(a::Foo) a1) {}", "void f(MACRO(b::Bar) a1) {}", "", ""},
        {"void f(MACRO(a::Foo a1)) {}", "void f(MACRO(b::Bar a1)) {}", "", ""},
        {"a::Foo::Nested ns;", "b::Bar::Nested ns;", "", ""},
        {"auto t = a::Foo::Constant;", "auto t = b::Bar::Constant;", "", ""},
        {"a::Foo::Nested ns;", "a::Foo::Nested2 ns;", "a::Foo::Nested",
         "a::Foo::Nested2"},

        // use namespace and typedefs
        {"using a::Foo; Foo gA;", "using b::Bar; b::Bar gA;", "", ""},
        {"using a::Foo; void f(Foo gA) {}", "using b::Bar; void f(Bar gA) {}",
         "", ""},
        {"using a::Foo; namespace x { Foo gA; }",
         "using b::Bar; namespace x { Bar gA; }", "", ""},
        {"struct S { using T = a::Foo; T a_; };",
         "struct S { using T = b::Bar; T a_; };", "", ""},
        {"using T = a::Foo; T gA;", "using T = b::Bar; T gA;", "", ""},
        {"typedef a::Foo T; T gA;", "typedef b::Bar T; T gA;", "", ""},
        {"typedef MACRO(a::Foo) T; T gA;", "typedef MACRO(b::Bar) T; T gA;", "",
         ""},

        // struct members and other oddities
        {"struct S : public a::Foo {};", "struct S : public b::Bar {};", "",
         ""},
        {"struct F { void f(a::Foo a1) {} };",
         "struct F { void f(b::Bar a1) {} };", "", ""},
        {"struct F { a::Foo a_; };", "struct F { b::Bar a_; };", "", ""},
        {"struct F { ptr<a::Foo> a_; };", "struct F { ptr<b::Bar> a_; };", "",
         ""},

        {"void f() { a::Foo::Nested ne; }", "void f() { b::Bar::Nested ne; }",
         "", ""},
        {"void f() { a::Goo::Nested ne; }", "void f() { a::Goo::Nested ne; }",
         "", ""},
        {"void f() { a::Foo::Nested::NestedEnum e; }",
         "void f() { b::Bar::Nested::NestedEnum e; }", "", ""},
        {"void f() { auto e = a::Foo::Nested::NestedEnum::E1; }",
         "void f() { auto e = b::Bar::Nested::NestedEnum::E1; }", "", ""},
        {"void f() { auto e = a::Foo::Nested::E1; }",
         "void f() { auto e = b::Bar::Nested::E1; }", "", ""},

        // templates
        {"template <typename T> struct Foo { T t; };\n"
         "void f() { Foo<a::Foo> foo; }",
         "template <typename T> struct Foo { T t; };\n"
         "void f() { Foo<b::Bar> foo; }",
         "", ""},
        {"template <typename T> struct Foo { a::Foo a; };",
         "template <typename T> struct Foo { b::Bar a; };", "", ""},
        {"template <typename T> void f(T t) {}\n"
         "void g() { f<a::Foo>(a::Foo()); }",
         "template <typename T> void f(T t) {}\n"
         "void g() { f<b::Bar>(b::Bar()); }",
         "", ""},
        {"template <typename T> int f() { return 1; }\n"
         "template <> int f<a::Foo>() { return 2; }\n"
         "int g() { return f<a::Foo>(); }",
         "template <typename T> int f() { return 1; }\n"
         "template <> int f<b::Bar>() { return 2; }\n"
         "int g() { return f<b::Bar>(); }",
         "", ""},
        {"struct Foo { template <typename T> T foo(); };\n"
         "void g() { Foo f;  auto a = f.template foo<a::Foo>(); }",
         "struct Foo { template <typename T> T foo(); };\n"
         "void g() { Foo f;  auto a = f.template foo<b::Bar>(); }",
         "", ""},

        // The following two templates are distilled from regressions found in
        // unique_ptr<> and type_traits.h
        {"template <typename T> struct outer {\n"
         "     typedef T type;\n"
         "     type Baz();\n"
         "    };\n"
         "    outer<a::Foo> g_A;",
         "template <typename T> struct outer {\n"
         "      typedef T type;\n"
         "      type Baz();\n"
         "    };\n"
         "    outer<b::Bar> g_A;",
         "", ""},
        {"template <typename T> struct nested { typedef T type; };\n"
         "template <typename T> struct outer { typename nested<T>::type Foo(); "
         "};\n"
         "outer<a::Foo> g_A;",
         "template <typename T> struct nested { typedef T type; };\n"
         "template <typename T> struct outer { typename nested<T>::type Foo(); "
         "};\n"
         "outer<b::Bar> g_A;",
         "", ""},

        // macros
        {"#define FOO(T, t) T t\n"
         "void f() { FOO(a::Foo, a1); FOO(a::Foo, a2); }",
         "#define FOO(T, t) T t\n"
         "void f() { FOO(b::Bar, a1); FOO(b::Bar, a2); }",
         "", ""},
        {"#define FOO(n) a::Foo n\n"
         " void f() { FOO(a1); FOO(a2); }",
         "#define FOO(n) b::Bar n\n"
         " void f() { FOO(a1); FOO(a2); }",
         "", ""},

        // Pointer to member functions
        {"auto gA = &a::Foo::func;", "auto gA = &b::Bar::func;", "", ""},
        {"using a::Foo; auto gA = &Foo::func;",
         "using b::Bar; auto gA = &b::Bar::func;", "", ""},
        {"using a::Foo; namespace x { auto gA = &Foo::func; }",
         "using b::Bar; namespace x { auto gA = &Bar::func; }", "", ""},
        {"typedef a::Foo T; auto gA = &T::func;",
         "typedef b::Bar T; auto gA = &T::func;", "", ""},
        {"auto gA = &MACRO(a::Foo)::func;", "auto gA = &MACRO(b::Bar)::func;",
         "", ""},

        // Short match inside a namespace
        {"namespace a { void f(Foo a1) {} }",
         "namespace a { void f(b::Bar a1) {} }", "", ""},

        // Correct match.
        {"using a::Foo; struct F { ptr<Foo> a_; };",
         "using b::Bar; struct F { ptr<Bar> a_; };", "", ""},

        // avoid false positives
        {"void f(b::Foo a) {}", "void f(b::Foo a) {}", "", ""},
        {"namespace b { void f(Foo a) {} }", "namespace b { void f(Foo a) {} }",
         "", ""},

        // friends, everyone needs friends.
        {"class Foo { int i; friend class a::Foo; };",
         "class Foo { int i; friend class b::Bar; };", "", ""},
    })), );

TEST_P(RenameClassTest, RenameClasses) {
  auto Param = GetParam();
  std::string OldName = Param.OldName.empty() ? "a::Foo" : Param.OldName;
  std::string NewName = Param.NewName.empty() ? "b::Bar" : Param.NewName;
  std::string Actual = runClangRenameOnCode(Param.Before, OldName, NewName);
  CompareSnippets(Param.After, Actual);
}

class NamespaceDetectionTest : public ClangRenameTest {
protected:
  NamespaceDetectionTest() {
    AppendToHeader(R"(
         class Old {};
         namespace o1 {
         class Old {};
         namespace o2 {
         class Old {};
         namespace o3 {
         class Old {};
         }  // namespace o3
         }  // namespace o2
         }  // namespace o1
     )");
  }
};

INSTANTIATE_TEST_CASE_P(
    RenameClassTest, NamespaceDetectionTest,
    ::testing::ValuesIn(std::vector<Case>({
        // Test old and new namespace overlap.
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { New moo; } } }",
         "o1::o2::o3::Old", "o1::o2::o3::New"},
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { n3::New moo; } } }",
         "o1::o2::o3::Old", "o1::o2::n3::New"},
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { n2::n3::New moo; } } }",
         "o1::o2::o3::Old", "o1::n2::n3::New"},
        {"namespace o1 { namespace o2 { Old moo; } }",
         "namespace o1 { namespace o2 { New moo; } }", "::o1::o2::Old",
         "::o1::o2::New"},
        {"namespace o1 { namespace o2 { Old moo; } }",
         "namespace o1 { namespace o2 { n2::New moo; } }", "::o1::o2::Old",
         "::o1::n2::New"},
        {"namespace o1 { namespace o2 { Old moo; } }",
         "namespace o1 { namespace o2 { ::n1::n2::New moo; } }",
         "::o1::o2::Old", "::n1::n2::New"},
        {"namespace o1 { namespace o2 { Old moo; } }",
         "namespace o1 { namespace o2 { n1::n2::New moo; } }", "::o1::o2::Old",
         "n1::n2::New"},

        // Test old and new namespace with differing depths.
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { New moo; } } }",
         "o1::o2::o3::Old", "::o1::New"},
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { New moo; } } }",
         "o1::o2::o3::Old", "::o1::o2::New"},
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { New moo; } } }",
         "o1::o2::o3::Old", "o1::New"},
        {"namespace o1 { namespace o2 { namespace o3 { Old moo; } } }",
         "namespace o1 { namespace o2 { namespace o3 { New moo; } } }",
         "o1::o2::o3::Old", "o1::o2::New"},
        {"Old moo;", "o1::New moo;", "::Old", "o1::New"},
        {"Old moo;", "o1::New moo;", "Old", "o1::New"},
        {"namespace o1 { ::Old moo; }", "namespace o1 { New moo; }", "Old",
         "o1::New"},
        {"namespace o1 { namespace o2 {  Old moo; } }",
         "namespace o1 { namespace o2 {  ::New moo; } }", "::o1::o2::Old",
         "::New"},
        {"namespace o1 { namespace o2 {  Old moo; } }",
         "namespace o1 { namespace o2 {  New moo; } }", "::o1::o2::Old", "New"},

        // Test moving into the new namespace at different levels.
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { New moo; } }", "::o1::o2::Old",
         "::n1::n2::New"},
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { New moo; } }", "::o1::o2::Old",
         "n1::n2::New"},
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { o2::New moo; } }", "::o1::o2::Old",
         "::n1::o2::New"},
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { o2::New moo; } }", "::o1::o2::Old",
         "n1::o2::New"},
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { ::o1::o2::New moo; } }",
         "::o1::o2::Old", "::o1::o2::New"},
        {"namespace n1 { namespace n2 { o1::o2::Old moo; } }",
         "namespace n1 { namespace n2 { o1::o2::New moo; } }", "::o1::o2::Old",
         "o1::o2::New"},

        // Test friends declarations.
        {"class Foo { friend class o1::Old; };",
         "class Foo { friend class o1::New; };", "o1::Old", "o1::New"},
        {"class Foo { int i; friend class o1::Old; };",
         "class Foo { int i; friend class ::o1::New; };", "::o1::Old",
         "::o1::New"},
        {"namespace o1 { class Foo { int i; friend class Old; }; }",
         "namespace o1 { class Foo { int i; friend class New; }; }", "o1::Old",
         "o1::New"},
        {"namespace o1 { class Foo { int i; friend class Old; }; }",
         "namespace o1 { class Foo { int i; friend class New; }; }",
         "::o1::Old", "::o1::New"},
    })), );

TEST_P(NamespaceDetectionTest, RenameClasses) {
  auto Param = GetParam();
  std::string Actual =
      runClangRenameOnCode(Param.Before, Param.OldName, Param.NewName);
  CompareSnippets(Param.After, Actual);
}

class TemplatedClassRenameTest : public ClangRenameTest {
protected:
  TemplatedClassRenameTest() {
    AppendToHeader(R"(
           template <typename T> struct Old {
             T t_;
             T f() { return T(); };
             static T s(T t) { return t; }
           };
           namespace ns {
           template <typename T> struct Old {
             T t_;
             T f() { return T(); };
             static T s(T t) { return t; }
           };
           }  // namespace ns

           namespace o1 {
           namespace o2 {
           namespace o3 {
           template <typename T> struct Old {
             T t_;
             T f() { return T(); };
             static T s(T t) { return t; }
           };
           }  // namespace o3
           }  // namespace o2
           }  // namespace o1
       )");
  }
};

INSTANTIATE_TEST_CASE_P(
    RenameClassTests, TemplatedClassRenameTest,
    ::testing::ValuesIn(std::vector<Case>({
        {"Old<int> gI; Old<bool> gB;", "New<int> gI; New<bool> gB;", "Old",
         "New"},
        {"ns::Old<int> gI; ns::Old<bool> gB;",
         "ns::New<int> gI; ns::New<bool> gB;", "ns::Old", "ns::New"},
        {"auto gI = &Old<int>::f; auto gB = &Old<bool>::f;",
         "auto gI = &New<int>::f; auto gB = &New<bool>::f;", "Old", "New"},
        {"auto gI = &ns::Old<int>::f;", "auto gI = &ns::New<int>::f;",
         "ns::Old", "ns::New"},

        {"int gI = Old<int>::s(0); bool gB = Old<bool>::s(false);",
         "int gI = New<int>::s(0); bool gB = New<bool>::s(false);", "Old",
         "New"},
        {"int gI = ns::Old<int>::s(0); bool gB = ns::Old<bool>::s(false);",
         "int gI = ns::New<int>::s(0); bool gB = ns::New<bool>::s(false);",
         "ns::Old", "ns::New"},

        {"struct S { Old<int*> o_; };", "struct S { New<int*> o_; };", "Old",
         "New"},
        {"struct S { ns::Old<int*> o_; };", "struct S { ns::New<int*> o_; };",
         "ns::Old", "ns::New"},

        {"auto a = reinterpret_cast<Old<int>*>(new Old<int>);",
         "auto a = reinterpret_cast<New<int>*>(new New<int>);", "Old", "New"},
        {"auto a = reinterpret_cast<ns::Old<int>*>(new ns::Old<int>);",
         "auto a = reinterpret_cast<ns::New<int>*>(new ns::New<int>);",
         "ns::Old", "ns::New"},
        {"auto a = reinterpret_cast<const Old<int>*>(new Old<int>);",
         "auto a = reinterpret_cast<const New<int>*>(new New<int>);", "Old",
         "New"},
        {"auto a = reinterpret_cast<const ns::Old<int>*>(new ns::Old<int>);",
         "auto a = reinterpret_cast<const ns::New<int>*>(new ns::New<int>);",
         "ns::Old", "ns::New"},

        {"Old<bool>& foo();", "New<bool>& foo();", "Old", "New"},
        {"ns::Old<bool>& foo();", "ns::New<bool>& foo();", "ns::Old",
         "ns::New"},
        {"o1::o2::o3::Old<bool>& foo();", "o1::o2::o3::New<bool>& foo();",
         "o1::o2::o3::Old", "o1::o2::o3::New"},
        {"namespace ns { Old<bool>& foo(); }",
         "namespace ns { New<bool>& foo(); }", "ns::Old", "ns::New"},
        {"const Old<bool>& foo();", "const New<bool>& foo();", "Old", "New"},
        {"const ns::Old<bool>& foo();", "const ns::New<bool>& foo();",
         "ns::Old", "ns::New"},

        // FIXME: figure out why this only works when Moo gets
        // specialized at some point.
        {"template <typename T> struct Moo { Old<T> o_; }; Moo<int> m;",
         "template <typename T> struct Moo { New<T> o_; }; Moo<int> m;", "Old",
         "New"},
        {"template <typename T> struct Moo { ns::Old<T> o_; }; Moo<int> m;",
         "template <typename T> struct Moo { ns::New<T> o_; }; Moo<int> m;",
         "ns::Old", "ns::New"},
    })), );

TEST_P(TemplatedClassRenameTest, RenameTemplateClasses) {
  auto Param = GetParam();
  std::string Actual =
      runClangRenameOnCode(Param.Before, Param.OldName, Param.NewName);
  CompareSnippets(Param.After, Actual);
}

TEST_F(ClangRenameTest, RenameClassWithOutOfLineMembers) {
  std::string Before = R"(
      class Old {
       public:
        Old();
        ~Old();

        Old* next();

       private:
        Old* next_;
      };

      Old::Old() {}
      Old::~Old() {}
      Old* Old::next() { return next_; }
    )";
  std::string Expected = R"(
      class New {
       public:
        New();
        ~New();

        New* next();

       private:
        New* next_;
      };

      New::New() {}
      New::~New() {}
      New* New::next() { return next_; }
    )";
  std::string After = runClangRenameOnCode(Before, "Old", "New");
  CompareSnippets(Expected, After);
}

TEST_F(ClangRenameTest, RenameClassWithInlineMembers) {
  std::string Before = R"(
      class Old {
       public:
        Old() {}
        ~Old() {}

        Old* next() { return next_; }

       private:
        Old* next_;
      };
    )";
  std::string Expected = R"(
      class New {
       public:
        New() {}
        ~New() {}

        New* next() { return next_; }

       private:
        New* next_;
      };
    )";
  std::string After = runClangRenameOnCode(Before, "Old", "New");
  CompareSnippets(Expected, After);
}

// FIXME: no prefix qualifiers being added to the class definition and
// constructor.
TEST_F(ClangRenameTest, RenameClassWithNamespaceWithInlineMembers) {
  std::string Before = R"(
      namespace ns {
      class Old {
       public:
        Old() {}
        ~Old() {}

        Old* next() { return next_; }

       private:
        Old* next_;
      };
      }  // namespace ns
    )";
  std::string Expected = R"(
      namespace ns {
      class ns::New {
       public:
        ns::New() {}
        ~New() {}

        New* next() { return next_; }

       private:
        New* next_;
      };
      }  // namespace ns
    )";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "ns::New");
  CompareSnippets(Expected, After);
}

// FIXME: no prefix qualifiers being added to the class definition and
// constructor.
TEST_F(ClangRenameTest, RenameClassWithNamespaceWithOutOfInlineMembers) {
  std::string Before = R"(
      namespace ns {
      class Old {
       public:
        Old();
        ~Old();

        Old* next();

       private:
        Old* next_;
      };

      Old::Old() {}
      Old::~Old() {}
      Old* Old::next() { return next_; }
      }  // namespace ns
    )";
  std::string Expected = R"(
      namespace ns {
      class ns::New {
       public:
        ns::New();
        ~New();

        New* next();

       private:
        New* next_;
      };

      New::ns::New() {}
      New::~New() {}
      New* New::next() { return next_; }
      }  // namespace ns
    )";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "ns::New");
  CompareSnippets(Expected, After);
}

// FIXME: no prefix qualifiers being added to the definition.
TEST_F(ClangRenameTest, RenameClassInInheritedConstructor) {
  // `using Base::Base;` will generate an implicit constructor containing usage
  // of `::ns::Old` which should not be matched.
  std::string Before = R"(
      namespace ns {
      class Old {
        int x;
      };
      class Base {
       protected:
        Old *moo_;
       public:
        Base(Old *moo) : moo_(moo) {}
      };
      class Derived : public Base {
       public:
         using Base::Base;
      };
      }  // namespace ns
      int main() {
        ::ns::Old foo;
        ::ns::Derived d(&foo);
        return 0;
      })";
  std::string Expected = R"(
      namespace ns {
      class ns::New {
        int x;
      };
      class Base {
       protected:
        New *moo_;
       public:
        Base(New *moo) : moo_(moo) {}
      };
      class Derived : public Base {
       public:
         using Base::Base;
      };
      }  // namespace ns
      int main() {
        ::ns::New foo;
        ::ns::Derived d(&foo);
        return 0;
      })";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "ns::New");
  CompareSnippets(Expected, After);
}

TEST_F(ClangRenameTest, DontRenameReferencesInImplicitFunction) {
  std::string Before = R"(
      namespace ns {
      class Old {
      };
      } // namespace ns
      struct S {
        int y;
        ns::Old old;
      };
      void f() {
        S s1, s2, s3;
        // This causes an implicit assignment operator to be created.
        s1 = s2 = s3;
      }
      )";
  std::string Expected = R"(
      namespace ns {
      class ::new_ns::New {
      };
      } // namespace ns
      struct S {
        int y;
        ::new_ns::New old;
      };
      void f() {
        S s1, s2, s3;
        // This causes an implicit assignment operator to be created.
        s1 = s2 = s3;
      }
      )";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "::new_ns::New");
  CompareSnippets(Expected, After);
}

// FIXME: no prefix qualifiers being adding to the definition.
TEST_F(ClangRenameTest, ReferencesInLambdaFunctionParameters) {
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
      class Old {};
      void f() {
        function<void(Old)> func;
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
      class ::new_ns::New {};
      void f() {
        function<void(::new_ns::New)> func;
      }
      }  // namespace ns)";
  std::string After = runClangRenameOnCode(Before, "ns::Old", "::new_ns::New");
  CompareSnippets(Expected, After);
}

} // anonymous namespace
} // namespace test
} // namespace clang_rename
} // namesdpace clang

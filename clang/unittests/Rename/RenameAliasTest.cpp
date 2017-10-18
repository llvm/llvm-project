//===-- RenameAliasTest.cpp - unit tests for renaming alias ---------------===//
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

class RenameAliasTest : public ClangRenameTest {
public:
  RenameAliasTest() {
    AppendToHeader(R"(
        #define MACRO(x) x
        namespace some_ns {
        class A {
         public:
          void foo() {}
          struct Nested {
           enum NestedEnum {
             E1, E2,
           };
          };
        };
        } // namespace some_ns
        namespace a {
        typedef some_ns::A TA;
        using UA = some_ns::A;
        } // namespace a
        namespace b {
        typedef some_ns::A TA;
        using UA = some_ns::A;
        }
        template <typename T> class ptr {};
        template <typename T>

        using TPtr = ptr<int>;
    )");
  }
};

INSTANTIATE_TEST_CASE_P(
    RenameAliasTests, RenameAliasTest,
    testing::ValuesIn(std::vector<Case>({
        // basic functions
        {"void f(a::TA a1) {}", "void f(b::TB a1) {}", "a::TA", "b::TB"},
        {"void f(a::UA a1) {}", "void f(b::UB a1) {}", "a::UA", "b::UB"},
        {"void f(a::TA* a1) {}", "void f(b::TB* a1) {}", "a::TA", "b::TB"},
        {"void f(a::TA** a1) {}", "void f(b::TB** a1) {}", "a::TA", "b::TB"},
        {"a::TA f() { return a::TA(); }", "b::TB f() { return b::TB(); }",
         "a::TA", "b::TB"},
        {"a::TA f() { return a::UA(); }", "b::TB f() { return a::UA(); }",
         "a::TA", "b::TB"},
        {"a::TA f() { return a::UA(); }", "a::TA f() { return b::UB(); }",
         "a::UA", "b::UB"},
        {"void f() { a::TA a; }", "void f() { b::TB a; }", "a::TA", "b::TB"},
        {"void f(const a::TA& a1) {}", "void f(const b::TB& a1) {}", "a::TA",
         "b::TB"},
        {"void f(const a::UA& a1) {}", "void f(const b::UB& a1) {}", "a::UA",
         "b::UB"},
        {"void f(const a::TA* a1) {}", "void f(const b::TB* a1) {}", "a::TA",
         "b::TB"},
        {"namespace a { void f(TA a1) {} }",
         "namespace a { void f(b::TB a1) {} }", "a::TA", "b::TB"},
        {"void f(MACRO(a::TA) a1) {}", "void f(MACRO(b::TB) a1) {}", "a::TA",
         "b::TB"},
        {"void f(MACRO(a::TA a1)) {}", "void f(MACRO(b::TB a1)) {}", "a::TA",
         "b::TB"},

        // shorten/add namespace.
        {"namespace b { void f(a::UA a1) {} }",
         "namespace b {void f(UB a1) {} }", "a::UA", "b::UB"},
        {"namespace a { void f(UA a1) {} }",
         "namespace a {void f(b::UB a1) {} }", "a::UA", "b::UB"},

        // use namespace and typedefs
        {"struct S { using T = a::TA; T a_; };",
         "struct S { using T = b::TB; T a_; };", "a::TA", "b::TB"},
        {"using T = a::TA; T gA;", "using T = b::TB; T gA;", "a::TA", "b::TB"},
        {"using T = a::UA; T gA;", "using T = b::UB; T gA;", "a::UA", "b::UB"},
        {"typedef a::TA T; T gA;", "typedef b::TB T; T gA;", "a::TA", "b::TB"},
        {"typedef a::UA T; T gA;", "typedef b::UB T; T gA;", "a::UA", "b::UB"},
        {"typedef MACRO(a::TA) T; T gA;", "typedef MACRO(b::TB) T; T gA;",
         "a::TA", "b::TB"},

        // types in using shadows.
        {"using a::TA; TA gA;", "using b::TB; b::TB gA;", "a::TA", "b::TB"},
        {"using a::UA; UA gA;", "using b::UB; b::UB gA;", "a::UA", "b::UB"},

        // struct members and other oddities
        {"struct S : public a::TA {};", "struct S : public b::TB {};", "a::TA",
         "b::TB"},
        {"struct S : public a::UA {};", "struct S : public b::UB {};", "a::UA",
         "b::UB"},
        {"struct F { void f(a::TA a1) {} };",
         "struct F { void f(b::TB a1) {} };", "a::TA", "b::TB"},
        {"struct F { a::TA a_; };", "struct F { b::TB a_; };", "a::TA",
         "b::TB"},
        {"struct F { ptr<a::TA> a_; };", "struct F { ptr<b::TB> a_; };",
         "a::TA", "b::TB"},
        {"struct F { ptr<a::UA> a_; };", "struct F { ptr<b::UB> a_; };",
         "a::UA", "b::UB"},

        // types in nested name specifiers
        {"void f() { a::TA::Nested ne; }", "void f() { b::TB::Nested ne; }",
         "a::TA", "b::TB"},
        {"void f() { a::UA::Nested ne; }", "void f() { b::UB::Nested ne; }",
         "a::UA", "b::UB"},
        {"void f() { a::TA::Nested::NestedEnum e; }",
         "void f() { b::TB::Nested::NestedEnum e; }", "a::TA", "b::TB"},
        {"void f() { auto e = a::TA::Nested::NestedEnum::E1; }",
         "void f() { auto e = b::TB::Nested::NestedEnum::E1; }", "a::TA",
         "b::TB"},
        {"void f() { auto e = a::TA::Nested::E1; }",
         "void f() { auto e = b::TB::Nested::E1; }", "a::TA", "b::TB"},

        // templates
        {"template <typename T> struct Foo { T t; }; void f() { Foo<a::TA> "
         "foo; }",
         "template <typename T> struct Foo { T t; }; void f() { Foo<b::TB> "
         "foo; }",
         "a::TA", "b::TB"},
        {"template <typename T> struct Foo { a::TA a; };",
         "template <typename T> struct Foo { b::TB a; };", "a::TA", "b::TB"},
        {"template <typename T> void f(T t) {} void g() { f<a::TA>(a::TA()); }",
         "template <typename T> void f(T t) {} void g() { f<b::TB>(b::TB()); }",
         "a::TA", "b::TB"},
        {"template <typename T> void f(T t) {} void g() { f<a::UA>(a::UA()); }",
         "template <typename T> void f(T t) {} void g() { f<b::UB>(b::UB()); }",
         "a::UA", "b::UB"},
        {"template <typename T> int f() { return 1; } template <> int "
         "f<a::TA>() { return 2; } int g() { return f<a::TA>(); }",
         "template <typename T> int f() { return 1; } template <> int "
         "f<b::TB>() { return 2; } int g() { return f<b::TB>(); }",
         "a::TA", "b::TB"},
        {"struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "auto a = f.template foo<a::TA>(); }",
         "struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "auto a = f.template foo<b::TB>(); }",
         "a::TA", "b::TB"},
        {"struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "auto a = f.template foo<a::UA>(); }",
         "struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "auto a = f.template foo<b::UB>(); }",
         "a::UA", "b::UB"},

        // The following two templates are distilled from regressions found in
        // unique_ptr<> and type_traits.h
        {"template <typename T> struct outer { typedef T type; type Baz(); }; "
         "outer<a::TA> g_A;",
         "template <typename T> struct outer { typedef T type; type Baz(); }; "
         "outer<b::TB> g_A;",
         "a::TA", "b::TB"},
        {"template <typename T> struct nested { typedef T type; }; template "
         "<typename T> struct outer { typename nested<T>::type Foo(); }; "
         "outer<a::TA> g_A;",
         "template <typename T> struct nested { typedef T type; }; template "
         "<typename T> struct outer { typename nested<T>::type Foo(); }; "
         "outer<b::TB> g_A;",
         "a::TA", "b::TB"},

        // macros
        {"#define FOO(T, t) T t\nvoid f() { FOO(a::TA, a1); FOO(a::TA, a2); }",
         "#define FOO(T, t) T t\nvoid f() { FOO(b::TB, a1); FOO(b::TB, a2); }",
         "a::TA", "b::TB"},
        {"#define FOO(n) a::TA n\nvoid f() { FOO(a1); FOO(a2); }",
         "#define FOO(n) b::TB n\nvoid f() { FOO(a1); FOO(a2); }", "a::TA",
         "b::TB"},
        {"#define FOO(n) a::UA n\nvoid f() { FOO(a1); FOO(a2); }",
         "#define FOO(n) b::UB n\nvoid f() { FOO(a1); FOO(a2); }", "a::UA",
         "b::UB"},

        // Pointer to member functions
        {"auto gA = &a::TA::foo;", "auto gA = &b::TB::foo;", "a::TA", "b::TB"},
        {"using a::TA; auto gA = &TA::foo;",
         "using b::TB; auto gA = &b::TB::foo;", "a::TA", "b::TB"},
        {"typedef a::TA T; auto gA = &T::foo;",
         "typedef b::TB T; auto gA = &T::foo;", "a::TA", "b::TB"},
        {"auto gA = &MACRO(a::TA)::foo;", "auto gA = &MACRO(b::TB)::foo;",
         "a::TA", "b::TB"},

        // templated using alias.
        {"void f(TPtr<int> p) {}", "void f(NewTPtr<int> p) {}", "TPtr",
         "NewTPtr"},
        {"void f(::TPtr<int> p) {}", "void f(::NewTPtr<int> p) {}", "TPtr",
         "NewTPtr"},
    })), );

TEST_P(RenameAliasTest, RenameAlias) {
  auto Param = GetParam();
  assert(!Param.OldName.empty());
  assert(!Param.NewName.empty());
  std::string Actual =
      runClangRenameOnCode(Param.Before, Param.OldName, Param.NewName);
  CompareSnippets(Param.After, Actual);
}

TEST_F(RenameAliasTest, RenameTypedefDefinitions) {
  std::string Before = R"(
    class X {};
    typedef X TOld;
    )";
  std::string Expected = R"(
    class X {};
    typedef X TNew;
    )";
  std::string After = runClangRenameOnCode(Before, "TOld", "TNew");
  CompareSnippets(Expected, After);
}

TEST_F(RenameAliasTest, RenameUsingAliasDefinitions) {
  std::string Before = R"(
    class X {};
    using UOld = X;
    )";
  std::string Expected = R"(
    class X {};
    using UNew = X;
    )";
  std::string After = runClangRenameOnCode(Before, "UOld", "UNew");
  CompareSnippets(Expected, After);
}

TEST_F(RenameAliasTest, RenameTemplatedAliasDefinitions) {
  std::string Before = R"(
    template <typename T>
    class X { T t; };

    template <typename T>
    using Old = X<T>;
    )";
  std::string Expected = R"(
    template <typename T>
    class X { T t; };

    template <typename T>
    using New = X<T>;
    )";
  std::string After = runClangRenameOnCode(Before, "Old", "New");
  CompareSnippets(Expected, After);
}

TEST_F(RenameAliasTest, RenameAliasesInNamespaces) {
  std::string Before = R"(
    namespace x { class X {}; }
    namespace ns {
    using UOld = x::X;
    }
    )";
  std::string Expected = R"(
    namespace x { class X {}; }
    namespace ns {
    using UNew = x::X;
    }
    )";
  std::string After = runClangRenameOnCode(Before, "ns::UOld", "ns::UNew");
  CompareSnippets(Expected, After);
}

TEST_F(RenameAliasTest, AliasesInMacros) {
  std::string Before = R"(
    namespace x { class Old {}; }
    namespace ns {
    #define REF(alias) alias alias_var;

    #define ALIAS(old) \
      using old##Alias = x::old; \
      REF(old##Alias);

    ALIAS(Old);

    OldAlias old_alias;
    }
    )";
  std::string Expected = R"(
    namespace x { class Old {}; }
    namespace ns {
    #define REF(alias) alias alias_var;

    #define ALIAS(old) \
      using old##Alias = x::old; \
      REF(old##Alias);

    ALIAS(Old);

    NewAlias old_alias;
    }
    )";
  std::string After =
      runClangRenameOnCode(Before, "ns::OldAlias", "ns::NewAlias");
  CompareSnippets(Expected, After);
}

} // anonymous namespace
} // namespace test
} // namespace clang_rename
} // namesdpace clang

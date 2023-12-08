#include "ClangRenameTest.h"

namespace clang {
namespace clang_rename {
namespace test {
namespace {

class RenameEnumTest : public ClangRenameTest {
public:
  RenameEnumTest() {
    AppendToHeader(R"(
        #define MACRO(x) x
        namespace a {
        enum A1 { Red };
        enum class A2 { Blue };
        struct C {
         enum NestedEnum { White };
         enum class NestedScopedEnum { Black };
        };
        namespace d {
        enum A3 { Orange };
        } // namespace d
        enum A4 { Pink };
        } // namespace a
        enum A5 { Green };)");
  }
};

INSTANTIATE_TEST_SUITE_P(
    RenameEnumTests, RenameEnumTest,
    testing::ValuesIn(std::vector<Case>({
        {"void f(a::A2 arg) { a::A2 t = a::A2::Blue; }",
         "void f(b::B2 arg) { b::B2 t = b::B2::Blue; }", "a::A2", "b::B2"},
        {"void f() { a::A1* t1; }", "void f() { b::B1* t1; }", "a::A1",
         "b::B1"},
        {"void f() { a::A2* t1; }", "void f() { b::B2* t1; }", "a::A2",
         "b::B2"},
        {"void f() { enum a::A2 t = a::A2::Blue; }",
         "void f() { enum b::B2 t = b::B2::Blue; }", "a::A2", "b::B2"},
        {"void f() { enum a::A2 t = a::A2::Blue; }",
         "void f() { enum b::B2 t = b::B2::Blue; }", "a::A2", "b::B2"},

        {"void f() { a::A1 t = a::Red; }", "void f() { b::B1 t = b::B1::Red; }",
         "a::A1", "b::B1"},
        {"void f() { a::A1 t = a::A1::Red; }",
         "void f() { b::B1 t = b::B1::Red; }", "a::A1", "b::B1"},
        {"void f() { auto t = a::Red; }", "void f() { auto t = b::B1::Red; }",
         "a::A1", "b::B1"},
        {"namespace b { void f() { a::A1 t = a::Red; } }",
         "namespace b { void f() { B1 t = B1::Red; } }", "a::A1", "b::B1"},
        {"void f() { a::d::A3 t = a::d::Orange; }",
         "void f() { a::b::B3 t = a::b::B3::Orange; }", "a::d::A3", "a::b::B3"},
        {"namespace a { void f() { a::d::A3 t = a::d::Orange; } }",
         "namespace a { void f() { b::B3 t = b::B3::Orange; } }", "a::d::A3",
         "a::b::B3"},
        {"void f() { A5 t = Green; }", "void f() { B5 t = Green; }", "A5",
         "B5"},
        // FIXME: the new namespace qualifier should be added to the unscoped
        // enum constant.
        {"namespace a { void f() { auto t = Green; } }",
         "namespace a { void f() { auto t = Green; } }", "a::A1", "b::B1"},

        // namespace qualifiers
        {"namespace a { void f(A1 a1) {} }",
         "namespace a { void f(b::B1 a1) {} }", "a::A1", "b::B1"},
        {"namespace a { void f(A2 a2) {} }",
         "namespace a { void f(b::B2 a2) {} }", "a::A2", "b::B2"},
        {"namespace b { void f(a::A1 a1) {} }",
         "namespace b { void f(B1 a1) {} }", "a::A1", "b::B1"},
        {"namespace b { void f(a::A2 a2) {} }",
         "namespace b { void f(B2 a2) {} }", "a::A2", "b::B2"},

        // nested enums
        {"void f() { a::C::NestedEnum t = a::C::White; }",
         "void f() { a::C::NewNestedEnum t = a::C::NewNestedEnum::White; }",
         "a::C::NestedEnum", "a::C::NewNestedEnum"},
        {"void f() { a::C::NestedScopedEnum t = a::C::NestedScopedEnum::Black; "
         "}",
         "void f() { a::C::NewNestedScopedEnum t = "
         "a::C::NewNestedScopedEnum::Black; }",
         "a::C::NestedScopedEnum", "a::C::NewNestedScopedEnum"},

        // macros
        {"void f(MACRO(a::A1) a1) {}", "void f(MACRO(b::B1) a1) {}", "a::A1",
         "b::B1"},
        {"void f(MACRO(a::A2) a2) {}", "void f(MACRO(b::B2) a2) {}", "a::A2",
         "b::B2"},
        {"#define FOO(T, t) T t\nvoid f() { FOO(a::A1, a1); }",
         "#define FOO(T, t) T t\nvoid f() { FOO(b::B1, a1); }", "a::A1",
         "b::B1"},
        {"#define FOO(T, t) T t\nvoid f() { FOO(a::A2, a2); }",
         "#define FOO(T, t) T t\nvoid f() { FOO(b::B2, a2); }", "a::A2",
         "b::B2"},
        {"#define FOO(n) a::A1 n\nvoid f() { FOO(a1); FOO(a2); }",
         "#define FOO(n) b::B1 n\nvoid f() { FOO(a1); FOO(a2); }", "a::A1",
         "b::B1"},

        // using and type alias
        {"using a::A1; A1 gA;", "using b::B1; b::B1 gA;", "a::A1", "b::B1"},
        {"using a::A2; A2 gA;", "using b::B2; b::B2 gA;", "a::A2", "b::B2"},
        {"struct S { using T = a::A1; T a_; };",
         "struct S { using T = b::B1; T a_; };", "a::A1", "b::B1"},
        {"using T = a::A1; T gA;", "using T = b::B1; T gA;", "a::A1", "b::B1"},
        {"using T = a::A2; T gA;", "using T = b::B2; T gA;", "a::A2", "b::B2"},
        {"typedef a::A1 T; T gA;", "typedef b::B1 T; T gA;", "a::A1", "b::B1"},
        {"typedef a::A2 T; T gA;", "typedef b::B2 T; T gA;", "a::A2", "b::B2"},
        {"typedef MACRO(a::A1) T; T gA;", "typedef MACRO(b::B1) T; T gA;",
         "a::A1", "b::B1"},

        // templates
        {"template<typename T> struct Foo { T t; }; void f() { Foo<a::A1> "
         "foo1; }",
         "template<typename T> struct Foo { T t; }; void f() { Foo<b::B1> "
         "foo1; }",
         "a::A1", "b::B1"},
        {"template<typename T> struct Foo { T t; }; void f() { Foo<a::A2> "
         "foo2; }",
         "template<typename T> struct Foo { T t; }; void f() { Foo<b::B2> "
         "foo2; }",
         "a::A2", "b::B2"},
        {"template<typename T> struct Foo { a::A1 a1; };",
         "template<typename T> struct Foo { b::B1 a1; };", "a::A1", "b::B1"},
        {"template<typename T> struct Foo { a::A2 a2; };",
         "template<typename T> struct Foo { b::B2 a2; };", "a::A2", "b::B2"},
        {"template<typename T> int f() { return 1; } template<> int f<a::A1>() "
         "{ return 2; } int g() { return f<a::A1>(); }",
         "template<typename T> int f() { return 1; } template<> int f<b::B1>() "
         "{ return 2; } int g() { return f<b::B1>(); }",
         "a::A1", "b::B1"},
        {"template<typename T> int f() { return 1; } template<> int f<a::A2>() "
         "{ return 2; } int g() { return f<a::A2>(); }",
         "template<typename T> int f() { return 1; } template<> int f<b::B2>() "
         "{ return 2; } int g() { return f<b::B2>(); }",
         "a::A2", "b::B2"},
        {"struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "f.foo<a::A1>(); }",
         "struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "f.foo<b::B1>(); }",
         "a::A1", "b::B1"},
        {"struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "f.foo<a::A2>(); }",
         "struct Foo { template <typename T> T foo(); }; void g() { Foo f;  "
         "f.foo<b::B2>(); }",
         "a::A2", "b::B2"},
    })) );

TEST_P(RenameEnumTest, RenameEnums) {
  auto Param = GetParam();
  assert(!Param.OldName.empty());
  assert(!Param.NewName.empty());
  std::string Actual =
      runClangRenameOnCode(Param.Before, Param.OldName, Param.NewName);
  CompareSnippets(Param.After, Actual);
}

TEST_F(RenameEnumTest, RenameEnumDecl) {
  std::string Before = R"(
      namespace ns {
      enum Old1 { Blue };
      }
  )";
  std::string Expected = R"(
      namespace ns {
      enum New1 { Blue };
      }
  )";
  std::string After = runClangRenameOnCode(Before, "ns::Old1", "ns::New1");
  CompareSnippets(Expected, After);
}

TEST_F(RenameEnumTest, RenameScopedEnumDecl) {
  std::string Before = R"(
      namespace ns {
      enum class Old1 { Blue };
      }
  )";
  std::string Expected = R"(
      namespace ns {
      enum class New1 { Blue };
      }
  )";
  std::string After = runClangRenameOnCode(Before, "ns::Old1", "ns::New1");
  CompareSnippets(Expected, After);
}

} // anonymous namespace
} // namespace test
} // namespace clang_rename
} // namesdpace clang

//===- unittest/Format/QualifierFixerTest.cpp - Formatting unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Format/QualifierAlignmentFixer.h"
#include "FormatTestBase.h"
#include "TestLexer.h"

#define DEBUG_TYPE "format-qualifier-fixer-test"

namespace clang {
namespace format {
namespace test {
namespace {

#define CHECK_PARSE(TEXT, FIELD, VALUE)                                        \
  EXPECT_NE(VALUE, Style.FIELD) << "Initial value already the same!";          \
  EXPECT_EQ(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

#define FAIL_PARSE(TEXT, FIELD, VALUE)                                         \
  EXPECT_NE(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

class QualifierFixerTest : public FormatTestBase {
protected:
  TokenList annotate(StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).annotate(Code);
  }
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
};

TEST_F(QualifierFixerTest, RotateTokens) {
  // TODO add test
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("const"),
            tok::kw_const);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("volatile"),
            tok::kw_volatile);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("inline"),
            tok::kw_inline);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("static"),
            tok::kw_static);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("restrict"),
            tok::kw_restrict);
  EXPECT_EQ(LeftRightQualifierAlignmentFixer::getTokenFromQualifier("friend"),
            tok::kw_friend);
}

TEST_F(QualifierFixerTest, FailQualifierInvalidConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile, apples, type]",
             QualifierOrder,
             std::vector<std::string>({"const", "volatile", "apples", "type"}));
}

TEST_F(QualifierFixerTest, FailQualifierDuplicateConfiguration) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile, const, type]",
             QualifierOrder,
             std::vector<std::string>({"const", "volatile", "const", "type"}));
}

TEST_F(QualifierFixerTest, FailQualifierMissingType) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\n"
             "QualifierOrder: [const, volatile ]",
             QualifierOrder,
             std::vector<std::string>({
                 "const",
                 "volatile",
             }));
}

TEST_F(QualifierFixerTest, FailQualifierEmptyOrder) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom\nQualifierOrder: []", QualifierOrder,
             std::vector<std::string>({}));
}

TEST_F(QualifierFixerTest, FailQualifierMissingOrder) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  FAIL_PARSE("QualifierAlignment: Custom", QualifierOrder,
             std::vector<std::string>());
}

TEST_F(QualifierFixerTest, QualifierLeft) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("QualifierAlignment: Left", QualifierOrder,
              std::vector<std::string>({"const", "volatile", "type"}));
}

TEST_F(QualifierFixerTest, QualifierRight) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_Cpp;
  CHECK_PARSE("QualifierAlignment: Right", QualifierOrder,
              std::vector<std::string>({"type", "const", "volatile"}));
}

TEST_F(QualifierFixerTest, QualifiersCustomOrder) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"friend", "inline",   "constexpr", "static",
                          "const",  "volatile", "type"};

  verifyFormat("const volatile int a;", Style);
  verifyFormat("const volatile int a;", "volatile const int a;", Style);
  verifyFormat("const volatile int a;", "int const volatile a;", Style);
  verifyFormat("const volatile int a;", "int volatile const a;", Style);
  verifyFormat("const volatile int a;", "const int volatile a;", Style);

  verifyFormat("static const volatile int a;", "const static int volatile a;",
               Style);
  verifyFormat("inline static const volatile int a;",
               "const static inline int volatile a;", Style);

  verifyFormat("constexpr static int a;", "static constexpr int a;", Style);
  verifyFormat("constexpr static int A;", "static constexpr int A;", Style);
  verifyFormat("constexpr static int Bar;", "static constexpr int Bar;", Style);
  verifyFormat("constexpr static LPINT Bar;", "static constexpr LPINT Bar;",
               Style);
  verifyFormat("const const int a;", "const int const a;", Style);

  verifyFormat(
      "friend constexpr auto operator<=>(const foo &, const foo &) = default;",
      "constexpr friend auto operator<=>(const foo &, const foo &) = default;",
      Style);
  verifyFormat(
      "friend constexpr bool operator==(const foo &, const foo &) = default;",
      "constexpr bool friend operator==(const foo &, const foo &) = default;",
      Style);
}

TEST_F(QualifierFixerTest, LeftRightQualifier) {
  FormatStyle Style = getLLVMStyle();

  // keep the const style unaltered
  verifyFormat("const int a;", Style);
  verifyFormat("const int *a;", Style);
  verifyFormat("const int &a;", Style);
  verifyFormat("const int &&a;", Style);
  verifyFormat("int const b;", Style);
  verifyFormat("int const *b;", Style);
  verifyFormat("int const &b;", Style);
  verifyFormat("int const &&b;", Style);
  verifyFormat("int const *const b;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("const Foo a;", Style);
  verifyFormat("const Foo *a;", Style);
  verifyFormat("const Foo &a;", Style);
  verifyFormat("const Foo &&a;", Style);
  verifyFormat("Foo const b;", Style);
  verifyFormat("Foo const *b;", Style);
  verifyFormat("Foo const &b;", Style);
  verifyFormat("Foo const &&b;", Style);
  verifyFormat("Foo const *const b;", Style);

  verifyFormat("LLVM_NODISCARD const int &Foo();", Style);
  verifyFormat("LLVM_NODISCARD int const &Foo();", Style);

  verifyFormat("volatile const int *restrict;", Style);
  verifyFormat("const volatile int *restrict;", Style);
  verifyFormat("const int volatile *restrict;", Style);
}

TEST_F(QualifierFixerTest, RightQualifier) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const", "volatile"};

  verifyFormat("int const a;", Style);
  verifyFormat("int const *a;", Style);
  verifyFormat("int const &a;", Style);
  verifyFormat("int const &&a;", Style);
  verifyFormat("int const b;", Style);
  verifyFormat("int const *b;", Style);
  verifyFormat("int const &b;", Style);
  verifyFormat("int const &&b;", Style);
  verifyFormat("int const *const b;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("Foo const a;", Style);
  verifyFormat("Foo const *a;", Style);
  verifyFormat("Foo const &a;", Style);
  verifyFormat("Foo const &&a;", Style);
  verifyFormat("Foo const b;", Style);
  verifyFormat("Foo const *b;", Style);
  verifyFormat("Foo const &b;", Style);
  verifyFormat("Foo const &&b;", Style);
  verifyFormat("Foo const *const b;", Style);
  verifyFormat("Foo *const b;", Style);
  verifyFormat("Foo const *const b;", Style);
  verifyFormat("auto const v = get_value();", Style);
  verifyFormat("long long const &a;", Style);
  verifyFormat("unsigned char const *a;", Style);
  verifyFormat("int main(int const argc, char const *const *const argv)",
               Style);

  verifyFormat("LLVM_NODISCARD int const &Foo();", Style);
  verifyFormat("SourceRange getSourceRange() const override LLVM_READONLY",
               Style);
  verifyFormat("void foo() const override;", Style);
  verifyFormat("void foo() const override LLVM_READONLY;", Style);
  verifyFormat("void foo() const final;", Style);
  verifyFormat("void foo() const final LLVM_READONLY;", Style);
  verifyFormat("void foo() const LLVM_READONLY;", Style);
  verifyFormat("void foo() const volatile override;", Style);
  verifyFormat("void foo() const volatile override LLVM_READONLY;", Style);
  verifyFormat("void foo() const volatile final;", Style);
  verifyFormat("void foo() const volatile final LLVM_READONLY;", Style);
  verifyFormat("void foo() const volatile LLVM_READONLY;", Style);

  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      "template <typename Func> explicit Action(const Action<Func>& action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(Action<Func> const &action);",
      "template <typename Func>\nexplicit Action(const Action<Func>& action);",
      Style);

  verifyFormat("int const a;", "const int a;", Style);
  verifyFormat("int const *a;", "const int *a;", Style);
  verifyFormat("int const &a;", "const int &a;", Style);
  verifyFormat("foo(int const &a)", "foo(const int &a)", Style);
  verifyFormat("unsigned char *a;", Style);
  verifyFormat("unsigned char const *a;", "const unsigned char *a;", Style);
  verifyFormat("vector<int, int const, int &, int const &> args1",
               "vector<int, const int, int &, const int &> args1", Style);
  verifyFormat("unsigned int const &get_nu() const",
               "const unsigned int &get_nu() const", Style);
  verifyFormat("Foo<int> const &a", "const Foo<int> &a", Style);
  verifyFormat("Foo<int>::iterator const &a", "const Foo<int>::iterator &a",
               Style);
  verifyFormat("::Foo<int>::iterator const &a", "const ::Foo<int>::iterator &a",
               Style);

  verifyFormat("Foo(int a, "
               "unsigned b, // c-style args\n"
               "    Bar const &c);",
               "Foo(int a, "
               "unsigned b, // c-style args\n"
               "    const Bar &c);",
               Style);

  verifyFormat("int const volatile;", "volatile const int;", Style);
  verifyFormat("int const volatile;", "const volatile int;", Style);
  verifyFormat("int const volatile;", "const int volatile;", Style);

  verifyFormat("int const volatile *restrict;", "volatile const int *restrict;",
               Style);
  verifyFormat("int const volatile *restrict;", "const volatile int *restrict;",
               Style);
  verifyFormat("int const volatile *restrict;", "const int volatile *restrict;",
               Style);

  verifyFormat("long long int const volatile;", "const long long int volatile;",
               Style);
  verifyFormat("long long int const volatile;", "long const long int volatile;",
               Style);
  verifyFormat("long long int const volatile;", "long long volatile int const;",
               Style);
  verifyFormat("long long int const volatile;", "long volatile long const int;",
               Style);
  verifyFormat("long long int const volatile;", "const long long volatile int;",
               Style);

  verifyFormat("static int const bat;", "static const int bat;", Style);
  verifyFormat("static int const bat;", Style);

  // static is not configured, unchanged on the left of the right hand
  // qualifiers.
  verifyFormat("int static const volatile;", "volatile const int static;",
               Style);
  verifyFormat("int static const volatile;", "const volatile int static;",
               Style);
  verifyFormat("int static const volatile;", "const int volatile static;",
               Style);
  verifyFormat("Foo static const volatile;", "volatile const Foo static;",
               Style);
  verifyFormat("Foo static const volatile;", "const volatile Foo static;",
               Style);
  verifyFormat("Foo static const volatile;", "const Foo volatile static;",
               Style);

  verifyFormat("Foo inline static const;", "const Foo inline static;", Style);
  verifyFormat("Foo inline static const;", "Foo const inline static;", Style);
  verifyFormat("Foo inline static const;", "Foo inline const static;", Style);
  verifyFormat("Foo inline static const;", Style);

  verifyFormat("Foo<T volatile>::Bar<Type const, 5> const volatile A:: *;",
               "volatile const Foo<volatile T>::Bar<const Type, 5> A::*;",
               Style);

  verifyFormat("int const Foo<int>::bat = 0;", "const int Foo<int>::bat = 0;",
               Style);
  verifyFormat("int const Foo<int>::bat = 0;", Style);
  verifyFormat("void fn(Foo<T> const &i);", "void fn(const Foo<T> &i);", Style);
  verifyFormat("int const Foo<int>::fn() {", Style);
  verifyFormat("Foo<Foo<int>> const *p;", "const Foo<Foo<int>> *p;", Style);
  verifyFormat(
      "Foo<Foo<int>> const *p = const_cast<Foo<Foo<int>> const *>(&ffi);",
      "const Foo<Foo<int>> *p = const_cast<const Foo<Foo<int>> *>(&ffi);",
      Style);

  verifyFormat("void fn(Foo<T> const &i);", "void fn(const Foo<T> &i);", Style);
  verifyFormat("void fns(ns::S const &s);", "void fns(const ns::S &s);", Style);
  verifyFormat("void fns(::ns::S const &s);", "void fns(const ::ns::S &s);",
               Style);
  verifyFormat("void fn(ns::Foo<T> const &i);", "void fn(const ns::Foo<T> &i);",
               Style);
  verifyFormat("void fns(ns::ns2::S const &s);",
               "void fns(const ns::ns2::S &s);", Style);
  verifyFormat("void fn(ns::Foo<Bar<T>> const &i);",
               "void fn(const ns::Foo<Bar<T>> &i);", Style);
  verifyFormat("void fn(ns::ns2::Foo<Bar<T>> const &i);",
               "void fn(const ns::ns2::Foo<Bar<T>> &i);", Style);
  verifyFormat("void fn(ns::ns2::Foo<Bar<T, U>> const &i);",
               "void fn(const ns::ns2::Foo<Bar<T, U>> &i);", Style);

  verifyFormat("LocalScope const *Scope = nullptr;",
               "const LocalScope* Scope = nullptr;", Style);
  verifyFormat("struct DOTGraphTraits<Stmt const *>",
               "struct DOTGraphTraits<const Stmt *>", Style);

  verifyFormat(
      "bool tools::addXRayRuntime(ToolChain const &TC, ArgList const &Args) {",
      "bool tools::addXRayRuntime(const ToolChain&TC, const ArgList &Args) {",
      Style);
  verifyFormat("Foo<Foo<int> const> P;", "Foo<const Foo<int>> P;", Style);
  verifyFormat("Foo<Foo<int> const> P;\n#if 0\n#else\n#endif",
               "Foo<const Foo<int>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("auto const i = 0;", "const auto i = 0;", Style);
  verifyFormat("auto const &ir = i;", "const auto &ir = i;", Style);
  verifyFormat("auto const *ip = &i;", "const auto *ip = &i;", Style);

  verifyFormat("void f(Concept auto const &x);",
               "void f(const Concept auto &x);", Style);
  verifyFormat("void f(std::integral auto const &x);",
               "void f(const std::integral auto &x);", Style);

  verifyFormat("auto lambda = [] { int const i = 0; };",
               "auto lambda = [] { const int i = 0; };", Style);

  verifyFormat("Foo<Foo<int> const> P;\n#if 0\n#else\n#endif",
               "Foo<const Foo<int>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Bar<Bar<int const> const> P;\n#if 0\n#else\n#endif",
               "Bar<Bar<const int> const> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Baz<Baz<int const> const> P;\n#if 0\n#else\n#endif",
               "Baz<const Baz<const int>> P;\n#if 0\n#else\n#endif", Style);

  // verifyFormat("#if 0\nBoo<Boo<int const> const> P;\n#else\n#endif",
  //             "#if 0\nBoo<const Boo<const int>> P;\n#else\n#endif", Style);

  verifyFormat("int const P;\n#if 0\n#else\n#endif",
               "const int P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("unsigned long const a;", "const unsigned long a;", Style);
  verifyFormat("unsigned long long const a;", "const unsigned long long a;",
               Style);

  // Multiple template parameters.
  verifyFormat("Bar<std::Foo const, 32>", "Bar<const std::Foo, 32>", Style);
  // Variable declaration based on template type.
  verifyFormat("Bar<std::Foo const> bar", "Bar<const std::Foo> bar", Style);

  // Using typename for a nested dependent type name.
  verifyFormat("typename Foo::iterator const;", "const typename Foo::iterator;",
               Style);

  // Don't move past C-style struct/class.
  verifyFormat("void foo(const struct A a);", Style);
  verifyFormat("void foo(const class A a);", Style);

  // Don't move past struct/class combined declaration and variable
  // definition.
  verifyFormat("const struct {\n} var;", Style);
  verifyFormat("struct {\n} const var;", Style);
  verifyFormat("const class {\n} var;", Style);
  verifyFormat("class {\n} const var;", Style);

  // Leave left qualifers unchanged for combined declaration and variable
  // definition.
  verifyFormat("volatile const class {\n} var;", Style);
  verifyFormat("const volatile class {\n} var;", Style);
  // Also do no sorting with respect to not-configured tokens.
  verifyFormat("const static volatile class {\n} var;", Style);
  // Sort right qualifiers for combined declaration and variable definition.
  verifyFormat("class {\n} const volatile var;", Style);
  verifyFormat("class {\n} const volatile var;",
               "class {\n} volatile const var;", Style);
  // Static keyword is not configured, should end up on the left of the right
  // side.
  verifyFormat("class {\n} static const volatile var;", Style);
  verifyFormat("class {\n} static const volatile var;",
               "class {\n} volatile static const var;", Style);

  // ::template for dependent names
  verifyFormat("::template Foo<T> const volatile var;",
               "const volatile ::template Foo<T> var;", Style);
  verifyFormat("typename ::template Foo<T> const volatile var;",
               "const volatile typename ::template Foo<T> var;", Style);
  verifyFormat("typename Bar::template Foo<T>::T const;",
               "const typename Bar::template Foo<T>::T;", Style);
  verifyFormat("typename Bar::template Foo<T>::T const volatile;",
               "const volatile typename Bar::template Foo<T>::T;", Style);

  // typename ::
  verifyFormat("typename ::Bar<int> const;", "const typename ::Bar<int>;",
               Style);
  // typename ::template
  verifyFormat("typename ::template Bar<int> const;",
               "const typename ::template Bar<int>;", Style);

  verifyFormat("foo<Bar<Baz> const>();", "foo<const Bar<Baz>>();", Style);
  verifyFormat("foo<Bar<Baz> const>();", "foo<const Bar<Baz> >();", Style);
  verifyFormat("Bar<32, Foo<25> const>;", "Bar<32, const Foo<25>>;", Style);
  verifyFormat("A<B<C<D> const> const>;", "A<const B<const C<D>>>;", Style);
  verifyFormat("A<B<C<D const> const> const>;", "A<const B<const C<const D>>>;",
               Style);

  // Don't move past decltype, typeof, or _Atomic.
  verifyFormat("const decltype(foo)", Style);
  verifyFormat("const typeof(foo)", Style);
  verifyFormat("const _Atomic(foo)", Style);

  // Comments
  const int ColumnLimit = Style.ColumnLimit;
  Style.ColumnLimit = 200;
  verifyFormat("/*c*/ Foo const *foo;", "const /*c*/ Foo *foo;", Style);
  verifyFormat("Foo const /*c*/ *foo;", "const Foo /*c*/ *foo;", Style);
  verifyFormat("Foo const * /*c*/ foo;", "const Foo * /*c*/ foo;", Style);

  verifyFormat("/*comment*/ std::vector<int> const v;",
               "const /*comment*/ std::vector<int> v;", Style);
  verifyFormat("std /*comment*/ ::vector<int> const v;",
               "const std /*comment*/ ::vector<int> v;", Style);
  verifyFormat("std::/*comment*/ vector<int> const v;",
               "const std::/*comment*/ vector<int> v;", Style);
  verifyFormat("std::vector /*comment*/<int> const v;",
               "const std::vector /*comment*/ <int> v;", Style);
  verifyFormat("std::vector</*comment*/ int> const v;",
               "const std::vector</*comment*/ int> v;", Style);
  verifyFormat("std::vector<int /*comment*/> const v;",
               "const std::vector<int /*comment*/> v;", Style);
  verifyFormat("std::vector<int> const /*comment*/ v;",
               "const std::vector<int> /*comment*/ v;", Style);

  verifyFormat("std::vector</*comment*/ int const> v;",
               "std::vector</*comment*/ const int> v;", Style);
  verifyFormat("std::vector</*comment*/ int const> v;",
               "std::vector<const /*comment*/ int> v;", Style);
  verifyFormat("std::vector<int const /*comment*/> v;",
               "std::vector<const int /*comment*/> v;", Style);
  verifyFormat("std::vector</*comment*/ Foo const> v;",
               "std::vector</*comment*/ const Foo> v;", Style);
  verifyFormat("std::vector</*comment*/ Foo const> v;",
               "std::vector<const /*comment*/ Foo> v;", Style);
  verifyFormat("std::vector<Foo const /*comment*/> v;",
               "std::vector<const Foo /*comment*/> v;", Style);

  verifyFormat("typename C<T>::template B<T> const;",
               "const typename C<T>::template B<T>;", Style);
  verifyFormat("/*c*/ typename C<T>::template B<T> const;",
               "const /*c*/ typename C<T>::template B<T>;", Style);
  verifyFormat("typename /*c*/ C<T>::template B<T> const;",
               "const typename /*c*/ C<T>::template B<T>;", Style);
  verifyFormat("typename C /*c*/<T>::template B<T> const;",
               "const typename C /*c*/<T>::template B<T>;", Style);
  verifyFormat("typename C<T> /*c*/ ::template B<T> const;",
               "const typename C<T> /*c*/ ::template B<T>;", Style);
  verifyFormat("typename C<T>::/*c*/ template B<T> const;",
               "const typename C<T>::/*c*/ template B<T>;", Style);
  verifyFormat("typename C<T>::template /*c*/ B<T> const;",
               "const typename C<T>::template /*c*/B<T>;", Style);
  verifyFormat("typename C<T>::template B<T> const /*c*/;",
               "const typename C<T>::template B<T>/*c*/;", Style);

  verifyFormat("/*c*/ /*c*/ typename /*c*/ C /*c*/<T> /*c*/ ::/*c*/ template "
               "/*c*/ B /*c*/<T> const /*c*/ v;",
               "/*c*/ const /*c*/ typename /*c*/ C /*c*/<T> /*c*/ "
               "::/*c*/template /*c*/ B /*c*/<T> /*c*/ v;",
               Style);

  verifyFormat("/*c*/ unsigned /*c*/ long const /*c*/ a;",
               "const /*c*/ unsigned /*c*/ long /*c*/ a;", Style);
  verifyFormat("unsigned /*c*/ long /*c*/ long const a;",
               "const unsigned /*c*/ long /*c*/ long a;", Style);

  // Not changed
  verifyFormat("foo() /*c*/ const", Style);
  verifyFormat("const /*c*/ struct a;", Style);
  verifyFormat("const /*c*/ class a;", Style);
  verifyFormat("const /*c*/ decltype(v) a;", Style);
  verifyFormat("const /*c*/ typeof(v) a;", Style);
  verifyFormat("const /*c*/ _Atomic(v) a;", Style);
  verifyFormat("const decltype /*c*/ (v) a;", Style);
  verifyFormat("const /*c*/ class {\n} volatile /*c*/ foo = {};", Style);

  Style.ColumnLimit = ColumnLimit;

  // Don't adjust macros
  verifyFormat("const INTPTR a;", Style);

  // Pointers to members
  verifyFormat("int S:: *a;", Style);
  verifyFormat("int const S:: *a;", "const int S:: *a;", Style);
  verifyFormat("int const S:: *const a;", "const int S::* const a;", Style);
  verifyFormat("int A:: *const A:: *p1;", Style);
  verifyFormat("float (C:: *p)(int);", Style);
  verifyFormat("float (C:: *const p)(int);", Style);
  verifyFormat("float (C:: *p)(int) const;", Style);
  verifyFormat("float const (C:: *p)(int);", "const float (C::*p)(int);",
               Style);
}

TEST_F(QualifierFixerTest, LeftQualifier) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};

  verifyFormat("const int a;", Style);
  verifyFormat("const int *a;", Style);
  verifyFormat("const int &a;", Style);
  verifyFormat("const int &&a;", Style);
  verifyFormat("const int b;", Style);
  verifyFormat("const int *b;", Style);
  verifyFormat("const int &b;", Style);
  verifyFormat("const int &&b;", Style);
  verifyFormat("const int *const b;", Style);
  verifyFormat("int *const c;", Style);

  verifyFormat("const Foo a;", Style);
  verifyFormat("const Foo *a;", Style);
  verifyFormat("const Foo &a;", Style);
  verifyFormat("const Foo &&a;", Style);
  verifyFormat("const Foo b;", Style);
  verifyFormat("const Foo *b;", Style);
  verifyFormat("const Foo &b;", Style);
  verifyFormat("const Foo &&b;", Style);
  verifyFormat("const Foo *const b;", Style);
  verifyFormat("Foo *const b;", Style);
  verifyFormat("const Foo *const b;", Style);

  verifyFormat("LLVM_NODISCARD const int &Foo();", Style);

  verifyFormat("const char a[];", Style);
  verifyFormat("const auto v = get_value();", Style);
  verifyFormat("const long long &a;", Style);
  verifyFormat("const unsigned char *a;", Style);
  verifyFormat("const unsigned char *a;", "unsigned char const *a;", Style);
  verifyFormat("const Foo<int> &a", "Foo<int> const &a", Style);
  verifyFormat("const Foo<int>::iterator &a", "Foo<int>::iterator const &a",
               Style);
  verifyFormat("const ::Foo<int>::iterator &a", "::Foo<int>::iterator const &a",
               Style);

  verifyFormat("const int a;", "int const a;", Style);
  verifyFormat("const int *a;", "int const *a;", Style);
  verifyFormat("const int &a;", "int const &a;", Style);
  verifyFormat("foo(const int &a)", "foo(int const &a)", Style);
  verifyFormat("unsigned char *a;", Style);
  verifyFormat("const unsigned int &get_nu() const",
               "unsigned int const &get_nu() const", Style);

  verifyFormat("const volatile int;", "volatile const int;", Style);
  verifyFormat("const volatile int;", Style);
  verifyFormat("const volatile int;", "const int volatile;", Style);

  verifyFormat("const volatile int *restrict;", "volatile const int *restrict;",
               Style);
  verifyFormat("const volatile int *restrict;", Style);
  verifyFormat("const volatile int *restrict;", "const int volatile *restrict;",
               Style);

  verifyFormat("const volatile long long int;", "volatile long long int const;",
               Style);
  verifyFormat("const volatile long long int;", "volatile long long const int;",
               Style);
  verifyFormat("const volatile long long int;", "long long volatile int const;",
               Style);
  verifyFormat("const volatile long long int;", "long volatile long int const;",
               Style);
  verifyFormat("const volatile long long int;", "const long long volatile int;",
               Style);

  verifyFormat("SourceRange getSourceRange() const override LLVM_READONLY;",
               Style);

  verifyFormat("void foo() const override;", Style);
  verifyFormat("void foo() const override LLVM_READONLY;", Style);
  verifyFormat("void foo() const final;", Style);
  verifyFormat("void foo() const final LLVM_READONLY;", Style);
  verifyFormat("void foo() const LLVM_READONLY;", Style);

  verifyFormat(
      "template <typename Func> explicit Action(const Action<Func> &action);",
      Style);
  verifyFormat(
      "template <typename Func> explicit Action(const Action<Func> &action);",
      "template <typename Func> explicit Action(Action<Func> const &action);",
      Style);

  verifyFormat("static const int bat;", Style);
  verifyFormat("static const int bat;", "static int const bat;", Style);

  verifyFormat("static const int Foo<int>::bat = 0;", Style);
  verifyFormat("static const int Foo<int>::bat = 0;",
               "static int const Foo<int>::bat = 0;", Style);

  verifyFormat("void fn(const Foo<T> &i);");

  verifyFormat("const int Foo<int>::bat = 0;", Style);
  verifyFormat("const int Foo<int>::bat = 0;", "int const Foo<int>::bat = 0;",
               Style);
  verifyFormat("void fn(const Foo<T> &i);", "void fn( Foo<T> const &i);",
               Style);
  verifyFormat("const int Foo<int>::fn() {", "int const Foo<int>::fn() {",
               Style);
  verifyFormat("const Foo<Foo<int>> *p;", "Foo<Foo<int>> const *p;", Style);
  verifyFormat(
      "const Foo<Foo<int>> *p = const_cast<const Foo<Foo<int>> *>(&ffi);",
      "const Foo<Foo<int>> *p = const_cast<Foo<Foo<int>> const *>(&ffi);",
      Style);

  verifyFormat("void fn(const Foo<T> &i);", "void fn(Foo<T> const &i);", Style);
  verifyFormat("void fns(const ns::S &s);", "void fns(ns::S const &s);", Style);
  verifyFormat("void fns(const ::ns::S &s);", "void fns(::ns::S const &s);",
               Style);
  verifyFormat("void fn(const ns::Foo<T> &i);", "void fn(ns::Foo<T> const &i);",
               Style);
  verifyFormat("void fns(const ns::ns2::S &s);",
               "void fns(ns::ns2::S const &s);", Style);
  verifyFormat("void fn(const ns::Foo<Bar<T>> &i);",
               "void fn(ns::Foo<Bar<T>> const &i);", Style);
  verifyFormat("void fn(const ns::ns2::Foo<Bar<T>> &i);",
               "void fn(ns::ns2::Foo<Bar<T>> const &i);", Style);
  verifyFormat("void fn(const ns::ns2::Foo<Bar<T, U>> &i);",
               "void fn(ns::ns2::Foo<Bar<T, U>> const &i);", Style);

  verifyFormat("const auto i = 0;", "auto const i = 0;", Style);
  verifyFormat("const auto &ir = i;", "auto const &ir = i;", Style);
  verifyFormat("const auto *ip = &i;", "auto const *ip = &i;", Style);

  verifyFormat("void f(const Concept auto &x);",
               "void f(Concept auto const &x);", Style);
  verifyFormat("void f(const std::integral auto &x);",
               "void f(std::integral auto const &x);", Style);

  verifyFormat("auto lambda = [] { const int i = 0; };",
               "auto lambda = [] { int const i = 0; };", Style);

  verifyFormat("Foo<const Foo<int>> P;\n#if 0\n#else\n#endif",
               "Foo<Foo<int> const> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("Foo<Foo<const int>> P;\n#if 0\n#else\n#endif",
               "Foo<Foo<int const>> P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("const int P;\n#if 0\n#else\n#endif",
               "int const P;\n#if 0\n#else\n#endif", Style);

  verifyFormat("const unsigned long a;", "unsigned long const a;", Style);
  verifyFormat("const unsigned long long a;", "unsigned long long const a;",
               Style);

  verifyFormat("const long long unsigned a;", "long const long unsigned a;",
               Style);

  verifyFormat("const std::Foo", Style);
  verifyFormat("const std::Foo<>", Style);
  verifyFormat("const std::Foo < int", "const std::Foo<int", Style);
  verifyFormat("const std::Foo<int>", Style);

  // Multiple template parameters.
  verifyFormat("Bar<const std::Foo, 32>;", "Bar<std::Foo const, 32>;", Style);

  // Variable declaration based on template type.
  verifyFormat("Bar<const std::Foo> bar;", "Bar<std::Foo const> bar;", Style);

  // Using typename for a dependent name.
  verifyFormat("const typename Foo::iterator;", "typename Foo::iterator const;",
               Style);

  // Don't move past C-style struct/class.
  verifyFormat("void foo(struct A const a);", Style);
  verifyFormat("void foo(class A const a);", Style);

  // Don't move past struct/class combined declaration and variable
  // definition.
  verifyFormat("const struct {\n} var;", Style);
  verifyFormat("struct {\n} const var;", Style);
  verifyFormat("const class {\n} var;", Style);
  verifyFormat("class {\n} const var;", Style);

  // Sort left qualifiers for struct/class combined declaration and variable
  // definition.
  verifyFormat("const volatile class {\n} var;", Style);
  verifyFormat("const volatile class {\n} var;",
               "volatile const class {\n} var;", Style);
  // Leave right qualifers unchanged for struct/class combined declaration and
  // variable definition.
  verifyFormat("class {\n} const volatile var;", Style);
  verifyFormat("class {\n} volatile const var;", Style);

  verifyFormat("foo<const Bar<Baz<T>>>();", "foo<Bar<Baz<T>> const>();", Style);
  verifyFormat("foo<const Bar<Baz<T>>>();", "foo<Bar<Baz<T> > const>();",
               Style);
  verifyFormat("Bar<32, const Foo<25>>;", "Bar<32, Foo<25> const>;", Style);
  verifyFormat("A<const B<const C<D>>>;", "A<B<C<D> const> const>;", Style);
  verifyFormat("A<const B<const C<const D>>>;", "A<B<C<D const> const> const>;",
               Style);

  // Don't move past decltype, typeof, or _Atomic.
  verifyFormat("decltype(foo) const", Style);
  verifyFormat("typeof(foo) const", Style);
  verifyFormat("_Atomic(foo) const", Style);

  // ::template for dependent names
  verifyFormat("const volatile ::template Foo<T> var;",
               "::template Foo<T> const volatile var;", Style);
  verifyFormat("const volatile typename ::template Foo<T> var;",
               "typename ::template Foo<T> const volatile var;", Style);
  verifyFormat("const typename Bar::template Foo<T>::T;",
               "typename Bar::template Foo<T>::T const;", Style);
  verifyFormat("const volatile typename Bar::template Foo<T>::T;",
               "typename Bar::template Foo<T>::T const volatile;", Style);

  // typename ::
  verifyFormat("const typename ::Bar<int>;", "typename ::Bar<int> const;",
               Style);
  // typename ::template
  verifyFormat("const typename ::template Bar<int>;",
               "typename ::template Bar<int> const;", Style);

  // Comments
  const int ColumnLimit = Style.ColumnLimit;
  Style.ColumnLimit = 200;
  verifyFormat("/*c*/ const Foo *foo;", "/*c*/ Foo const *foo;", Style);
  verifyFormat("const Foo /*c*/ *foo;", "Foo const /*c*/ *foo;", Style);
  verifyFormat("const Foo * /*c*/ foo;", "Foo const * /*c*/ foo;", Style);

  verifyFormat("/*comment*/ const std::vector<int> v;",
               "/*comment*/ std::vector<int> const v;", Style);
  verifyFormat("const std /*comment*/ ::vector<int> v;",
               "std /*comment*/ ::vector<int> const v;", Style);
  verifyFormat("const std::/*comment*/ vector<int> v;",
               "std::/*comment*/ vector<int> const v;", Style);
  verifyFormat("const std::vector /*comment*/<int> v;",
               "std::vector /*comment*/<int> const v;", Style);
  verifyFormat("const std::vector</*comment*/ int> v;",
               "std::vector</*comment*/ int> const v;", Style);
  verifyFormat("const std::vector<int /*comment*/> v;",
               "std::vector<int /*comment*/> const v;", Style);
  verifyFormat("const std::vector<int> /*comment*/ v;",
               "std::vector<int> /*comment*/ const v;", Style);

  verifyFormat("std::vector</*comment*/ const int> v;",
               "std::vector</*comment*/ int const> v;", Style);
  verifyFormat("std::vector<const int /*comment*/> v;",
               "std::vector<int /*comment*/ const> v;", Style);
  verifyFormat("std::vector<const int /*comment*/> v;",
               "std::vector<int const /*comment*/> v;", Style);
  verifyFormat("std::vector</*comment*/ const Foo> v;",
               "std::vector</*comment*/ Foo const> v;", Style);
  verifyFormat("std::vector<const Foo /*comment*/> v;",
               "std::vector<Foo /*comment*/ const> v;", Style);
  verifyFormat("std::vector<const Foo /*comment*/> v;",
               "std::vector<Foo const /*comment*/> v;", Style);

  verifyFormat("const typename C<T>::template B<T>;",
               "typename C<T>::template B<T> const;", Style);
  verifyFormat("/*c*/ const typename C<T>::template B<T>;",
               "/*c*/ typename C<T>::template B<T> const;", Style);
  verifyFormat("const typename /*c*/ C<T>::template B<T>;",
               "typename /*c*/ C<T>::template B<T> const;", Style);
  verifyFormat("const typename C /*c*/<T>::template B<T>;",
               "typename C /*c*/<T>::template B<T> const;", Style);
  verifyFormat("const typename C<T> /*c*/ ::template B<T>;",
               "typename C<T> /*c*/ ::template B<T> const;", Style);
  verifyFormat("const typename C<T>::/*c*/ template B<T>;",
               "typename C<T>::/*c*/ template B<T> const;", Style);
  verifyFormat("const typename C<T>::template /*c*/ B<T>;",
               "typename C<T>::template /*c*/ B<T> const;", Style);
  verifyFormat("const typename C<T>::template B<T> /*c*/;",
               "typename C<T>::template B<T> /*c*/ const;", Style);

  verifyFormat("/*c*/ const typename /*c*/ C /*c*/<T> /*c*/ ::/*c*/ template "
               "/*c*/ B /*c*/<T> /*c*/ v;",
               "/*c*/ typename /*c*/ C /*c*/<T> /*c*/ ::/*c*/ template /*c*/ B "
               "/*c*/<T> /*c*/ const v;",
               Style);

  verifyFormat("const unsigned /*c*/ long /*c*/ a;",
               "unsigned /*c*/ long /*c*/ const a;", Style);
  verifyFormat("const unsigned /*c*/ long /*c*/ long a;",
               "unsigned /*c*/ long /*c*/ long const a;", Style);

  // Not changed
  verifyFormat("foo() /*c*/ const", Style);
  verifyFormat("struct /*c*/ const a;", Style);
  verifyFormat("class /*c*/ const a;", Style);
  verifyFormat("decltype(v) /*c*/ const a;", Style);
  verifyFormat("typeof(v) /*c*/ const a;", Style);
  verifyFormat("_Atomic(v) /*c*/ const a;", Style);
  verifyFormat("decltype /*c*/ (v) const a;", Style);
  verifyFormat("const /*c*/ class {\n} /*c*/ volatile /*c*/ foo = {};", Style);

  Style.ColumnLimit = ColumnLimit;

  // Don't adjust macros
  verifyFormat("INTPTR const a;", Style);

  // Pointers to members
  verifyFormat("int S:: *a;", Style);
  verifyFormat("const int S:: *a;", "int const S:: *a;", Style);
  verifyFormat("const int S:: *const a;", "int const S::* const a;", Style);
  verifyFormat("int A:: *const A:: *p1;", Style);
  verifyFormat("float (C:: *p)(int);", Style);
  verifyFormat("float (C:: *const p)(int);", Style);
  verifyFormat("float (C:: *p)(int) const;", Style);
  verifyFormat("const float (C:: *p)(int);", "float const (C::*p)(int);",
               Style);
}

TEST_F(QualifierFixerTest, ConstVolatileQualifiersOrder) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};

  // The Default
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("const volatile int a;", Style);
  verifyFormat("const volatile int a;", "volatile const int a;", Style);
  verifyFormat("const volatile int a;", "int const volatile a;", Style);
  verifyFormat("const volatile int a;", "int volatile const a;", Style);
  verifyFormat("const volatile int a;", "const int volatile a;", Style);

  verifyFormat("const volatile Foo a;", Style);
  verifyFormat("const volatile Foo a;", "volatile const Foo a;", Style);
  verifyFormat("const volatile Foo a;", "Foo const volatile a;", Style);
  verifyFormat("const volatile Foo a;", "Foo volatile const a;", Style);
  verifyFormat("const volatile Foo a;", "const Foo volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const", "volatile"};

  verifyFormat("int const volatile a;", "const volatile int a;", Style);
  verifyFormat("int const volatile a;", "volatile const int a;", Style);
  verifyFormat("int const volatile a;", Style);
  verifyFormat("int const volatile a;", "int volatile const a;", Style);
  verifyFormat("int const volatile a;", "const int volatile a;", Style);

  verifyFormat("Foo const volatile a;", "const volatile Foo a;", Style);
  verifyFormat("Foo const volatile a;", "volatile const Foo a;", Style);
  verifyFormat("Foo const volatile a;", Style);
  verifyFormat("Foo const volatile a;", "Foo volatile const a;", Style);
  verifyFormat("Foo const volatile a;", "const Foo volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"volatile", "const", "type"};

  verifyFormat("volatile const int a;", "const volatile int a;", Style);
  verifyFormat("volatile const int a;", Style);
  verifyFormat("volatile const int a;", "int const volatile a;", Style);
  verifyFormat("volatile const int a;", "int volatile const a;", Style);
  verifyFormat("volatile const int a;", "const int volatile a;", Style);

  verifyFormat("volatile const Foo a;", "const volatile Foo a;", Style);
  verifyFormat("volatile const Foo a;", Style);
  verifyFormat("volatile const Foo a;", "Foo const volatile a;", Style);
  verifyFormat("volatile const Foo a;", "Foo volatile const a;", Style);
  verifyFormat("volatile const Foo a;", "const Foo volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "volatile", "const"};

  verifyFormat("int volatile const a;", "const volatile int a;", Style);
  verifyFormat("int volatile const a;", "volatile const int a;", Style);
  verifyFormat("int volatile const a;", "int const volatile a;", Style);
  verifyFormat("int volatile const a;", Style);
  verifyFormat("int volatile const a;", "const int volatile a;", Style);

  verifyFormat("Foo volatile const a;", "const volatile Foo a;", Style);
  verifyFormat("Foo volatile const a;", "volatile const Foo a;", Style);
  verifyFormat("Foo volatile const a;", "Foo const volatile a;", Style);
  verifyFormat("Foo volatile const a;", Style);
  verifyFormat("Foo volatile const a;", "const Foo volatile a;", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "volatile", "const"};

  verifyFormat("int volatile const a;", "const volatile int a;", Style);
  verifyFormat("int volatile const a;", "volatile const int a;", Style);
  verifyFormat("int volatile const a;", "int const volatile a;", Style);
  verifyFormat("int volatile const a;", Style);
  verifyFormat("int volatile const a;", "const int volatile a;", Style);

  verifyFormat("Foo volatile const a;", "const volatile Foo a;", Style);
  verifyFormat("Foo volatile const a;", "volatile const Foo a;", Style);
  verifyFormat("Foo volatile const a;", "Foo const volatile a;", Style);
  verifyFormat("Foo volatile const a;", Style);
  verifyFormat("Foo volatile const a;", "const Foo volatile a;", Style);
}

TEST_F(QualifierFixerTest, InlineStatics) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"inline", "static", "const", "volatile", "type"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("inline static const volatile int a;",
               "const inline static volatile int a;", Style);
  verifyFormat("inline static const volatile int a;",
               "volatile inline static const int a;", Style);
  verifyFormat("inline static const volatile int a;",
               "int const inline static  volatile a;", Style);
  verifyFormat("inline static const volatile int a;",
               "int volatile inline static  const a;", Style);
  verifyFormat("inline static const volatile int a;",
               "const int inline static  volatile a;", Style);
}

TEST_F(QualifierFixerTest, AmpEqual) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "type", "const"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)3);

  verifyFormat("foo(std::string const & = std::string()) const",
               "foo(const std::string & = std::string()) const", Style);
  verifyFormat("foo(std::string const & = std::string())",
               "foo(const std::string & = std::string())", Style);
}

TEST_F(QualifierFixerTest, MoveConstBeyondTypeSmall) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "const"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)2);

  verifyFormat("int const a;", "const int a;", Style);
  verifyFormat("int const *a;", "const int*a;", Style);
  verifyFormat("int const *a;", "const int *a;", Style);
  verifyFormat("int const &a;", "const int &a;", Style);
  verifyFormat("int const &&a;", "const int &&a;", Style);
}

TEST_F(QualifierFixerTest, MoveConstBeforeTypeSmall) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"const", "type"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)2);

  verifyFormat("const int a;", "int const a;", Style);
  verifyFormat("const int *a;", "int const *a;", Style);
  verifyFormat("const int *const a;", "int const *const a;", Style);

  verifyFormat("const int a = foo();", "int const a = foo();", Style);
  verifyFormat("const int *a = foo();", "int const *a = foo();", Style);
  verifyFormat("const int *const a = foo();", "int const *const a = foo();",
               Style);

  verifyFormat("const auto a = foo();", "auto const a = foo();", Style);
  verifyFormat("const auto *a = foo();", "auto const *a = foo();", Style);
  verifyFormat("const auto *const a = foo();", "auto const *const a = foo();",
               Style);
}

TEST_F(QualifierFixerTest, MoveConstBeyondType) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "inline", "type", "const", "volatile"};
  EXPECT_EQ(Style.QualifierOrder.size(), (size_t)5);

  verifyFormat("static inline int const volatile a;",
               "const inline static volatile int a;", Style);
  verifyFormat("static inline int const volatile a;",
               "volatile inline static const int a;", Style);
  verifyFormat("static inline int const volatile a;",
               "int const inline static  volatile a;", Style);
  verifyFormat("static inline int const volatile a;",
               "int volatile inline static  const a;", Style);
  verifyFormat("static inline int const volatile a;",
               "const int inline static  volatile a;", Style);

  verifyFormat("static inline int const volatile *const a;",
               "const int inline static  volatile *const a;", Style);

  verifyFormat("static inline Foo const volatile a;",
               "const inline static volatile Foo a;", Style);
  verifyFormat("static inline Foo const volatile a;",
               "volatile inline static const Foo a;", Style);
  verifyFormat("static inline Foo const volatile a;",
               "Foo const inline static  volatile a;", Style);
  verifyFormat("static inline Foo const volatile a;",
               "Foo volatile inline static  const a;", Style);
  verifyFormat("static inline Foo const volatile a;",
               "const Foo inline static  volatile a;", Style);

  verifyFormat("static inline Foo const volatile *const a;",
               "const Foo inline static volatile *const a;", Style);
}

TEST_F(QualifierFixerTest, PrepareLeftRightOrdering) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "inline", "type", "const", "volatile"};

  std::vector<std::string> Left;
  std::vector<std::string> Right;
  std::vector<tok::TokenKind> ConfiguredTokens;
  prepareLeftRightOrderingForQualifierAlignmentFixer(Style.QualifierOrder, Left,
                                                     Right, ConfiguredTokens);

  EXPECT_EQ(Left.size(), (size_t)2);
  EXPECT_EQ(Right.size(), (size_t)2);

  std::vector<std::string> LeftResult = {"inline", "static"};
  std::vector<std::string> RightResult = {"const", "volatile"};
  EXPECT_EQ(Left, LeftResult);
  EXPECT_EQ(Right, RightResult);
}

TEST_F(QualifierFixerTest, IsQualifierType) {

  std::vector<tok::TokenKind> ConfiguredTokens;
  ConfiguredTokens.push_back(tok::kw_const);
  ConfiguredTokens.push_back(tok::kw_static);
  ConfiguredTokens.push_back(tok::kw_inline);
  ConfiguredTokens.push_back(tok::kw_restrict);
  ConfiguredTokens.push_back(tok::kw_constexpr);
  ConfiguredTokens.push_back(tok::kw_friend);

  TestLexer lexer{Allocator, Buffers};
  const auto LangOpts = getFormattingLangOpts();

  auto Tokens = lexer.lex(
      "const static inline auto restrict int double long constexpr friend");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;

  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[0], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[1], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[2], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[3], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[4], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[5], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[6], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[7], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[8], ConfiguredTokens, LangOpts));
  EXPECT_TRUE(
      isConfiguredQualifierOrType(Tokens[9], ConfiguredTokens, LangOpts));

  EXPECT_TRUE(isQualifierOrType(Tokens[0], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[1], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[2], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[3], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[4], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[5], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[6], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[7], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[8], LangOpts));
  EXPECT_TRUE(isQualifierOrType(Tokens[9], LangOpts));

  auto NotTokens = lexer.lex("for while do Foo Bar ");
  ASSERT_EQ(NotTokens.size(), 6u) << Tokens;

  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[0], ConfiguredTokens, LangOpts));
  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[1], ConfiguredTokens, LangOpts));
  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[2], ConfiguredTokens, LangOpts));
  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[3], ConfiguredTokens, LangOpts));
  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[4], ConfiguredTokens, LangOpts));
  EXPECT_FALSE(
      isConfiguredQualifierOrType(NotTokens[5], ConfiguredTokens, LangOpts));

  EXPECT_FALSE(isQualifierOrType(NotTokens[0], LangOpts));
  EXPECT_FALSE(isQualifierOrType(NotTokens[1], LangOpts));
  EXPECT_FALSE(isQualifierOrType(NotTokens[2], LangOpts));
  EXPECT_FALSE(isQualifierOrType(NotTokens[3], LangOpts));
  EXPECT_FALSE(isQualifierOrType(NotTokens[4], LangOpts));
  EXPECT_FALSE(isQualifierOrType(NotTokens[5], LangOpts));
}

TEST_F(QualifierFixerTest, IsMacro) {

  auto Tokens = annotate("INT INTPR Foo int");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;

  EXPECT_TRUE(isPossibleMacro(Tokens[0]));
  EXPECT_TRUE(isPossibleMacro(Tokens[1]));
  EXPECT_FALSE(isPossibleMacro(Tokens[2]));
  EXPECT_FALSE(isPossibleMacro(Tokens[3]));
}

TEST_F(QualifierFixerTest, OverlappingQualifier) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("Foo(const Bar &name);", "Foo(Bar const &name);", Style);
}

TEST_F(QualifierFixerTest, DontPushQualifierThroughNonSpecifiedTypes) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "volatile", "type"};

  verifyFormat("inline static const int a;", Style);

  Style.QualifierOrder = {"static", "const", "type"};

  verifyFormat("inline static const int a;", Style);
  verifyFormat("static inline const int a;", Style);

  verifyFormat("static const int a;", "const static int a;", Style);

  Style.QualifierOrder = {"const", "volatile", "type"};
  // static is not configured, unchanged at right hand qualifiers.
  verifyFormat("const volatile int static;", "int volatile static const;",
               Style);
  verifyFormat("const volatile int static;", "int const static volatile;",
               Style);
  verifyFormat("const volatile int static;", "const int static volatile;",
               Style);
  verifyFormat("const volatile Foo static;", "Foo volatile static const;",
               Style);
  verifyFormat("const volatile Foo static;", "Foo const static volatile;",
               Style);
  verifyFormat("const volatile Foo static;", "const Foo static volatile;",
               Style);

  verifyFormat("inline static const Foo;", "inline static Foo const;", Style);
  verifyFormat("inline static const Foo;", Style);

  // Don't move qualifiers to the right for aestethics only.
  verifyFormat("inline const static Foo;", Style);
  verifyFormat("const inline static Foo;", Style);
}

TEST_F(QualifierFixerTest, QualifiersBrokenUpByPPDirectives) {
  auto Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"constexpr", "inline", "type"};

  verifyFormat("inline\n"
               "#if FOO\n"
               "    constexpr\n"
               "#endif\n"
               "    int i = 0;",
               Style);
}

TEST_F(QualifierFixerTest, UnsignedQualifier) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Left;
  Style.QualifierOrder = {"const", "type"};

  verifyFormat("Foo(const unsigned char *bytes)",
               "Foo(unsigned const char *bytes)", Style);

  Style.QualifierAlignment = FormatStyle::QAS_Right;
  Style.QualifierOrder = {"type", "const"};

  verifyFormat("Foo(unsigned char const *bytes)",
               "Foo(unsigned const char *bytes)", Style);
}

TEST_F(QualifierFixerTest, NoOpQualifierReplacements) {

  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "const", "type"};

  verifyFormat("static const uint32 foo[] = {0, 31};", Style);
  EXPECT_EQ(ReplacementCount, 0);

  verifyFormat("#define MACRO static const", Style);
  EXPECT_EQ(ReplacementCount, 0);

  verifyFormat("using sc = static const", Style);
  EXPECT_EQ(ReplacementCount, 0);
}

TEST_F(QualifierFixerTest, QualifierTemplates) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"static", "const", "type"};

  ReplacementCount = 0;
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("using A = B<>;", Style);
  verifyFormat("using A = B /**/<>;", Style);
  verifyFormat("template <class C> using A = B<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /* */<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /*foo*/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/ /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B<Foo</**/ C>, 1>;", Style);
  verifyFormat("template <class C> using A = /**/ B<Foo<C>, 1>;", Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("template <class C>\n"
               "using A = B // foo\n"
               "    <Foo<C>, 1>;",
               Style);

  ReplacementCount = 0;
  Style.QualifierOrder = {"type", "static", "const"};
  verifyFormat("using A = B<>;", Style);
  verifyFormat("using A = B /**/<>;", Style);
  verifyFormat("template <class C> using A = B<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /* */<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /*foo*/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B /**/ /**/<Foo<C>, 1>;", Style);
  verifyFormat("template <class C> using A = B<Foo</**/ C>, 1>;", Style);
  verifyFormat("template <class C> using A = /**/ B<Foo<C>, 1>;", Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("template <class C>\n"
               "using A = B // foo\n"
               "    <Foo<C>, 1>;",
               Style);
}

TEST_F(QualifierFixerTest, WithConstraints) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"constexpr", "type"};

  verifyFormat("template <typename T>\n"
               "  requires Concept<F>\n"
               "constexpr constructor();",
               Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept1<F> && Concept2<F>\n"
               "constexpr constructor();",
               Style);
}

TEST_F(QualifierFixerTest, DisableRegions) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"inline", "static", "const", "type"};

  ReplacementCount = 0;
  verifyFormat("// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on",
               Style);
  EXPECT_EQ(ReplacementCount, 0);
  verifyFormat("// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on\n"
               "inline static const int a = 0;",
               "// clang-format off\n"
               "int const inline static a = 0;\n"
               "// clang-format on\n"
               "int const inline static a = 0;",
               Style);
}

TEST_F(QualifierFixerTest, TemplatesRight) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"type", "const"};

  verifyFormat("template <typename T> Foo const f();",
               "template <typename T> const Foo f();", Style);
  verifyFormat("template <typename T> int const f();",
               "template <typename T> const int f();", Style);

  verifyFormat("template <T const> t;", "template <const T> t;", Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept<T const>\n"
               "Foo const f();",
               "template <typename T>\n"
               "  requires Concept<const T>\n"
               "const Foo f();",
               Style);
  verifyFormat("TemplateType<T const> t;", "TemplateType<const T> t;", Style);
  verifyFormat("TemplateType<Container const> t;",
               "TemplateType<const Container> t;", Style);
}

TEST_F(QualifierFixerTest, TemplatesLeft) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"const", "volatile", "type"};

  verifyFormat("template <typename T> const Foo f();",
               "template <typename T> Foo const f();", Style);
  verifyFormat("template <typename T> const int f();",
               "template <typename T> int const f();", Style);

  verifyFormat("template <const T> t;", "template <T const> t;", Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept<const T>\n"
               "const Foo f();",
               "template <typename T>\n"
               "  requires Concept<T const>\n"
               "Foo const f();",
               Style);
  verifyFormat("template <typename T>\n"
               "  requires Concept<const T>\n"
               "const volatile Foo f();",
               "template <typename T>\n"
               "  requires Concept<T const>\n"
               "volatile const Foo f();",
               Style);
  verifyFormat("TemplateType<const T> t;", "TemplateType<T const> t;", Style);
  verifyFormat("TemplateType<const Container> t;",
               "TemplateType<Container const> t;", Style);
}

TEST_F(QualifierFixerTest, Ranges) {
  FormatStyle Style = getLLVMStyle();
  Style.QualifierAlignment = FormatStyle::QAS_Custom;
  Style.QualifierOrder = {"const", "volatile", "type"};

  // Only the first line should be formatted; the second should remain as is.
  verifyFormat("template <typename T> const Foo f();\n"
               "template <typename T> Foo const f();",
               "template <typename T> Foo const f();\n"
               "template <typename T> Foo const f();",
               Style, {tooling::Range(0, 36)});

  // Only the middle line should be formatted; the first and last should remain
  // as is.
  verifyFormat("template <typename T> Foo const f();\n"
               "template <typename T> const Foo f();\n"
               "template <typename T> Foo const f();",
               "template <typename T> Foo const f();\n"
               "template <typename T> Foo const f();\n"
               "template <typename T> Foo const f();",
               Style, {tooling::Range(37, 36)});
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang

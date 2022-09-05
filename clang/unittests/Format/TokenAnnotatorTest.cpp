//===- unittest/Format/TokenAnnotatorTest.cpp - Formatting unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "FormatTestUtils.h"
#include "TestLexer.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {

// Not really the equality, but everything we need.
static bool operator==(const FormatToken &LHS,
                       const FormatToken &RHS) noexcept {
  return LHS.Tok.getKind() == RHS.Tok.getKind() &&
         LHS.getType() == RHS.getType();
}

namespace {

class TokenAnnotatorTest : public ::testing::Test {
protected:
  TokenList annotate(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).annotate(Code);
  }
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
};

#define EXPECT_TOKEN_KIND(FormatTok, Kind)                                     \
  EXPECT_EQ((FormatTok)->Tok.getKind(), Kind) << *(FormatTok)
#define EXPECT_TOKEN_TYPE(FormatTok, Type)                                     \
  EXPECT_EQ((FormatTok)->getType(), Type) << *(FormatTok)
#define EXPECT_TOKEN_PRECEDENCE(FormatTok, Prec)                               \
  EXPECT_EQ((FormatTok)->getPrecedence(), Prec) << *(FormatTok)
#define EXPECT_TOKEN(FormatTok, Kind, Type)                                    \
  do {                                                                         \
    EXPECT_TOKEN_KIND(FormatTok, Kind);                                        \
    EXPECT_TOKEN_TYPE(FormatTok, Type);                                        \
  } while (false)

TEST_F(TokenAnnotatorTest, UnderstandsUsesOfStarAndAmp) {
  auto Tokens = annotate("auto x = [](const decltype(x) &ptr) {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);

  Tokens = annotate("auto x = [](const decltype(x) *ptr) {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);

  Tokens = annotate("#define lambda [](const decltype(x) &ptr) {}");
  EXPECT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);

  Tokens = annotate("#define lambda [](const decltype(x) *ptr) {}");
  EXPECT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);

  Tokens = annotate("void f() {\n"
                    "  while (p < a && *p == 'a')\n"
                    "    p++;\n"
                    "}");
  EXPECT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_UnaryOperator);

  Tokens = annotate("case *x:");
  EXPECT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::star, TT_UnaryOperator);
  Tokens = annotate("case &x:");
  EXPECT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_UnaryOperator);

  Tokens = annotate("bool b = 3 == int{3} && true;\n");
  EXPECT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("struct {\n"
                    "} *ptr;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);
  Tokens = annotate("union {\n"
                    "} *ptr;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);
  Tokens = annotate("class {\n"
                    "} *ptr;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);

  Tokens = annotate("struct {\n"
                    "} &&ptr = {};");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("union {\n"
                    "} &&ptr = {};");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("class {\n"
                    "} &&ptr = {};");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("int i = int{42} * 2;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::star, TT_BinaryOperator);

  Tokens = annotate("delete[] *ptr;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);
  Tokens = annotate("delete[] **ptr;");
  EXPECT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[4], tok::star, TT_UnaryOperator);
  Tokens = annotate("delete[] *(ptr);");
  EXPECT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsUsesOfPlusAndMinus) {
  auto Tokens = annotate("x - 0");
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_BinaryOperator);
  Tokens = annotate("0 + 0");
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_BinaryOperator);
  Tokens = annotate("x + +0");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::plus, TT_UnaryOperator);
  Tokens = annotate("x ? -0 : +0");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::minus, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[5], tok::plus, TT_UnaryOperator);
  Tokens = annotate("(-0)");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("0, -0");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::minus, TT_UnaryOperator);
  Tokens = annotate("for (; -1;) {\n}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::minus, TT_UnaryOperator);
  Tokens = annotate("x = -1;");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::minus, TT_UnaryOperator);
  Tokens = annotate("x[-1]");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::minus, TT_UnaryOperator);
  Tokens = annotate("x = {-1};");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::minus, TT_UnaryOperator);
  Tokens = annotate("case -x:");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("co_await -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("co_return -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("co_yield -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("delete -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("return -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("throw -x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("sizeof -x");
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("co_await +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("co_return +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("co_yield +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("delete +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("return +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("throw +x;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("sizeof +x");
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
  Tokens = annotate("(int)-x");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::minus, TT_UnaryOperator);
  Tokens = annotate("(-x)");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::minus, TT_UnaryOperator);
  Tokens = annotate("!+x");
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::exclaim, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[1], tok::plus, TT_UnaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsClasses) {
  auto Tokens = annotate("class C {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_ClassLBrace);

  Tokens = annotate("const class C {} c;");
  EXPECT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::l_brace, TT_ClassLBrace);

  Tokens = annotate("const class {} c;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_ClassLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsStructs) {
  auto Tokens = annotate("struct S {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_StructLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsUnions) {
  auto Tokens = annotate("union U {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_UnionLBrace);

  Tokens = annotate("union U { void f() { return; } };");
  EXPECT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_UnionLBrace);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_FunctionLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsEnums) {
  auto Tokens = annotate("enum E {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_EnumLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsDefaultedAndDeletedFunctions) {
  auto Tokens = annotate("auto operator<=>(const T &) const & = default;");
  EXPECT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void F(T) && = delete;");
  EXPECT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_PointerOrReference);
}

TEST_F(TokenAnnotatorTest, UnderstandsVariables) {
  auto Tokens =
      annotate("inline bool var = is_integral_v<int> && is_signed_v<int>;");
  EXPECT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::ampamp, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsVariableTemplates) {
  auto Tokens =
      annotate("template <typename T> "
               "inline bool var = is_integral_v<int> && is_signed_v<int>;");
  EXPECT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsWhitespaceSensitiveMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.WhitespaceSensitiveMacros.push_back("FOO");

  auto Tokens = annotate("FOO(1+2 )\n", Style);
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_UntouchableMacroFunc);

  Tokens = annotate("FOO(a:b:c)\n", Style);
  EXPECT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_UntouchableMacroFunc);
}

TEST_F(TokenAnnotatorTest, UnderstandsDelete) {
  auto Tokens = annotate("delete (void *)p;");
  EXPECT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] (void *)p;");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] /*comment*/ (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[/*comment*/] (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete/*comment*/[] (void *)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsFunctionRefQualifiers) {
  auto Tokens = annotate("void f() &;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_PointerOrReference);

  Tokens = annotate("void operator=(T) &&;");
  EXPECT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void f() &;");
  EXPECT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void operator=(T) &;");
  EXPECT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);
}

TEST_F(TokenAnnotatorTest, UnderstandsRequiresClausesAndConcepts) {
  auto Tokens = annotate("template <typename T>\n"
                         "concept C = (Foo && Bar) && (Bar && Baz);");

  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[16], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T>\n"
                    "concept C = requires(T t) {\n"
                    "  { t.foo() };\n"
                    "} && Bar<T> && Baz<T>;");
  ASSERT_EQ(Tokens.size(), 35u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[9], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_RequiresExpressionLBrace);
  EXPECT_TOKEN(Tokens[23], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[28], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template<typename T>\n"
                    "requires C1<T> && (C21<T> || C22<T> && C2e<T>) && C3<T>\n"
                    "struct Foo;");
  ASSERT_EQ(Tokens.size(), 36u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_Unknown);
  EXPECT_EQ(Tokens[6]->FakeLParens.size(), 1u);
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[16], tok::pipepipe, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[21], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[27], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[31], tok::greater, TT_TemplateCloser);
  EXPECT_EQ(Tokens[31]->FakeRParens, 1u);
  EXPECT_TRUE(Tokens[31]->ClosesRequiresClause);

  Tokens =
      annotate("template<typename T>\n"
               "requires (C1<T> && (C21<T> || C22<T> && C2e<T>) && C3<T>)\n"
               "struct Foo;");
  ASSERT_EQ(Tokens.size(), 38u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_Unknown);
  EXPECT_EQ(Tokens[7]->FakeLParens.size(), 1u);
  EXPECT_TOKEN(Tokens[11], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[17], tok::pipepipe, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[22], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[28], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[32], tok::greater, TT_TemplateCloser);
  EXPECT_EQ(Tokens[32]->FakeRParens, 1u);
  EXPECT_TOKEN(Tokens[33], tok::r_paren, TT_Unknown);
  EXPECT_TRUE(Tokens[33]->ClosesRequiresClause);

  Tokens = annotate("template <typename T>\n"
                    "void foo(T) noexcept requires Bar<T>;");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[11], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("template <typename T>\n"
                    "struct S {\n"
                    "  void foo() const requires Bar<T>;\n"
                    "  void bar() const & requires Baz<T>;\n"
                    "  void bar() && requires Baz2<T>;\n"
                    "  void baz() const & noexcept requires Baz<T>;\n"
                    "  void baz() && noexcept requires Baz2<T>;\n"
                    "};\n"
                    "\n"
                    "void S::bar() const & requires Baz<T> { }");
  ASSERT_EQ(Tokens.size(), 85u) << Tokens;
  EXPECT_TOKEN(Tokens[13], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[25], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[36], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[49], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[61], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[77], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("void Class::member() && requires(Constant) {}");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("void Class::member() && requires(Constant<T>) {}");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_requires, TT_RequiresClause);

  Tokens =
      annotate("void Class::member() && requires(Namespace::Constant<T>) {}");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("void Class::member() && requires(typename "
                    "Namespace::Outer<T>::Inner::Constant) {}");
  ASSERT_EQ(Tokens.size(), 24u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("struct [[nodiscard]] zero_t {\n"
                    "  template<class T>\n"
                    "    requires requires { number_zero_v<T>; }\n"
                    "  [[nodiscard]] constexpr operator T() const { "
                    "return number_zero_v<T>; }\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 44u);
  EXPECT_TOKEN(Tokens[13], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[14], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[15], tok::l_brace, TT_RequiresExpressionLBrace);
  EXPECT_TOKEN(Tokens[21], tok::r_brace, TT_Unknown);
  EXPECT_EQ(Tokens[21]->MatchingParen, Tokens[15]);
  EXPECT_TRUE(Tokens[21]->ClosesRequiresClause);

  Tokens =
      annotate("template <class A, class B> concept C ="
               "std::same_as<std::iter_value_t<A>, std::iter_value_t<B>>;");
  ASSERT_EQ(Tokens.size(), 31u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_concept, TT_Unknown);
  EXPECT_TOKEN(Tokens[14], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[18], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[20], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[25], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[27], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[28], tok::greater, TT_TemplateCloser);

  Tokens = annotate("auto bar() -> int requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("auto bar() -> void requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("auto bar() -> MyType requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);

  Tokens =
      annotate("auto bar() -> SOME_MACRO_TYPE requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);

  Tokens =
      annotate("auto bar() -> qualified::type requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_requires, TT_RequiresClause);

  Tokens =
      annotate("auto bar() -> Template<type> requires(is_integral_v<T>) {}");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::kw_requires, TT_RequiresClause);
}

TEST_F(TokenAnnotatorTest, UnderstandsRequiresExpressions) {
  auto Tokens = annotate("bool b = requires(int i) { i + 5; };");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("if (requires(int i) { i + 5; }) return;");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("if (func() && requires(int i) { i + 5; }) return;");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(const T t) {});");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(const int t) {});");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(const T t) {});");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(int const* volatile t) {});");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(T const* volatile t) {});");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens =
      annotate("foo(requires(const typename Outer<T>::Inner * const t) {});");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[16], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("template <typename T>\n"
                    "concept C = requires(T T) {\n"
                    "  requires Bar<T> && Foo<T>;\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 28u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[9], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_RequiresExpressionLBrace);
  EXPECT_TOKEN(Tokens[14], tok::kw_requires,
               TT_RequiresClauseInARequiresExpression);

  Tokens = annotate("template <typename T>\n"
                    "concept C = requires(T T) {\n"
                    "  { t.func() } -> std::same_as<int>;"
                    "  requires Bar<T> && Foo<T>;\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 43u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[9], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_RequiresExpressionLBrace);
  EXPECT_TOKEN(Tokens[29], tok::kw_requires,
               TT_RequiresClauseInARequiresExpression);

  // Invalid Code, but we don't want to crash. See http://llvm.org/PR54350.
  Tokens = annotate("bool r10 = requires (struct new_struct { int x; } s) { "
                    "requires true; };");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_RequiresExpressionLBrace);
}

TEST_F(TokenAnnotatorTest, RequiresDoesNotChangeParsingOfTheRest) {
  const char *BaseCode = nullptr;
  const char *ConstrainedCode = nullptr;
  auto BaseTokenCount = 0u;
  auto RequiresTokenCount = 0u;
  auto PrefixTokenCount = 0u;

  auto TestRequires = [&](int Line) {
    const auto BaseTokens = annotate(BaseCode);
    const auto ConstrainedTokens = annotate(ConstrainedCode);

#define LINE " (Line " << Line << ')'

    ASSERT_EQ(BaseTokens.size(), BaseTokenCount) << BaseTokens << LINE;
    ASSERT_EQ(ConstrainedTokens.size(), BaseTokenCount + RequiresTokenCount)
        << LINE;

    for (auto I = 0u; I < BaseTokenCount; ++I) {
      EXPECT_EQ(
          *BaseTokens[I],
          *ConstrainedTokens[I < PrefixTokenCount ? I : I + RequiresTokenCount])
          << I << LINE;
    }

#undef LINE
  };

  BaseCode = "template<typename T>\n"
             "T Pi = 3.14;";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "T Pi = 3.14;";
  BaseTokenCount = 11;
  RequiresTokenCount = 5;
  PrefixTokenCount = 5;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "struct Bar;";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "struct Bar;";
  BaseTokenCount = 9;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "struct Bar {\n"
             "  T foo();\n"
             "  T bar();\n"
             "};";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "struct Bar {\n"
                    "  T foo();\n"
                    "  T bar();\n"
                    "};";
  BaseTokenCount = 21;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "Bar(T) -> Bar<T>;";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "Bar(T) -> Bar<T>;";
  BaseTokenCount = 16;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "T foo();";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "T foo();";
  BaseTokenCount = 11;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "T foo() {\n"
             "  auto bar = baz();\n"
             "  return bar + T{};\n"
             "}";
  ConstrainedCode = "template<typename T>\n"
                    "  requires Foo<T>\n"
                    "T foo() {\n"
                    "  auto bar = baz();\n"
                    "  return bar + T{};\n"
                    "}";
  BaseTokenCount = 26;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "T foo();";
  ConstrainedCode = "template<typename T>\n"
                    "T foo() requires Foo<T>;";
  BaseTokenCount = 11;
  PrefixTokenCount = 9;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "T foo() {\n"
             "  auto bar = baz();\n"
             "  return bar + T{};\n"
             "}";
  ConstrainedCode = "template<typename T>\n"
                    "T foo() requires Foo<T> {\n"
                    "  auto bar = baz();\n"
                    "  return bar + T{};\n"
                    "}";
  BaseTokenCount = 26;
  TestRequires(__LINE__);

  BaseCode = "template<typename T>\n"
             "Bar(T) -> Bar<typename T::I>;";
  ConstrainedCode = "template<typename T>\n"
                    "  requires requires(T &&t) {\n"
                    "             typename T::I;\n"
                    "           }\n"
                    "Bar(T) -> Bar<typename T::I>;";
  BaseTokenCount = 19;
  RequiresTokenCount = 14;
  PrefixTokenCount = 5;
  TestRequires(__LINE__);

  BaseCode = "struct [[nodiscard]] zero_t {\n"
             "  template<class T>\n"
             "  [[nodiscard]] constexpr operator T() const { return v<T>; }\n"
             "};";
  ConstrainedCode =
      "struct [[nodiscard]] zero_t {\n"
      "  template<class T>\n"
      "    requires requires { v<T>; }\n"
      "  [[nodiscard]] constexpr operator T() const { return v<T>; }\n"
      "};";
  BaseTokenCount = 35;
  RequiresTokenCount = 9;
  PrefixTokenCount = 13;
  TestRequires(__LINE__);

  BaseCode = "constexpr Foo(Foo const &other)\n"
             "    : value{other.value} {\n"
             "  do_magic();\n"
             "  do_more_magic();\n"
             "}";
  ConstrainedCode = "constexpr Foo(Foo const &other)\n"
                    "  requires std::is_copy_constructible<T>\n"
                    "    : value{other.value} {\n"
                    "  do_magic();\n"
                    "  do_more_magic();\n"
                    "}";
  BaseTokenCount = 26;
  RequiresTokenCount = 7;
  PrefixTokenCount = 8;
  TestRequires(__LINE__);

  BaseCode = "constexpr Foo(Foo const &other)\n"
             "    : value{other.value} {\n"
             "  do_magic();\n"
             "  do_more_magic();\n"
             "}";
  ConstrainedCode = "constexpr Foo(Foo const &other)\n"
                    "  requires (std::is_copy_constructible<T>)\n"
                    "    : value{other.value} {\n"
                    "  do_magic();\n"
                    "  do_more_magic();\n"
                    "}";
  RequiresTokenCount = 9;
  TestRequires(__LINE__);
}

TEST_F(TokenAnnotatorTest, UnderstandsAsm) {
  auto Tokens = annotate("__asm{\n"
                         "a:\n"
                         "};");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::kw_asm, TT_Unknown);
  EXPECT_TOKEN(Tokens[1], tok::l_brace, TT_InlineASMBrace);
  EXPECT_TOKEN(Tokens[4], tok::r_brace, TT_InlineASMBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsObjCBlock) {
  auto Tokens = annotate("int (^)() = ^ ()\n"
                         "  external_source_symbol() { //\n"
                         "  return 1;\n"
                         "};");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ObjCBlockLParen);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_ObjCBlockLBrace);

  Tokens = annotate("int *p = ^int*(){ //\n"
                    "  return nullptr;\n"
                    "}();");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_ObjCBlockLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsLambdas) {
  auto Tokens = annotate("[]() constexpr {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto {}");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto & {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto * {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsFunctionAnnotations) {
  auto Tokens = annotate("template <typename T>\n"
                         "DEPRECATED(\"Use NewClass::NewFunction instead.\")\n"
                         "string OldFunction(const string &parameter) {}");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_FunctionAnnotationRParen);

  Tokens = annotate("template <typename T>\n"
                    "A(T) noexcept;");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_Unknown);
}

TEST_F(TokenAnnotatorTest, UnderstandsVerilogOperators) {
  auto Annotate = [this](llvm::StringRef Code) {
    return annotate(Code, getLLVMStyle(FormatStyle::LK_Verilog));
  };
  // Test that unary operators get labeled as such and that operators like '++'
  // don't get split.
  tok::TokenKind Unary[] = {tok::plus,  tok::minus,    tok::exclaim,
                            tok::tilde, tok::amp,      tok::pipe,
                            tok::caret, tok::plusplus, tok::minusminus};
  for (auto Kind : Unary) {
    auto Tokens =
        Annotate(std::string("x = ") + tok::getPunctuatorSpelling(Kind) + "x;");
    ASSERT_EQ(Tokens.size(), 6u) << Tokens;
    EXPECT_TOKEN(Tokens[2], Kind, TT_UnaryOperator);
  }
  // Operators formed by joining two operators like '^~'. For some of these
  // joined operators, we don't have a separate type, so we only test for their
  // precedence.
  std::pair<prec::Level, std::string> JoinedBinary[] = {
      {prec::Comma, "<->"},       {prec::Assignment, "+="},
      {prec::Assignment, "-="},   {prec::Assignment, "*="},
      {prec::Assignment, "/="},   {prec::Assignment, "%="},
      {prec::Assignment, "&="},   {prec::Assignment, "^="},
      {prec::Assignment, "<<="},  {prec::Assignment, ">>="},
      {prec::Assignment, "<<<="}, {prec::Assignment, ">>>="},
      {prec::LogicalOr, "||"},    {prec::LogicalAnd, "&&"},
      {prec::Equality, "=="},     {prec::Equality, "!="},
      {prec::Equality, "==="},    {prec::Equality, "!=="},
      {prec::Equality, "==?"},    {prec::Equality, "!=?"},
      {prec::ExclusiveOr, "~^"},  {prec::ExclusiveOr, "^~"},
  };
  for (auto Operator : JoinedBinary) {
    auto Tokens = Annotate(std::string("x = x ") + Operator.second + " x;");
    ASSERT_EQ(Tokens.size(), 7u) << Tokens;
    EXPECT_TOKEN_TYPE(Tokens[3], TT_BinaryOperator);
    EXPECT_TOKEN_PRECEDENCE(Tokens[3], Operator.first);
  }
  // '~^' and '^~' can be unary as well as binary operators.
  auto Tokens = Annotate("x = ~^x;");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN_TYPE(Tokens[2], TT_UnaryOperator);
  Tokens = Annotate("x = ^~x;");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN_TYPE(Tokens[2], TT_UnaryOperator);
  // The unary operators '~&' and '~|' can only be unary operators. The current
  // implementation treats each of them as separate unary '~' and '&' or '|'
  // operators, which is enough for formatting purposes. In FormatTestVerilog,
  // there is a test that there is no space in between. And even if a new line
  // is inserted between the '~' and '|', the semantic meaning is the same as
  // the joined operator, so the CanBreakBefore property doesn't need to be
  // false for the second operator.
  Tokens = Annotate("x = ~&x;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::tilde, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[3], tok::amp, TT_UnaryOperator);
  Tokens = Annotate("x = ~|x;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::tilde, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[3], tok::pipe, TT_UnaryOperator);
  // Test for block label colons.
  Tokens = Annotate("begin : x\n"
                    "end : x");
  ASSERT_EQ(Tokens.size(), 7u);
  EXPECT_TOKEN(Tokens[1], tok::colon, TT_VerilogBlockLabelColon);
  EXPECT_TOKEN(Tokens[4], tok::colon, TT_VerilogBlockLabelColon);
  // Test that the dimension colon is annotated correctly.
  Tokens = Annotate("var [1 : 0] x;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::colon, TT_BitFieldColon);
  Tokens = Annotate("extern function [1 : 0] x;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::colon, TT_BitFieldColon);
  // Test case labels and ternary operators.
  Tokens = Annotate("case (x)\n"
                    "  x:\n"
                    "    x;\n"
                    "endcase\n");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_GotoLabelColon);
  Tokens = Annotate("case (x)\n"
                    "  x ? x : x:\n"
                    "    x;\n"
                    "endcase\n");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::question, TT_ConditionalExpr);
  EXPECT_TOKEN(Tokens[7], tok::colon, TT_ConditionalExpr);
  EXPECT_TOKEN(Tokens[9], tok::colon, TT_GotoLabelColon);
}

} // namespace
} // namespace format
} // namespace clang

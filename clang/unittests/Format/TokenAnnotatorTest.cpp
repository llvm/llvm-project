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

  Tokens = annotate("void f() { void (*fnptr)(char* foo); }");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_FunctionTypeLParen);
  // FIXME: The star of a function pointer probably makes more sense as
  // TT_PointerOrReference.
  EXPECT_TOKEN(Tokens[7], tok::star, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[12], tok::star, TT_PointerOrReference);

  Tokens = annotate("void f() { void (*fnptr)(t* foo); }");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_FunctionTypeLParen);
  EXPECT_TOKEN(Tokens[7], tok::star, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[12], tok::star, TT_PointerOrReference);

  Tokens = annotate("int f3() { return sizeof(Foo&); }");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);

  Tokens = annotate("int f4() { return sizeof(Foo&&); }");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("void f5() { int f6(Foo&, Bar&); }");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[12], tok::amp, TT_PointerOrReference);

  Tokens = annotate("void f7() { int f8(Foo&&, Bar&&); }");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[12], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("Type1 &val1 = val2;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_PointerOrReference);

  Tokens = annotate("Type1 *val1 = &val2;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::star, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_UnaryOperator);

  Tokens = annotate("val1 & val2;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1 & val2.member;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1 & val2.*member;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1.*member & val2;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1 & val2->*member;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1->member & val2;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1 & val2 & val3;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[3], tok::amp, TT_BinaryOperator);

  Tokens = annotate("val1 & val2 // comment\n"
                    "     & val3;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_BinaryOperator);

  Tokens =
      annotate("val1 & val2.member & val3.member() & val4 & val5->member;");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[5], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[13], tok::amp, TT_BinaryOperator);

  Tokens = annotate("class c {\n"
                    "  void func(type &a) { a & member; }\n"
                    "  anotherType &member;\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 22u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[12], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[17], tok::amp, TT_PointerOrReference);

  Tokens = annotate("struct S {\n"
                    "  auto Mem = C & D;\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::amp, TT_BinaryOperator);

  Tokens =
      annotate("template <typename T> void swap() noexcept(Bar<T> && Foo<T>);");
  ASSERT_EQ(Tokens.size(), 23u) << Tokens;
  EXPECT_TOKEN(Tokens[15], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T> struct S {\n"
                    "  explicit(Bar<T> && Foo<T>) S(const S &);\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 30u) << Tokens;
  EXPECT_TOKEN(Tokens[14], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <bool B = C && D> struct S {};");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T, bool B = C && D> struct S {};");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T, typename U = T&&> struct S {};");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("template <typename T = int (*)(int)> struct S {};");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_FunctionTypeLParen);
  EXPECT_TOKEN(Tokens[7], tok::star, TT_PointerOrReference);

  Tokens = annotate("Foo<A && B> a = {};");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("Foo<A &&> a = {};");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("template <enable_if_t<foo && !bar>* = nullptr> void f();");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::ampamp, TT_BinaryOperator);
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

  Tokens = annotate("class [[deprecated(\"\")]] C { int i; };");
  EXPECT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_ClassLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsStructs) {
  auto Tokens = annotate("struct S {};");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_StructLBrace);

  Tokens = annotate("struct EXPORT_MACRO [[nodiscard]] C { int i; };");
  EXPECT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_StructLBrace);

  Tokens = annotate("struct [[deprecated]] [[nodiscard]] C { int i; };");
  EXPECT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_StructLBrace);

  Tokens = annotate("template <typename T> struct S<const T[N]> {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[13], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_StructLBrace);

  Tokens = annotate("template <typename T> struct S<T const[N]> {};");
  EXPECT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[13], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_StructLBrace);

  Tokens = annotate("template <typename T, unsigned n> struct S<T const[n]> {\n"
                    "  void f(T const (&a)[n]);\n"
                    "};");
  EXPECT_EQ(Tokens.size(), 35u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[13], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[16], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[17], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[23], tok::l_paren, TT_FunctionTypeLParen);
  EXPECT_TOKEN(Tokens[24], tok::amp, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[27], tok::l_square, TT_ArraySubscriptLSquare);
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

TEST_F(TokenAnnotatorTest, UnderstandsTemplatesInMacros) {
  auto Tokens =
      annotate("#define FOO(typeName) \\\n"
               "  { #typeName, foo<FooType>(new foo<realClass>(#typeName)) }");
  ASSERT_EQ(Tokens.size(), 27u) << Tokens;
  EXPECT_TOKEN(Tokens[11], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[13], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[17], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[19], tok::greater, TT_TemplateCloser);
}

TEST_F(TokenAnnotatorTest, UnderstandsGreaterAfterTemplateCloser) {
  auto Tokens = annotate("if (std::tuple_size_v<T> > 0)");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[8], tok::greater, TT_BinaryOperator);
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

TEST_F(TokenAnnotatorTest, UnderstandsCasts) {
  auto Tokens = annotate("(void)p;");
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::r_paren, TT_CastRParen);

  Tokens = annotate("auto x = (Foo)p;");
  EXPECT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_CastRParen);

  Tokens = annotate("(std::vector<int>)p;");
  EXPECT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("return (Foo)p;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_CastRParen);

  Tokens = annotate("throw (Foo)p;");
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_CastRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsDynamicExceptionSpecifier) {
  auto Tokens = annotate("void f() throw(int);");
  EXPECT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_throw, TT_Unknown);
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

TEST_F(TokenAnnotatorTest, UnderstandsOverloadedOperators) {
  auto Tokens = annotate("x.operator+()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator=()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::equal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator+=()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::plusequal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator,()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::comma, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator()()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::r_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator[]()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  // EXPECT_TOKEN(Tokens[3], tok::l_square, TT_OverloadedOperator);
  // EXPECT_TOKEN(Tokens[4], tok::r_square, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"_a()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\" _a()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  // FIXME
  // EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"if()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"s()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\" s()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  // FIXME
  // EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);

  Tokens = annotate("int operator+(int);");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[2], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("auto operator=(T&) {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[2], tok::equal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("auto operator()() {}");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
}

TEST_F(TokenAnnotatorTest, OverloadedOperatorInTemplate) {
  struct {
    const char *Text;
    tok::TokenKind Kind;
  } Operators[] = {{"+", tok::plus},
                   {"-", tok::minus},
                   // FIXME:
                   // {"*", tok::star},
                   {"/", tok::slash},
                   {"%", tok::percent},
                   {"^", tok::caret},
                   // FIXME:
                   // {"&", tok::amp},
                   {"|", tok::pipe},
                   {"~", tok::tilde},
                   {"!", tok::exclaim},
                   {"=", tok::equal},
                   // FIXME:
                   // {"<", tok::less},
                   {">", tok::greater},
                   {"+=", tok::plusequal},
                   {"-=", tok::minusequal},
                   {"*=", tok::starequal},
                   {"/=", tok::slashequal},
                   {"%=", tok::percentequal},
                   {"^=", tok::caretequal},
                   {"&=", tok::ampequal},
                   {"|=", tok::pipeequal},
                   {"<<", tok::lessless},
                   {">>", tok::greatergreater},
                   {">>=", tok::greatergreaterequal},
                   {"<<=", tok::lesslessequal},
                   {"==", tok::equalequal},
                   {"!=", tok::exclaimequal},
                   {"<=", tok::lessequal},
                   {">=", tok::greaterequal},
                   {"<=>", tok::spaceship},
                   {"&&", tok::ampamp},
                   {"||", tok::pipepipe},
                   {"++", tok::plusplus},
                   {"--", tok::minusminus},
                   {",", tok::comma},
                   {"->*", tok::arrowstar},
                   {"->", tok::arrow}};

  for (const auto &Operator : Operators) {
    std::string Input("C<&operator");
    Input += Operator.Text;
    Input += " > a;";
    auto Tokens = annotate(std::string(Input));
    ASSERT_EQ(Tokens.size(), 9u) << Tokens;
    EXPECT_TOKEN(Tokens[1], tok::less, TT_TemplateOpener);
    EXPECT_TOKEN(Tokens[4], Operator.Kind, TT_OverloadedOperator);
    EXPECT_TOKEN(Tokens[5], tok::greater, TT_TemplateCloser);
  }

  auto Tokens = annotate("C<&operator< <X>> lt;");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[4], tok::less, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[8], tok::greater, TT_TemplateCloser);
}

TEST_F(TokenAnnotatorTest, UnderstandsRequiresClausesAndConcepts) {
  auto Tokens = annotate("template <typename T>\n"
                         "concept C = (Foo && Bar) && (Bar && Baz);");

  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[16], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("template <typename T>\n"
                    "concept C = Foo && !Bar;");

  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[10], tok::exclaim, TT_UnaryOperator);

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
                    "requires Bar<T> || Baz<T>\n"
                    "auto foo(T) -> int;");
  ASSERT_EQ(Tokens.size(), 24u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);
  EXPECT_EQ(Tokens[11]->FakeLParens.size(), 0u);
  EXPECT_TRUE(Tokens[14]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[20], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("template <typename T>\n"
                    "requires Bar<T>\n"
                    "bool foo(T) { return false; }");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[9]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[11], tok::identifier, TT_FunctionDeclarationName);

  Tokens = annotate("template <typename T>\n"
                    "requires Bar<T>\n"
                    "decltype(auto) foo(T) { return false; }");
  ASSERT_EQ(Tokens.size(), 24u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[9]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[14], tok::identifier, TT_FunctionDeclarationName);

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
  EXPECT_TOKEN(Tokens[24], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[25], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[35], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[36], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[47], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[49], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[59], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[61], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[76], tok::amp, TT_PointerOrReference);
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
  ASSERT_EQ(Tokens.size(), 44u) << Tokens;
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
  EXPECT_TOKEN(Tokens[6], tok::star, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(T const* volatile t) {});");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[6], tok::star, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(T& t) {});");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[5], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("foo(requires(T&& t) {});");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[5], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("bool foo = requires(T& t) {};");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[6], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens = annotate("bool foo = requires(T&& t) {};");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[6], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_RequiresExpressionLBrace);

  Tokens =
      annotate("foo(requires(const typename Outer<T>::Inner * const t) {});");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[12], tok::star, TT_PointerOrReference);
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

TEST_F(TokenAnnotatorTest, UnderstandsPragmaRegion) {
  // Everything after #pragma region should be ImplicitStringLiteral
  auto Tokens = annotate("#pragma region Foo(Bar: Hello)");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::identifier, TT_ImplicitStringLiteral);
  EXPECT_TOKEN(Tokens[6], tok::colon, TT_ImplicitStringLiteral);
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_ImplicitStringLiteral);

  // Make sure it's annotated correctly inside a function as well
  Tokens = annotate("void test(){\n#pragma region Foo(Bar: Hello)\n}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::identifier, TT_ImplicitStringLiteral);
  EXPECT_TOKEN(Tokens[11], tok::colon, TT_ImplicitStringLiteral);
  EXPECT_TOKEN(Tokens[12], tok::identifier, TT_ImplicitStringLiteral);
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
             "T foo();";
  ConstrainedCode = "template<typename T>\n"
                    "  requires(Foo<T>)\n"
                    "T foo();";
  BaseTokenCount = 11;
  RequiresTokenCount = 7;
  PrefixTokenCount = 5;
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

TEST_F(TokenAnnotatorTest, UnderstandsObjCMethodExpr) {
  auto Tokens = annotate("void f() {\n"
                         "  //\n"
                         "  BOOL a = [b.c n] > 1;\n"
                         "}");
  EXPECT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::l_square, TT_ObjCMethodExpr);
  EXPECT_TOKEN(Tokens[15], tok::greater, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsLambdas) {
  auto Tokens = annotate("[]() constexpr {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() consteval {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() mutable {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() static {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto {}");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto & {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto * {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] {}");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] noexcept {}");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[3], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] -> auto {}");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> () {}");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> {}");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename... T> () {}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename... T> {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <int... T> () {}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <int... T> {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <Foo... T> () {}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[9], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <Foo... T> {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  // Lambdas with a requires-clause
  Tokens = annotate("[] <typename T> (T t) requires Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[14]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[15], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> (T &&t) requires Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[8], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[11], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[15]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[16], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> (T t) requires Foo<T> || Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 23u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[19]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[20], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> (T t) -> T requires Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[12], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[16]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[17], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Bar<T> (T t) {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[15], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Bar<T> (T &&t) {}");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[16], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Foo<T> || Bar<T> (T t) {}");
  ASSERT_EQ(Tokens.size(), 23u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[15]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[20], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires true (T&& t) {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[7]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Bar<T> noexcept {}");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Bar<T> -> T {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[11], tok::arrow, TT_LambdaArrow);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T> requires Foo<T> (T t) requires Bar<T> {}");
  ASSERT_EQ(Tokens.size(), 23u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[6], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[10]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[15], tok::kw_requires, TT_RequiresClause);
  EXPECT_TRUE(Tokens[19]->ClosesRequiresClause);
  EXPECT_TOKEN(Tokens[20], tok::l_brace, TT_LambdaLBrace);
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

TEST_F(TokenAnnotatorTest, UnderstandsFunctionDeclarationNames) {
  auto Tokens = annotate("void f [[noreturn]] ();");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);

  Tokens = annotate("void f [[noreturn]] () {}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
}

TEST_F(TokenAnnotatorTest, UnderstandsC11GenericSelection) {
  auto Tokens = annotate("_Generic(x, int: 1, default: 0)");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::kw__Generic, TT_Unknown);
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_GenericSelectionColon);
  EXPECT_TOKEN(Tokens[9], tok::colon, TT_GenericSelectionColon);
}

TEST_F(TokenAnnotatorTest, UnderstandsTrailingReturnArrow) {
  auto Tokens = annotate("auto f() -> int;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("auto operator->() -> int;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::arrow, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("auto operator++(int) -> int;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("auto operator=() -> int;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("auto operator=(int) -> int;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("auto foo() -> auto { return Val; }");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);

  Tokens = annotate("struct S { auto bar() const -> int; };");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::arrow, TT_TrailingReturnArrow);

  // Not trailing return arrows
  Tokens = annotate("auto a = b->c;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_Unknown);

  Tokens = annotate("auto a = (b)->c;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::arrow, TT_Unknown);

  Tokens = annotate("auto a = b()->c;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::arrow, TT_Unknown);

  Tokens = annotate("auto a = b->c();");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_Unknown);

  Tokens = annotate("decltype(auto) a = b()->c;");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::arrow, TT_Unknown);

  Tokens = annotate("void f() { auto a = b->c(); }");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::arrow, TT_Unknown);

  Tokens = annotate("void f() { auto a = b()->c; }");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[11], tok::arrow, TT_Unknown);

  // Mixed
  Tokens = annotate("auto f() -> int { auto a = b()->c; }");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);
  EXPECT_TOKEN(Tokens[13], tok::arrow, TT_Unknown);
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
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
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
  // Non-blocking assignments.
  Tokens = Annotate("a <= b;");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::lessequal, TT_BinaryOperator);
  EXPECT_TOKEN_PRECEDENCE(Tokens[1], prec::Assignment);
  Tokens = Annotate("if (a <= b) break;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::lessequal, TT_BinaryOperator);
  EXPECT_TOKEN_PRECEDENCE(Tokens[3], prec::Relational);
  Tokens = Annotate("a <= b <= a;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::lessequal, TT_BinaryOperator);
  EXPECT_TOKEN_PRECEDENCE(Tokens[1], prec::Assignment);
  EXPECT_TOKEN(Tokens[3], tok::lessequal, TT_BinaryOperator);
  EXPECT_TOKEN_PRECEDENCE(Tokens[3], prec::Relational);

  // Port lists in module instantiation.
  Tokens = Annotate("module_x instance_1(port_1), instance_2(port_2);");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_VerilogInstancePortLParen);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_VerilogInstancePortLParen);
  Tokens = Annotate("module_x #(parameter) instance_1(port_1), "
                    "instance_2(port_2);");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_VerilogInstancePortLParen);
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_VerilogInstancePortLParen);
  EXPECT_TOKEN(Tokens[11], tok::l_paren, TT_VerilogInstancePortLParen);

  // Condition parentheses.
  Tokens = Annotate("assert (x);");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
  Tokens = Annotate("assert #0 (x);");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_ConditionLParen);
  Tokens = Annotate("assert final (x);");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_ConditionLParen);
  Tokens = Annotate("foreach (x[x]) continue;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
  Tokens = Annotate("repeat (x[x]) continue;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
  Tokens = Annotate("case (x) endcase;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
}

TEST_F(TokenAnnotatorTest, UnderstandConstructors) {
  auto Tokens = annotate("Class::Class() : BaseClass(), Member() {}");

  // The TT_Unknown is clearly not binding for the future, please adapt if those
  // tokens get annotated.
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_CtorInitializerColon);
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[9], tok::comma, TT_CtorInitializerComma);
  EXPECT_TOKEN(Tokens[10], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[11], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[12], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("Class::Class() : BaseClass{}, Member{} {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_CtorInitializerColon);
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::r_brace, TT_Unknown);
  EXPECT_TOKEN(Tokens[9], tok::comma, TT_CtorInitializerComma);
  EXPECT_TOKEN(Tokens[10], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_Unknown);
  EXPECT_TOKEN(Tokens[12], tok::r_brace, TT_Unknown);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_FunctionLBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsConditionParens) {
  auto Tokens = annotate("if (x) {}");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
  Tokens = annotate("if constexpr (x) {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_ConditionLParen);
  Tokens = annotate("if CONSTEXPR (x) {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_ConditionLParen);
  Tokens = annotate("if (x) {} else if (x) {}");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_ConditionLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_ConditionLParen);
}

TEST_F(TokenAnnotatorTest, CSharpNullableTypes) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_CSharp);

  auto Tokens = annotate("int? a;", Style);
  EXPECT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int? a = 1;", Style);
  EXPECT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int?)", Style);
  EXPECT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int?>", Style);
  EXPECT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("cond? id : id2", Style);
  EXPECT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_ConditionalExpr);

  Tokens = annotate("cond ? cond2 ? : id1 : id2", Style);
  EXPECT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_ConditionalExpr);
}

} // namespace
} // namespace format
} // namespace clang

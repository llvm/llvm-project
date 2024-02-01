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
#define EXPECT_BRACE_KIND(FormatTok, Kind)                                     \
  EXPECT_EQ(FormatTok->getBlockKind(), Kind) << *(FormatTok)
#define EXPECT_TOKEN(FormatTok, Kind, Type)                                    \
  do {                                                                         \
    EXPECT_TOKEN_KIND(FormatTok, Kind);                                        \
    EXPECT_TOKEN_TYPE(FormatTok, Type);                                        \
  } while (false)

TEST_F(TokenAnnotatorTest, UnderstandsUsesOfStarAndAmp) {
  auto Tokens = annotate("auto x = [](const decltype(x) &ptr) {};");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);

  Tokens = annotate("auto x = [](const decltype(x) *ptr) {};");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);

  Tokens = annotate("#define lambda [](const decltype(x) &ptr) {}");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw_decltype, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);

  Tokens = annotate("#define lambda [](const decltype(x) *ptr) {}");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_PointerOrReference);

  Tokens = annotate("void f() {\n"
                    "  while (p < a && *p == 'a')\n"
                    "    p++;\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[11], tok::star, TT_UnaryOperator);

  Tokens = annotate("case *x:");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::star, TT_UnaryOperator);
  Tokens = annotate("case &x:");
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::amp, TT_UnaryOperator);

  Tokens = annotate("bool b = 3 == int{3} && true;");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("struct {\n"
                    "} *ptr;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);
  Tokens = annotate("union {\n"
                    "} *ptr;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);
  Tokens = annotate("class {\n"
                    "} *ptr;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);

  Tokens = annotate("struct {\n"
                    "} &&ptr = {};");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("union {\n"
                    "} &&ptr = {};");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("class {\n"
                    "} &&ptr = {};");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::ampamp, TT_PointerOrReference);
  Tokens = annotate("int i = int{42} * 2;");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::star, TT_BinaryOperator);

  Tokens = annotate("delete[] *ptr;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);
  Tokens = annotate("delete[] **ptr;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[4], tok::star, TT_UnaryOperator);
  Tokens = annotate("delete[] *(ptr);");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::star, TT_UnaryOperator);

  Tokens = annotate("void f() { void (*fnptr)(char* foo); }");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_FunctionTypeLParen);
  // FIXME: The star of a function pointer probably makes more sense as
  // TT_PointerOrReference.
  EXPECT_TOKEN(Tokens[7], tok::star, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[12], tok::star, TT_PointerOrReference);

  Tokens = annotate("void f() { void (*fnptr)(t* foo); }");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
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

  Tokens =
      annotate("auto foo() noexcept(noexcept(bar()) && "
               "trait<std::decay_t<decltype(bar())>> && noexcept(baz())) {}");
  ASSERT_EQ(Tokens.size(), 38u) << Tokens;
  EXPECT_TOKEN(Tokens[12], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[27], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("foo = *i < *j && *j > *k;");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::less, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[7], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[10], tok::greater, TT_BinaryOperator);

  FormatStyle Style = getLLVMStyle();
  Style.TypeNames.push_back("MYI");
  Tokens = annotate("if (MYI *p{nullptr})", Style);
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_TypeName);
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);

  Style.TypeNames.push_back("Class");
  Tokens = annotate("if (Class *obj {getObj()})", Style);
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_TypeName);
  EXPECT_TOKEN(Tokens[3], tok::star, TT_PointerOrReference);

  Tokens = annotate("class Foo {\n"
                    "  void operator<() {}\n"
                    "  Foo &f;\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[5], tok::less, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_FunctionLBrace);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);
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
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_ClassLBrace);
  EXPECT_TOKEN(Tokens[3], tok::r_brace, TT_ClassRBrace);

  Tokens = annotate("const class C {} c;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::l_brace, TT_ClassLBrace);
  EXPECT_TOKEN(Tokens[4], tok::r_brace, TT_ClassRBrace);

  Tokens = annotate("const class {} c;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_ClassLBrace);
  EXPECT_TOKEN(Tokens[3], tok::r_brace, TT_ClassRBrace);

  Tokens = annotate("class [[deprecated(\"\")]] C { int i; };");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_ClassLBrace);
  EXPECT_TOKEN(Tokens[14], tok::r_brace, TT_ClassRBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsStructs) {
  auto Tokens = annotate("struct S {};");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[3], tok::r_brace, TT_StructRBrace);

  Tokens = annotate("struct EXPORT_MACRO [[nodiscard]] C { int i; };");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[12], tok::r_brace, TT_StructRBrace);

  Tokens = annotate("struct [[deprecated]] [[nodiscard]] C { int i; };");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[16], tok::r_brace, TT_StructRBrace);

  Tokens = annotate("template <typename T> struct S<const T[N]> {};");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[13], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[15], tok::r_brace, TT_StructRBrace);

  Tokens = annotate("template <typename T> struct S<T const[N]> {};");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[10], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[13], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[15], tok::r_brace, TT_StructRBrace);

  Tokens = annotate("template <typename T, unsigned n> struct S<T const[n]> {\n"
                    "  void f(T const (&a)[n]);\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 35u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[13], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[16], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[17], tok::l_brace, TT_StructLBrace);
  EXPECT_TOKEN(Tokens[23], tok::l_paren, TT_FunctionTypeLParen);
  EXPECT_TOKEN(Tokens[24], tok::amp, TT_UnaryOperator);
  EXPECT_TOKEN(Tokens[27], tok::l_square, TT_ArraySubscriptLSquare);
  EXPECT_TOKEN(Tokens[32], tok::r_brace, TT_StructRBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsUnions) {
  auto Tokens = annotate("union U {};");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_UnionLBrace);
  EXPECT_TOKEN(Tokens[3], tok::r_brace, TT_UnionRBrace);

  Tokens = annotate("union U { void f() { return; } };");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_UnionLBrace);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_FunctionLBrace);
  EXPECT_TOKEN(Tokens[11], tok::r_brace, TT_UnionRBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsEnums) {
  auto Tokens = annotate("enum E {};");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_EnumLBrace);
  EXPECT_TOKEN(Tokens[3], tok::r_brace, TT_EnumRBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsDefaultedAndDeletedFunctions) {
  auto Tokens = annotate("auto operator<=>(const T &) const & = default;");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void F(T) && = delete;");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_PointerOrReference);
}

TEST_F(TokenAnnotatorTest, UnderstandsVariables) {
  auto Tokens =
      annotate("inline bool var = is_integral_v<int> && is_signed_v<int>;");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::ampamp, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsVariableTemplates) {
  auto Tokens =
      annotate("template <typename T> "
               "inline bool var = is_integral_v<int> && is_signed_v<int>;");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
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

TEST_F(TokenAnnotatorTest, UnderstandsNonTemplateAngleBrackets) {
  auto Tokens = annotate("return a < b && c > d;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::less, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[6], tok::greater, TT_BinaryOperator);

  Tokens = annotate("a < 0 ? b : a > 0 ? c : d;");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::less, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_BinaryOperator);

  Tokens = annotate("ratio{-1, 2} < ratio{-1, 3} == -1 / 3 > -1 / 2;");
  ASSERT_EQ(Tokens.size(), 27u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::less, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[20], tok::greater, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsWhitespaceSensitiveMacros) {
  FormatStyle Style = getLLVMStyle();
  Style.WhitespaceSensitiveMacros.push_back("FOO");

  auto Tokens = annotate("FOO(1+2 )", Style);
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_UntouchableMacroFunc);

  Tokens = annotate("FOO(a:b:c)", Style);
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_UntouchableMacroFunc);
}

TEST_F(TokenAnnotatorTest, UnderstandsDelete) {
  auto Tokens = annotate("delete (void *)p;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] (void *)p;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[] /*comment*/ (void *)p;");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete[/*comment*/] (void *)p;");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("delete/*comment*/[] (void *)p;");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsCasts) {
  auto Tokens = annotate("(void)p;");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::r_paren, TT_CastRParen);

  Tokens = annotate("auto x = (Foo)p;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_CastRParen);

  Tokens = annotate("(std::vector<int>)p;");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_CastRParen);

  Tokens = annotate("return (Foo)p;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_CastRParen);

  Tokens = annotate("throw (Foo)p;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_CastRParen);

  Tokens = annotate("#define FOO(x) (((uint64_t)(x) * BAR) / 100)");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_CastRParen);
  EXPECT_TOKEN(Tokens[13], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[14], tok::star, TT_BinaryOperator);

  Tokens = annotate("#define foo(i) ((i) - bar)");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[10], tok::minus, TT_BinaryOperator);

  Tokens = annotate("return (Foo) & 10;");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_BinaryOperator);
}

TEST_F(TokenAnnotatorTest, UnderstandsDynamicExceptionSpecifier) {
  auto Tokens = annotate("void f() throw(int);");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_throw, TT_Unknown);
}

TEST_F(TokenAnnotatorTest, UnderstandsFunctionRefQualifiers) {
  auto Tokens = annotate("void f() &;");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_PointerOrReference);

  Tokens = annotate("void operator=(T) &&;");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void f() &;");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);

  Tokens = annotate("template <typename T> void operator=(T) &;");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);
}

TEST_F(TokenAnnotatorTest, UnderstandsOverloadedOperators) {
  auto Tokens = annotate("x.operator+()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator=()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::equal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator+=()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::plusequal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator,()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::comma, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator()()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::r_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator[]()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  // EXPECT_TOKEN(Tokens[3], tok::l_square, TT_OverloadedOperator);
  // EXPECT_TOKEN(Tokens[4], tok::r_square, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"_a()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\" _a()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"if()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\"s()");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_OverloadedOperatorLParen);
  Tokens = annotate("x.operator\"\" s()");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::kw_operator, TT_Unknown);
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

  Tokens = annotate("class Foo {\n"
                    "  int operator+(a* b);\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[5], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[8], tok::star, TT_PointerOrReference);

  Tokens = annotate("class Foo {\n"
                    "  int c = operator+(a * b);\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[7], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[10], tok::star, TT_BinaryOperator);

  Tokens = annotate("void foo() {\n"
                    "  operator+(a * b);\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[6], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[9], tok::star, TT_BinaryOperator);

  Tokens = annotate("return operator+(a * b, c & d) + operator+(a && b && c);");
  ASSERT_EQ(Tokens.size(), 24u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[2], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[5], tok::star, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[13], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[14], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[15], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[17], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[19], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("class Foo {\n"
                    "  void foo() {\n"
                    "    operator+(a * b);\n"
                    "  }\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::kw_operator, TT_Unknown);
  EXPECT_TOKEN(Tokens[9], tok::plus, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[10], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[12], tok::star, TT_BinaryOperator);

  Tokens = annotate("std::vector<Foo> operator()(Foo &foo);");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[5], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[6], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[9], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[11], tok::amp, TT_PointerOrReference);

  Tokens = annotate("decltype(auto) operator()(T &x);");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_TypeDeclarationParen);
  EXPECT_TOKEN(Tokens[4], tok::kw_operator, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[5], tok::l_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_OverloadedOperator);
  EXPECT_TOKEN(Tokens[7], tok::l_paren, TT_OverloadedOperatorLParen);
  EXPECT_TOKEN(Tokens[9], tok::amp, TT_PointerOrReference);
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

  Tokens = annotate("void foo() requires((A<T>) && C) {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[12], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("void foo() requires(((A<T>) && C)) {}");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[13], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("void foo() requires([](T&&){}(t)) {}");
  ASSERT_EQ(Tokens.size(), 21u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("void foo() requires([](T&& u){}(t)) {}");
  ASSERT_EQ(Tokens.size(), 22u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[10], tok::ampamp, TT_PointerOrReference);

  Tokens = annotate("void f() & requires(true) {}");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("void f() & requires(C<true, true>) {}");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("template <typename T>\n"
                    "concept C = (!Foo<T>) && Bar;");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_TOKEN(Tokens[15], tok::ampamp, TT_BinaryOperator);

  Tokens = annotate("void f() & requires(C<decltype(x)>) {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("auto f() -> int& requires(C<decltype(x)>) {}");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::amp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[7], tok::kw_requires, TT_RequiresClause);

  Tokens = annotate("bool x = t && requires(decltype(t) x) { x.foo(); };");
  ASSERT_EQ(Tokens.size(), 23u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresExpression);

  Tokens = annotate("bool x = t && requires(Foo<decltype(t)> x) { x.foo(); };");
  ASSERT_EQ(Tokens.size(), 26u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresExpression);

  Tokens = annotate("bool x = t && requires(Foo<C1 || C2> x) { x.foo(); };");
  ASSERT_EQ(Tokens.size(), 25u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw_requires, TT_RequiresExpression);
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

  Tokens = annotate("bool foo = requires(C<true, true> c) {\n"
                    "  { c.foo(); }\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 25u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::kw_requires, TT_RequiresExpression);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_RequiresExpressionLParen);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_RequiresExpressionLBrace);
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

  BaseCode = "template<typename T>\n"
             "ANNOTATE(\"S\"\n"
             "         \"S\")\n"
             "void foo();";
  ConstrainedCode = "template<typename T>\n"
                    "  requires(true)\n"
                    "ANNOTATE(\"S\"\n"
                    "         \"S\")\n"
                    "void foo();";
  BaseTokenCount = 16;
  RequiresTokenCount = 4;
  PrefixTokenCount = 5;
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
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
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
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto & {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);
  EXPECT_TOKEN(Tokens[7], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[]() -> auto * {}");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);
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
  EXPECT_TOKEN(Tokens[2], tok::arrow, TT_TrailingReturnArrow);
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
  EXPECT_TOKEN(Tokens[10], tok::arrow, TT_TrailingReturnArrow);
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
  EXPECT_TOKEN(Tokens[11], tok::arrow, TT_TrailingReturnArrow);
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

  Tokens = annotate("[] <typename T = int> (T t) {}");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <int I = 0> (T t) {}");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <bool b = false> (T t) {}");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <bool b = true && false> (T&& t) {}");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::ampamp, TT_BinaryOperator);
  EXPECT_TOKEN(Tokens[9], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[12], tok::ampamp, TT_PointerOrReference);
  EXPECT_TOKEN(Tokens[15], tok::l_brace, TT_LambdaLBrace);

  Tokens = annotate("[] <typename T = int> requires Foo<T> (T t) {}");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::l_square, TT_LambdaLSquare);
  EXPECT_TOKEN(Tokens[2], tok::less, TT_TemplateOpener);
  EXPECT_TOKEN(Tokens[7], tok::greater, TT_TemplateCloser);
  EXPECT_TOKEN(Tokens[8], tok::kw_requires, TT_RequiresClause);
  EXPECT_TOKEN(Tokens[17], tok::l_brace, TT_LambdaLBrace);
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

  Tokens = annotate("#define FOO Foo::\n"
                    "FOO Foo();");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_FunctionDeclarationName);

  Tokens = annotate("struct Foo {\n"
                    "  Bar (*func)();\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_FunctionTypeLParen);

  auto Style = getLLVMStyle();
  Style.TypeNames.push_back("time_t");
  Tokens = annotate("int iso_time(time_t);", Style);
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_TypeName);
}

TEST_F(TokenAnnotatorTest, UnderstandsCtorAndDtorDeclNames) {
  auto Tokens = annotate("class Foo { public: Foo(); };");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::identifier, TT_CtorDtorDeclName);

  Tokens = annotate("class Foo { public: ~Foo(); };");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::identifier, TT_CtorDtorDeclName);

  Tokens = annotate("struct Foo { [[deprecated]] Foo() {} };");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[8], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Foo { [[deprecated]] ~Foo() {} };");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Foo { Foo() [[deprecated]] {} };");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Foo { ~Foo() [[deprecated]] {} };");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Foo { [[deprecated]] explicit Foo() {} };");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[9], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[12], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Foo { virtual [[deprecated]] ~Foo() {} };");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("Foo::Foo() {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("Foo::~Foo() {}");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_FunctionLBrace);

  Tokens = annotate("struct Test {\n"
                    "  Test()\n"
                    "      : l([] {\n"
                    "          Short::foo();\n"
                    "        }) {}\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 25u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_CtorDtorDeclName);
  EXPECT_TOKEN(Tokens[14], tok::identifier, TT_Unknown);
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

  Tokens = annotate("#define P(ptr) auto p = (ptr)->p");
  ASSERT_EQ(Tokens.size(), 15u) << Tokens;
  EXPECT_TOKEN(Tokens[12], tok::arrow, TT_Unknown);

  Tokens = annotate("void f() FOO(foo->bar);");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::arrow, TT_Unknown);

  // Mixed
  Tokens = annotate("auto f() -> int { auto a = b()->c; }");
  ASSERT_EQ(Tokens.size(), 18u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::arrow, TT_TrailingReturnArrow);
  EXPECT_TOKEN(Tokens[13], tok::arrow, TT_Unknown);
}

TEST_F(TokenAnnotatorTest, UnderstandHashInMacro) {
  auto Tokens = annotate("#define Foo(Bar) \\\n"
                         "  { \\\n"
                         "    #Bar \\\n"
                         "  }");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);
  EXPECT_BRACE_KIND(Tokens[9], BK_Block);

  Tokens = annotate("#define Foo(Bar) \\\n"
                    "  { #Bar }");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);
  EXPECT_BRACE_KIND(Tokens[9], BK_Block);
}

TEST_F(TokenAnnotatorTest, UnderstandsAttributeMacros) {
  // '__attribute__' has special handling.
  auto Tokens = annotate("__attribute__(X) void Foo(void);");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::kw___attribute, TT_Unknown);
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_AttributeRParen);

  // Generic macro has no special handling in this location.
  Tokens = annotate("A(X) void Foo(void);");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_Unknown);

  // Add a custom AttributeMacro. Test that it has the same behavior.
  FormatStyle Style = getLLVMStyle();
  Style.AttributeMacros.push_back("A");

  // An "AttributeMacro" gets annotated like '__attribute__'.
  Tokens = annotate("A(X) void Foo(void);", Style);
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_AttributeMacro);
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_AttributeRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsAttributeMacrosOnObjCDecl) {
  // '__attribute__' has special handling.
  auto Tokens = annotate("__attribute__(X) @interface Foo");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::kw___attribute, TT_Unknown);
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_AttributeRParen);

  // Generic macro has no special handling in this location.
  Tokens = annotate("A(X) @interface Foo");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  // Note: Don't check token-type as a random token in this position is hard to
  // reason about.
  EXPECT_TOKEN_KIND(Tokens[0], tok::identifier);
  EXPECT_TOKEN_KIND(Tokens[1], tok::l_paren);

  // Add a custom AttributeMacro. Test that it has the same behavior.
  FormatStyle Style = getLLVMStyle();
  Style.AttributeMacros.push_back("A");

  // An "AttributeMacro" gets annotated like '__attribute__'.
  Tokens = annotate("A(X) @interface Foo", Style);
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_AttributeMacro);
  EXPECT_TOKEN(Tokens[1], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::r_paren, TT_AttributeRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsAttributeMacrosOnObjCMethodDecl) {
  // '__attribute__' has special handling.
  auto Tokens = annotate("- (id)init __attribute__(X);");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::kw___attribute, TT_Unknown);
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_AttributeRParen);

  // Generic macro has no special handling in this location.
  Tokens = annotate("- (id)init A(X);");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  // Note: Don't check token-type as a random token in this position is hard to
  // reason about.
  EXPECT_TOKEN_KIND(Tokens[5], tok::identifier);
  EXPECT_TOKEN_KIND(Tokens[6], tok::l_paren);

  // Add a custom AttributeMacro. Test that it has the same behavior.
  FormatStyle Style = getLLVMStyle();
  Style.AttributeMacros.push_back("A");

  // An "AttributeMacro" gets annotated like '__attribute__'.
  Tokens = annotate("- (id)init A(X);", Style);
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::identifier, TT_AttributeMacro);
  EXPECT_TOKEN(Tokens[6], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_AttributeRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsAttributeMacrosOnObjCProperty) {
  // '__attribute__' has special handling.
  auto Tokens = annotate("@property(weak) id delegate __attribute__(X);");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::kw___attribute, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_AttributeRParen);

  // Generic macro has no special handling in this location.
  Tokens = annotate("@property(weak) id delegate A(X);");
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  // Note: Don't check token-type as a random token in this position is hard to
  // reason about.
  EXPECT_TOKEN_KIND(Tokens[7], tok::identifier);
  EXPECT_TOKEN_KIND(Tokens[8], tok::l_paren);

  // Add a custom AttributeMacro. Test that it has the same behavior.
  FormatStyle Style = getLLVMStyle();
  Style.AttributeMacros.push_back("A");

  // An "AttributeMacro" gets annotated like '__attribute__'.
  Tokens = annotate("@property(weak) id delegate A(X);", Style);
  ASSERT_EQ(Tokens.size(), 13u) << Tokens;
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_AttributeMacro);
  EXPECT_TOKEN(Tokens[8], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[10], tok::r_paren, TT_AttributeRParen);
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
      {prec::Comma, "->"},        {prec::Comma, "<->"},
      {prec::Assignment, "+="},   {prec::Assignment, "-="},
      {prec::Assignment, "*="},   {prec::Assignment, "/="},
      {prec::Assignment, "%="},   {prec::Assignment, "&="},
      {prec::Assignment, "^="},   {prec::Assignment, "<<="},
      {prec::Assignment, ">>="},  {prec::Assignment, "<<<="},
      {prec::Assignment, ">>>="}, {prec::LogicalOr, "||"},
      {prec::LogicalAnd, "&&"},   {prec::Equality, "=="},
      {prec::Equality, "!="},     {prec::Equality, "==="},
      {prec::Equality, "!=="},    {prec::Equality, "==?"},
      {prec::Equality, "!=?"},    {prec::ExclusiveOr, "~^"},
      {prec::ExclusiveOr, "^~"},
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
  Tokens = Annotate("module test\n"
                    "    (input wire [7 : 0] a[7 : 0]);\n"
                    "endmodule");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::identifier, TT_VerilogDimensionedTypeName);
  EXPECT_TOKEN(Tokens[7], tok::colon, TT_BitFieldColon);
  EXPECT_TOKEN(Tokens[13], tok::colon, TT_BitFieldColon);
  // Test case labels and ternary operators.
  Tokens = Annotate("case (x)\n"
                    "  x:\n"
                    "    x;\n"
                    "endcase");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_CaseLabelColon);
  Tokens = Annotate("case (x)\n"
                    "  x ? x : x:\n"
                    "    x;\n"
                    "endcase");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::question, TT_ConditionalExpr);
  EXPECT_TOKEN(Tokens[7], tok::colon, TT_ConditionalExpr);
  EXPECT_TOKEN(Tokens[9], tok::colon, TT_CaseLabelColon);
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

  // Sensitivity list. The TT_Unknown type is clearly not binding for the
  // future, please adapt if those tokens get annotated.  This test is only here
  // to prevent the comma from being annotated as TT_VerilogInstancePortComma.
  Tokens = Annotate("always @(posedge x, posedge y);");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[5], tok::comma, TT_Unknown);
  EXPECT_TOKEN(Tokens[8], tok::r_paren, TT_Unknown);

  // String literals in concatenation.
  Tokens = Annotate("x = {\"\"};");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_StringInConcatenation);
  Tokens = Annotate("x = {\"\", \"\"};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_StringInConcatenation);
  EXPECT_TOKEN(Tokens[5], tok::string_literal, TT_StringInConcatenation);
  Tokens = Annotate("x = '{{\"\"}};");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::string_literal, TT_StringInConcatenation);
  // Cases where the string should not be annotated that type.  Fix the
  // `TT_Unknown` if needed in the future.
  Tokens = Annotate("x = {\"\" == \"\"};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::string_literal, TT_Unknown);
  EXPECT_TOKEN(Tokens[5], tok::string_literal, TT_Unknown);
  Tokens = Annotate("x = {(\"\")};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::string_literal, TT_Unknown);
  Tokens = Annotate("x = '{\"\"};");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::string_literal, TT_Unknown);
}

TEST_F(TokenAnnotatorTest, UnderstandTableGenTokens) {
  auto Style = getLLVMStyle(FormatStyle::LK_TableGen);
  ASSERT_TRUE(Style.isTableGen());

  TestLexer Lexer(Allocator, Buffers, Style);
  AdditionalKeywords Keywords(Lexer.IdentTable);
  auto Annotate = [&Lexer](llvm::StringRef Code) {
    return Lexer.annotate(Code);
  };

  // Additional keywords representation test.
  auto Tokens = Annotate("def foo : Bar<1>;");
  ASSERT_TRUE(Keywords.isTableGenKeyword(*Tokens[0]));
  ASSERT_TRUE(Keywords.isTableGenDefinition(*Tokens[0]));
  ASSERT_TRUE(Tokens[0]->is(Keywords.kw_def));
  ASSERT_TRUE(Tokens[1]->is(TT_StartOfName));

  // Code, the multiline string token.
  Tokens = Annotate("[{ code is multiline string }]");
  ASSERT_EQ(Tokens.size(), 2u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::string_literal, TT_TableGenMultiLineString);
  EXPECT_FALSE(Tokens[0]->IsMultiline);
  // Case with multiple lines.
  Tokens = Annotate("[{ It can break\n"
                    "   across lines and the line breaks\n"
                    "   are retained in \n"
                    "   the string. }]");
  ASSERT_EQ(Tokens.size(), 2u) << Tokens;
  EXPECT_TOKEN(Tokens[0], tok::string_literal, TT_TableGenMultiLineString);
  EXPECT_EQ(Tokens[0]->ColumnWidth, sizeof("[{ It can break\n") - 1);
  EXPECT_TRUE(Tokens[0]->IsMultiline);
  EXPECT_EQ(Tokens[0]->LastLineColumnWidth, sizeof("   the string. }]") - 1);

  // Numeric literals.
  Tokens = Annotate("1234");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);
  Tokens = Annotate("-1");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);
  Tokens = Annotate("+1234");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);
  Tokens = Annotate("0b0110");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);
  Tokens = Annotate("0x1abC");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);

  // Identifier tokens. In TableGen, identifiers can begin with a number.
  // In ambiguous cases, the lexer tries to lex it as a number.
  // Even if the try fails, it does not fall back to identifier lexing and
  // regard as an error.
  // The ambiguity is not documented. The result of those tests are based on the
  // implementation of llvm::TGLexer::LexToken.
  // This is invalid syntax of number, but not an identifier.
  Tokens = Annotate("0x1234x");
  EXPECT_TOKEN(Tokens[0], tok::numeric_constant, TT_Unknown);
  Tokens = Annotate("identifier");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_Unknown);
  // Identifier beginning with a number.
  Tokens = Annotate("0x");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_Unknown);
  Tokens = Annotate("2dVector");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_Unknown);
  Tokens = Annotate("01234Vector");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_Unknown);

  // Structured statements.
  Tokens = Annotate("class Foo {}");
  EXPECT_TOKEN(Tokens[2], tok::l_brace, TT_FunctionLBrace);
  Tokens = Annotate("def Def: Foo {}");
  EXPECT_TOKEN(Tokens[2], tok::colon, TT_InheritanceColon);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);
  Tokens = Annotate("if cond then {} else {}");
  EXPECT_TOKEN(Tokens[3], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_ElseLBrace);
  Tokens = Annotate("defset Foo Def2 = {}");
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);

  // Bang Operators.
  Tokens = Annotate("!foreach");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_TableGenBangOperator);
  Tokens = Annotate("!if");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_TableGenBangOperator);
  Tokens = Annotate("!cond");
  EXPECT_TOKEN(Tokens[0], tok::identifier, TT_TableGenCondOperator);
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
  EXPECT_BRACE_KIND(Tokens[13], BK_Block);

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
  EXPECT_BRACE_KIND(Tokens[13], BK_Block);

  Tokens = annotate("class Class {\n"
                    "  Class() : BaseClass() {\n"
                    "#if 0\n"
                    "    // comment\n"
                    "#endif\n"
                    "  }\n"
                    "  Class f();\n"
                    "}");
  ASSERT_EQ(Tokens.size(), 25u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::colon, TT_CtorInitializerColon);
  EXPECT_TOKEN(Tokens[10], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[10], BK_Block);
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
  ASSERT_EQ(Tokens.size(), 5u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int? a = 1;", Style);
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int?)", Style);
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("int?>", Style);
  ASSERT_EQ(Tokens.size(), 4u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_CSharpNullable);

  Tokens = annotate("cond? id : id2", Style);
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_ConditionalExpr);

  Tokens = annotate("cond ? cond2 ? : id1 : id2", Style);
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::question, TT_ConditionalExpr);
}

TEST_F(TokenAnnotatorTest, UnderstandsLabels) {
  auto Tokens = annotate("{ x: break; }");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::colon, TT_GotoLabelColon);
  Tokens = annotate("{ case x: break; }");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::colon, TT_CaseLabelColon);
  Tokens = annotate("{ x: { break; } }");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::colon, TT_GotoLabelColon);
  Tokens = annotate("{ case x: { break; } }");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::colon, TT_CaseLabelColon);
}

TEST_F(TokenAnnotatorTest, UnderstandsNestedBlocks) {
  // The closing braces are not annotated. It doesn't seem to cause a problem.
  // So we only test for the opening braces.
  auto Tokens = annotate("{\n"
                         "  {\n"
                         "    { int a = 0; }\n"
                         "  }\n"
                         "  {}\n"
                         "}");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[0], BK_Block);
  EXPECT_BRACE_KIND(Tokens[1], BK_Block);
  EXPECT_BRACE_KIND(Tokens[2], BK_Block);
  EXPECT_BRACE_KIND(Tokens[10], BK_Block);
}

TEST_F(TokenAnnotatorTest, UnderstandDesignatedInitializers) {
  auto Tokens = annotate("SomeStruct { .a = 1 };");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[1], BK_BracedInit);
  EXPECT_TOKEN(Tokens[2], tok::period, TT_DesignatedInitializerPeriod);

  Tokens = annotate("SomeStruct { .a = 1, .b = 2 };");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[1], BK_BracedInit);
  EXPECT_TOKEN(Tokens[2], tok::period, TT_DesignatedInitializerPeriod);
  EXPECT_TOKEN(Tokens[7], tok::period, TT_DesignatedInitializerPeriod);

  Tokens = annotate("SomeStruct {\n"
                    "#ifdef FOO\n"
                    "  .a = 1,\n"
                    "#endif\n"
                    "  .b = 2\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 19u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[1], BK_BracedInit);
  EXPECT_TOKEN(Tokens[5], tok::period, TT_DesignatedInitializerPeriod);
  EXPECT_TOKEN(Tokens[12], tok::period, TT_DesignatedInitializerPeriod);

  Tokens = annotate("SomeStruct {\n"
                    "#if defined FOO\n"
                    "  .a = 1,\n"
                    "#endif\n"
                    "  .b = 2\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 20u) << Tokens;
  EXPECT_BRACE_KIND(Tokens[1], BK_BracedInit);
  EXPECT_TOKEN(Tokens[6], tok::period, TT_DesignatedInitializerPeriod);
  EXPECT_TOKEN(Tokens[13], tok::period, TT_DesignatedInitializerPeriod);

  Tokens = annotate("Foo foo[] = {[0]{}};");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::l_square, TT_DesignatedInitializerLSquare);
  EXPECT_BRACE_KIND(Tokens[9], BK_BracedInit);
}

TEST_F(TokenAnnotatorTest, UnderstandsJavaScript) {
  auto Annotate = [this](llvm::StringRef Code) {
    return annotate(Code, getLLVMStyle(FormatStyle::LK_JavaScript));
  };

  // Dictionary.
  auto Tokens = Annotate("var x = {'x' : 1, 'y' : 2};");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::l_brace, TT_DictLiteral);
  EXPECT_TOKEN(Tokens[4], tok::string_literal, TT_SelectorName);
  EXPECT_TOKEN(Tokens[5], tok::colon, TT_DictLiteral);
  EXPECT_TOKEN(Tokens[8], tok::string_literal, TT_SelectorName);
  EXPECT_TOKEN(Tokens[9], tok::colon, TT_DictLiteral);
  // Change when we need to annotate these.
  EXPECT_BRACE_KIND(Tokens[3], BK_Unknown);
  EXPECT_BRACE_KIND(Tokens[11], BK_Unknown);
  EXPECT_TOKEN(Tokens[11], tok::r_brace, TT_Unknown);
}

TEST_F(TokenAnnotatorTest, UnderstandsAttributes) {
  auto Tokens = annotate("bool foo __attribute__((unused));");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_StartOfName);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[4], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[7], tok::r_paren, TT_AttributeRParen);

  Tokens = annotate("bool foo __declspec(dllimport);");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_AttributeRParen);

  Tokens = annotate("bool __attribute__((unused)) foo;");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_AttributeRParen);
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_StartOfName);

  Tokens = annotate("void __attribute__((x)) Foo();");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_Unknown);
  EXPECT_TOKEN(Tokens[6], tok::r_paren, TT_AttributeRParen);
  EXPECT_TOKEN(Tokens[7], tok::identifier, TT_FunctionDeclarationName);

  FormatStyle Style = getLLVMStyle();
  Style.AttributeMacros.push_back("FOO");
  Tokens = annotate("bool foo FOO(unused);", Style);
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_AttributeMacro);
  EXPECT_TOKEN(Tokens[3], tok::l_paren, TT_AttributeLParen);
  EXPECT_TOKEN(Tokens[5], tok::r_paren, TT_AttributeRParen);
}

TEST_F(TokenAnnotatorTest, UnderstandsControlStatements) {
  auto Tokens = annotate("while (true) {}");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[5], tok::r_brace, TT_ControlStatementRBrace);

  Tokens = annotate("for (;;) {}");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[6], tok::r_brace, TT_ControlStatementRBrace);

  Tokens = annotate("do {} while (true);");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[2], tok::r_brace, TT_ControlStatementRBrace);

  Tokens = annotate("if (true) {} else if (false) {} else {}");
  ASSERT_EQ(Tokens.size(), 17u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[5], tok::r_brace, TT_ControlStatementRBrace);
  EXPECT_TOKEN(Tokens[11], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[12], tok::r_brace, TT_ControlStatementRBrace);
  EXPECT_TOKEN(Tokens[14], tok::l_brace, TT_ElseLBrace);
  EXPECT_TOKEN(Tokens[15], tok::r_brace, TT_ElseRBrace);

  Tokens = annotate("switch (foo) {}");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_ControlStatementLBrace);
  EXPECT_TOKEN(Tokens[5], tok::r_brace, TT_ControlStatementRBrace);
}

TEST_F(TokenAnnotatorTest, UnderstandsDoWhile) {
  auto Tokens = annotate("do { ++i; } while ( i > 5 );");
  ASSERT_EQ(Tokens.size(), 14u) << Tokens;
  EXPECT_TOKEN(Tokens[6], tok::kw_while, TT_DoWhile);

  Tokens = annotate("do ++i; while ( i > 5 );");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[4], tok::kw_while, TT_DoWhile);
}

TEST_F(TokenAnnotatorTest, StartOfName) {
  auto Tokens = annotate("#pragma clang diagnostic push");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[4], tok::identifier, TT_Unknown);

  Tokens = annotate("#pragma clang diagnostic ignored \"-Wzero-length-array\"");
  ASSERT_EQ(Tokens.size(), 7u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[4], tok::identifier, TT_Unknown);

  Tokens = annotate("#define FOO Foo foo");
  ASSERT_EQ(Tokens.size(), 6u) << Tokens;
  EXPECT_TOKEN(Tokens[2], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[3], tok::identifier, TT_Unknown);
  EXPECT_TOKEN(Tokens[4], tok::identifier, TT_StartOfName);
}

TEST_F(TokenAnnotatorTest, BraceKind) {
  auto Tokens = annotate("void f() {};");
  ASSERT_EQ(Tokens.size(), 8u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[4], BK_Block);
  EXPECT_BRACE_KIND(Tokens[5], BK_Block);

  Tokens = annotate("class Foo<int> f() {}");
  ASSERT_EQ(Tokens.size(), 11u) << Tokens;
  EXPECT_TOKEN(Tokens[5], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[8], BK_Block);
  EXPECT_BRACE_KIND(Tokens[9], BK_Block);

  Tokens = annotate("template <typename T> class Foo<T> f() {}");
  ASSERT_EQ(Tokens.size(), 16u) << Tokens;
  EXPECT_TOKEN(Tokens[10], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[13], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[13], BK_Block);
  EXPECT_BRACE_KIND(Tokens[14], BK_Block);

  Tokens = annotate("void f() override {};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[5], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[5], BK_Block);
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);

  Tokens = annotate("void f() noexcept(false) {};");
  ASSERT_EQ(Tokens.size(), 12u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[8], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[8], BK_Block);
  EXPECT_BRACE_KIND(Tokens[9], BK_Block);

  Tokens = annotate("auto f() -> void {};");
  ASSERT_EQ(Tokens.size(), 10u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[6], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);
  EXPECT_BRACE_KIND(Tokens[7], BK_Block);

  Tokens = annotate("void f() { /**/ };");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[4], BK_Block);
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);

  Tokens = annotate("void f() { //\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[4], BK_Block);
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);

  Tokens = annotate("void f() {\n"
                    "  //\n"
                    "};");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_TOKEN(Tokens[1], tok::identifier, TT_FunctionDeclarationName);
  EXPECT_TOKEN(Tokens[4], tok::l_brace, TT_FunctionLBrace);
  EXPECT_BRACE_KIND(Tokens[4], BK_Block);
  EXPECT_BRACE_KIND(Tokens[6], BK_Block);
}

TEST_F(TokenAnnotatorTest, StreamOperator) {
  auto Tokens = annotate("\"foo\\n\" << aux << \"foo\\n\" << \"foo\";");
  ASSERT_EQ(Tokens.size(), 9u) << Tokens;
  EXPECT_FALSE(Tokens[1]->MustBreakBefore);
  EXPECT_FALSE(Tokens[3]->MustBreakBefore);
  // Only break between string literals if the former ends with \n.
  EXPECT_TRUE(Tokens[5]->MustBreakBefore);
}

} // namespace
} // namespace format
} // namespace clang

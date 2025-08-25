//===- unittests/AST/TypePrinterTest.cpp --- Type printer tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for QualType::print() and related methods.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

static void PrintType(raw_ostream &Out, const ASTContext *Context,
                      const QualType *T,
                      PrintingPolicyAdjuster PolicyAdjuster) {
  assert(T && !T->isNull() && "Expected non-null Type");
  PrintingPolicy Policy = Context->getPrintingPolicy();
  if (PolicyAdjuster)
    PolicyAdjuster(Policy);
  T->print(Out, Policy);
}

::testing::AssertionResult
PrintedTypeMatches(StringRef Code, const std::vector<std::string> &Args,
                   const DeclarationMatcher &NodeMatch,
                   StringRef ExpectedPrinted,
                   PrintingPolicyAdjuster PolicyAdjuster) {
  return PrintedNodeMatches<QualType>(Code, Args, NodeMatch, ExpectedPrinted,
                                      "", PrintType, PolicyAdjuster);
}

} // unnamed namespace

TEST(TypePrinter, TemplateId) {
  std::string Code = R"cpp(
    namespace N {
      template <typename> struct Type {};

      template <typename T>
      void Foo(const Type<T> &Param);
    }
  )cpp";
  auto Matcher = parmVarDecl(hasType(qualType().bind("id")));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "const Type<T> &",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = false; }));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "const N::Type<T> &",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = true; }));
}

TEST(TypePrinter, TemplateId2) {
  std::string Code = R"cpp(
      template <template <typename ...> class TemplatedType>
      void func(TemplatedType<int> Param);
    )cpp";
  auto Matcher = parmVarDecl(hasType(qualType().bind("id")));

  // Regression test ensuring we do not segfault getting the QualType as a
  // string.
  ASSERT_TRUE(PrintedTypeMatches(Code, {}, Matcher, "<int>",
                                 [](PrintingPolicy &Policy) {
                                   Policy.FullyQualifiedName = true;
                                   Policy.PrintAsCanonical = true;
                                 }));
}

TEST(TypePrinter, ParamsUglified) {
  llvm::StringLiteral Code = R"cpp(
    template <typename _Tp, template <typename> class __f>
    const __f<_Tp&> *A = nullptr;
  )cpp";
  auto Clean = [](PrintingPolicy &Policy) {
    Policy.CleanUglifiedParameters = true;
  };

  ASSERT_TRUE(PrintedTypeMatches(Code, {},
                                 varDecl(hasType(qualType().bind("id"))),
                                 "const __f<_Tp &> *", nullptr));
  ASSERT_TRUE(PrintedTypeMatches(Code, {},
                                 varDecl(hasType(qualType().bind("id"))),
                                 "const f<Tp &> *", Clean));
}

TEST(TypePrinter, TemplateSpecializationFullyQualified) {
  llvm::StringLiteral Code = R"cpp(
    namespace shared {
    namespace a {
    template <typename T>
    struct S {};
    }  // namespace a
    namespace b {
    struct Foo {};
    }  // namespace b
    using Alias = a::S<b::Foo>;
    }  // namespace shared
  )cpp";

  auto Matcher = typedefNameDecl(hasName("::shared::Alias"),
                                 hasType(qualType().bind("id")));
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "a::S<b::Foo>",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = false; }));
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "shared::a::S<shared::b::Foo>",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = true; }));
}

TEST(TypePrinter, TemplateIdWithNTTP) {
  constexpr char Code[] = R"cpp(
    template <int N>
    struct Str {
      constexpr Str(char const (&s)[N]) { __builtin_memcpy(value, s, N); }
      char value[N];
    };
    template <Str> class ASCII {};

    ASCII<"this nontype template argument is too long to print"> x;
  )cpp";
  auto Matcher = classTemplateSpecializationDecl(
      hasName("ASCII"), has(cxxConstructorDecl(
                            isMoveConstructor(),
                            has(parmVarDecl(hasType(qualType().bind("id")))))));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++20"}, Matcher,
      R"(ASCII<Str<52>{"this nontype template argument is [...]"}> &&)",
      [](PrintingPolicy &Policy) {
        Policy.EntireContentsOfLargeArray = false;
      }));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++20"}, Matcher,
      R"(ASCII<Str<52>{"this nontype template argument is too long to print"}> &&)",
      [](PrintingPolicy &Policy) {
        Policy.EntireContentsOfLargeArray = true;
      }));
}

TEST(TypePrinter, TemplateArgumentsSubstitution) {
  constexpr char Code[] = R"cpp(
       template <typename Y> class X {};
       typedef X<int> A;
       int foo() {
          return sizeof(A);
       }
  )cpp";
  auto Matcher = typedefNameDecl(hasName("A"), hasType(qualType().bind("id")));
  ASSERT_TRUE(PrintedTypeMatches(Code, {}, Matcher, "X<int>",
                                 [](PrintingPolicy &Policy) {
                                   Policy.SuppressTagKeyword = false;
                                   Policy.SuppressScope = true;
                                 }));
}

TEST(TypePrinter, TemplateArgumentsSubstitution_Expressions) {
  /// Tests clang::isSubstitutedDefaultArgument on TemplateArguments
  /// that are of kind TemplateArgument::Expression
  constexpr char Code[] = R"cpp(
    constexpr bool func() { return true; }

    template <typename T1 = int,
              int      T2 = 42,
              T1       T3 = 43,
              int      T4 = sizeof(T1),
              bool     T5 = func()
              >
    struct Foo {
    };

    Foo<int, 40 + 2> X;
  )cpp";

  auto AST = tooling::buildASTFromCodeWithArgs(Code, /*Args=*/{"-std=c++20"});
  ASTContext &Ctx = AST->getASTContext();

  auto const *CTD = selectFirst<ClassTemplateDecl>(
      "id", match(classTemplateDecl(hasName("Foo")).bind("id"), Ctx));
  ASSERT_NE(CTD, nullptr);
  auto const *CTSD = *CTD->specializations().begin();
  ASSERT_NE(CTSD, nullptr);
  auto const *Params = CTD->getTemplateParameters();
  ASSERT_NE(Params, nullptr);
  auto const &ArgList = CTSD->getTemplateArgs();

  auto createBinOpExpr = [&](uint32_t LHS, uint32_t RHS,
                             uint32_t Result) -> ConstantExpr * {
    const int numBits = 32;
    clang::APValue ResultVal{llvm::APSInt(llvm::APInt(numBits, Result))};
    auto *LHSInt = IntegerLiteral::Create(Ctx, llvm::APInt(numBits, LHS),
                                          Ctx.UnsignedIntTy, {});
    auto *RHSInt = IntegerLiteral::Create(Ctx, llvm::APInt(numBits, RHS),
                                          Ctx.UnsignedIntTy, {});
    auto *BinOp = BinaryOperator::Create(
        Ctx, LHSInt, RHSInt, BinaryOperatorKind::BO_Add, Ctx.UnsignedIntTy,
        ExprValueKind::VK_PRValue, ExprObjectKind::OK_Ordinary, {}, {});
    return ConstantExpr::Create(Ctx, dyn_cast<Expr>(BinOp), ResultVal);
  };

  {
    // Arg is an integral '42'
    auto const &Arg = ArgList.get(1);
    ASSERT_EQ(Arg.getKind(), TemplateArgument::Integral);

    // Param has default expr which evaluates to '42'
    auto const *Param = Params->getParam(1);

    EXPECT_TRUE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }

  {
    // Arg is an integral '41'
    llvm::APInt Int(32, 41);
    TemplateArgument Arg(Ctx, llvm::APSInt(Int), Ctx.UnsignedIntTy);

    // Param has default expr which evaluates to '42'
    auto const *Param = Params->getParam(1);

    EXPECT_FALSE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }

  {
    // Arg is an integral '4'
    llvm::APInt Int(32, 4);
    TemplateArgument Arg(Ctx, llvm::APSInt(Int), Ctx.UnsignedIntTy);

    // Param has is value-dependent expression (i.e., sizeof(T))
    auto const *Param = Params->getParam(3);

    EXPECT_FALSE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }

  {
    const int LHS = 40;
    const int RHS = 2;
    const int Result = 42;
    auto *ConstExpr = createBinOpExpr(LHS, RHS, Result);
    // Arg is instantiated with '40 + 2'
    TemplateArgument Arg(ConstExpr, /*IsCanonical=*/false);

    // Param has default expr of '42'
    auto const *Param = Params->getParam(1);

    EXPECT_TRUE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }

  {
    const int LHS = 40;
    const int RHS = 1;
    const int Result = 41;
    auto *ConstExpr = createBinOpExpr(LHS, RHS, Result);

    // Arg is instantiated with '40 + 1'
    TemplateArgument Arg(ConstExpr, /*IsCanonical=*/false);

    // Param has default expr of '42'
    auto const *Param = Params->getParam(1);

    EXPECT_FALSE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }

  {
    const int LHS = 4;
    const int RHS = 0;
    const int Result = 4;
    auto *ConstExpr = createBinOpExpr(LHS, RHS, Result);

    // Arg is instantiated with '4 + 0'
    TemplateArgument Arg(ConstExpr, /*IsCanonical=*/false);

    // Param has is value-dependent expression (i.e., sizeof(T))
    auto const *Param = Params->getParam(3);

    EXPECT_FALSE(clang::isSubstitutedDefaultArgument(
        Ctx, Arg, Param, ArgList.asArray(), Params->getDepth()));
  }
}

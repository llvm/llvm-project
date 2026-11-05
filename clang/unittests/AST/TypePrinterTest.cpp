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
#include "clang/Frontend/ASTUnit.h"
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

TEST(TypePrinter, TemplateArgumentExpressionFullyQualified) {
  llvm::StringLiteral Code = R"cpp(
    namespace ns {
      template <class T> inline constexpr bool pred_v = sizeof(T) > 0;
      template <bool, class T> struct ei {};
      template <class T> struct ei<true, T> { using type = T; };
      template <bool B, class T> using enable_if_t = typename ei<B, T>::type;
      struct Ret {};
      template <class T> enable_if_t<pred_v<T>, Ret> f() { return {}; }
    }
    inline auto *ns_f_int = &ns::f<int>;
  )cpp";

  auto Matcher = functionDecl(hasName("::ns::f"), isTemplateInstantiation(),
                              returns(qualType().bind("id")));
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++17"}, Matcher, "enable_if_t<pred_v<int>, Ret>",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = false; }));
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++17"}, Matcher,
      "ns::enable_if_t<ns::pred_v<int>, ns::Ret>",
      [](PrintingPolicy &Policy) { Policy.FullyQualifiedName = true; }));
}

TEST(TypePrinter, TemplateSpecializationWithDependentSizedArrayType) {
  llvm::StringLiteral Code = R"cpp(
    template<typename Container>
    struct View {};

    template<int Count>
    struct View<int[Count]> {
      using Container = int[Count];
    };
  )cpp";

  auto Matcher =
      typeAliasDecl(hasName("Container"), hasType(qualType().bind("id")));
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, Matcher, "int[Count]",
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

TEST(TypePrinter, NestedNameSpecifiers) {
  constexpr char Code[] = R"cpp(
    void level1() {
      struct Inner {
        Inner(int) {
          struct {
            union {} u;
          } imem;
        }
      };
    }
  )cpp";

  // Types scoped immediately inside a function don't print the function name in
  // their scope.
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, varDecl(hasName("imem"), hasType(qualType().bind("id"))),
      "struct (unnamed)", [](PrintingPolicy &Policy) {
        Policy.FullyQualifiedName = true;
        Policy.AnonymousTagNameStyle =
            llvm::to_underlying(PrintingPolicy::AnonymousTagMode::Plain);
      }));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, varDecl(hasName("imem"), hasType(qualType().bind("id"))),
      "struct (unnamed)", [](PrintingPolicy &Policy) {
        Policy.FullyQualifiedName = false;
        Policy.AnonymousTagNameStyle =
            llvm::to_underlying(PrintingPolicy::AnonymousTagMode::Plain);
      }));

  // Further levels of nesting print the entire scope.
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, fieldDecl(hasName("u"), hasType(qualType().bind("id"))),
      "union level1()::Inner::Inner(int)::(unnamed struct)::(unnamed)",
      [](PrintingPolicy &Policy) {
        Policy.FullyQualifiedName = true;
        Policy.AnonymousTagNameStyle =
            llvm::to_underlying(PrintingPolicy::AnonymousTagMode::Plain);
      }));

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, fieldDecl(hasName("u"), hasType(qualType().bind("id"))),
      "union (unnamed)", [](PrintingPolicy &Policy) {
        Policy.FullyQualifiedName = false;
        Policy.AnonymousTagNameStyle =
            llvm::to_underlying(PrintingPolicy::AnonymousTagMode::Plain);
      }));
}

TEST(TypePrinter, NestedNameSpecifiersTypedef) {
  constexpr char Code[] = R"cpp(
    typedef union {
      struct {
        struct {
          unsigned int baz;
        } bar;
      };
    } foo;
  )cpp";

  ASSERT_TRUE(PrintedTypeMatches(
      Code, {}, fieldDecl(hasName("bar"), hasType(qualType().bind("id"))),
      "struct foo::(anonymous struct)::(unnamed)", [](PrintingPolicy &Policy) {
        Policy.FullyQualifiedName = true;
        Policy.AnonymousTagNameStyle =
            llvm::to_underlying(PrintingPolicy::AnonymousTagMode::Plain);
      }));
}

TEST(TypePrinter, TypePrintLimitTemplateArguments) {
  // Each level uses the previous one twice, so the type is 2^40 'int's when
  // spelled out in full.
  constexpr char Code[] = R"cpp(
    template <class A, class B> struct F {};
    template <int N> struct Chain {
      using type = F<typename Chain<N - 1>::type, typename Chain<N - 1>::type>;
    };
    template <> struct Chain<0> { using type = int; };
    Chain<40>::type huge;
  )cpp";
  auto Matcher = varDecl(hasName("huge"), hasType(qualType().bind("id")));

  // Arguments are printed left to right until the limit is reached. After
  // that, the ones that would open another list are omitted.
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++11", "-ftemplate-depth=100"}, Matcher,
      "F<F<F<F<F<F<[2 args omitted]>, [arg omitted]>, [arg omitted]>, [arg "
      "omitted]>, [arg omitted]>, [arg omitted]>",
      [](PrintingPolicy &Policy) {
        Policy.PrintAsCanonical = true;
        Policy.TypePrintLimit = 5;
      }));

  // The limit applies to the whole type, however it is nested.
  ASSERT_TRUE(PrintedTypeMatches(
      Code, {"-std=c++11", "-ftemplate-depth=100"}, Matcher,
      "F<F<F<F<[2 args omitted]>, [arg omitted]>, [arg omitted]>, [arg "
      "omitted]>",
      [](PrintingPolicy &Policy) {
        Policy.PrintAsCanonical = true;
        Policy.TypePrintLimit = 3;
      }));
}

TEST(TypePrinter, TypePrintLimitLeavesArePrinted) {
  // The arguments that don't open another list are printed, whatever the limit.
  constexpr char Code[] = R"cpp(
    template <class A, class B> struct F {};
    template <class... Ts> struct G {};
    G<int, F<int, int>, char, F<int, int>, F<int, int>> mixed;
  )cpp";
  auto Matcher = varDecl(hasName("mixed"), hasType(qualType().bind("id")));

  ASSERT_TRUE(PrintedTypeMatches(Code, {"-std=c++11"}, Matcher,
                                 "G<int, F<int, int>, char, [2 args omitted]>",
                                 [](PrintingPolicy &Policy) {
                                   Policy.PrintAsCanonical = true;
                                   Policy.TypePrintLimit = 3;
                                 }));

  // And without a limit, nothing is omitted.
  ASSERT_TRUE(
      PrintedTypeMatches(Code, {"-std=c++11"}, Matcher,
                         "G<int, F<int, int>, char, F<int, int>, F<int, int>>",
                         [](PrintingPolicy &Policy) {
                           Policy.PrintAsCanonical = true;
                           Policy.TypePrintLimit = 0;
                         }));
}

TEST(TypePrinter, TypePrintLimitFunctionParameters) {
  // Build the type directly, as a function type that uses the previous one for
  // both of its parameters, so that the test does not depend on how well Sema
  // copes with it. It would be 2^40 function types if spelled out in full.
  auto AST = tooling::buildASTFromCode("");
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  QualType T =
      Ctx.getFunctionType(Ctx.VoidTy, {}, FunctionProtoType::ExtProtoInfo());
  for (int I = 0; I < 40; ++I) {
    QualType Param = Ctx.getPointerType(T);
    T = Ctx.getFunctionType(Ctx.VoidTy, {Param, Param},
                            FunctionProtoType::ExtProtoInfo());
  }

  PrintingPolicy Policy = Ctx.getPrintingPolicy();
  Policy.TypePrintLimit = 5;
  EXPECT_EQ(T.getAsString(Policy),
            "void (void (*)(void (*)(void (*)(void (*)(void (*)([2 args "
            "omitted]), [arg omitted]), [arg omitted]), [arg omitted]), [arg "
            "omitted]), [arg omitted])");
}

TEST(TypePrinter, TypePrintLimitDefault) {
  auto AST = tooling::buildASTFromCode("");
  ASSERT_TRUE(AST);
  EXPECT_EQ(AST->getASTContext().getPrintingPolicy().TypePrintLimit, 32768u);
}

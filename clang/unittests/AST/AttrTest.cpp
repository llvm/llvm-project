//===- unittests/AST/AttrTests.cpp --- Attribute tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

using clang::ast_matchers::constantExpr;
using clang::ast_matchers::equals;
using clang::ast_matchers::functionDecl;
using clang::ast_matchers::has;
using clang::ast_matchers::hasDescendant;
using clang::ast_matchers::hasName;
using clang::ast_matchers::integerLiteral;
using clang::ast_matchers::match;
using clang::ast_matchers::selectFirst;
using clang::ast_matchers::stringLiteral;
using clang::ast_matchers::varDecl;
using clang::tooling::buildASTFromCode;
using clang::tooling::buildASTFromCodeWithArgs;

TEST(Attr, Doc) {
  EXPECT_THAT(Attr::getDocumentation(attr::Used).str(),
              testing::HasSubstr("The compiler must emit the definition even "
                                 "if it appears to be unused"));
}

const FunctionDecl *getFunctionNode(ASTUnit *AST, const std::string &Name) {
  auto Result =
      match(functionDecl(hasName(Name)).bind("fn"), AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  return Result[0].getNodeAs<FunctionDecl>("fn");
}

const VarDecl *getVariableNode(ASTUnit *AST, const std::string &Name) {
  auto Result = match(varDecl(hasName(Name)).bind("var"), AST->getASTContext());
  EXPECT_EQ(Result.size(), 1u);
  return Result[0].getNodeAs<VarDecl>("var");
}

template <class ModifiedTypeLoc>
void AssertAnnotatedAs(TypeLoc TL, llvm::StringRef annotation,
                       ModifiedTypeLoc &ModifiedTL,
                       const AnnotateTypeAttr **AnnotateOut = nullptr) {
  const auto AttributedTL = TL.getAs<AttributedTypeLoc>();
  ASSERT_FALSE(AttributedTL.isNull());
  ModifiedTL = AttributedTL.getModifiedLoc().getAs<ModifiedTypeLoc>();
  ASSERT_TRUE(ModifiedTL);

  ASSERT_NE(AttributedTL.getAttr(), nullptr);
  const auto *Annotate = dyn_cast<AnnotateTypeAttr>(AttributedTL.getAttr());
  ASSERT_NE(Annotate, nullptr);
  EXPECT_EQ(Annotate->getAnnotation(), annotation);
  if (AnnotateOut) {
    *AnnotateOut = Annotate;
  }
}

TEST(Attr, AnnotateType) {

  // Test that the AnnotateType attribute shows up in the right places and that
  // it stores its arguments correctly.

  auto AST = buildASTFromCode(R"cpp(
    void f(int* [[clang::annotate_type("foo", "arg1", 2)]] *,
           int [[clang::annotate_type("bar")]]);

    int [[clang::annotate_type("int")]] * [[clang::annotate_type("ptr")]]
      array[10] [[clang::annotate_type("arr")]];

    void (* [[clang::annotate_type("funcptr")]] fp)(void);

    struct S { int mem; };
    int [[clang::annotate_type("int")]]
    S::* [[clang::annotate_type("ptr_to_mem")]] ptr_to_member = &S::mem;

    // Function Type Attributes
    __attribute__((noreturn)) int f_noreturn();

    #define NO_RETURN __attribute__((noreturn))
    NO_RETURN int f_macro_attribue();

    int (__attribute__((noreturn)) f_paren_attribute)();

    int (
      NO_RETURN
      (
        __attribute__((warn_unused_result))
        (f_w_paren_and_attr)
      )
    ) ();
  )cpp");

  {
    const FunctionDecl *Func = getFunctionNode(AST.get(), "f");

    // First parameter.
    const auto PointerTL = Func->getParamDecl(0)
                               ->getTypeSourceInfo()
                               ->getTypeLoc()
                               .getAs<PointerTypeLoc>();
    ASSERT_FALSE(PointerTL.isNull());
    PointerTypeLoc PointerPointerTL;
    const AnnotateTypeAttr *Annotate;
    AssertAnnotatedAs(PointerTL.getPointeeLoc(), "foo", PointerPointerTL,
                      &Annotate);

    EXPECT_EQ(Annotate->args_size(), 2u);
    const auto *StringLit = selectFirst<StringLiteral>(
        "str", match(constantExpr(hasDescendant(stringLiteral().bind("str"))),
                     *Annotate->args_begin()[0], AST->getASTContext()));
    ASSERT_NE(StringLit, nullptr);
    EXPECT_EQ(StringLit->getString(), "arg1");
    EXPECT_EQ(match(constantExpr(has(integerLiteral(equals(2u)).bind("int"))),
                    *Annotate->args_begin()[1], AST->getASTContext())
                  .size(),
              1u);

    // Second parameter.
    BuiltinTypeLoc IntTL;
    AssertAnnotatedAs(Func->getParamDecl(1)->getTypeSourceInfo()->getTypeLoc(),
                      "bar", IntTL);
    EXPECT_EQ(IntTL.getType(), AST->getASTContext().IntTy);
  }

  {
    const VarDecl *Var = getVariableNode(AST.get(), "array");

    ArrayTypeLoc ArrayTL;
    AssertAnnotatedAs(Var->getTypeSourceInfo()->getTypeLoc(), "arr", ArrayTL);
    PointerTypeLoc PointerTL;
    AssertAnnotatedAs(ArrayTL.getElementLoc(), "ptr", PointerTL);
    BuiltinTypeLoc IntTL;
    AssertAnnotatedAs(PointerTL.getPointeeLoc(), "int", IntTL);
    EXPECT_EQ(IntTL.getType(), AST->getASTContext().IntTy);
  }

  {
    const VarDecl *Var = getVariableNode(AST.get(), "fp");

    PointerTypeLoc PointerTL;
    AssertAnnotatedAs(Var->getTypeSourceInfo()->getTypeLoc(), "funcptr",
                      PointerTL);
    ASSERT_TRUE(
        PointerTL.getPointeeLoc().IgnoreParens().getAs<FunctionTypeLoc>());
  }

  {
    const VarDecl *Var = getVariableNode(AST.get(), "ptr_to_member");

    MemberPointerTypeLoc MemberPointerTL;
    AssertAnnotatedAs(Var->getTypeSourceInfo()->getTypeLoc(), "ptr_to_mem",
                      MemberPointerTL);
    BuiltinTypeLoc IntTL;
    AssertAnnotatedAs(MemberPointerTL.getPointeeLoc(), "int", IntTL);
    EXPECT_EQ(IntTL.getType(), AST->getASTContext().IntTy);
  }

  {
    const FunctionDecl *Func = getFunctionNode(AST.get(), "f_noreturn");
    const FunctionTypeLoc FTL = Func->getFunctionTypeLoc();
    const FunctionType *FT = FTL.getTypePtr();

    EXPECT_TRUE(FT->getNoReturnAttr());
  }

  {
    for (auto should_have_func_type_loc : {
             "f_macro_attribue",
             "f_paren_attribute",
             "f_w_paren_and_attr",
         }) {
      llvm::errs() << "O: " << should_have_func_type_loc << "\n";
      const FunctionDecl *Func =
          getFunctionNode(AST.get(), should_have_func_type_loc);

      EXPECT_TRUE(Func->getFunctionTypeLoc());
    }
  }

  // The following test verifies getFunctionTypeLoc returns a type
  // which takes into account the attribute (instead of only the nake
  // type).
  //
  // This is hard to do with C/C++ because it seems using a function
  // type attribute with a C/C++ function declaration only results
  // with either:
  //
  // 1. It does NOT produce any AttributedType (for example it only
  //   sets one flag of the FunctionType's ExtInfo, e.g. NoReturn).
  // 2. It produces an AttributedType with modified type and
  //   equivalent type that are equal (for example, that's what
  //   happens with Calling Convention attributes).
  //
  // Fortunately, ObjC has one specific function type attribute that
  // creates an AttributedType with different modified type and
  // equivalent type.
  auto AST_ObjC = buildASTFromCodeWithArgs(
      R"objc(
    __attribute__((ns_returns_retained)) id f();
  )objc",
      {"-fobjc-arc", "-fsyntax-only", "-fobjc-runtime=macosx-10.7"},
      "input.mm");
  {
    const FunctionDecl *f = getFunctionNode(AST_ObjC.get(), "f");
    const FunctionTypeLoc FTL = f->getFunctionTypeLoc();

    const FunctionType *FT = FTL.getTypePtr();
    EXPECT_TRUE(FT->getExtInfo().getProducesResult());
  }

  // Test type annotation on an `__auto_type` type in C mode.
  AST = buildASTFromCodeWithArgs(R"c(
    __auto_type [[clang::annotate_type("auto")]] auto_var = 1;
  )c",
                                 {},
                                 "input.c");

  {
    const VarDecl *Var = getVariableNode(AST.get(), "auto_var");

    AutoTypeLoc AutoTL;
    AssertAnnotatedAs(Var->getTypeSourceInfo()->getTypeLoc(), "auto", AutoTL);
  }
}

TEST(Attr, RegularKeywordAttribute) {
  auto AST = clang::tooling::buildASTFromCode("");
  auto &Ctx = AST->getASTContext();
  auto Funcref = clang::WebAssemblyFuncrefAttr::CreateImplicit(Ctx);
  EXPECT_EQ(Funcref->getSyntax(), clang::AttributeCommonInfo::AS_Keyword);
  ASSERT_FALSE(Funcref->isRegularKeywordAttribute());

  auto Streaming = clang::ArmStreamingAttr::CreateImplicit(Ctx);
  EXPECT_EQ(Streaming->getSyntax(), clang::AttributeCommonInfo::AS_Keyword);
  ASSERT_TRUE(Streaming->isRegularKeywordAttribute());
}

} // namespace

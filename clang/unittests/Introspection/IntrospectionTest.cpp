//===- unittest/Introspection/IntrospectionTest.cpp ----------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for AST location API introspection.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/NodeIntrospection.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

template <typename T, typename MapType>
std::vector<std::pair<std::string, T>>
FormatExpected(const MapType &Accessors) {
  std::vector<std::pair<std::string, T>> Result;
  llvm::transform(llvm::make_filter_range(Accessors,
                                          [](const auto &Accessor) {
                                            return Accessor.first.isValid();
                                          }),
                  std::back_inserter(Result), [](const auto &Accessor) {
                    return std::make_pair(
                        LocationCallFormatterCpp::format(*Accessor.second),
                        Accessor.first);
                  });
  return Result;
}

#define STRING_LOCATION_PAIR(INSTANCE, LOC) Pair(#LOC, INSTANCE->LOC)

#define STRING_LOCATION_STDPAIR(INSTANCE, LOC)                                 \
  std::make_pair(std::string(#LOC), INSTANCE->LOC)

/**
  A test formatter for a hypothetical language which needs
  neither casts nor '->'.
*/
class LocationCallFormatterSimple {
public:
  static void print(const LocationCall &Call, llvm::raw_ostream &OS) {
    if (Call.isCast()) {
      if (const LocationCall *On = Call.on())
        print(*On, OS);
      return;
    }
    if (const LocationCall *On = Call.on()) {
      print(*On, OS);
      OS << '.';
    }
    OS << Call.name() << "()";
  }

  static std::string format(const LocationCall &Call) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    print(Call, OS);
    OS.flush();
    return Result;
  }
};

TEST(Introspection, SourceLocations_CallContainer) {
  SourceLocationMap slm;
  SharedLocationCall Prefix;
  slm.insert(std::make_pair(
      SourceLocation(),
      llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "getSourceRange")));
  EXPECT_EQ(slm.size(), 1u);

  auto callTypeLoc =
      llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "getTypeLoc");
  slm.insert(std::make_pair(
      SourceLocation(),
      llvm::makeIntrusiveRefCnt<LocationCall>(callTypeLoc, "getSourceRange")));
  EXPECT_EQ(slm.size(), 2u);
}

TEST(Introspection, SourceLocations_CallContainer2) {
  SourceRangeMap slm;
  SharedLocationCall Prefix;
  slm.insert(
      std::make_pair(SourceRange(), llvm::makeIntrusiveRefCnt<LocationCall>(
                                        Prefix, "getCXXOperatorNameRange")));
  EXPECT_EQ(slm.size(), 1u);

  slm.insert(std::make_pair(
      SourceRange(),
      llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "getSourceRange")));
  EXPECT_EQ(slm.size(), 2u);
}

TEST(Introspection, SourceLocations_CallChainFormatting) {
  SharedLocationCall Prefix;
  auto chainedCall = llvm::makeIntrusiveRefCnt<LocationCall>(
      llvm::makeIntrusiveRefCnt<LocationCall>(Prefix, "getTypeLoc"),
      "getSourceRange");
  EXPECT_EQ(LocationCallFormatterCpp::format(*chainedCall),
            "getTypeLoc().getSourceRange()");
}

TEST(Introspection, SourceLocations_Formatter) {
  SharedLocationCall Prefix;
  auto chainedCall = llvm::makeIntrusiveRefCnt<LocationCall>(
      llvm::makeIntrusiveRefCnt<LocationCall>(
          llvm::makeIntrusiveRefCnt<LocationCall>(
              llvm::makeIntrusiveRefCnt<LocationCall>(
                  Prefix, "getTypeSourceInfo", LocationCall::ReturnsPointer),
              "getTypeLoc"),
          "getAs<clang::TypeSpecTypeLoc>", LocationCall::IsCast),
      "getNameLoc");

  EXPECT_EQ("getTypeSourceInfo()->getTypeLoc().getAs<clang::TypeSpecTypeLoc>()."
            "getNameLoc()",
            LocationCallFormatterCpp::format(*chainedCall));
  EXPECT_EQ("getTypeSourceInfo().getTypeLoc().getNameLoc()",
            LocationCallFormatterSimple::format(*chainedCall));
}

TEST(Introspection, SourceLocations_Stmt) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST = buildASTFromCode("void foo() {} void bar() { foo(); }", "foo.cpp",
                              std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          callExpr(callee(functionDecl(hasName("foo")))).bind("fooCall"))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  auto *FooCall = BoundNodes[0].getNodeAs<CallExpr>("fooCall");

  auto Result = NodeIntrospection::GetLocations(FooCall);

  auto ExpectedLocations =
    FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(STRING_LOCATION_PAIR(FooCall, getBeginLoc()),
                           STRING_LOCATION_PAIR(FooCall, getEndLoc()),
                           STRING_LOCATION_PAIR(FooCall, getExprLoc()),
                           STRING_LOCATION_PAIR(FooCall, getRParenLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(STRING_LOCATION_PAIR(
                                  FooCall, getSourceRange())));
}

TEST(Introspection, SourceLocations_Decl) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
namespace ns1 {
namespace ns2 {
template <typename T, typename U> struct Foo {};
template <typename T, typename U> struct Bar {
  struct Nested {
    template <typename A, typename B>
    Foo<A, B> method(int i, bool b) const noexcept(true);
  };
};
} // namespace ns2
} // namespace ns1

template <typename T, typename U>
template <typename A, typename B>
ns1::ns2::Foo<A, B> ns1::ns2::Bar<T, U>::Nested::method(int i, bool b) const
    noexcept(true) {}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          cxxMethodDecl(hasName("method"), isDefinition()).bind("method"))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *MethodDecl = BoundNodes[0].getNodeAs<CXXMethodDecl>("method");

  auto Result = NodeIntrospection::GetLocations(MethodDecl);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  // clang-format off
  std::vector<std::pair<std::string, SourceLocation>> ActualLocations{
STRING_LOCATION_STDPAIR(MethodDecl, getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getBodyRBrace()),
STRING_LOCATION_STDPAIR(MethodDecl, getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getInnerLocStart()),
STRING_LOCATION_STDPAIR(MethodDecl, getLocation()),
STRING_LOCATION_STDPAIR(MethodDecl, getNameInfo().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getNameInfo().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getNameInfo().getLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getOuterLocStart()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getLParenLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getLocalRangeBegin()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getLocalRangeEnd()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getRParenLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSpecEndLoc()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSpecStartLoc())
  };
  // clang-format on

  EXPECT_EQ(ExpectedLocations, ActualLocations);

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  llvm::sort(ExpectedRanges, [](const auto &LHS, const auto &RHS) {
    return LHS.first < RHS.first;
  });

  // clang-format off
  EXPECT_EQ(
            llvm::ArrayRef(ExpectedRanges),
      (ArrayRef<std::pair<std::string, SourceRange>>{
STRING_LOCATION_STDPAIR(MethodDecl, getExceptionSpecSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getNameInfo().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getParametersSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getPrefix().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getPrefix().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getPrefix().getTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getQualifierLoc().getTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getReturnTypeSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getExceptionSpecRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getParensRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getAs<clang::FunctionTypeLoc>().getReturnLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getPrefix().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::ElaboratedTypeLoc>().getQualifierLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_STDPAIR(MethodDecl, getTypeSourceInfo()->getTypeLoc().getSourceRange())
  }));
  // clang-format on
}

TEST(Introspection, SourceLocations_NNS) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
namespace ns
{
  struct A {
  void foo();
};
}
void ns::A::foo() {}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(nestedNameSpecifierLoc().bind("nns"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *NNS = BoundNodes[0].getNodeAs<NestedNameSpecifierLoc>("nns");

  auto Result = NodeIntrospection::GetLocations(*NNS);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  EXPECT_EQ(
      llvm::ArrayRef(ExpectedLocations),
      (ArrayRef<std::pair<std::string, SourceLocation>>{
          STRING_LOCATION_STDPAIR(NNS, getBeginLoc()),
          STRING_LOCATION_STDPAIR(NNS, getEndLoc()),
          STRING_LOCATION_STDPAIR(NNS, getLocalBeginLoc()),
          STRING_LOCATION_STDPAIR(NNS, getLocalEndLoc()),
          STRING_LOCATION_STDPAIR(NNS, getPrefix().getBeginLoc()),
          STRING_LOCATION_STDPAIR(NNS, getPrefix().getEndLoc()),
          STRING_LOCATION_STDPAIR(NNS, getPrefix().getLocalBeginLoc()),
          STRING_LOCATION_STDPAIR(NNS, getPrefix().getLocalEndLoc()),
          STRING_LOCATION_STDPAIR(
              NNS, getTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
          STRING_LOCATION_STDPAIR(NNS, getTypeLoc().getBeginLoc()),
          STRING_LOCATION_STDPAIR(NNS, getTypeLoc().getEndLoc())}));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(NNS, getPrefix().getLocalSourceRange()),
          STRING_LOCATION_PAIR(NNS, getPrefix().getSourceRange()),
          STRING_LOCATION_PAIR(NNS, getLocalSourceRange()),
          STRING_LOCATION_PAIR(NNS, getSourceRange()),
          STRING_LOCATION_PAIR(NNS, getTypeLoc().getSourceRange()),
          STRING_LOCATION_PAIR(NNS, getTypeLoc().getLocalSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Type) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<typename T>
  struct A {
  void foo();
};

void foo()
{
  A<int> a;
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(TA, getLocation()),
STRING_LOCATION_PAIR(TA,
  getTypeSourceInfo()->getTypeLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_PAIR(TA,
  getTypeSourceInfo()->getTypeLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(
    TA, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(
    TA, getTypeSourceInfo()->getTypeLoc().getEndLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(TA, getSourceRange()),
          STRING_LOCATION_PAIR(
              TA, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
          STRING_LOCATION_PAIR(
              TA, getTypeSourceInfo()->getTypeLoc().getLocalSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Decl) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<void(*Ty)()>
void test2() {}
void doNothing() {}
void test() {
    test2<doNothing>();
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Nullptr) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<void(*Ty)()>
void test2() {}
void doNothing() {}
void test() {
    test2<nullptr>();
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Integral) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<int>
void test2() {}
void test() {
    test2<42>();
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Template) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<typename T> class A;
template <template <typename> class T> void foo();
void bar()
{
  foo<A>();
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation()),
                           STRING_LOCATION_PAIR(TA, getTemplateNameLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_TemplateExpansion) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST = buildASTFromCodeWithArgs(
      R"cpp(
template<template<typename> class ...> class B { };
  template<template<typename> class ...T> class C {
  B<T...> testTemplateExpansion;
};
)cpp",
      {"-fno-delayed-template-parsing"}, "foo.cpp", "clang-tool",
      std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation()),
                           STRING_LOCATION_PAIR(TA, getTemplateNameLoc()),
                           STRING_LOCATION_PAIR(TA, getTemplateEllipsisLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Expression) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<int, int = 0> class testExpr;
template<int I> class testExpr<I> { };
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getLocation())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(STRING_LOCATION_PAIR(TA, getSourceRange())));
}

TEST(Introspection, SourceLocations_TA_Pack) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST = buildASTFromCodeWithArgs(
      R"cpp(
template<typename... T> class A {};
void foo()
{
    A<int> ai;
}
)cpp",
      {"-fno-delayed-template-parsing"}, "foo.cpp", "clang-tool",
      std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(templateArgumentLoc().bind("ta"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TA = BoundNodes[0].getNodeAs<TemplateArgumentLoc>("ta");

  auto Result = NodeIntrospection::GetLocations(*TA);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(TA, getLocation()),
STRING_LOCATION_PAIR(TA,
  getTypeSourceInfo()->getTypeLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_PAIR(TA,
  getTypeSourceInfo()->getTypeLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(
    TA, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(
    TA, getTypeSourceInfo()->getTypeLoc().getEndLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(TA, getSourceRange()),
          STRING_LOCATION_PAIR(
              TA, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
          STRING_LOCATION_PAIR(
              TA, getTypeSourceInfo()->getTypeLoc().getLocalSourceRange())));
}

TEST(Introspection, SourceLocations_CXXCtorInitializer_base) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
struct A {
};

struct B : A {
  B() : A() {}
};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxConstructorDecl(
          hasAnyConstructorInitializer(cxxCtorInitializer().bind("init"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *CtorInit = BoundNodes[0].getNodeAs<CXXCtorInitializer>("init");

  auto Result = NodeIntrospection::GetLocations(CtorInit);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(CtorInit, getLParenLoc()),
STRING_LOCATION_PAIR(CtorInit, getRParenLoc()),
STRING_LOCATION_PAIR(CtorInit, getSourceLocation()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getEndLoc())
 ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getSourceRange())));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXCtorInitializer_member) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
struct A {
  int m_i;
  A() : m_i(42) {}
};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxConstructorDecl(
          hasAnyConstructorInitializer(cxxCtorInitializer().bind("init"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *CtorInit = BoundNodes[0].getNodeAs<CXXCtorInitializer>("init");

  auto Result = NodeIntrospection::GetLocations(CtorInit);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
                  STRING_LOCATION_PAIR(CtorInit, getLParenLoc()),
                  STRING_LOCATION_PAIR(CtorInit, getMemberLocation()),
                  STRING_LOCATION_PAIR(CtorInit, getRParenLoc()),
                  STRING_LOCATION_PAIR(CtorInit, getSourceLocation())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(STRING_LOCATION_PAIR(
                                  CtorInit, getSourceRange())));
}

TEST(Introspection, SourceLocations_CXXCtorInitializer_ctor) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
struct C {
  C() : C(42) {}
  C(int) {}
};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxConstructorDecl(
          hasAnyConstructorInitializer(cxxCtorInitializer().bind("init"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *CtorInit = BoundNodes[0].getNodeAs<CXXCtorInitializer>("init");

  auto Result = NodeIntrospection::GetLocations(CtorInit);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedLocations,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(CtorInit, getLParenLoc()),
STRING_LOCATION_PAIR(CtorInit, getRParenLoc()),
STRING_LOCATION_PAIR(CtorInit, getSourceLocation()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(CtorInit, getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXCtorInitializer_pack) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST = buildASTFromCodeWithArgs(
      R"cpp(
template<typename... T>
struct Templ {
};

template<typename... T>
struct D : Templ<T...> {
  D(T... t) : Templ<T>(t)... {}
};
)cpp",
      {"-fno-delayed-template-parsing"}, "foo.cpp", "clang-tool",
      std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxConstructorDecl(
          hasAnyConstructorInitializer(cxxCtorInitializer().bind("init"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *CtorInit = BoundNodes[0].getNodeAs<CXXCtorInitializer>("init");

  auto Result = NodeIntrospection::GetLocations(CtorInit);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  // clang-format off
  EXPECT_EQ(
     llvm::ArrayRef(ExpectedLocations),
      (ArrayRef<std::pair<std::string, SourceLocation>>{
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getEllipsisLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getLParenLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getMemberLocation()),
STRING_LOCATION_STDPAIR(CtorInit, getRParenLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getSourceLocation()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc())
  }));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getBaseClassLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(CtorInit, getSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXBaseSpecifier_plain) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
class A {};
class B : A {};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxRecordDecl(hasDirectBase(
          cxxBaseSpecifier(hasType(asString("A"))).bind("base"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Base = BoundNodes[0].getNodeAs<CXXBaseSpecifier>("base");

  auto Result = NodeIntrospection::GetLocations(Base);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getBaseTypeLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getSourceRange())
    ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXBaseSpecifier_accessspec) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
class A {};
class B : public A {};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxRecordDecl(hasDirectBase(
          cxxBaseSpecifier(hasType(asString("A"))).bind("base"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Base = BoundNodes[0].getNodeAs<CXXBaseSpecifier>("base");

  auto Result = NodeIntrospection::GetLocations(Base);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getBeginLoc()),
STRING_LOCATION_PAIR(Base, getBaseTypeLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXBaseSpecifier_virtual) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
class A {};
class B {};
class C : virtual B, A {};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxRecordDecl(hasDirectBase(
          cxxBaseSpecifier(hasType(asString("A"))).bind("base"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Base = BoundNodes[0].getNodeAs<CXXBaseSpecifier>("base");

  auto Result = NodeIntrospection::GetLocations(Base);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getBaseTypeLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXBaseSpecifier_template_base) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
template<typename T, typename U>
class A {};
class B : A<int, bool> {};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes =
      ast_matchers::match(decl(hasDescendant(cxxRecordDecl(
                              hasDirectBase(cxxBaseSpecifier().bind("base"))))),
                          TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Base = BoundNodes[0].getNodeAs<CXXBaseSpecifier>("base");

  auto Result = NodeIntrospection::GetLocations(Base);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getBaseTypeLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getBeginLoc()),
STRING_LOCATION_PAIR(Base, getBeginLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getTemplateNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getLAngleLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getAs<clang::TemplateSpecializationTypeLoc>().getRAngleLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::ElaboratedTypeLoc>().getNamedTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_CXXBaseSpecifier_pack) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST = buildASTFromCodeWithArgs(
      R"cpp(
template<typename... T>
struct Templ : T... {
};
)cpp",
      {"-fno-delayed-template-parsing"}, "foo.cpp", "clang-tool",
      std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes =
      ast_matchers::match(decl(hasDescendant(cxxRecordDecl(
                              hasDirectBase(cxxBaseSpecifier().bind("base"))))),
                          TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Base = BoundNodes[0].getNodeAs<CXXBaseSpecifier>("base");

  auto Result = NodeIntrospection::GetLocations(Base);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getBaseTypeLoc()),
STRING_LOCATION_PAIR(Base, getEllipsisLoc()),
STRING_LOCATION_PAIR(Base, getBeginLoc()),
STRING_LOCATION_PAIR(Base, getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getEndLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getBeginLoc())
  ));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(
STRING_LOCATION_PAIR(Base, getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(Base, getTypeSourceInfo()->getTypeLoc().getLocalSourceRange())
  ));
  // clang-format on
}

TEST(Introspection, SourceLocations_FunctionProtoTypeLoc) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
int foo();
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(loc(functionProtoType()).bind("tl"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TL = BoundNodes[0].getNodeAs<TypeLoc>("tl");
  auto Result = NodeIntrospection::GetLocations(*TL);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  // clang-format off
  EXPECT_EQ(
      llvm::ArrayRef(ExpectedLocations),
          (ArrayRef<std::pair<std::string, SourceLocation>>{
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getLParenLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getLocalRangeBegin()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getLocalRangeEnd()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getRParenLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(TL, getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getEndLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getEndLoc())
        }));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(TL, getAs<clang::FunctionTypeLoc>().getParensRange()),
STRING_LOCATION_PAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getAs<clang::FunctionTypeLoc>().getReturnLoc().getSourceRange()),
STRING_LOCATION_PAIR(TL, getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(TL, getSourceRange())
          ));
  // clang-format on
}

TEST(Introspection, SourceLocations_PointerTypeLoc) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
int* i;
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          varDecl(hasName("i"), hasDescendant(loc(pointerType()).bind("tl"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TL = BoundNodes[0].getNodeAs<TypeLoc>("tl");
  auto Result = NodeIntrospection::GetLocations(*TL);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  // clang-format off
  EXPECT_EQ(
      llvm::ArrayRef(ExpectedLocations),
      (ArrayRef<std::pair<std::string, SourceLocation>>{
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getEndLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getSigilLoc()),
STRING_LOCATION_STDPAIR(TL, getAs<clang::PointerTypeLoc>().getStarLoc()),
STRING_LOCATION_STDPAIR(TL, getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getEndLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getAs<clang::BuiltinTypeLoc>().getBuiltinLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getAs<clang::BuiltinTypeLoc>().getNameLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getBeginLoc()),
STRING_LOCATION_STDPAIR(TL, getNextTypeLoc().getEndLoc())
}));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  // clang-format off
  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
STRING_LOCATION_PAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getAs<clang::PointerTypeLoc>().getPointeeLoc().getSourceRange()),
STRING_LOCATION_PAIR(TL, getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getNextTypeLoc().getLocalSourceRange()),
STRING_LOCATION_PAIR(TL, getNextTypeLoc().getSourceRange()),
STRING_LOCATION_PAIR(TL, getSourceRange())
          ));
  // clang-format on
}

#ifndef _WIN32
// This test doesn't work on windows due to use of the typeof extension.
TEST(Introspection, SourceLocations_TypeOfTypeLoc) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
typeof (static_cast<void *>(0)) i;
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(
          varDecl(hasName("i"), hasDescendant(loc(type()).bind("tl"))))),
      TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *TL = BoundNodes[0].getNodeAs<TypeLoc>("tl");
  auto Result = NodeIntrospection::GetLocations(*TL);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  EXPECT_THAT(ExpectedLocations,
              UnorderedElementsAre(
                  STRING_LOCATION_PAIR(TL, getBeginLoc()),
                  STRING_LOCATION_PAIR(TL, getEndLoc()),
                  STRING_LOCATION_PAIR(
                      TL, getAs<clang::TypeOfExprTypeLoc>().getTypeofLoc()),
                  STRING_LOCATION_PAIR(
                      TL, getAs<clang::TypeOfExprTypeLoc>().getLParenLoc()),
                  STRING_LOCATION_PAIR(
                      TL, getAs<clang::TypeOfExprTypeLoc>().getRParenLoc())));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(
                  STRING_LOCATION_PAIR(TL, getLocalSourceRange()),
                  STRING_LOCATION_PAIR(TL, getSourceRange()),
                  STRING_LOCATION_PAIR(
                      TL, getAs<clang::TypeOfExprTypeLoc>().getParensRange())));
}
#endif

TEST(Introspection, SourceLocations_DeclarationNameInfo_Dtor) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
class Foo
{
  ~Foo() {}
};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxDestructorDecl(hasName("~Foo")).bind("dtor"))), TU,
      Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Dtor = BoundNodes[0].getNodeAs<CXXDestructorDecl>("dtor");
  auto NI = Dtor->getNameInfo();
  auto Result = NodeIntrospection::GetLocations(NI);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  // clang-format off
  EXPECT_EQ(
      llvm::ArrayRef(ExpectedLocations),
      (ArrayRef<std::pair<std::string, SourceLocation>>{
          STRING_LOCATION_STDPAIR((&NI), getBeginLoc()),
          STRING_LOCATION_STDPAIR((&NI), getEndLoc()),
          STRING_LOCATION_STDPAIR((&NI), getLoc()),
          STRING_LOCATION_STDPAIR((&NI),
getNamedTypeInfo()->getTypeLoc().getAs<clang::TypeSpecTypeLoc>().getNameLoc()),
          STRING_LOCATION_STDPAIR(
              (&NI), getNamedTypeInfo()->getTypeLoc().getBeginLoc()),
          STRING_LOCATION_STDPAIR(
              (&NI), getNamedTypeInfo()->getTypeLoc().getEndLoc())}));
  // clang-format on

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(
              (&NI), getNamedTypeInfo()->getTypeLoc().getLocalSourceRange()),
          STRING_LOCATION_PAIR(
              (&NI), getNamedTypeInfo()->getTypeLoc().getSourceRange()),
          STRING_LOCATION_PAIR((&NI), getSourceRange())));
}

TEST(Introspection, SourceLocations_DeclarationNameInfo_CRef) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();

  auto AST = buildASTFromCodeWithArgs(
      R"cpp(
template<typename T>
struct MyContainer
{
    template <typename U>
    void pushBack();
};

template<typename T>
void foo()
{
    MyContainer<T> mc;
    mc.template pushBack<int>();
}
)cpp",
      {"-fno-delayed-template-parsing"}, "foo.cpp", "clang-tool",
      std::make_shared<PCHContainerOperations>());

  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxDependentScopeMemberExpr(hasMemberName("pushBack")).bind("member"))), TU,
      Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Member = BoundNodes[0].getNodeAs<CXXDependentScopeMemberExpr>("member");
  auto Result = NodeIntrospection::GetLocations(Member);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  EXPECT_EQ(
      llvm::ArrayRef(ExpectedLocations),
      (ArrayRef<std::pair<std::string, SourceLocation>>{
          STRING_LOCATION_STDPAIR(Member, getBeginLoc()),
          STRING_LOCATION_STDPAIR(Member, getEndLoc()),
          STRING_LOCATION_STDPAIR(Member, getExprLoc()),
          STRING_LOCATION_STDPAIR(Member, getLAngleLoc()),
          STRING_LOCATION_STDPAIR(Member, getMemberLoc()),
          STRING_LOCATION_STDPAIR(Member, getMemberNameInfo().getBeginLoc()),
          STRING_LOCATION_STDPAIR(Member, getMemberNameInfo().getEndLoc()),
          STRING_LOCATION_STDPAIR(Member, getMemberNameInfo().getLoc()),
          STRING_LOCATION_STDPAIR(Member, getOperatorLoc()),
          STRING_LOCATION_STDPAIR(Member, getRAngleLoc()),
          STRING_LOCATION_STDPAIR(Member, getTemplateKeywordLoc())}));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(
      ExpectedRanges,
      UnorderedElementsAre(
          STRING_LOCATION_PAIR(Member, getMemberNameInfo().getSourceRange()),
          STRING_LOCATION_PAIR(Member, getSourceRange())
          ));
}

TEST(Introspection, SourceLocations_DeclarationNameInfo_ConvOp) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
class Foo
{
  bool operator==(const Foo&) const { return false; }
};
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(cxxMethodDecl().bind("opeq"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *Opeq = BoundNodes[0].getNodeAs<CXXMethodDecl>("opeq");
  auto NI = Opeq->getNameInfo();
  auto Result = NodeIntrospection::GetLocations(NI);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  EXPECT_EQ(llvm::ArrayRef(ExpectedLocations),
            (ArrayRef<std::pair<std::string, SourceLocation>>{
                STRING_LOCATION_STDPAIR((&NI), getBeginLoc()),
                STRING_LOCATION_STDPAIR((&NI), getEndLoc()),
                STRING_LOCATION_STDPAIR((&NI), getLoc())}));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges,
              UnorderedElementsAre(
                  STRING_LOCATION_PAIR((&NI), getSourceRange()),
                  STRING_LOCATION_PAIR((&NI), getCXXOperatorNameRange())));
}

TEST(Introspection, SourceLocations_DeclarationNameInfo_LitOp) {
  if (!NodeIntrospection::hasIntrospectionSupport())
    GTEST_SKIP();
  auto AST =
      buildASTFromCode(R"cpp(
long double operator"" _identity ( long double val )
{
    return val;
}
)cpp",
                       "foo.cpp", std::make_shared<PCHContainerOperations>());
  auto &Ctx = AST->getASTContext();
  auto &TU = *Ctx.getTranslationUnitDecl();

  auto BoundNodes = ast_matchers::match(
      decl(hasDescendant(functionDecl().bind("litop"))), TU, Ctx);

  EXPECT_EQ(BoundNodes.size(), 1u);

  const auto *LitOp = BoundNodes[0].getNodeAs<FunctionDecl>("litop");
  auto NI = LitOp->getNameInfo();
  auto Result = NodeIntrospection::GetLocations(NI);

  auto ExpectedLocations =
      FormatExpected<SourceLocation>(Result.LocationAccessors);

  llvm::sort(ExpectedLocations);

  EXPECT_EQ(llvm::ArrayRef(ExpectedLocations),
            (ArrayRef<std::pair<std::string, SourceLocation>>{
                STRING_LOCATION_STDPAIR((&NI), getBeginLoc()),
                STRING_LOCATION_STDPAIR((&NI), getCXXLiteralOperatorNameLoc()),
                STRING_LOCATION_STDPAIR((&NI), getEndLoc()),
                STRING_LOCATION_STDPAIR((&NI), getLoc())}));

  auto ExpectedRanges = FormatExpected<SourceRange>(Result.RangeAccessors);

  EXPECT_THAT(ExpectedRanges, UnorderedElementsAre(STRING_LOCATION_PAIR(
                                  (&NI), getSourceRange())));
}

//===- unittest/AST/ASTImporterObjCTest.cpp -============================--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the correct import of AST nodes related to Objective-C and
// Objective-C++.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclContextInternals.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "gtest/gtest.h"

#include "ASTImporterFixtures.h"

using namespace clang::ast_matchers;
using namespace clang;

namespace {
struct ImportObjCDecl : ASTImporterOptionSpecificTestBase {};
} // namespace

TEST_P(ImportObjCDecl, ImplicitlyDeclareSelf) {
  Decl *FromTU = getTuDecl(R"(
                           __attribute__((objc_root_class))
                           @interface Root
                           @end
                           @interface C : Root
                             -(void)method;
                           @end
                           @implementation C
                             -(void)method {}
                           @end
                           )",
                           Lang_OBJCXX, "input.mm");
  auto *FromMethod = LastDeclMatcher<ObjCMethodDecl>().match(
      FromTU, namedDecl(hasName("method")));
  ASSERT_TRUE(FromMethod);
  auto ToMethod = Import(FromMethod, Lang_OBJCXX);
  ASSERT_TRUE(ToMethod);

  // Both methods should have their implicit parameters.
  EXPECT_TRUE(FromMethod->getSelfDecl() != nullptr);
  EXPECT_TRUE(ToMethod->getSelfDecl() != nullptr);
}

TEST_P(ImportObjCDecl, ObjPropertyNameConflict) {
  // Tests that properties that share the same name are correctly imported.
  // This is only possible with one instance and one class property.
  Decl *FromTU = getTuDecl(R"(
                           @interface DupProp{}
                           @property (class) int prop;
                           @property int prop;
                           @end
                           )",
                           Lang_OBJCXX, "input.mm");
  auto *FromClass = FirstDeclMatcher<ObjCInterfaceDecl>().match(
      FromTU, namedDecl(hasName("DupProp")));
  auto ToClass = Import(FromClass, Lang_OBJCXX);
  ASSERT_TRUE(ToClass);
  // We should have one class and one instance property.
  ASSERT_EQ(
      1, std::distance(ToClass->classprop_begin(), ToClass->classprop_end()));
  ASSERT_EQ(1,
            std::distance(ToClass->instprop_begin(), ToClass->instprop_end()));
  for (clang::ObjCPropertyDecl *prop : ToClass->properties()) {
    // All properties should have a getter and a setter.
    ASSERT_TRUE(prop->getGetterMethodDecl());
    ASSERT_TRUE(prop->getSetterMethodDecl());
    // The getters/setters should be able to find the right associated property.
    ASSERT_EQ(prop->getGetterMethodDecl()->findPropertyDecl(), prop);
    ASSERT_EQ(prop->getSetterMethodDecl()->findPropertyDecl(), prop);
  }
}

TEST_P(ImportObjCDecl, ImportObjCTypeParamDecl) {
  Decl *FromTU = getTuDecl(
      R"(
        @interface X <FirstParam: id>
        @end
      )",
      Lang_OBJCXX, "input.mm");
  auto *FromInterfaceDecl = FirstDeclMatcher<ObjCInterfaceDecl>().match(
      FromTU, namedDecl(hasName("X")));
  auto *FromTypeParamDecl =
      FromInterfaceDecl->getTypeParamListAsWritten()->front();

  auto *ToTypeParamDeclImported = Import(FromTypeParamDecl, Lang_OBJCXX);
  ASSERT_TRUE(ToTypeParamDeclImported);
}

static const auto ObjCTestArrayForRunOptions =
    std::array<std::vector<std::string>, 2>{
        {std::vector<std::string>{"-fno-objc-arc"},
         std::vector<std::string>{"-fobjc-arc"}}};

const auto ObjCTestValuesForRunOptions =
    ::testing::ValuesIn(ObjCTestArrayForRunOptions);

INSTANTIATE_TEST_SUITE_P(ParameterizedTests, ImportObjCDecl,
                         ObjCTestValuesForRunOptions);

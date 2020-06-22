//===- InternalNamesTest.cpp -- InternalNames unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InternalNames.h"
#include "gtest/gtest.h"
#include <string>

using namespace fir;
using llvm::SmallVector;
using llvm::StringRef;

struct DeconstructedName {
  DeconstructedName(llvm::ArrayRef<std::string> modules,
      llvm::Optional<std::string> host, llvm::StringRef name,
      llvm::ArrayRef<std::int64_t> kinds)
      : modules{modules.begin(), modules.end()}, host{host}, name{name},
        kinds{kinds.begin(), kinds.end()} {}

  bool isObjEqual(const NameUniquer::DeconstructedName &actualObj) {
    if ((actualObj.name == name) && (actualObj.modules == modules) &&
        (actualObj.host == host) && (actualObj.kinds == kinds)) {
      return true;
    }
    return false;
  }

private:
  llvm::SmallVector<std::string, 2> modules;
  llvm::Optional<std::string> host;
  std::string name;
  llvm::SmallVector<std::int64_t, 4> kinds;
};

void validateDeconstructedName(
    std::pair<NameUniquer::NameKind, NameUniquer::DeconstructedName> &actual,
    NameUniquer::NameKind &expectedNameKind,
    struct DeconstructedName &components) {
  EXPECT_EQ(actual.first, expectedNameKind)
      << "Possible error: NameKind mismatch";
  ASSERT_TRUE(components.isObjEqual(actual.second))
      << "Possible error: DeconstructedName mismatch";
}

TEST(InternalNamesTest, doCommonBlockTest) {
  NameUniquer obj;
  std::string actual = obj.doCommonBlock("hello");
  std::string actualBlank = obj.doCommonBlock("");
  std::string expectedMangledName = "_QBhello";
  std::string expectedMangledNameBlank = "_QB";
  ASSERT_EQ(actual, expectedMangledName);
  ASSERT_EQ(actualBlank, expectedMangledNameBlank);
}

TEST(InternalNamesTest, doGeneratedTest) {
  NameUniquer obj;
  std::string actual = obj.doGenerated("@MAIN");
  std::string expectedMangledName = "_QQ@MAIN";
  ASSERT_EQ(actual, expectedMangledName);

  std::string actual1 = obj.doGenerated("@_ZNSt8ios_base4InitC1Ev");
  std::string expectedMangledName1 = "_QQ@_ZNSt8ios_base4InitC1Ev";
  ASSERT_EQ(actual1, expectedMangledName1);

  std::string actual2 = obj.doGenerated("_QQ@MAIN");
  std::string expectedMangledName2 = "_QQ_QQ@MAIN";
  ASSERT_EQ(actual2, expectedMangledName2);
}

TEST(InternalNamesTest, doConstantTest) {
  NameUniquer obj;
  std::string actual = obj.doConstant({"mod1", "mod2"}, {"foo"}, "Hello");
  std::string expectedMangledName = "_QMmod1Smod2FfooEChello";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doProcedureTest) {
  NameUniquer obj;
  std::string actual = obj.doProcedure({"mod1", "mod2"}, {}, "HeLLo");
  std::string expectedMangledName = "_QMmod1Smod2Phello";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doTypeTest) {
  NameUniquer obj;
  std::string actual = obj.doType({}, {}, "mytype", {4, -1});
  std::string expectedMangledName = "_QTmytypeK4KN1";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doIntrinsicTypeDescriptorTest) {
  using IntrinsicType = fir::NameUniquer::IntrinsicType;
  NameUniquer obj;
  std::string actual =
      obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::REAL, 42);
  std::string expectedMangledName = "_QCrealK42";
  ASSERT_EQ(actual, expectedMangledName);

  actual = obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::REAL, {});
  expectedMangledName = "_QCrealK0";
  ASSERT_EQ(actual, expectedMangledName);

  actual = obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::INTEGER, 3);
  expectedMangledName = "_QCintegerK3";
  ASSERT_EQ(actual, expectedMangledName);

  actual = obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::LOGICAL, 2);
  expectedMangledName = "_QClogicalK2";
  ASSERT_EQ(actual, expectedMangledName);

  actual = obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::CHARACTER, 4);
  expectedMangledName = "_QCcharacterK4";
  ASSERT_EQ(actual, expectedMangledName);

  actual = obj.doIntrinsicTypeDescriptor({}, {}, IntrinsicType::COMPLEX, 4);
  expectedMangledName = "_QCcomplexK4";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doDispatchTableTest) {
  NameUniquer obj;
  std::string actual = obj.doDispatchTable({}, {}, "MyTYPE", {2, 8, 18});
  std::string expectedMangledName = "_QDTmytypeK2K8K18";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doTypeDescriptorTest) {
  NameUniquer obj;
  std::string actual = obj.doTypeDescriptor(
      {StringRef("moD1")}, {StringRef("foo")}, "MyTYPE", {2, 8});
  std::string expectedMangledName = "_QMmod1FfooCTmytypeK2K8";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doVariableTest) {
  NameUniquer obj;
  std::string actual = obj.doVariable(
      {"mod1", "mod2"}, {""}, "intvar"); // Function is present and is blank.
  std::string expectedMangledName = "_QMmod1Smod2FEintvar";
  ASSERT_EQ(actual, expectedMangledName);

  std::string actual2 = obj.doVariable(
      {"mod1", "mod2"}, {}, "intVariable"); // Function is not present.
  std::string expectedMangledName2 = "_QMmod1Smod2Eintvariable";
  ASSERT_EQ(actual2, expectedMangledName2);
}

TEST(InternalNamesTest, doProgramEntry) {
  NameUniquer obj;
  llvm::StringRef actual = obj.doProgramEntry();
  std::string expectedMangledName = "_QQmain";
  ASSERT_EQ(actual.str(), expectedMangledName);
}

TEST(InternalNamesTest, deconstructTest) {
  NameUniquer obj;
  std::pair actual = obj.deconstruct("_QBhello");
  auto expectedNameKind = NameUniquer::NameKind::COMMON;
  struct DeconstructedName expectedComponents {
    {}, {}, "hello", {}
  };
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);
}

TEST(InternalNamesTest, complexdeconstructTest) {
  using NameKind = fir::NameUniquer::NameKind;
  NameUniquer obj;
  std::pair actual = obj.deconstruct("_QMmodSs1modSs2modFsubPfun");
  auto expectedNameKind = NameKind::PROCEDURE;
  struct DeconstructedName expectedComponents = {
      {"mod", "s1mod", "s2mod"}, {"sub"}, "fun", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QPsub");
  expectedNameKind = NameKind::PROCEDURE;
  expectedComponents = {{}, {}, "sub", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QBvariables");
  expectedNameKind = NameKind::COMMON;
  expectedComponents = {{}, {}, "variables", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QMmodEintvar");
  expectedNameKind = NameKind::VARIABLE;
  expectedComponents = {{"mod"}, {}, "intvar", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QMmodECpi");
  expectedNameKind = NameKind::CONSTANT;
  expectedComponents = {{"mod"}, {}, "pi", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QTyourtypeK4KN6");
  expectedNameKind = NameKind::DERIVED_TYPE;
  expectedComponents = {{}, {}, "yourtype", {4, -6}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = obj.deconstruct("_QDTt");
  expectedNameKind = NameKind::DISPATCH_TABLE;
  expectedComponents = {{}, {}, "t", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);
}

// main() from gtest_main

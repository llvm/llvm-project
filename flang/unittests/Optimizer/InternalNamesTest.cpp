//===- InternalNamesTest.cpp -- InternalNames unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/InternalNames.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>

using namespace fir;
using llvm::SmallVector;

struct DeconstructedName {
  DeconstructedName(llvm::StringRef name) : name{name} {}
  DeconstructedName(llvm::ArrayRef<std::string> modules,
      llvm::ArrayRef<std::string> procs, std::int64_t blockId,
      llvm::StringRef name, llvm::ArrayRef<std::int64_t> kinds)
      : modules{modules.begin(), modules.end()}, procs{procs.begin(),
                                                     procs.end()},
        blockId{blockId}, name{name}, kinds{kinds.begin(), kinds.end()} {}

  bool isObjEqual(const NameUniquer::DeconstructedName &actualObj) {
    return actualObj.modules == modules && actualObj.procs == procs &&
        actualObj.blockId == blockId && actualObj.name == name &&
        actualObj.kinds == kinds;
  }

  llvm::SmallVector<std::string> modules;
  llvm::SmallVector<std::string> procs;
  std::int64_t blockId;
  std::string name;
  llvm::SmallVector<std::int64_t> kinds;
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
  std::string actual = NameUniquer::doCommonBlock("hello");
  std::string actualBlank = NameUniquer::doCommonBlock("");
  std::string expectedMangledName = "_QChello";
  std::string expectedMangledNameBlank = "_QC";
  ASSERT_EQ(actual, expectedMangledName);
  ASSERT_EQ(actualBlank, expectedMangledNameBlank);
}

TEST(InternalNamesTest, doGeneratedTest) {
  std::string actual = NameUniquer::doGenerated("@MAIN");
  std::string expectedMangledName = "_QQ@MAIN";
  ASSERT_EQ(actual, expectedMangledName);

  std::string actual1 = NameUniquer::doGenerated("@_ZNSt8ios_base4InitC1Ev");
  std::string expectedMangledName1 = "_QQ@_ZNSt8ios_base4InitC1Ev";
  ASSERT_EQ(actual1, expectedMangledName1);

  std::string actual2 = NameUniquer::doGenerated("_QQ@MAIN");
  std::string expectedMangledName2 = "_QQ_QQ@MAIN";
  ASSERT_EQ(actual2, expectedMangledName2);
}

TEST(InternalNamesTest, doConstantTest) {
  std::string actual =
      NameUniquer::doConstant({"mod1", "mod2"}, {"foo"}, 0, "Hello");
  std::string expectedMangledName = "_QMmod1Smod2FfooEChello";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doProcedureTest) {
  std::string actual = NameUniquer::doProcedure({"mod1", "mod2"}, {}, "HeLLo");
  std::string expectedMangledName = "_QMmod1Smod2Phello";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doTypeTest) {
  std::string actual = NameUniquer::doType({}, {}, 0, "mytype", {4, -1});
  std::string expectedMangledName = "_QTmytypeK4KN1";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doIntrinsicTypeDescriptorTest) {
  using IntrinsicType = fir::NameUniquer::IntrinsicType;
  std::string actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::REAL, 42);
  std::string expectedMangledName = "_QYIrealK42";
  ASSERT_EQ(actual, expectedMangledName);

  actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::REAL, {});
  expectedMangledName = "_QYIrealK0";
  ASSERT_EQ(actual, expectedMangledName);

  actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::INTEGER, 3);
  expectedMangledName = "_QYIintegerK3";
  ASSERT_EQ(actual, expectedMangledName);

  actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::LOGICAL, 2);
  expectedMangledName = "_QYIlogicalK2";
  ASSERT_EQ(actual, expectedMangledName);

  actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::CHARACTER, 4);
  expectedMangledName = "_QYIcharacterK4";
  ASSERT_EQ(actual, expectedMangledName);

  actual = NameUniquer::doIntrinsicTypeDescriptor(
      {}, {}, 0, IntrinsicType::COMPLEX, 4);
  expectedMangledName = "_QYIcomplexK4";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doDispatchTableTest) {
  std::string actual =
      NameUniquer::doDispatchTable({}, {}, 0, "MyTYPE", {2, 8, 18});
  std::string expectedMangledName = "_QDTmytypeK2K8K18";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, doVariableTest) {
  std::string actual = NameUniquer::doVariable(
      {"mod1", "mod2"}, {""}, 0, "intvar"); // Function is present and is blank.
  std::string expectedMangledName = "_QMmod1Smod2FEintvar";
  ASSERT_EQ(actual, expectedMangledName);

  std::string actual2 = NameUniquer::doVariable(
      {"mod1", "mod2"}, {}, 0, "intVariable"); // Function is not present.
  std::string expectedMangledName2 = "_QMmod1Smod2Eintvariable";
  ASSERT_EQ(actual2, expectedMangledName2);
}

TEST(InternalNamesTest, doProgramEntry) {
  llvm::StringRef actual = NameUniquer::doProgramEntry();
  std::string expectedMangledName = "_QQmain";
  ASSERT_EQ(actual.str(), expectedMangledName);
}

TEST(InternalNamesTest, doNamelistGroup) {
  std::string actual = NameUniquer::doNamelistGroup({"mod1"}, {}, "nlg");
  std::string expectedMangledName = "_QMmod1Nnlg";
  ASSERT_EQ(actual, expectedMangledName);
}

TEST(InternalNamesTest, deconstructTest) {
  std::pair actual = NameUniquer::deconstruct("_QChello");
  auto expectedNameKind = NameUniquer::NameKind::COMMON;
  struct DeconstructedName expectedComponents {
    {}, {}, 0, "hello", {}
  };
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);
}

TEST(InternalNamesTest, complexdeconstructTest) {
  using NameKind = fir::NameUniquer::NameKind;
  std::pair actual = NameUniquer::deconstruct("_QMmodSs1modSs2modFsubPfun");
  auto expectedNameKind = NameKind::PROCEDURE;
  struct DeconstructedName expectedComponents = {
      {"mod", "s1mod", "s2mod"}, {"sub"}, 0, "fun", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QPsub");
  expectedNameKind = NameKind::PROCEDURE;
  expectedComponents = {{}, {}, 0, "sub", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QCvariables");
  expectedNameKind = NameKind::COMMON;
  expectedComponents = {{}, {}, 0, "variables", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QMmodEintvar");
  expectedNameKind = NameKind::VARIABLE;
  expectedComponents = {{"mod"}, {}, 0, "intvar", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QMmodECpi");
  expectedNameKind = NameKind::CONSTANT;
  expectedComponents = {{"mod"}, {}, 0, "pi", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QTyourtypeK4KN6");
  expectedNameKind = NameKind::DERIVED_TYPE;
  expectedComponents = {{}, {}, 0, "yourtype", {4, -6}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QDTt");
  expectedNameKind = NameKind::DISPATCH_TABLE;
  expectedComponents = {{}, {}, 0, "t", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);

  actual = NameUniquer::deconstruct("_QFmstartNmpitop");
  expectedNameKind = NameKind::NAMELIST_GROUP;
  expectedComponents = {{}, {"mstart"}, 0, "mpitop", {}};
  validateDeconstructedName(actual, expectedNameKind, expectedComponents);
}

TEST(InternalNamesTest, needExternalNameMangling) {
  ASSERT_FALSE(
      NameUniquer::needExternalNameMangling("_QMmodSs1modSs2modFsubPfun"));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling("omp_num_thread"));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling(""));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling("_QDTmytypeK2K8K18"));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling("exit_"));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling("_QFfooEx"));
  ASSERT_FALSE(NameUniquer::needExternalNameMangling("_QFmstartNmpitop"));
  ASSERT_TRUE(NameUniquer::needExternalNameMangling("_QPfoo"));
  ASSERT_TRUE(NameUniquer::needExternalNameMangling("_QPbar"));
  ASSERT_TRUE(NameUniquer::needExternalNameMangling("_QCa"));
}

TEST(InternalNamesTest, isExternalFacingUniquedName) {
  std::pair result = NameUniquer::deconstruct("_QMmodSs1modSs2modFsubPfun");

  ASSERT_FALSE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("omp_num_thread");
  ASSERT_FALSE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("");
  ASSERT_FALSE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("_QDTmytypeK2K8K18");
  ASSERT_FALSE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("exit_");
  ASSERT_FALSE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("_QPfoo");
  ASSERT_TRUE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("_QPbar");
  ASSERT_TRUE(NameUniquer::isExternalFacingUniquedName(result));
  result = NameUniquer::deconstruct("_QCa");
  ASSERT_TRUE(NameUniquer::isExternalFacingUniquedName(result));
}

TEST(InternalNamesTest, getTypeDescriptorName) {
  std::string derivedTypeName = "_QMdispatch1Tp1";
  std::string expectedBindingTableName = "_QMdispatch1E.dt.p1";
  ASSERT_EQ(expectedBindingTableName,
      fir::NameUniquer::getTypeDescriptorName(derivedTypeName));
  ASSERT_EQ("", fir::NameUniquer::getTypeDescriptorName("_QMdispatch1Pp1"));
}

TEST(InternalNamesTest, getTypeDescriptorBindingTableName) {
  std::string derivedTypeName = "_QMdispatch1Tp1";
  std::string expectedBindingTableName = "_QMdispatch1E.v.p1";
  ASSERT_EQ(expectedBindingTableName,
      fir::NameUniquer::getTypeDescriptorBindingTableName(derivedTypeName));
  ASSERT_EQ("",
      fir::NameUniquer::getTypeDescriptorBindingTableName("_QMdispatch1Pp1"));
}

// main() from gtest_main

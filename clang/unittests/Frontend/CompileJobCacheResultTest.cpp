//===- unittests/Frontend/CompileJobCacheResultTest.cpp - CI tests //------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompileJobCacheResult.h"
#include "FaultingCAS.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::cas;
using namespace llvm::cas;
using llvm::Succeeded;
using Output = CompileJobCacheResult::Output;
using OutputKind = CompileJobCacheResult::OutputKind;

std::vector<Output> getAllOutputs(CompileJobCacheResult Result) {
  std::vector<Output> Outputs;
  llvm::cantFail(Result.forEachOutput([&](Output O) {
    Outputs.push_back(O);
    return llvm::Error::success();
  }));
  return Outputs;
}

TEST(CompileJobCacheResultTest, Empty) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  CompileJobCacheResult::Builder B;
  std::optional<ObjectRef> Result;
  ASSERT_THAT_ERROR(B.build(*CAS).moveInto(Result), Succeeded());

  std::optional<CompileJobCacheResult> Proxy;
  auto Schema = CompileJobResultSchema::create(*CAS);
  ASSERT_THAT_ERROR(Schema.takeError(), Succeeded());
  ASSERT_THAT_ERROR(Schema->load(*Result).moveInto(Proxy), Succeeded());

  EXPECT_EQ(Proxy->getNumOutputs(), 0u);
}

TEST(CompileJobCacheResultTest, AddOutputs) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto Obj1 = llvm::cantFail(CAS->storeFromString({}, "obj1"));
  auto Obj2 = llvm::cantFail(CAS->storeFromString({}, "obj2"));

  std::vector<Output> Expected = {
      Output{Obj1, OutputKind::MainOutput},
      Output{Obj2, OutputKind::Dependencies},
  };

  CompileJobCacheResult::Builder B;
  for (const auto &Output : Expected)
    B.addOutput(Output.Kind, Output.Object);

  std::optional<ObjectRef> Result;
  ASSERT_THAT_ERROR(B.build(*CAS).moveInto(Result), Succeeded());

  std::optional<CompileJobCacheResult> Proxy;
  auto Schema = CompileJobResultSchema::create(*CAS);
  ASSERT_THAT_ERROR(Schema.takeError(), Succeeded());
  ASSERT_THAT_ERROR(Schema->load(*Result).moveInto(Proxy), Succeeded());

  EXPECT_EQ(Proxy->getNumOutputs(), 2u);
  auto Actual = getAllOutputs(*Proxy);

  EXPECT_EQ(Actual, Expected);
}

TEST(CompileJobCacheResultTest, AddKindMap) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();

  auto Obj1 = llvm::cantFail(CAS->storeFromString({}, "obj1"));
  auto Obj2 = llvm::cantFail(CAS->storeFromString({}, "obj2"));
  auto Obj3 = llvm::cantFail(CAS->storeFromString({}, "err"));

  std::vector<Output> Expected = {
      Output{Obj1, OutputKind::MainOutput},
      Output{Obj2, OutputKind::Dependencies},
  };

  CompileJobCacheResult::Builder B;
  B.addKindMap(OutputKind::MainOutput, "/main");
  B.addKindMap(OutputKind::Dependencies, "/deps");
  ASSERT_THAT_ERROR(B.addOutput("/main", Obj1), Succeeded());
  ASSERT_THAT_ERROR(B.addOutput("/deps", Obj2), Succeeded());

  EXPECT_THAT_ERROR(B.addOutput("/other", Obj3), llvm::Failed());

  std::optional<ObjectRef> Result;
  ASSERT_THAT_ERROR(B.build(*CAS).moveInto(Result), Succeeded());

  std::optional<CompileJobCacheResult> Proxy;
  auto Schema = CompileJobResultSchema::create(*CAS);
  ASSERT_THAT_ERROR(Schema.takeError(), Succeeded());
  ASSERT_THAT_ERROR(Schema->load(*Result).moveInto(Proxy), Succeeded());

  EXPECT_EQ(Proxy->getNumOutputs(), 2u);
  auto Actual = getAllOutputs(*Proxy);

  EXPECT_EQ(Actual, Expected);
}

TEST(CompileJobCacheResultTest, SchemaCreateStoreFailure) {
  FaultingCAS CAS(createInMemoryCAS(), /*FailStoreAtCall=*/0);

  auto Schema = CompileJobResultSchema::create(CAS);
  EXPECT_THAT_ERROR(Schema.takeError(),
                    llvm::FailedWithMessage("injected store error"));
}

TEST(CompileJobCacheResultTest, BuilderBuildStoreFailure) {
  // There are 2 stores
  for (unsigned I = 0; I < 2; ++I) {
    FaultingCAS CAS(createInMemoryCAS(), /*FailStoreAtCall=*/I);
    CompileJobCacheResult::Builder B;
    std::optional<ObjectRef> Result;
    EXPECT_THAT_ERROR(B.build(CAS).moveInto(Result),
                      llvm::FailedWithMessage("injected store error"));
  }
}

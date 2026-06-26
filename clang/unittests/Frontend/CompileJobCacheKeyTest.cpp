//===- unittests/Frontend/CompileJobCacheKeyTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompileJobCacheKey.h"
#include "FaultingCAS.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm::cas;

namespace {
struct CompileJobCacheKeyStoreFailureTest
    : public ::testing::TestWithParam<unsigned> {};
} // end anonymous namespace

TEST_P(CompileJobCacheKeyStoreFailureTest, EmitsDiagAndReturnsNullopt) {
  auto Inner = createInMemoryCAS();
  // Create the fake input file.
  ObjectRef InputRef = llvm::cantFail(Inner->storeFromString({}, "input"));
  std::string InputCASID = Inner->getID(InputRef).toString();

  // Inject an error at the store indexed by the test param.
  FaultingCAS CAS(std::move(Inner), GetParam());

  DiagnosticOptions DiagOpts;
  auto *DiagBuffer = new TextDiagnosticBuffer();
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*llvm::vfs::getRealFileSystem(),
                                          DiagOpts, DiagBuffer);

  CompilerInvocation Invocation;
  ASSERT_TRUE(CompilerInvocation::CreateFromArgs(Invocation, {}, *Diags));
  Invocation.getFrontendOpts().CASInputFileCASID = InputCASID;

  auto Key = createCompileJobCacheKey(CAS, *Diags, Invocation);
  EXPECT_FALSE(Key.has_value());

  EXPECT_EQ(DiagBuffer->getNumErrors(), 1u);
  if (DiagBuffer->getNumErrors())
    EXPECT_THAT(DiagBuffer->err_begin()->second,
                ::testing::HasSubstr("failed to store to CAS"));
}

// There are 5 store calls along the success path of createCompileJobCacheKey
// when CASInputFileCASID is used (command-line, "-cc1", version, schema
// kind-ref inside Builder::build, and the final NamedValues object). Injecting
// a failure at any of them must produce an err_cas_store diagnostic instead of
// the previous cantFail crash.
INSTANTIATE_TEST_SUITE_P(AllStoreCalls, CompileJobCacheKeyStoreFailureTest,
                         ::testing::Range(0u, 5u));

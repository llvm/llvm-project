//===- unittests/Interpreter/InterpreterExtensionsTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Clang's Interpreter library.
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/Interpreter/Interpreter.h"

#include "clang/AST/Expr.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <system_error>

#if defined(_AIX) || defined(__MVS__)
#define CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
#endif

using namespace clang;
namespace {

class InterpreterExtensionsTest : public InterpreterTestBase {
protected:
  void SetUp() override {
#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
    GTEST_SKIP();
#endif
  }

  static void SetUpTestSuite() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  }

public:
  // Some tests require a arm-registered-target
  static bool IsARMTargetRegistered() {
    llvm::Triple TT;
    TT.setArch(llvm::Triple::arm);
    TT.setVendor(llvm::Triple::UnknownVendor);
    TT.setOS(llvm::Triple::UnknownOS);

    std::string UnusedErr;
    return llvm::TargetRegistry::lookupTarget(TT, UnusedErr);
  }
};

struct OutOfProcInterpreter : public Interpreter {
  OutOfProcInterpreter(
      std::unique_ptr<CompilerInstance> CI, llvm::Error &ErrOut,
      std::unique_ptr<clang::ASTConsumer> Consumer,
      std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder = nullptr)
      : Interpreter(std::move(CI), ErrOut, std::move(JITBuilder),
                    std::move(Consumer)) {}
};

TEST_F(InterpreterExtensionsTest, FindRuntimeInterface) {
// FIXME : WebAssembly doesn't currently support Jit (see
// https: // github.com/llvm/llvm-project/pull/150977#discussion_r2237521095).
// so this check of HostSupportsJIT has been skipped
// over until support is added, and HostSupportsJIT can return true.
#ifndef __EMSCRIPTEN__
  if (!HostSupportsJIT())
    GTEST_SKIP();
#endif
  clang::IncrementalCompilerBuilder CB;
  llvm::Error ErrOut = llvm::Error::success();
  auto CI = cantFail(CB.CreateCpp());
  // Do not attach the default consumer which is specialized for in-process.
  class NoopConsumer : public ASTConsumer {};
  std::unique_ptr<ASTConsumer> C = std::make_unique<NoopConsumer>();
  OutOfProcInterpreter I(std::move(CI), ErrOut, std::move(C),
                         /*JITBuilder=*/nullptr);
  cantFail(std::move(ErrOut));
  cantFail(I.Parse("int a = 1; a"));
  cantFail(I.Parse("int b = 2; b"));
  cantFail(I.Parse("int c = 3; c"));

  // Make sure no clang::Value logic is attached by the Interpreter.
  Value V1;
  llvm::cantFail(I.ParseAndExecute("int x = 42;"));
  llvm::cantFail(I.ParseAndExecute("x", &V1));
  EXPECT_FALSE(V1.isValid());
  EXPECT_FALSE(V1.hasValue());
}

class CustomJBInterpreter : public Interpreter {
  using CustomJITBuilderCreatorFunction =
      std::function<llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>()>;
  CustomJITBuilderCreatorFunction JBCreator = nullptr;

public:
  CustomJBInterpreter(std::unique_ptr<CompilerInstance> CI, llvm::Error &ErrOut,
                      std::unique_ptr<llvm::orc::LLJITBuilder> JB)
      : Interpreter(std::move(CI), ErrOut, std::move(JB)) {}

  ~CustomJBInterpreter() override {
    // Skip cleanUp() because it would trigger LLJIT default dtors
    Interpreter::ResetExecutor();
  }

  llvm::Error CreateExecutor() { return Interpreter::CreateExecutor(); }
};

TEST_F(InterpreterExtensionsTest, DefaultCrossJIT) {
  if (!IsARMTargetRegistered())
    GTEST_SKIP();

  IncrementalCompilerBuilder CB;
  CB.SetTargetTriple("armv6-none-eabi");
  auto CI = cantFail(CB.CreateCpp());
  llvm::Error ErrOut = llvm::Error::success();
  CustomJBInterpreter Interp(std::move(CI), ErrOut, nullptr);
  cantFail(std::move(ErrOut));
}

TEST_F(InterpreterExtensionsTest, CustomCrossJIT) {
  if (!IsARMTargetRegistered())
    GTEST_SKIP();

  std::string TargetTriple = "armv6-none-eabi";

  IncrementalCompilerBuilder CB;
  CB.SetTargetTriple(TargetTriple);
  auto CI = cantFail(CB.CreateCpp());

  using namespace llvm::orc;
  LLJIT *JIT = nullptr;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Objs;
  auto JTMB = JITTargetMachineBuilder(llvm::Triple(TargetTriple));
  JTMB.setCPU("cortex-m0plus");

  auto JB = std::make_unique<LLJITBuilder>();
  JB->setJITTargetMachineBuilder(JTMB);
  JB->setPlatformSetUp(setUpInactivePlatform);
  JB->setNotifyCreatedCallback([&](LLJIT &J) {
    ObjectLayer &ObjLayer = J.getObjLinkingLayer();
    auto *JITLinkObjLayer = llvm::dyn_cast<ObjectLinkingLayer>(&ObjLayer);
    JITLinkObjLayer->setReturnObjectBuffer(
        [&Objs](std::unique_ptr<llvm::MemoryBuffer> MB) {
          Objs.push_back(std::move(MB));
        });
    JIT = &J;
    return llvm::Error::success();
  });

  llvm::Error ErrOut = llvm::Error::success();
  CustomJBInterpreter Interp(std::move(CI), ErrOut, std::move(JB));
  cantFail(std::move(ErrOut));

  EXPECT_EQ(0U, Objs.size());
  cantFail(Interp.ParseAndExecute("int a = 1;"));
  ASSERT_NE(JIT, nullptr); // But it is, because JBCreator was never called
  ExecutorAddr Addr = cantFail(JIT->lookup("a"));
  EXPECT_NE(0U, Addr.getValue());
  EXPECT_EQ(1U, Objs.size());
}

} // end anonymous namespace

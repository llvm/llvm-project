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

#include "clang/Interpreter/Interpreter.h"

#include "clang/AST/Expr.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <system_error>

#if defined(_AIX)
#define CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
#endif

using namespace clang;
namespace {

static bool HostSupportsJit() {
  auto J = llvm::orc::LLJITBuilder().create();
  if (J)
    return true;
  LLVMConsumeError(llvm::wrap(J.takeError()));
  return false;
}

// Some tests require a arm-registered-target
static bool IsARMTargetRegistered() {
  llvm::Triple TT;
  TT.setArch(llvm::Triple::arm);
  TT.setVendor(llvm::Triple::UnknownVendor);
  TT.setOS(llvm::Triple::UnknownOS);

  std::string UnusedErr;
  return llvm::TargetRegistry::lookupTarget(TT.str(), UnusedErr);
}

struct LLVMInitRAII {
  LLVMInitRAII() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  }
  ~LLVMInitRAII() { llvm::llvm_shutdown(); }
} LLVMInit;

class TestCreateResetExecutor : public Interpreter {
public:
  TestCreateResetExecutor(std::unique_ptr<CompilerInstance> CI,
                          llvm::Error &Err)
      : Interpreter(std::move(CI), Err) {}

  llvm::Error testCreateJITBuilderError() {
    JB = nullptr;
    return Interpreter::CreateExecutor();
  }

  llvm::Error testCreateExecutor() {
    JB = std::make_unique<llvm::orc::LLJITBuilder>();
    return Interpreter::CreateExecutor();
  }

  void resetExecutor() { Interpreter::ResetExecutor(); }

private:
  llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
  CreateJITBuilder(CompilerInstance &CI) override {
    if (JB)
      return std::move(JB);
    return llvm::make_error<llvm::StringError>("TestError", std::error_code());
  }

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;
};

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(InterpreterExtensionsTest, DISABLED_ExecutorCreateReset) {
#else
TEST(InterpreterExtensionsTest, ExecutorCreateReset) {
#endif
  // Make sure we can create the executer on the platform.
  if (!HostSupportsJit())
    GTEST_SKIP();

  clang::IncrementalCompilerBuilder CB;
  llvm::Error ErrOut = llvm::Error::success();
  TestCreateResetExecutor Interp(cantFail(CB.CreateCpp()), ErrOut);
  cantFail(std::move(ErrOut));
  EXPECT_THAT_ERROR(Interp.testCreateJITBuilderError(),
                    llvm::FailedWithMessage("TestError"));
  cantFail(Interp.testCreateExecutor());
  Interp.resetExecutor();
  cantFail(Interp.testCreateExecutor());
  EXPECT_THAT_ERROR(Interp.testCreateExecutor(),
                    llvm::FailedWithMessage("Operation failed. "
                                            "Execution engine exists"));
}

class RecordRuntimeIBMetrics : public Interpreter {
  struct NoopRuntimeInterfaceBuilder : public RuntimeInterfaceBuilder {
    NoopRuntimeInterfaceBuilder(Sema &S) : S(S) {}

    TransformExprFunction *getPrintValueTransformer() override {
      TransformerQueries += 1;
      return &noop;
    }

    static ExprResult noop(RuntimeInterfaceBuilder *Builder, Expr *E,
                           ArrayRef<Expr *> FixedArgs) {
      auto *B = static_cast<NoopRuntimeInterfaceBuilder *>(Builder);
      B->TransformedExprs += 1;
      return B->S.ActOnFinishFullExpr(E, /*DiscardedValue=*/false);
    }

    Sema &S;
    size_t TransformedExprs = 0;
    size_t TransformerQueries = 0;
  };

public:
  // Inherit with using wouldn't make it public
  RecordRuntimeIBMetrics(std::unique_ptr<CompilerInstance> CI, llvm::Error &Err)
      : Interpreter(std::move(CI), Err) {}

  std::unique_ptr<RuntimeInterfaceBuilder> FindRuntimeInterface() override {
    assert(RuntimeIBPtr == nullptr && "We create the builder only once");
    Sema &S = getCompilerInstance()->getSema();
    auto RuntimeIB = std::make_unique<NoopRuntimeInterfaceBuilder>(S);
    RuntimeIBPtr = RuntimeIB.get();
    return RuntimeIB;
  }

  NoopRuntimeInterfaceBuilder *RuntimeIBPtr = nullptr;
};

TEST(InterpreterExtensionsTest, FindRuntimeInterface) {
  clang::IncrementalCompilerBuilder CB;
  llvm::Error ErrOut = llvm::Error::success();
  RecordRuntimeIBMetrics Interp(cantFail(CB.CreateCpp()), ErrOut);
  cantFail(std::move(ErrOut));
  cantFail(Interp.Parse("int a = 1; a"));
  cantFail(Interp.Parse("int b = 2; b"));
  cantFail(Interp.Parse("int c = 3; c"));
  EXPECT_EQ(3U, Interp.RuntimeIBPtr->TransformedExprs);
  EXPECT_EQ(1U, Interp.RuntimeIBPtr->TransformerQueries);
}

class CustomJBInterpreter : public Interpreter {
  using CustomJITBuilderCreatorFunction =
      std::function<llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>()>;
  CustomJITBuilderCreatorFunction JBCreator = nullptr;

public:
  CustomJBInterpreter(std::unique_ptr<CompilerInstance> CI, llvm::Error &ErrOut)
      : Interpreter(std::move(CI), ErrOut) {}

  ~CustomJBInterpreter() override {
    // Skip cleanUp() because it would trigger LLJIT default dtors
    Interpreter::ResetExecutor();
  }

  void setCustomJITBuilderCreator(CustomJITBuilderCreatorFunction Fn) {
    JBCreator = std::move(Fn);
  }

  llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>
  CreateJITBuilder(CompilerInstance &CI) override {
    if (JBCreator)
      return JBCreator();
    return Interpreter::CreateJITBuilder(CI);
  }

  llvm::Error CreateExecutor() { return Interpreter::CreateExecutor(); }
};

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(InterpreterExtensionsTest, DISABLED_DefaultCrossJIT) {
#else
TEST(InterpreterExtensionsTest, DefaultCrossJIT) {
#endif
  if (!IsARMTargetRegistered())
    GTEST_SKIP();

  IncrementalCompilerBuilder CB;
  CB.SetTargetTriple("armv6-none-eabi");
  auto CI = cantFail(CB.CreateCpp());
  llvm::Error ErrOut = llvm::Error::success();
  CustomJBInterpreter Interp(std::move(CI), ErrOut);
  cantFail(std::move(ErrOut));
  cantFail(Interp.CreateExecutor());
}

#ifdef CLANG_INTERPRETER_PLATFORM_CANNOT_CREATE_LLJIT
TEST(InterpreterExtensionsTest, DISABLED_CustomCrossJIT) {
#else
TEST(InterpreterExtensionsTest, CustomCrossJIT) {
#endif
  if (!IsARMTargetRegistered())
    GTEST_SKIP();

  std::string TargetTriple = "armv6-none-eabi";

  IncrementalCompilerBuilder CB;
  CB.SetTargetTriple(TargetTriple);
  auto CI = cantFail(CB.CreateCpp());
  llvm::Error ErrOut = llvm::Error::success();
  CustomJBInterpreter Interp(std::move(CI), ErrOut);
  cantFail(std::move(ErrOut));

  using namespace llvm::orc;
  LLJIT *JIT = nullptr;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Objs;
  Interp.setCustomJITBuilderCreator([&]() {
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
    return JB;
  });

  EXPECT_EQ(0U, Objs.size());
  cantFail(Interp.CreateExecutor());
  cantFail(Interp.ParseAndExecute("int a = 1;"));
  ExecutorAddr Addr = cantFail(JIT->lookup("a"));
  EXPECT_NE(0U, Addr.getValue());
  EXPECT_EQ(1U, Objs.size());
}

} // end anonymous namespace

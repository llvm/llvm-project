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
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Testing/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <system_error>

using namespace clang;
namespace {

static bool HostSupportsJit() {
  auto J = llvm::orc::LLJITBuilder().create();
  if (J)
    return true;
  LLVMConsumeError(llvm::wrap(J.takeError()));
  return false;
}

struct LLVMInitRAII {
  LLVMInitRAII() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
  ~LLVMInitRAII() { llvm::llvm_shutdown(); }
} LLVMInit;

class TestCreateResetExecutor : public Interpreter {
public:
  TestCreateResetExecutor(std::unique_ptr<CompilerInstance> CI,
                          llvm::Error &Err)
      : Interpreter(std::move(CI), Err) {}

  llvm::Error testCreateExecutor() { return Interpreter::CreateExecutor(); }

  void resetExecutor() { Interpreter::ResetExecutor(); }
};

#ifdef _AIX
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

} // end anonymous namespace

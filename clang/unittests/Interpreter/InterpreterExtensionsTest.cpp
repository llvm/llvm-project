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

class CustomJBInterpreter : public Interpreter {
  using CustomJITBuilderCreatorFunction =
      std::function<llvm::Expected<std::unique_ptr<llvm::orc::LLJITBuilder>>()>;
  CustomJITBuilderCreatorFunction JBCreator = nullptr;

public:
  CustomJBInterpreter(std::unique_ptr<CompilerInstance> CI, llvm::Error &ErrOut,
                      std::unique_ptr<clang::IncrementalExecutorBuilder> IEB)
      : Interpreter(std::move(CI), ErrOut, std::move(IEB)) {}

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
  auto IEB = std::make_unique<IncrementalExecutorBuilder>();
  IEB->JITBuilder = std::move(JB);
  llvm::Error ErrOut = llvm::Error::success();
  CustomJBInterpreter Interp(std::move(CI), ErrOut, std::move(IEB));
  cantFail(std::move(ErrOut));

  EXPECT_EQ(0U, Objs.size());
  cantFail(Interp.ParseAndExecute("int a = 1;"));
  ASSERT_NE(JIT, nullptr); // But it is, because JBCreator was never called
  ExecutorAddr Addr = cantFail(JIT->lookup("a"));
  EXPECT_NE(0U, Addr.getValue());
  EXPECT_EQ(1U, Objs.size());
}

TEST_F(InterpreterExtensionsTest, CustomIncrementalExecutor) {
  struct RecordingIncrementalExecutor : public clang::IncrementalExecutor {
    mutable unsigned RanCtors = false;
    unsigned Added = 0;
    unsigned Removed = 0;

    // Default ctor is fine; builder.create() will just return this IE.
    RecordingIncrementalExecutor() = default;

    llvm::Error addModule(clang::PartialTranslationUnit &PTU) override {
      Added++;
      return llvm::Error::success();
    }

    llvm::Error removeModule(clang::PartialTranslationUnit &PTU) override {
      Removed++;
      return llvm::Error::success();
    }

    llvm::Error runCtors() const override {
      RanCtors++;
      return llvm::Error::success();
    }

    llvm::Error cleanUp() override { return llvm::Error::success(); }

    llvm::Expected<llvm::orc::ExecutorAddr>
    getSymbolAddress(llvm::StringRef /*Name*/,
                     SymbolNameKind /*NameKind*/) const override {
      // Return an error here; test doesn't need a real address.
      return llvm::make_error<llvm::StringError>(
          "not implemented in test", llvm::inconvertibleErrorCode());
    }

    llvm::Error LoadDynamicLibrary(const char * /*name*/) override {
      return llvm::Error::success();
    }
  };

  // Prepare a builder that hands out our recording executor.
  auto B = std::make_unique<IncrementalExecutorBuilder>();
  B->IE = std::make_unique<RecordingIncrementalExecutor>();

  IncrementalCompilerBuilder CB;
  auto CI = cantFail(CB.CreateCpp());

  auto I = cantFail(Interpreter::create(std::move(CI), std::move(B)));
  ASSERT_TRUE(I);

  const auto &Rec = static_cast<RecordingIncrementalExecutor &>(
      cantFail(I->getExecutionEngine()));
  unsigned NumInitAdded = Rec.Added;
  unsigned NumInitRanCtors = Rec.RanCtors;
  unsigned NumInitRemoved = Rec.Removed;

  cantFail(I->ParseAndExecute("int a = 1;"));

  EXPECT_TRUE(Rec.Added == NumInitAdded + 1);
  EXPECT_TRUE(Rec.RanCtors == NumInitRanCtors + 1);
  EXPECT_TRUE(Rec.Removed == NumInitRemoved);
}

} // end anonymous namespace

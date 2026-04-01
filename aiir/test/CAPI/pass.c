//===- pass.c - Simple test of C APIs -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: aiir-capi-pass-test 2>&1 | FileCheck %s
 */

#include "aiir-c/Pass.h"
#include "aiir-c/Dialect/Func.h"
#include "aiir-c/IR.h"
#include "aiir-c/RegisterEverything.h"
#include "aiir-c/Transforms.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void registerAllUpstreamDialects(AiirContext ctx) {
  AiirDialectRegistry registry = aiirDialectRegistryCreate();
  aiirRegisterAllDialects(registry);
  aiirContextAppendDialectRegistry(ctx, registry);
  aiirDialectRegistryDestroy(registry);
}

void testRunPassOnModule(void) {
  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  const char *funcAsm = //
      "func.func @foo(%arg0 : i32) -> i32 {   \n"
      "  %res = arith.addi %arg0, %arg0 : i32 \n"
      "  return %res : i32                    \n"
      "}                                      \n";
  AiirOperation func =
      aiirOperationCreateParse(ctx, aiirStringRefCreateFromCString(funcAsm),
                               aiirStringRefCreateFromCString("funcAsm"));
  if (aiirOperationIsNull(func)) {
    fprintf(stderr, "Unexpected failure parsing asm.\n");
    exit(EXIT_FAILURE);
  }

  // Run the print-op-stats pass on the top-level module:
  // CHECK-LABEL: Operations encountered:
  // CHECK: arith.addi        , 1
  // CHECK: func.func      , 1
  // CHECK: func.return        , 1
  {
    AiirPassManager pm = aiirPassManagerCreate(ctx);
    AiirPass printOpStatPass = aiirCreateTransformsPrintOpStatsPass();
    aiirPassManagerAddOwnedPass(pm, printOpStatPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, func);
    if (aiirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running pass manager.\n");
      exit(EXIT_FAILURE);
    }
    aiirPassManagerDestroy(pm);
  }
  aiirOperationDestroy(func);
  aiirContextDestroy(ctx);
}

void testRunPassOnNestedModule(void) {
  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  const char *moduleAsm = //
      "module {                                   \n"
      "  func.func @foo(%arg0 : i32) -> i32 {     \n"
      "    %res = arith.addi %arg0, %arg0 : i32   \n"
      "    return %res : i32                      \n"
      "  }                                        \n"
      "  module {                                 \n"
      "    func.func @bar(%arg0 : f32) -> f32 {   \n"
      "      %res = arith.addf %arg0, %arg0 : f32 \n"
      "      return %res : f32                    \n"
      "    }                                      \n"
      "  }                                        \n"
      "}                                          \n";
  AiirOperation module =
      aiirOperationCreateParse(ctx, aiirStringRefCreateFromCString(moduleAsm),
                               aiirStringRefCreateFromCString("moduleAsm"));
  if (aiirOperationIsNull(module))
    exit(1);

  // Run the print-op-stats pass on functions under the top-level module:
  // CHECK-LABEL: Operations encountered:
  // CHECK: arith.addi        , 1
  // CHECK: func.func      , 1
  // CHECK: func.return        , 1
  {
    AiirPassManager pm = aiirPassManagerCreate(ctx);
    AiirOpPassManager nestedFuncPm = aiirPassManagerGetNestedUnder(
        pm, aiirStringRefCreateFromCString("func.func"));
    AiirPass printOpStatPass = aiirCreateTransformsPrintOpStatsPass();
    aiirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(success))
      exit(2);
    aiirPassManagerDestroy(pm);
  }
  // Run the print-op-stats pass on functions under the nested module:
  // CHECK-LABEL: Operations encountered:
  // CHECK: arith.addf        , 1
  // CHECK: func.func      , 1
  // CHECK: func.return        , 1
  {
    AiirPassManager pm = aiirPassManagerCreate(ctx);
    AiirOpPassManager nestedModulePm = aiirPassManagerGetNestedUnder(
        pm, aiirStringRefCreateFromCString("builtin.module"));
    AiirOpPassManager nestedFuncPm = aiirOpPassManagerGetNestedUnder(
        nestedModulePm, aiirStringRefCreateFromCString("func.func"));
    AiirPass printOpStatPass = aiirCreateTransformsPrintOpStatsPass();
    aiirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(success))
      exit(2);
    aiirPassManagerDestroy(pm);
  }

  aiirOperationDestroy(module);
  aiirContextDestroy(ctx);
}

static void printToStderr(AiirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

static void dontPrint(AiirStringRef str, void *userData) {
  (void)str;
  (void)userData;
}

void testPrintPassPipeline(void) {
  AiirContext ctx = aiirContextCreate();
  AiirPassManager pm = aiirPassManagerCreateOnOperation(
      ctx, aiirStringRefCreateFromCString("any"));
  // Populate the pass-manager
  AiirOpPassManager nestedModulePm = aiirPassManagerGetNestedUnder(
      pm, aiirStringRefCreateFromCString("builtin.module"));
  AiirOpPassManager nestedFuncPm = aiirOpPassManagerGetNestedUnder(
      nestedModulePm, aiirStringRefCreateFromCString("func.func"));
  AiirPass printOpStatPass = aiirCreateTransformsPrintOpStatsPass();
  aiirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);

  // Print the top level pass manager
  //      CHECK: Top-level: any(
  // CHECK-SAME:   builtin.module(func.func(print-op-stats{json=false}))
  // CHECK-SAME: )
  fprintf(stderr, "Top-level: ");
  aiirPrintPassPipeline(aiirPassManagerGetAsOpPassManager(pm), printToStderr,
                        NULL);
  fprintf(stderr, "\n");

  // Print the pipeline nested one level down
  // CHECK: Nested Module: builtin.module(func.func(print-op-stats{json=false}))
  fprintf(stderr, "Nested Module: ");
  aiirPrintPassPipeline(nestedModulePm, printToStderr, NULL);
  fprintf(stderr, "\n");

  // Print the pipeline nested two levels down
  // CHECK: Nested Module>Func: func.func(print-op-stats{json=false})
  fprintf(stderr, "Nested Module>Func: ");
  aiirPrintPassPipeline(nestedFuncPm, printToStderr, NULL);
  fprintf(stderr, "\n");

  aiirPassManagerDestroy(pm);
  aiirContextDestroy(ctx);
}

void testParsePassPipeline(void) {
  AiirContext ctx = aiirContextCreate();
  AiirPassManager pm = aiirPassManagerCreate(ctx);
  // Try parse a pipeline.
  AiirLogicalResult status = aiirParsePassPipeline(
      aiirPassManagerGetAsOpPassManager(pm),
      aiirStringRefCreateFromCString(
          "builtin.module(func.func(print-op-stats{json=false}))"),
      printToStderr, NULL);
  // Expect a failure, we haven't registered the print-op-stats pass yet.
  if (aiirLogicalResultIsSuccess(status)) {
    fprintf(
        stderr,
        "Unexpected success parsing pipeline without registering the pass\n");
    exit(EXIT_FAILURE);
  }
  // Try again after registrating the pass.
  aiirRegisterTransformsPrintOpStatsPass();
  status = aiirParsePassPipeline(
      aiirPassManagerGetAsOpPassManager(pm),
      aiirStringRefCreateFromCString(
          "builtin.module(func.func(print-op-stats{json=false}))"),
      printToStderr, NULL);
  // Expect a failure, we haven't registered the print-op-stats pass yet.
  if (aiirLogicalResultIsFailure(status)) {
    fprintf(stderr,
            "Unexpected failure parsing pipeline after registering the pass\n");
    exit(EXIT_FAILURE);
  }

  // CHECK: Round-trip: builtin.module(func.func(print-op-stats{json=false}))
  fprintf(stderr, "Round-trip: ");
  aiirPrintPassPipeline(aiirPassManagerGetAsOpPassManager(pm), printToStderr,
                        NULL);
  fprintf(stderr, "\n");

  // Try appending a pass:
  status = aiirOpPassManagerAddPipeline(
      aiirPassManagerGetAsOpPassManager(pm),
      aiirStringRefCreateFromCString("func.func(print-op-stats{json=false})"),
      printToStderr, NULL);
  if (aiirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure appending pipeline\n");
    exit(EXIT_FAILURE);
  }
  //      CHECK: Appended: builtin.module(
  // CHECK-SAME:   func.func(print-op-stats{json=false}),
  // CHECK-SAME:   func.func(print-op-stats{json=false})
  // CHECK-SAME: )
  fprintf(stderr, "Appended: ");
  aiirPrintPassPipeline(aiirPassManagerGetAsOpPassManager(pm), printToStderr,
                        NULL);
  fprintf(stderr, "\n");

  aiirPassManagerDestroy(pm);
  aiirContextDestroy(ctx);
}

void testParseErrorCapture(void) {
  // CHECK-LABEL: testParseErrorCapture:
  fprintf(stderr, "\nTEST: testParseErrorCapture:\n");

  AiirContext ctx = aiirContextCreate();
  AiirPassManager pm = aiirPassManagerCreate(ctx);
  AiirOpPassManager opm = aiirPassManagerGetAsOpPassManager(pm);
  AiirStringRef invalidPipeline = aiirStringRefCreateFromCString("invalid");

  // CHECK: aiirParsePassPipeline:
  // CHECK: expected pass pipeline to be wrapped with the anchor operation type
  fprintf(stderr, "aiirParsePassPipeline:\n");
  if (aiirLogicalResultIsSuccess(
          aiirParsePassPipeline(opm, invalidPipeline, printToStderr, NULL)))
    exit(EXIT_FAILURE);
  fprintf(stderr, "\n");

  // CHECK: aiirOpPassManagerAddPipeline:
  // CHECK: 'invalid' does not refer to a registered pass or pass pipeline
  fprintf(stderr, "aiirOpPassManagerAddPipeline:\n");
  if (aiirLogicalResultIsSuccess(aiirOpPassManagerAddPipeline(
          opm, invalidPipeline, printToStderr, NULL)))
    exit(EXIT_FAILURE);
  fprintf(stderr, "\n");

  // Make sure all output is going through the callback.
  // CHECK: dontPrint: <>
  fprintf(stderr, "dontPrint: <");
  if (aiirLogicalResultIsSuccess(
          aiirParsePassPipeline(opm, invalidPipeline, dontPrint, NULL)))
    exit(EXIT_FAILURE);
  if (aiirLogicalResultIsSuccess(
          aiirOpPassManagerAddPipeline(opm, invalidPipeline, dontPrint, NULL)))
    exit(EXIT_FAILURE);
  fprintf(stderr, ">\n");

  aiirPassManagerDestroy(pm);
  aiirContextDestroy(ctx);
}

struct TestExternalPassUserData {
  int constructCallCount;
  int destructCallCount;
  int initializeCallCount;
  int cloneCallCount;
  int runCallCount;
};
typedef struct TestExternalPassUserData TestExternalPassUserData;

void testConstructExternalPass(void *userData) {
  ++((TestExternalPassUserData *)userData)->constructCallCount;
}

void testDestructExternalPass(void *userData) {
  ++((TestExternalPassUserData *)userData)->destructCallCount;
}

AiirLogicalResult testInitializeExternalPass(AiirContext ctx, void *userData) {
  ++((TestExternalPassUserData *)userData)->initializeCallCount;
  return aiirLogicalResultSuccess();
}

AiirLogicalResult testInitializeFailingExternalPass(AiirContext ctx,
                                                    void *userData) {
  ++((TestExternalPassUserData *)userData)->initializeCallCount;
  return aiirLogicalResultFailure();
}

void *testCloneExternalPass(void *userData) {
  ++((TestExternalPassUserData *)userData)->cloneCallCount;
  return userData;
}

void testRunExternalPass(AiirOperation op, AiirExternalPass pass,
                         void *userData) {
  ++((TestExternalPassUserData *)userData)->runCallCount;
}

void testRunExternalFuncPass(AiirOperation op, AiirExternalPass pass,
                             void *userData) {
  ++((TestExternalPassUserData *)userData)->runCallCount;
  AiirStringRef opName = aiirIdentifierStr(aiirOperationGetName(op));
  if (!aiirStringRefEqual(opName,
                          aiirStringRefCreateFromCString("func.func"))) {
    aiirExternalPassSignalFailure(pass);
  }
}

void testRunFailingExternalPass(AiirOperation op, AiirExternalPass pass,
                                void *userData) {
  ++((TestExternalPassUserData *)userData)->runCallCount;
  aiirExternalPassSignalFailure(pass);
}

AiirExternalPassCallbacks makeTestExternalPassCallbacks(
    AiirLogicalResult (*initializePass)(AiirContext ctx, void *userData),
    void (*runPass)(AiirOperation op, AiirExternalPass, void *userData)) {
  return (AiirExternalPassCallbacks){testConstructExternalPass,
                                     testDestructExternalPass, initializePass,
                                     testCloneExternalPass, runPass};
}

void testExternalPass(void) {
  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  const char *moduleAsm = //
      "module {                                 \n"
      "  func.func @foo(%arg0 : i32) -> i32 {   \n"
      "    %res = arith.addi %arg0, %arg0 : i32 \n"
      "    return %res : i32                    \n"
      "  }                                      \n"
      "}";
  AiirOperation module =
      aiirOperationCreateParse(ctx, aiirStringRefCreateFromCString(moduleAsm),
                               aiirStringRefCreateFromCString("moduleAsm"));
  if (aiirOperationIsNull(module)) {
    fprintf(stderr, "Unexpected failure parsing module.\n");
    exit(EXIT_FAILURE);
  }

  AiirStringRef description = aiirStringRefCreateFromCString("");
  AiirStringRef emptyOpName = aiirStringRefCreateFromCString("");

  AiirTypeIDAllocator typeIDAllocator = aiirTypeIDAllocatorCreate();

  // Run a generic pass
  {
    AiirTypeID passID = aiirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    AiirStringRef name = aiirStringRefCreateFromCString("TestExternalPass");
    AiirStringRef argument =
        aiirStringRefCreateFromCString("test-external-pass");
    TestExternalPassUserData userData = {0};

    AiirPass externalPass = aiirCreateExternalPass(
        passID, name, argument, description, emptyOpName, 0, NULL,
        makeTestExternalPassCallbacks(NULL, testRunExternalPass), &userData);

    if (userData.constructCallCount != 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    AiirPassManager pm = aiirPassManagerCreate(ctx);
    aiirPassManagerAddOwnedPass(pm, externalPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running external pass.\n");
      exit(EXIT_FAILURE);
    }

    if (userData.runCallCount != 1) {
      fprintf(stderr, "Expected runCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    aiirPassManagerDestroy(pm);

    if (userData.destructCallCount != userData.constructCallCount) {
      fprintf(stderr, "Expected destructCallCount to be equal to "
                      "constructCallCount\n");
      exit(EXIT_FAILURE);
    }
  }

  // Run a func operation pass
  {
    AiirTypeID passID = aiirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    AiirStringRef name = aiirStringRefCreateFromCString("TestExternalFuncPass");
    AiirStringRef argument =
        aiirStringRefCreateFromCString("test-external-func-pass");
    TestExternalPassUserData userData = {0};
    AiirDialectHandle funcHandle = aiirGetDialectHandle__func__();
    AiirStringRef funcOpName = aiirStringRefCreateFromCString("func.func");

    AiirPass externalPass = aiirCreateExternalPass(
        passID, name, argument, description, funcOpName, 1, &funcHandle,
        makeTestExternalPassCallbacks(NULL, testRunExternalFuncPass),
        &userData);

    if (userData.constructCallCount != 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    AiirPassManager pm = aiirPassManagerCreate(ctx);
    AiirOpPassManager nestedFuncPm =
        aiirPassManagerGetNestedUnder(pm, funcOpName);
    aiirOpPassManagerAddOwnedPass(nestedFuncPm, externalPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running external operation pass.\n");
      exit(EXIT_FAILURE);
    }

    // Since this is a nested pass, it can be cloned and run in parallel
    if (userData.cloneCallCount != userData.constructCallCount - 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    // The pass should only be run once this there is only one func op
    if (userData.runCallCount != 1) {
      fprintf(stderr, "Expected runCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    aiirPassManagerDestroy(pm);

    if (userData.destructCallCount != userData.constructCallCount) {
      fprintf(stderr, "Expected destructCallCount to be equal to "
                      "constructCallCount\n");
      exit(EXIT_FAILURE);
    }
  }

  // Run a pass with `initialize` set
  {
    AiirTypeID passID = aiirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    AiirStringRef name = aiirStringRefCreateFromCString("TestExternalPass");
    AiirStringRef argument =
        aiirStringRefCreateFromCString("test-external-pass");
    TestExternalPassUserData userData = {0};

    AiirPass externalPass = aiirCreateExternalPass(
        passID, name, argument, description, emptyOpName, 0, NULL,
        makeTestExternalPassCallbacks(testInitializeExternalPass,
                                      testRunExternalPass),
        &userData);

    if (userData.constructCallCount != 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    AiirPassManager pm = aiirPassManagerCreate(ctx);
    aiirPassManagerAddOwnedPass(pm, externalPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running external pass.\n");
      exit(EXIT_FAILURE);
    }

    if (userData.initializeCallCount != 1) {
      fprintf(stderr, "Expected initializeCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    if (userData.runCallCount != 1) {
      fprintf(stderr, "Expected runCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    // Run the pass again and confirm that the initializeCallCount is still 1.
    AiirLogicalResult second_success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsFailure(second_success)) {
      fprintf(stderr, "Unexpected failure running external pass.\n");
      exit(EXIT_FAILURE);
    }

    if (userData.initializeCallCount != 1) {
      fprintf(stderr, "Expected initializeCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    if (userData.runCallCount != 2) {
      fprintf(stderr, "Expected runCallCount to be 2\n");
      exit(EXIT_FAILURE);
    }

    aiirPassManagerDestroy(pm);

    if (userData.destructCallCount != userData.constructCallCount) {
      fprintf(stderr, "Expected destructCallCount to be equal to "
                      "constructCallCount\n");
      exit(EXIT_FAILURE);
    }
  }

  // Run a pass that fails during `initialize`
  {
    AiirTypeID passID = aiirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    AiirStringRef name =
        aiirStringRefCreateFromCString("TestExternalFailingPass");
    AiirStringRef argument =
        aiirStringRefCreateFromCString("test-external-failing-pass");
    TestExternalPassUserData userData = {0};

    AiirPass externalPass = aiirCreateExternalPass(
        passID, name, argument, description, emptyOpName, 0, NULL,
        makeTestExternalPassCallbacks(testInitializeFailingExternalPass,
                                      testRunExternalPass),
        &userData);

    if (userData.constructCallCount != 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    AiirPassManager pm = aiirPassManagerCreate(ctx);
    aiirPassManagerAddOwnedPass(pm, externalPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsSuccess(success)) {
      fprintf(
          stderr,
          "Expected failure running pass manager on failing external pass.\n");
      exit(EXIT_FAILURE);
    }

    if (userData.initializeCallCount != 1) {
      fprintf(stderr, "Expected initializeCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    if (userData.runCallCount != 0) {
      fprintf(stderr, "Expected runCallCount to be 0\n");
      exit(EXIT_FAILURE);
    }

    aiirPassManagerDestroy(pm);

    if (userData.destructCallCount != userData.constructCallCount) {
      fprintf(stderr, "Expected destructCallCount to be equal to "
                      "constructCallCount\n");
      exit(EXIT_FAILURE);
    }
  }

  // Run a pass that fails during `run`
  {
    AiirTypeID passID = aiirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    AiirStringRef name =
        aiirStringRefCreateFromCString("TestExternalFailingPass");
    AiirStringRef argument =
        aiirStringRefCreateFromCString("test-external-failing-pass");
    TestExternalPassUserData userData = {0};

    AiirPass externalPass = aiirCreateExternalPass(
        passID, name, argument, description, emptyOpName, 0, NULL,
        makeTestExternalPassCallbacks(NULL, testRunFailingExternalPass),
        &userData);

    if (userData.constructCallCount != 1) {
      fprintf(stderr, "Expected constructCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    AiirPassManager pm = aiirPassManagerCreate(ctx);
    aiirPassManagerAddOwnedPass(pm, externalPass);
    AiirLogicalResult success = aiirPassManagerRunOnOp(pm, module);
    if (aiirLogicalResultIsSuccess(success)) {
      fprintf(
          stderr,
          "Expected failure running pass manager on failing external pass.\n");
      exit(EXIT_FAILURE);
    }

    if (userData.runCallCount != 1) {
      fprintf(stderr, "Expected runCallCount to be 1\n");
      exit(EXIT_FAILURE);
    }

    aiirPassManagerDestroy(pm);

    if (userData.destructCallCount != userData.constructCallCount) {
      fprintf(stderr, "Expected destructCallCount to be equal to "
                      "constructCallCount\n");
      exit(EXIT_FAILURE);
    }
  }

  aiirTypeIDAllocatorDestroy(typeIDAllocator);
  aiirOperationDestroy(module);
  aiirContextDestroy(ctx);
}

int main(void) {
  testRunPassOnModule();
  testRunPassOnNestedModule();
  testPrintPassPipeline();
  testParsePassPipeline();
  testParseErrorCapture();
  testExternalPass();
  return 0;
}

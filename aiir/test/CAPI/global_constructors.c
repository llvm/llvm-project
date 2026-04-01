//===- global_constructors.c - Test JIT with the global constructors ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: target=aarch64{{.*}}, target=arm64{{.*}}
/* RUN: aiir-capi-global-constructors-test 2>&1 | FileCheck %s
 */
/* REQUIRES: host-supports-jit
 */
// XFAIL: system-aix

#include "aiir-c/Conversion.h"
#include "aiir-c/ExecutionEngine.h"
#include "aiir-c/IR.h"
#include "aiir-c/RegisterEverything.h"

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

void lowerModuleToLLVM(AiirContext ctx, AiirModule module) {
  AiirPassManager pm = aiirPassManagerCreate(ctx);
  AiirOpPassManager opm = aiirPassManagerGetNestedUnder(
      pm, aiirStringRefCreateFromCString("func.func"));
  aiirPassManagerAddOwnedPass(pm, aiirCreateConversionConvertFuncToLLVMPass());
  aiirOpPassManagerAddOwnedPass(
      opm, aiirCreateConversionArithToLLVMConversionPass());
  AiirLogicalResult status =
      aiirPassManagerRunOnOp(pm, aiirModuleGetOperation(module));
  if (aiirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure running pass pipeline\n");
    exit(2);
  }
  aiirPassManagerDestroy(pm);
}

// Helper variable to track callback invocations
static int initCnt = 0;

// Callback function that will be called during JIT initialization
static void initCallback(void) { initCnt += 1; }

// CHECK-LABEL: Running test 'testGlobalCtorJitCallback'
void testGlobalCtorJitCallback(void) {
  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  // Create module with global constructor that calls our callback
  AiirModule module = aiirModuleCreateParse(
      ctx, aiirStringRefCreateFromCString(
               // clang-format off
"module {                                                                       \n"
"  llvm.aiir.global_ctors ctors = [@ctor], priorities = [0 : i32], data = [#llvm.zero] \n"
"  llvm.func @ctor() {                                                          \n"
"    func.call @init_callback() : () -> ()                                      \n"
"    llvm.return                                                                \n"
"  }                                                                            \n"
"  func.func private @init_callback() attributes { llvm.emit_c_interface }      \n"
"}                                                                              \n"
               // clang-format on
               ));

  lowerModuleToLLVM(ctx, module);
  aiirRegisterAllLLVMTranslations(ctx);

  // Create execution engine with initialization disabled
  AiirExecutionEngine jit = aiirExecutionEngineCreate(
      module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL,
      /*enableObjectDump=*/false, /*enablePIC=*/false);

  if (aiirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Execution engine creation failed");
    exit(2);
  }

  // Register callback symbol before initialization
  aiirExecutionEngineRegisterSymbol(
      jit, aiirStringRefCreateFromCString("_aiir_ciface_init_callback"),
      (void *)(uintptr_t)initCallback);

  aiirExecutionEngineInitialize(jit);

  // CHECK: Init count: 1
  printf("Init count: %d\n", initCnt);

  aiirExecutionEngineDestroy(jit);
  aiirModuleDestroy(module);
  aiirContextDestroy(ctx);
}

int main(void) {

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define TEST(test)                                                             \
  printf("Running test '" STRINGIFY(test) "'\n");                              \
  test();
  TEST(testGlobalCtorJitCallback);
  return 0;
}

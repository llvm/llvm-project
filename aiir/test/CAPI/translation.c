//===- translation.c - Test AIIR Target translations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-translation-test 2>&1 | FileCheck %s

#include "llvm-c/Core.h"
#include "llvm-c/Support.h"
#include "llvm-c/Types.h"

#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/Dialect/LLVM.h"
#include "aiir-c/IR.h"
#include "aiir-c/RegisterEverything.h"
#include "aiir-c/Support.h"
#include "aiir-c/Target/LLVMIR.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testToLLVMIR()
static void testToLLVMIR(AiirContext ctx) {
  fprintf(stderr, "testToLLVMIR()\n");
  LLVMContextRef llvmCtx = LLVMContextCreate();

  const char *moduleString = "llvm.func @add(%arg0: i64, %arg1: i64) -> i64 { \
                                %0 = llvm.add %arg0, %arg1  : i64 \
                                llvm.return %0 : i64 \
                             }";

  aiirRegisterAllLLVMTranslations(ctx);

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));

  AiirOperation operation = aiirModuleGetOperation(module);

  LLVMModuleRef llvmModule = aiirTranslateModuleToLLVMIR(operation, llvmCtx);

  // clang-format off
  // CHECK: define i64 @add(i64 %[[arg1:.*]], i64 %[[arg2:.*]]) {
  // CHECK-NEXT:   %[[arg3:.*]] = add i64 %[[arg1]], %[[arg2]]
  // CHECK-NEXT:   ret i64 %[[arg3]]
  // CHECK-NEXT: }
  // clang-format on
  LLVMDumpModule(llvmModule);

  LLVMDisposeModule(llvmModule);
  aiirModuleDestroy(module);
  LLVMContextDispose(llvmCtx);
}

// CHECK-LABEL: testTypeToFromLLVMIRTranslator
static void testTypeToFromLLVMIRTranslator(AiirContext ctx) {
  fprintf(stderr, "testTypeToFromLLVMIRTranslator\n");
  LLVMContextRef llvmCtx = LLVMContextCreate();

  LLVMTypeRef llvmTy = LLVMInt32TypeInContext(llvmCtx);
  AiirTypeFromLLVMIRTranslator fromLLVMTranslator =
      aiirTypeFromLLVMIRTranslatorCreate(ctx);
  AiirType aiirTy =
      aiirTypeFromLLVMIRTranslatorTranslateType(fromLLVMTranslator, llvmTy);
  // CHECK: i32
  aiirTypeDump(aiirTy);

  AiirTypeToLLVMIRTranslator toLLVMTranslator =
      aiirTypeToLLVMIRTranslatorCreate(llvmCtx);
  LLVMTypeRef llvmTy2 =
      aiirTypeToLLVMIRTranslatorTranslateType(toLLVMTranslator, aiirTy);
  // CHECK: i32
  LLVMDumpType(llvmTy2);
  fprintf(stderr, "\n");

  aiirTypeFromLLVMIRTranslatorDestroy(fromLLVMTranslator);
  aiirTypeToLLVMIRTranslatorDestroy(toLLVMTranslator);
  LLVMContextDispose(llvmCtx);
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__llvm__(), ctx);
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("llvm"));
  testToLLVMIR(ctx);
  testTypeToFromLLVMIRTranslator(ctx);
  aiirContextDestroy(ctx);
  return 0;
}

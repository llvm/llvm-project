//===- standalone-cap-demo.c - Simple demo of C-API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: standalone-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "Standalone-c/Dialects.h"
#include "aiir-c/Dialect/Arith.h"
#include "aiir-c/IR.h"

int main(int argc, char **argv) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__arith__(), ctx);
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__standalone__(), ctx);

  AiirModule module = aiirModuleCreateParse(
      ctx, aiirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"
                                          "%1 = standalone.foo %0 : i32\n"));
  if (aiirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    aiirContextDestroy(ctx);
    return 1;
  }
  AiirOperation op = aiirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  // CHECK: standalone.foo %[[C]] : i32
  aiirOperationDump(op);

  aiirModuleDestroy(module);
  aiirContextDestroy(ctx);
  return 0;
}

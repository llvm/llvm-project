//===- transform_interpreter.c - Test of the Transform interpreter C API --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-transform-interpreter-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/Transform.h"
#include "aiir-c/Dialect/Transform/Interpreter.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#include <stdio.h>
#include <stdlib.h>

int testApplyNamedSequence(AiirContext ctx) {
  fprintf(stderr, "%s\n", __func__);

  const char module[] =
      "module attributes {transform.with_named_sequence} {"
      "  transform.named_sequence @__transform_main(%root: !transform.any_op) {"
      "    transform.print %root { name = \"from interpreter\" }: "
      "!transform.any_op"
      "    transform.yield"
      "  }"
      "}";

  AiirStringRef moduleStringRef = aiirStringRefCreateFromCString(module);
  AiirStringRef nameStringRef = aiirStringRefCreateFromCString("inline-module");

  AiirOperation root =
      aiirOperationCreateParse(ctx, moduleStringRef, nameStringRef);
  if (aiirOperationIsNull(root))
    return 1;
  AiirBlock body = aiirRegionGetFirstBlock(aiirOperationGetRegion(root, 0));
  AiirOperation entry = aiirBlockGetFirstOperation(body);

  AiirTransformOptions options = aiirTransformOptionsCreate();
  aiirTransformOptionsEnableExpensiveChecks(options, true);
  aiirTransformOptionsEnforceSingleTopLevelTransformOp(options, true);

  AiirLogicalResult result =
      aiirTransformApplyNamedSequence(root, entry, root, options);
  aiirTransformOptionsDestroy(options);
  aiirOperationDestroy(root);
  if (aiirLogicalResultIsFailure(result))
    return 2;

  return 0;
}
// CHECK-LABEL: testApplyNamedSequence
// CHECK: from interpreter
// CHECK: transform.named_sequence @__transform_main
// CHECK:   transform.print %arg0
// CHECK:   transform.yield

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__transform__(), ctx);
  int result = testApplyNamedSequence(ctx);
  aiirContextDestroy(ctx);
  if (result)
    return result;

  return EXIT_SUCCESS;
}

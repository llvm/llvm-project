//===- transform.c - Test of Transform dialect C API ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-transform-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/Transform.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testAnyOpType
void testAnyOpType(AiirContext ctx) {
  fprintf(stderr, "testAnyOpType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!transform.any_op"));
  AiirType constructedType = aiirTransformAnyOpTypeGet(ctx);

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse AnyOpType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct AnyOpType");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: parsedType isa AnyOpType: 1
  fprintf(stderr, "parsedType isa AnyOpType: %d\n",
          aiirTypeIsATransformAnyOpType(parsedType));
  // CHECK: parsedType isa OperationType: 0
  fprintf(stderr, "parsedType isa OperationType: %d\n",
          aiirTypeIsATransformOperationType(parsedType));

  // CHECK: !transform.any_op
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testOperationType
void testOperationType(AiirContext ctx) {
  fprintf(stderr, "testOperationType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!transform.op<\"foo.bar\">"));
  AiirType constructedType = aiirTransformOperationTypeGet(
      ctx, aiirStringRefCreateFromCString("foo.bar"));

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse AnyOpType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct AnyOpType");

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: parsedType isa AnyOpType: 0
  fprintf(stderr, "parsedType isa AnyOpType: %d\n",
          aiirTypeIsATransformAnyOpType(parsedType));
  // CHECK: parsedType isa OperationType: 1
  fprintf(stderr, "parsedType isa OperationType: %d\n",
          aiirTypeIsATransformOperationType(parsedType));

  // CHECK: operation name equal: 1
  AiirStringRef operationName =
      aiirTransformOperationTypeGetOperationName(constructedType);
  fprintf(stderr, "operation name equal: %d\n",
          aiirStringRefEqual(operationName,
                             aiirStringRefCreateFromCString("foo.bar")));

  // CHECK: !transform.op<"foo.bar">
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__transform__(), ctx);
  testAnyOpType(ctx);
  testOperationType(ctx);
  aiirContextDestroy(ctx);
  return EXIT_SUCCESS;
}

//===- pdl.c - Test of PDL dialect C API ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-pdl-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/PDL.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

// CHECK-LABEL: testAttributeType
void testAttributeType(AiirContext ctx) {
  fprintf(stderr, "testAttributeType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!pdl.attribute"));
  AiirType constructedType = aiirPDLAttributeTypeGet(ctx);

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 1
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 1
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.attribute
  aiirTypeDump(parsedType);
  // CHECK: !pdl.attribute
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testOperationType
void testOperationType(AiirContext ctx) {
  fprintf(stderr, "testOperationType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!pdl.operation"));
  AiirType constructedType = aiirPDLOperationTypeGet(ctx);

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 1 
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 1
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.operation
  aiirTypeDump(parsedType);
  // CHECK: !pdl.operation
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testRangeType
void testRangeType(AiirContext ctx) {
  fprintf(stderr, "testRangeType\n");

  AiirType typeType = aiirPDLTypeTypeGet(ctx);
  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!pdl.range<type>"));
  AiirType constructedType = aiirPDLRangeTypeGet(typeType);
  AiirType elementType = aiirPDLRangeTypeGetElementType(constructedType);

  assert(!aiirTypeIsNull(typeType) && "couldn't get PDLTypeType");
  assert(!aiirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 1 
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 1 
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));
  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(typeType, elementType));

  // CHECK: !pdl.range<type>
  aiirTypeDump(parsedType);
  // CHECK: !pdl.range<type>
  aiirTypeDump(constructedType);
  // CHECK: !pdl.type
  aiirTypeDump(elementType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testTypeType
void testTypeType(AiirContext ctx) {
  fprintf(stderr, "testTypeType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!pdl.type"));
  AiirType constructedType = aiirPDLTypeTypeGet(ctx);

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 1 
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 0
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 1
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 0
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.type
  aiirTypeDump(parsedType);
  // CHECK: !pdl.type
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

// CHECK-LABEL: testValueType
void testValueType(AiirContext ctx) {
  fprintf(stderr, "testValueType\n");

  AiirType parsedType = aiirTypeParseGet(
      ctx, aiirStringRefCreateFromCString("!pdl.value"));
  AiirType constructedType = aiirPDLValueTypeGet(ctx);

  assert(!aiirTypeIsNull(parsedType) && "couldn't parse PDLAttributeType");
  assert(!aiirTypeIsNull(constructedType) && "couldn't construct PDLAttributeType");

  // CHECK: parsedType isa PDLType: 1
  fprintf(stderr, "parsedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(parsedType));
  // CHECK: parsedType isa PDLAttributeType: 0
  fprintf(stderr, "parsedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(parsedType));
  // CHECK: parsedType isa PDLOperationType: 0
  fprintf(stderr, "parsedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(parsedType));
  // CHECK: parsedType isa PDLRangeType: 0
  fprintf(stderr, "parsedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(parsedType));
  // CHECK: parsedType isa PDLTypeType: 0
  fprintf(stderr, "parsedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(parsedType));
  // CHECK: parsedType isa PDLValueType: 1
  fprintf(stderr, "parsedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(parsedType));

  // CHECK: constructedType isa PDLType: 1
  fprintf(stderr, "constructedType isa PDLType: %d\n", 
      aiirTypeIsAPDLType(constructedType));
  // CHECK: constructedType isa PDLAttributeType: 0
  fprintf(stderr, "constructedType isa PDLAttributeType: %d\n", 
      aiirTypeIsAPDLAttributeType(constructedType));
  // CHECK: constructedType isa PDLOperationType: 0
  fprintf(stderr, "constructedType isa PDLOperationType: %d\n", 
      aiirTypeIsAPDLOperationType(constructedType));
  // CHECK: constructedType isa PDLRangeType: 0
  fprintf(stderr, "constructedType isa PDLRangeType: %d\n", 
      aiirTypeIsAPDLRangeType(constructedType));
  // CHECK: constructedType isa PDLTypeType: 0
  fprintf(stderr, "constructedType isa PDLTypeType: %d\n", 
      aiirTypeIsAPDLTypeType(constructedType));
  // CHECK: constructedType isa PDLValueType: 1
  fprintf(stderr, "constructedType isa PDLValueType: %d\n", 
      aiirTypeIsAPDLValueType(constructedType));

  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirTypeEqual(parsedType, constructedType));

  // CHECK: !pdl.value
  aiirTypeDump(parsedType);
  // CHECK: !pdl.value
  aiirTypeDump(constructedType);

  fprintf(stderr, "\n\n");
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__pdl__(), ctx);
  testAttributeType(ctx);
  testOperationType(ctx);
  testRangeType(ctx);
  testTypeType(ctx);
  testValueType(ctx);
  aiirContextDestroy(ctx);
  return EXIT_SUCCESS;
}

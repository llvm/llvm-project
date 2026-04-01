//===- smt.c - Test of SMT APIs -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: aiir-capi-smt-test 2>&1 | FileCheck %s
 */

#include "aiir-c/Dialect/SMT.h"
#include "aiir-c/Dialect/Func.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir-c/Target/ExportSMTLIB.h"
#include <assert.h>
#include <stdio.h>

void dumpCallback(AiirStringRef message, void *userData) {
  fprintf(stderr, "%.*s", (int)message.length, message.data);
}

void testExportSMTLIB(AiirContext ctx) {
  // clang-format off
  const char *testSMT = 
    "func.func @test() {\n"
    "  smt.solver() : () -> () { }\n"
    "  return\n"
    "}\n";
  // clang-format on

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(testSMT));

  AiirLogicalResult result = aiirTranslateModuleToSMTLIB(
      module, dumpCallback, NULL, false, false, true);
  (void)result;
  assert(aiirLogicalResultIsSuccess(result));

  // CHECK: ; solver scope 0
  // CHECK-NEXT: (reset)

  result = aiirTranslateModuleToSMTLIB(module, dumpCallback, NULL, false, false,
                                       false);
  assert(aiirLogicalResultIsSuccess(result));
  (void)result;

  // CHECK-NOT: (reset)
  aiirModuleDestroy(module);
}

void testSMTType(AiirContext ctx) {
  AiirType boolType = aiirSMTTypeGetBool(ctx);
  AiirType intType = aiirSMTTypeGetInt(ctx);
  AiirType arrayType = aiirSMTTypeGetArray(ctx, intType, boolType);
  AiirType bvType = aiirSMTTypeGetBitVector(ctx, 32);
  AiirType funcType =
      aiirSMTTypeGetSMTFunc(ctx, 2, (AiirType[]){intType, boolType}, boolType);
  AiirType sortType = aiirSMTTypeGetSort(
      ctx, aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("sort")), 0,
      NULL);

  // CHECK: !smt.bool
  aiirTypeDump(boolType);
  // CHECK: !smt.int
  aiirTypeDump(intType);
  // CHECK: !smt.array<[!smt.int -> !smt.bool]>
  aiirTypeDump(arrayType);
  // CHECK: !smt.bv<32>
  aiirTypeDump(bvType);
  // CHECK: !smt.func<(!smt.int, !smt.bool) !smt.bool>
  aiirTypeDump(funcType);
  // CHECK: !smt.sort<"sort">
  aiirTypeDump(sortType);

  // CHECK: bool_is_any_non_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(boolType)
                      ? "bool_is_any_non_func_smt_value_type\n"
                      : "bool_is_func_smt_value_type\n");
  // CHECK: int_is_any_non_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(intType)
                      ? "int_is_any_non_func_smt_value_type\n"
                      : "int_is_func_smt_value_type\n");
  // CHECK: array_is_any_non_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(arrayType)
                      ? "array_is_any_non_func_smt_value_type\n"
                      : "array_is_func_smt_value_type\n");
  // CHECK: bit_vector_is_any_non_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(bvType)
                      ? "bit_vector_is_any_non_func_smt_value_type\n"
                      : "bit_vector_is_func_smt_value_type\n");
  // CHECK: sort_is_any_non_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(sortType)
                      ? "sort_is_any_non_func_smt_value_type\n"
                      : "sort_is_func_smt_value_type\n");
  // CHECK: smt_func_is_func_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnyNonFuncSMTValueType(funcType)
                      ? "smt_func_is_any_non_func_smt_value_type\n"
                      : "smt_func_is_func_smt_value_type\n");

  // CHECK: bool_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(boolType)
                      ? "bool_is_any_smt_value_type\n"
                      : "bool_is_not_any_smt_value_type\n");
  // CHECK: int_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(intType)
                      ? "int_is_any_smt_value_type\n"
                      : "int_is_not_any_smt_value_type\n");
  // CHECK: array_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(arrayType)
                      ? "array_is_any_smt_value_type\n"
                      : "array_is_not_any_smt_value_type\n");
  // CHECK: array_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(bvType)
                      ? "array_is_any_smt_value_type\n"
                      : "array_is_not_any_smt_value_type\n");
  // CHECK: smt_func_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(funcType)
                      ? "smt_func_is_any_smt_value_type\n"
                      : "smt_func_is_not_any_smt_value_type\n");
  // CHECK: sort_is_any_smt_value_type
  fprintf(stderr, aiirSMTTypeIsAnySMTValueType(sortType)
                      ? "sort_is_any_smt_value_type\n"
                      : "sort_is_not_any_smt_value_type\n");

  // CHECK: int_type_is_not_a_bool
  fprintf(stderr, aiirSMTTypeIsABool(intType) ? "int_type_is_a_bool\n"
                                              : "int_type_is_not_a_bool\n");
  // CHECK: bool_type_is_not_a_int
  fprintf(stderr, aiirSMTTypeIsAInt(boolType) ? "bool_type_is_a_int\n"
                                              : "bool_type_is_not_a_int\n");
  // CHECK: bv_type_is_not_a_array
  fprintf(stderr, aiirSMTTypeIsAArray(bvType) ? "bv_type_is_a_array\n"
                                              : "bv_type_is_not_a_array\n");
  // CHECK: array_type_is_not_a_bit_vector
  fprintf(stderr, aiirSMTTypeIsABitVector(arrayType)
                      ? "array_type_is_a_bit_vector\n"
                      : "array_type_is_not_a_bit_vector\n");
  // CHECK: sort_type_is_not_a_smt_func
  fprintf(stderr, aiirSMTTypeIsASMTFunc(sortType)
                      ? "sort_type_is_a_smt_func\n"
                      : "sort_type_is_not_a_smt_func\n");
  // CHECK: func_type_is_not_a_sort
  fprintf(stderr, aiirSMTTypeIsASort(funcType) ? "func_type_is_a_sort\n"
                                               : "func_type_is_not_a_sort\n");
}

void testSMTAttribute(AiirContext ctx) {
  // CHECK: slt_is_BVCmpPredicate
  fprintf(stderr, aiirSMTAttrCheckBVCmpPredicate(
                      ctx, aiirStringRefCreateFromCString("slt"))
                      ? "slt_is_BVCmpPredicate\n"
                      : "slt_is_not_BVCmpPredicate\n");
  // CHECK: lt_is_not_BVCmpPredicate
  fprintf(stderr, aiirSMTAttrCheckBVCmpPredicate(
                      ctx, aiirStringRefCreateFromCString("lt"))
                      ? "lt_is_BVCmpPredicate\n"
                      : "lt_is_not_BVCmpPredicate\n");
  // CHECK: slt_is_not_IntPredicate
  fprintf(stderr, aiirSMTAttrCheckIntPredicate(
                      ctx, aiirStringRefCreateFromCString("slt"))
                      ? "slt_is_IntPredicate\n"
                      : "slt_is_not_IntPredicate\n");
  // CHECK: lt_is_IntPredicate
  fprintf(stderr, aiirSMTAttrCheckIntPredicate(
                      ctx, aiirStringRefCreateFromCString("lt"))
                      ? "lt_is_IntPredicate\n"
                      : "lt_is_not_IntPredicate\n");

  // CHECK: #smt.bv<5> : !smt.bv<32>
  aiirAttributeDump(aiirSMTAttrGetBitVector(ctx, 5, 32));
  // CHECK: 0 : i64
  aiirAttributeDump(
      aiirSMTAttrGetBVCmpPredicate(ctx, aiirStringRefCreateFromCString("slt")));
  // CHECK: 0 : i64
  aiirAttributeDump(
      aiirSMTAttrGetIntPredicate(ctx, aiirStringRefCreateFromCString("lt")));
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleLoadDialect(aiirGetDialectHandle__smt__(), ctx);
  aiirDialectHandleLoadDialect(aiirGetDialectHandle__func__(), ctx);
  testExportSMTLIB(ctx);
  testSMTType(ctx);
  testSMTAttribute(ctx);

  aiirContextDestroy(ctx);

  return 0;
}

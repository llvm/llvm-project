//===- llvm.c - Test of llvm APIs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-llvm-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/LLVM.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/DebugInfo.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testTypeCreation()
static void testTypeCreation(AiirContext ctx) {
  fprintf(stderr, "testTypeCreation()\n");
  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType i32 = aiirIntegerTypeGet(ctx, 32);
  AiirType i64 = aiirIntegerTypeGet(ctx, 64);

  const char *ptr_text = "!llvm.ptr";
  AiirType ptr = aiirLLVMPointerTypeGet(ctx, 0);
  AiirType ptr_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(ptr_text));
  // CHECK: !llvm.ptr: 1
  fprintf(stderr, "%s: %d\n", ptr_text, aiirTypeEqual(ptr, ptr_ref));

  const char *ptr_addr_text = "!llvm.ptr<42>";
  AiirType ptr_addr = aiirLLVMPointerTypeGet(ctx, 42);
  AiirType ptr_addr_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(ptr_addr_text));
  // CHECK: !llvm.ptr<42>: 1
  fprintf(stderr, "%s: %d\n", ptr_addr_text,
          aiirTypeEqual(ptr_addr, ptr_addr_ref));

  const char *voidt_text = "!llvm.void";
  AiirType voidt = aiirLLVMVoidTypeGet(ctx);
  AiirType voidt_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(voidt_text));
  // CHECK: !llvm.void: 1
  fprintf(stderr, "%s: %d\n", voidt_text, aiirTypeEqual(voidt, voidt_ref));

  const char *i32_4_text = "!llvm.array<4 x i32>";
  AiirType i32_4 = aiirLLVMArrayTypeGet(i32, 4);
  AiirType i32_4_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(i32_4_text));
  // CHECK: !llvm.array<4 x i32>: 1
  fprintf(stderr, "%s: %d\n", i32_4_text, aiirTypeEqual(i32_4, i32_4_ref));
  // CHECK: array_isa: 1
  fprintf(stderr, "array_isa: %d\n", aiirTypeIsALLVMArrayType(i32_4));
  // CHECK: array_element_type: 1
  fprintf(stderr, "array_element_type: %d\n",
          aiirTypeEqual(aiirLLVMArrayTypeGetElementType(i32_4), i32));
  // CHECK: array_num_elements: 4
  fprintf(stderr, "array_num_elements: %u\n",
          aiirLLVMArrayTypeGetNumElements(i32_4));

  const char *i8_i32_i64_text = "!llvm.func<i8 (i32, i64)>";
  const AiirType i32_i64_arr[] = {i32, i64};
  AiirType i8_i32_i64 = aiirLLVMFunctionTypeGet(i8, 2, i32_i64_arr, false);
  AiirType i8_i32_i64_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(i8_i32_i64_text));
  // CHECK: !llvm.func<i8 (i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i8_i32_i64_text,
          aiirTypeEqual(i8_i32_i64, i8_i32_i64_ref));

  const char *i32_i64_s_text = "!llvm.struct<(i32, i64)>";
  AiirType i32_i64_s = aiirLLVMStructTypeLiteralGet(ctx, 2, i32_i64_arr, false);
  AiirType i32_i64_s_ref =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString(i32_i64_s_text));
  // CHECK: !llvm.struct<(i32, i64)>: 1
  fprintf(stderr, "%s: %d\n", i32_i64_s_text,
          aiirTypeEqual(i32_i64_s, i32_i64_s_ref));
}

// CHECK-LABEL: testStructTypeCreation
static int testStructTypeCreation(AiirContext ctx) {
  fprintf(stderr, "testStructTypeCreation\n");

  // CHECK: !llvm.struct<()>
  aiirTypeDump(aiirLLVMStructTypeLiteralGet(ctx, /*nFieldTypes=*/0,
                                            /*fieldTypes=*/NULL,
                                            /*isPacked=*/false));

  AiirType i8 = aiirIntegerTypeGet(ctx, 8);
  AiirType i32 = aiirIntegerTypeGet(ctx, 32);
  AiirType i64 = aiirIntegerTypeGet(ctx, 64);
  AiirType i8_i32_i64[] = {i8, i32, i64};
  // CHECK: !llvm.struct<(i8, i32, i64)>
  aiirTypeDump(
      aiirLLVMStructTypeLiteralGet(ctx, sizeof(i8_i32_i64) / sizeof(AiirType),
                                   i8_i32_i64, /*isPacked=*/false));
  // CHECK: !llvm.struct<(i32)>
  aiirTypeDump(aiirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false));
  AiirType i32_i32[] = {i32, i32};
  // CHECK: !llvm.struct<packed (i32, i32)>
  aiirTypeDump(aiirLLVMStructTypeLiteralGet(
      ctx, sizeof(i32_i32) / sizeof(AiirType), i32_i32, /*isPacked=*/true));

  AiirType literal =
      aiirLLVMStructTypeLiteralGet(ctx, sizeof(i8_i32_i64) / sizeof(AiirType),
                                   i8_i32_i64, /*isPacked=*/false);
  // CHECK: num elements: 3
  // CHECK: i8
  // CHECK: i32
  // CHECK: i64
  fprintf(stderr, "num elements: %" PRIdPTR "\n",
          aiirLLVMStructTypeGetNumElementTypes(literal));
  for (intptr_t i = 0; i < 3; ++i) {
    aiirTypeDump(aiirLLVMStructTypeGetElementType(literal, i));
  }

  if (!aiirTypeEqual(
          aiirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false),
          aiirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false))) {
    return 1;
  }
  if (aiirTypeEqual(
          aiirLLVMStructTypeLiteralGet(ctx, 1, &i32, /*isPacked=*/false),
          aiirLLVMStructTypeLiteralGet(ctx, 1, &i64, /*isPacked=*/false))) {
    return 2;
  }

  // CHECK: !llvm.struct<"foo", opaque>
  // CHECK: !llvm.struct<"bar", opaque>
  aiirTypeDump(aiirLLVMStructTypeIdentifiedGet(
      ctx, aiirStringRefCreateFromCString("foo")));
  aiirTypeDump(aiirLLVMStructTypeIdentifiedGet(
      ctx, aiirStringRefCreateFromCString("bar")));

  if (!aiirTypeEqual(aiirLLVMStructTypeIdentifiedGet(
                         ctx, aiirStringRefCreateFromCString("foo")),
                     aiirLLVMStructTypeIdentifiedGet(
                         ctx, aiirStringRefCreateFromCString("foo")))) {
    return 3;
  }
  if (aiirTypeEqual(aiirLLVMStructTypeIdentifiedGet(
                        ctx, aiirStringRefCreateFromCString("foo")),
                    aiirLLVMStructTypeIdentifiedGet(
                        ctx, aiirStringRefCreateFromCString("bar")))) {
    return 4;
  }

  AiirType fooStruct = aiirLLVMStructTypeIdentifiedGet(
      ctx, aiirStringRefCreateFromCString("foo"));
  AiirStringRef name = aiirLLVMStructTypeGetIdentifier(fooStruct);
  if (memcmp(name.data, "foo", name.length))
    return 5;
  if (!aiirLLVMStructTypeIsOpaque(fooStruct))
    return 6;

  AiirType i32_i64[] = {i32, i64};
  AiirLogicalResult result =
      aiirLLVMStructTypeSetBody(fooStruct, sizeof(i32_i64) / sizeof(AiirType),
                                i32_i64, /*isPacked=*/false);
  if (!aiirLogicalResultIsSuccess(result))
    return 7;

  // CHECK: !llvm.struct<"foo", (i32, i64)>
  aiirTypeDump(fooStruct);
  if (aiirLLVMStructTypeIsOpaque(fooStruct))
    return 8;
  if (aiirLLVMStructTypeIsPacked(fooStruct))
    return 9;
  if (!aiirTypeEqual(aiirLLVMStructTypeIdentifiedGet(
                         ctx, aiirStringRefCreateFromCString("foo")),
                     fooStruct)) {
    return 10;
  }

  AiirType barStruct = aiirLLVMStructTypeIdentifiedGet(
      ctx, aiirStringRefCreateFromCString("bar"));
  result = aiirLLVMStructTypeSetBody(barStruct, 1, &i32, /*isPacked=*/true);
  if (!aiirLogicalResultIsSuccess(result))
    return 11;

  // CHECK: !llvm.struct<"bar", packed (i32)>
  aiirTypeDump(barStruct);
  if (!aiirLLVMStructTypeIsPacked(barStruct))
    return 12;

  // Same body, should succeed.
  result =
      aiirLLVMStructTypeSetBody(fooStruct, sizeof(i32_i64) / sizeof(AiirType),
                                i32_i64, /*isPacked=*/false);
  if (!aiirLogicalResultIsSuccess(result))
    return 13;

  // Different body, should fail.
  result = aiirLLVMStructTypeSetBody(fooStruct, 1, &i32, /*isPacked=*/false);
  if (aiirLogicalResultIsSuccess(result))
    return 14;

  // Packed flag differs, should fail.
  result = aiirLLVMStructTypeSetBody(barStruct, 1, &i32, /*isPacked=*/false);
  if (aiirLogicalResultIsSuccess(result))
    return 15;

  // Should have a different name.
  // CHECK: !llvm.struct<"foo{{[^"]+}}
  aiirTypeDump(aiirLLVMStructTypeIdentifiedNewGet(
      ctx, aiirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
      /*fieldTypes=*/NULL, /*isPacked=*/false));

  // Two freshly created "new" types must differ.
  if (aiirTypeEqual(
          aiirLLVMStructTypeIdentifiedNewGet(
              ctx, aiirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
              /*fieldTypes=*/NULL, /*isPacked=*/false),
          aiirLLVMStructTypeIdentifiedNewGet(
              ctx, aiirStringRefCreateFromCString("foo"), /*nFieldTypes=*/0,
              /*fieldTypes=*/NULL, /*isPacked=*/false))) {
    return 16;
  }

  AiirType opaque = aiirLLVMStructTypeOpaqueGet(
      ctx, aiirStringRefCreateFromCString("opaque"));
  // CHECK: !llvm.struct<"opaque", opaque>
  aiirTypeDump(opaque);
  if (!aiirLLVMStructTypeIsOpaque(opaque))
    return 17;

  return 0;
}

// CHECK-LABEL: testLLVMAttributes
static void testLLVMAttributes(AiirContext ctx) {
  fprintf(stderr, "testLLVMAttributes\n");

  // CHECK: #llvm.linkage<internal>
  aiirAttributeDump(aiirLLVMLinkageAttrGet(ctx, AiirLLVMLinkageInternal));
  // CHECK: #llvm.cconv<ccc>
  aiirAttributeDump(aiirLLVMCConvAttrGet(ctx, AiirLLVMCConvC));
  // CHECK: #llvm<comdat any>
  aiirAttributeDump(aiirLLVMComdatAttrGet(ctx, AiirLLVMComdatAny));
}

// CHECK-LABEL: testDebugInfoAttributes
static void testDebugInfoAttributes(AiirContext ctx) {
  fprintf(stderr, "testDebugInfoAttributes\n");

  AiirAttribute foo =
      aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("foo"));
  AiirAttribute bar =
      aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("bar"));

  AiirAttribute none = aiirUnitAttrGet(ctx);
  AiirAttribute id = aiirDistinctAttrCreate(none);
  AiirAttribute recId0 = aiirDistinctAttrCreate(none);
  AiirAttribute recId1 = aiirDistinctAttrCreate(none);

  // CHECK: #llvm.di_null_type
  aiirAttributeDump(aiirLLVMDINullTypeAttrGet(ctx));

  // CHECK: #llvm.di_basic_type<name = "foo", sizeInBits =
  // CHECK-SAME: 64, encoding = DW_ATE_signed>
  AiirAttribute di_type =
      aiirLLVMDIBasicTypeAttrGet(ctx, 0, foo, 64, AiirLLVMTypeEncodingSigned);
  aiirAttributeDump(di_type);

  AiirAttribute file = aiirLLVMDIFileAttrGet(ctx, foo, bar);

  // CHECK: #llvm.di_file<"foo" in "bar">
  aiirAttributeDump(file);

  AiirAttribute compile_unit = aiirLLVMDICompileUnitAttrGet(
      ctx, id, LLVMDWARFSourceLanguageC99, file, foo, false,
      AiirLLVMDIEmissionKindFull, false, AiirLLVMDINameTableKindDefault, bar, 0,
      NULL);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  aiirAttributeDump(compile_unit);

  AiirAttribute di_module = aiirLLVMDIModuleAttrGet(
      ctx, file, compile_unit, foo,
      aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("")), bar, foo, 1,
      0);
  // CHECK: #llvm.di_module<{{.*}}>
  aiirAttributeDump(di_module);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  aiirAttributeDump(aiirLLVMDIModuleAttrGetScope(di_module));

  // CHECK: 1 : i32
  aiirAttributeDump(aiirLLVMDIFlagsAttrGet(ctx, 0x1));

  // CHECK: #llvm.di_lexical_block<{{.*}}>
  aiirAttributeDump(
      aiirLLVMDILexicalBlockAttrGet(ctx, compile_unit, file, 1, 2));

  // CHECK: #llvm.di_lexical_block_file<{{.*}}>
  aiirAttributeDump(
      aiirLLVMDILexicalBlockFileAttrGet(ctx, compile_unit, file, 3));

  // CHECK: #llvm.di_local_variable<{{.*}}>
  AiirAttribute local_var = aiirLLVMDILocalVariableAttrGet(
      ctx, compile_unit, foo, file, 1, 0, 8, di_type, 0);
  aiirAttributeDump(local_var);
  // CHECK: #llvm.di_derived_type<{{.*}}>
  // CHECK-NOT: dwarfAddressSpace
  aiirAttributeDump(aiirLLVMDIDerivedTypeAttrGet(
      ctx, 0, bar, file, 1, compile_unit, di_type, 64, 8, 0,
      AIIR_CAPI_DWARF_ADDRESS_SPACE_NULL, 0, di_type));

  // CHECK: #llvm.di_derived_type<{{.*}} dwarfAddressSpace = 3{{.*}}>
  aiirAttributeDump(aiirLLVMDIDerivedTypeAttrGet(
      ctx, 0, bar, file, 1, compile_unit, di_type, 64, 8, 0, 3, 0, di_type));

  AiirAttribute subroutine_type =
      aiirLLVMDISubroutineTypeAttrGet(ctx, 0x0, 1, &di_type);

  // CHECK: #llvm.di_subroutine_type<{{.*}}>
  aiirAttributeDump(subroutine_type);

  AiirAttribute di_subprogram_self_rec =
      aiirLLVMDISubprogramAttrGetRecSelf(recId0);
  AiirAttribute di_imported_entity = aiirLLVMDIImportedEntityAttrGet(
      ctx, 0, di_subprogram_self_rec, di_module, file, 1, foo, 1, &local_var);

  aiirAttributeDump(di_imported_entity);
  // CHECK: #llvm.di_imported_entity<{{.*}}>

  AiirAttribute di_annotation = aiirLLVMDIAnnotationAttrGet(
      ctx, aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("foo")),
      aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("bar")));

  aiirAttributeDump(di_annotation);
  // CHECK: #llvm.di_annotation<{{.*}}>

  AiirAttribute di_subprogram = aiirLLVMDISubprogramAttrGet(
      ctx, recId0, false, id, compile_unit, compile_unit, foo, bar, file, 1, 2,
      0, subroutine_type, 1, &di_imported_entity, 1, &di_annotation);
  // CHECK: #llvm.di_subprogram<{{.*}}>
  aiirAttributeDump(di_subprogram);

  // CHECK: #llvm.di_compile_unit<{{.*}}>
  aiirAttributeDump(aiirLLVMDISubprogramAttrGetScope(di_subprogram));

  // CHECK: #llvm.di_file<{{.*}}>
  aiirAttributeDump(aiirLLVMDISubprogramAttrGetFile(di_subprogram));

  // CHECK: #llvm.di_subroutine_type<{{.*}}>
  aiirAttributeDump(aiirLLVMDISubprogramAttrGetType(di_subprogram));

  AiirAttribute expression_elem =
      aiirLLVMDIExpressionElemAttrGet(ctx, 1, 1, &(uint64_t){1});

  // CHECK: #llvm<di_expression_elem(1)>
  aiirAttributeDump(expression_elem);

  AiirAttribute expression =
      aiirLLVMDIExpressionAttrGet(ctx, 1, &expression_elem);
  // CHECK: #llvm.di_expression<[(1)]>
  aiirAttributeDump(expression);

  AiirAttribute string_type =
      aiirLLVMDIStringTypeAttrGet(ctx, 0x0, foo, 16, 0, local_var, expression,
                                  expression, AiirLLVMTypeEncodingSigned);
  // CHECK: #llvm.di_string_type<{{.*}}>
  aiirAttributeDump(string_type);

  // CHECK: #llvm.di_composite_type<recId = {{.*}}, isRecSelf = true>
  aiirAttributeDump(aiirLLVMDICompositeTypeAttrGetRecSelf(recId1));

  // CHECK: #llvm.di_composite_type<{{.*}}>
  aiirAttributeDump(aiirLLVMDICompositeTypeAttrGet(
      ctx, recId1, false, 0, foo, file, 1, compile_unit, di_type, 0, 64, 8, 1,
      &di_type, expression, expression, expression, expression));
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__llvm__(), ctx);
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("llvm"));
  testTypeCreation(ctx);
  int result = testStructTypeCreation(ctx);
  testLLVMAttributes(ctx);
  testDebugInfoAttributes(ctx);
  aiirContextDestroy(ctx);
  if (result)
    fprintf(stderr, "FAILED: code %d", result);
  return result;
}

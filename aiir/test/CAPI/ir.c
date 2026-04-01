//===- ir.c - Simple test of C APIs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: aiir-capi-ir-test 2>&1 | FileCheck %s
 */

#include "aiir-c/IR.h"
#include "aiir-c/AffineExpr.h"
#include "aiir-c/AffineMap.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/Diagnostics.h"
#include "aiir-c/Dialect/Func.h"
#include "aiir-c/IntegerSet.h"
#include "aiir-c/RegisterEverything.h"
#include "aiir-c/Support.h"

#include <assert.h>
#include <inttypes.h>
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

struct ResourceDeleteUserData {
  const char *name;
};
static struct ResourceDeleteUserData resourceI64BlobUserData = {
    "resource_i64_blob"};
static void reportResourceDelete(void *userData, const void *data, size_t size,
                                 size_t align) {
  fprintf(stderr, "reportResourceDelete: %s\n",
          ((struct ResourceDeleteUserData *)userData)->name);
}

void populateLoopBody(AiirContext ctx, AiirBlock loopBody,
                      AiirLocation location, AiirBlock funcBody) {
  AiirValue iv = aiirBlockGetArgument(loopBody, 0);
  AiirValue funcArg0 = aiirBlockGetArgument(funcBody, 0);
  AiirValue funcArg1 = aiirBlockGetArgument(funcBody, 1);
  AiirType f32Type =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString("f32"));

  AiirOperationState loadLHSState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("memref.load"), location);
  AiirValue loadLHSOperands[] = {funcArg0, iv};
  aiirOperationStateAddOperands(&loadLHSState, 2, loadLHSOperands);
  aiirOperationStateAddResults(&loadLHSState, 1, &f32Type);
  AiirOperation loadLHS = aiirOperationCreate(&loadLHSState);
  aiirBlockAppendOwnedOperation(loopBody, loadLHS);

  AiirOperationState loadRHSState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("memref.load"), location);
  AiirValue loadRHSOperands[] = {funcArg1, iv};
  aiirOperationStateAddOperands(&loadRHSState, 2, loadRHSOperands);
  aiirOperationStateAddResults(&loadRHSState, 1, &f32Type);
  AiirOperation loadRHS = aiirOperationCreate(&loadRHSState);
  aiirBlockAppendOwnedOperation(loopBody, loadRHS);

  AiirOperationState addState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.addf"), location);
  AiirValue addOperands[] = {aiirOperationGetResult(loadLHS, 0),
                             aiirOperationGetResult(loadRHS, 0)};
  aiirOperationStateAddOperands(&addState, 2, addOperands);
  aiirOperationStateAddResults(&addState, 1, &f32Type);
  AiirOperation add = aiirOperationCreate(&addState);
  aiirBlockAppendOwnedOperation(loopBody, add);

  AiirOperationState storeState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("memref.store"), location);
  AiirValue storeOperands[] = {aiirOperationGetResult(add, 0), funcArg0, iv};
  aiirOperationStateAddOperands(&storeState, 3, storeOperands);
  AiirOperation store = aiirOperationCreate(&storeState);
  aiirBlockAppendOwnedOperation(loopBody, store);

  AiirOperationState yieldState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("scf.yield"), location);
  AiirOperation yield = aiirOperationCreate(&yieldState);
  aiirBlockAppendOwnedOperation(loopBody, yield);
}

AiirModule makeAndDumpAdd(AiirContext ctx, AiirLocation location) {
  AiirModule moduleOp = aiirModuleCreateEmpty(location);
  AiirBlock moduleBody = aiirModuleGetBody(moduleOp);

  AiirType memrefType =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString("memref<?xf32>"));
  AiirType funcBodyArgTypes[] = {memrefType, memrefType};
  AiirLocation funcBodyArgLocs[] = {location, location};
  AiirRegion funcBodyRegion = aiirRegionCreate();
  AiirBlock funcBody =
      aiirBlockCreate(sizeof(funcBodyArgTypes) / sizeof(AiirType),
                      funcBodyArgTypes, funcBodyArgLocs);
  aiirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  AiirAttribute funcTypeAttr = aiirAttributeParseGet(
      ctx,
      aiirStringRefCreateFromCString("(memref<?xf32>, memref<?xf32>) -> ()"));
  AiirAttribute funcNameAttr =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("\"add\""));
  AiirNamedAttribute funcAttrs[] = {
      aiirNamedAttributeGet(
          aiirIdentifierGet(ctx,
                            aiirStringRefCreateFromCString("function_type")),
          funcTypeAttr),
      aiirNamedAttributeGet(
          aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("sym_name")),
          funcNameAttr)};
  AiirOperationState funcState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("func.func"), location);
  aiirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  aiirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  AiirOperation func = aiirOperationCreate(&funcState);
  aiirBlockInsertOwnedOperation(moduleBody, 0, func);

  AiirType indexType =
      aiirTypeParseGet(ctx, aiirStringRefCreateFromCString("index"));
  AiirAttribute indexZeroLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("0 : index"));
  AiirNamedAttribute indexZeroValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")),
      indexZeroLiteral);
  AiirOperationState constZeroState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), location);
  aiirOperationStateAddResults(&constZeroState, 1, &indexType);
  aiirOperationStateAddAttributes(&constZeroState, 1, &indexZeroValueAttr);
  AiirOperation constZero = aiirOperationCreate(&constZeroState);
  aiirBlockAppendOwnedOperation(funcBody, constZero);

  AiirValue funcArg0 = aiirBlockGetArgument(funcBody, 0);
  AiirValue constZeroValue = aiirOperationGetResult(constZero, 0);
  AiirValue dimOperands[] = {funcArg0, constZeroValue};
  AiirOperationState dimState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("memref.dim"), location);
  aiirOperationStateAddOperands(&dimState, 2, dimOperands);
  aiirOperationStateAddResults(&dimState, 1, &indexType);
  AiirOperation dim = aiirOperationCreate(&dimState);
  aiirBlockAppendOwnedOperation(funcBody, dim);

  AiirRegion loopBodyRegion = aiirRegionCreate();
  AiirBlock loopBody = aiirBlockCreate(0, NULL, NULL);
  aiirBlockAddArgument(loopBody, indexType, location);
  aiirRegionAppendOwnedBlock(loopBodyRegion, loopBody);

  AiirAttribute indexOneLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("1 : index"));
  AiirNamedAttribute indexOneValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")),
      indexOneLiteral);
  AiirOperationState constOneState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), location);
  aiirOperationStateAddResults(&constOneState, 1, &indexType);
  aiirOperationStateAddAttributes(&constOneState, 1, &indexOneValueAttr);
  AiirOperation constOne = aiirOperationCreate(&constOneState);
  aiirBlockAppendOwnedOperation(funcBody, constOne);

  AiirValue dimValue = aiirOperationGetResult(dim, 0);
  AiirValue constOneValue = aiirOperationGetResult(constOne, 0);
  AiirValue loopOperands[] = {constZeroValue, dimValue, constOneValue};
  AiirOperationState loopState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("scf.for"), location);
  aiirOperationStateAddOperands(&loopState, 3, loopOperands);
  aiirOperationStateAddOwnedRegions(&loopState, 1, &loopBodyRegion);
  AiirOperation loop = aiirOperationCreate(&loopState);
  aiirBlockAppendOwnedOperation(funcBody, loop);

  populateLoopBody(ctx, loopBody, location, funcBody);

  AiirOperationState retState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("func.return"), location);
  AiirOperation ret = aiirOperationCreate(&retState);
  aiirBlockAppendOwnedOperation(funcBody, ret);

  AiirOperation module = aiirModuleGetOperation(moduleOp);
  aiirOperationDump(module);
  // clang-format off
  // CHECK: module {
  // CHECK:   func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
  // CHECK:     %[[C1:.*]] = arith.constant 1 : index
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
  // CHECK:       %[[LHS:.*]] = memref.load %[[ARG0]][%[[I]]] : memref<?xf32>
  // CHECK:       %[[RHS:.*]] = memref.load %[[ARG1]][%[[I]]] : memref<?xf32>
  // CHECK:       %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
  // CHECK:       memref.store %[[SUM]], %[[ARG0]][%[[I]]] : memref<?xf32>
  // CHECK:     }
  // CHECK:     return
  // CHECK:   }
  // CHECK: }
  // clang-format on

  return moduleOp;
}

struct OpListNode {
  AiirOperation op;
  struct OpListNode *next;
};
typedef struct OpListNode OpListNode;

struct ModuleStats {
  unsigned numOperations;
  unsigned numAttributes;
  unsigned numBlocks;
  unsigned numRegions;
  unsigned numValues;
  unsigned numBlockArguments;
  unsigned numOpResults;
};
typedef struct ModuleStats ModuleStats;

int collectStatsSingle(OpListNode *head, ModuleStats *stats) {
  AiirOperation operation = head->op;
  stats->numOperations += 1;
  stats->numValues += aiirOperationGetNumResults(operation);
  stats->numAttributes += aiirOperationGetNumAttributes(operation);

  unsigned numRegions = aiirOperationGetNumRegions(operation);

  stats->numRegions += numRegions;

  intptr_t numResults = aiirOperationGetNumResults(operation);
  for (intptr_t i = 0; i < numResults; ++i) {
    AiirValue result = aiirOperationGetResult(operation, i);
    if (!aiirValueIsAOpResult(result))
      return 1;
    if (aiirValueIsABlockArgument(result))
      return 2;
    if (!aiirOperationEqual(operation, aiirOpResultGetOwner(result)))
      return 3;
    if (i != aiirOpResultGetResultNumber(result))
      return 4;
    ++stats->numOpResults;
  }

  AiirRegion region = aiirOperationGetFirstRegion(operation);
  while (!aiirRegionIsNull(region)) {
    for (AiirBlock block = aiirRegionGetFirstBlock(region);
         !aiirBlockIsNull(block); block = aiirBlockGetNextInRegion(block)) {
      ++stats->numBlocks;
      intptr_t numArgs = aiirBlockGetNumArguments(block);
      stats->numValues += numArgs;
      for (intptr_t j = 0; j < numArgs; ++j) {
        AiirValue arg = aiirBlockGetArgument(block, j);
        if (!aiirValueIsABlockArgument(arg))
          return 5;
        if (aiirValueIsAOpResult(arg))
          return 6;
        if (!aiirBlockEqual(block, aiirBlockArgumentGetOwner(arg)))
          return 7;
        if (j != aiirBlockArgumentGetArgNumber(arg))
          return 8;
        ++stats->numBlockArguments;
      }

      for (AiirOperation child = aiirBlockGetFirstOperation(block);
           !aiirOperationIsNull(child);
           child = aiirOperationGetNextInBlock(child)) {
        OpListNode *node = malloc(sizeof(OpListNode));
        node->op = child;
        node->next = head->next;
        head->next = node;
      }
    }
    region = aiirRegionGetNextInOperation(region);
  }
  return 0;
}

int collectStats(AiirOperation operation) {
  OpListNode *head = malloc(sizeof(OpListNode));
  head->op = operation;
  head->next = NULL;

  ModuleStats stats;
  stats.numOperations = 0;
  stats.numAttributes = 0;
  stats.numBlocks = 0;
  stats.numRegions = 0;
  stats.numValues = 0;
  stats.numBlockArguments = 0;
  stats.numOpResults = 0;

  do {
    int retval = collectStatsSingle(head, &stats);
    if (retval) {
      free(head);
      return retval;
    }
    OpListNode *next = head->next;
    free(head);
    head = next;
  } while (head);

  if (stats.numValues != stats.numBlockArguments + stats.numOpResults)
    return 100;

  fprintf(stderr, "@stats\n");
  fprintf(stderr, "Number of operations: %u\n", stats.numOperations);
  fprintf(stderr, "Number of attributes: %u\n", stats.numAttributes);
  fprintf(stderr, "Number of blocks: %u\n", stats.numBlocks);
  fprintf(stderr, "Number of regions: %u\n", stats.numRegions);
  fprintf(stderr, "Number of values: %u\n", stats.numValues);
  fprintf(stderr, "Number of block arguments: %u\n", stats.numBlockArguments);
  fprintf(stderr, "Number of op results: %u\n", stats.numOpResults);
  // clang-format off
  // CHECK-LABEL: @stats
  // CHECK: Number of operations: 12
  // CHECK: Number of attributes: 5
  // CHECK: Number of blocks: 3
  // CHECK: Number of regions: 3
  // CHECK: Number of values: 9
  // CHECK: Number of block arguments: 3
  // CHECK: Number of op results: 6
  // clang-format on
  return 0;
}

static void printToStderr(AiirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

static void printFirstOfEach(AiirContext ctx, AiirOperation operation) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  AiirRegion region = aiirOperationGetRegion(operation, 0);
  AiirBlock block = aiirRegionGetFirstBlock(region);
  AiirOperation function = aiirBlockGetFirstOperation(block);
  region = aiirOperationGetRegion(function, 0);
  AiirOperation parentOperation = function;
  block = aiirRegionGetFirstBlock(region);
  operation = aiirBlockGetFirstOperation(block);
  assert(aiirModuleIsNull(aiirModuleFromOperation(operation)));

  // Verify that parent operation and block report correctly.
  // CHECK: Parent operation eq: 1
  fprintf(stderr, "Parent operation eq: %d\n",
          aiirOperationEqual(aiirOperationGetParentOperation(operation),
                             parentOperation));
  // CHECK: Block eq: 1
  fprintf(stderr, "Block eq: %d\n",
          aiirBlockEqual(aiirOperationGetBlock(operation), block));
  // CHECK: Block parent operation eq: 1
  fprintf(
      stderr, "Block parent operation eq: %d\n",
      aiirOperationEqual(aiirBlockGetParentOperation(block), parentOperation));
  // CHECK: Block parent region eq: 1
  fprintf(stderr, "Block parent region eq: %d\n",
          aiirRegionEqual(aiirBlockGetParentRegion(block), region));

  // In the module we created, the first operation of the first function is
  // an "memref.dim", which has an attribute and a single result that we can
  // use to test the printing mechanism.
  aiirBlockPrint(block, printToStderr, NULL);
  fprintf(stderr, "\n");
  fprintf(stderr, "First operation: ");
  aiirOperationPrint(operation, printToStderr, NULL);
  fprintf(stderr, "\n");
  // clang-format off
  // CHECK:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK:   %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
  // CHECK:     %[[LHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:     %[[RHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
  // CHECK:     memref.store %[[SUM]], %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:   }
  // CHECK: return
  // CHECK: First operation: {{.*}} = arith.constant 0 : index
  // clang-format on

  // Get the operation name and print it.
  AiirIdentifier ident = aiirOperationGetName(operation);
  AiirStringRef identStr = aiirIdentifierStr(ident);
  fprintf(stderr, "Operation name: '");
  for (size_t i = 0; i < identStr.length; ++i)
    fputc(identStr.data[i], stderr);
  fprintf(stderr, "'\n");
  // CHECK: Operation name: 'arith.constant'

  // Get the identifier again and verify equal.
  AiirIdentifier identAgain = aiirIdentifierGet(ctx, identStr);
  fprintf(stderr, "Identifier equal: %d\n",
          aiirIdentifierEqual(ident, identAgain));
  // CHECK: Identifier equal: 1

  // Get the block terminator and print it.
  AiirOperation terminator = aiirBlockGetTerminator(block);
  fprintf(stderr, "Terminator: ");
  aiirOperationPrint(terminator, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Terminator: func.return

  // Get the attribute by name.
  bool hasValueAttr = aiirOperationHasInherentAttributeByName(
      operation, aiirStringRefCreateFromCString("value"));
  if (hasValueAttr)
    // CHECK: Has attr "value"
    fprintf(stderr, "Has attr \"value\"");

  AiirAttribute valueAttr0 = aiirOperationGetInherentAttributeByName(
      operation, aiirStringRefCreateFromCString("value"));
  fprintf(stderr, "Get attr \"value\": ");
  aiirAttributePrint(valueAttr0, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Get attr "value": 0 : index

  // Get a non-existing attribute and assert that it is null (sanity).
  fprintf(stderr, "does_not_exist is null: %d\n",
          aiirAttributeIsNull(aiirOperationGetDiscardableAttributeByName(
              operation, aiirStringRefCreateFromCString("does_not_exist"))));
  // CHECK: does_not_exist is null: 1

  // Get result 0 and its type.
  AiirValue value = aiirOperationGetResult(operation, 0);
  fprintf(stderr, "Result 0: ");
  aiirValuePrint(value, printToStderr, NULL);
  fprintf(stderr, "\n");
  fprintf(stderr, "Value is null: %d\n", aiirValueIsNull(value));
  // CHECK: Result 0: {{.*}} = arith.constant 0 : index
  // CHECK: Value is null: 0

  AiirType type = aiirValueGetType(value);
  fprintf(stderr, "Result 0 type: ");
  aiirTypePrint(type, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Result 0 type: index

  // Set a discardable attribute.
  aiirOperationSetDiscardableAttributeByName(
      operation, aiirStringRefCreateFromCString("custom_attr"),
      aiirBoolAttrGet(ctx, 1));
  fprintf(stderr, "Op with set attr: ");
  aiirOperationPrint(operation, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Op with set attr: {{.*}} {custom_attr = true}

  // Remove the attribute.
  fprintf(stderr, "Remove attr: %d\n",
          aiirOperationRemoveDiscardableAttributeByName(
              operation, aiirStringRefCreateFromCString("custom_attr")));
  fprintf(stderr, "Remove attr again: %d\n",
          aiirOperationRemoveDiscardableAttributeByName(
              operation, aiirStringRefCreateFromCString("custom_attr")));
  fprintf(stderr, "Removed attr is null: %d\n",
          aiirAttributeIsNull(aiirOperationGetDiscardableAttributeByName(
              operation, aiirStringRefCreateFromCString("custom_attr"))));
  // CHECK: Remove attr: 1
  // CHECK: Remove attr again: 0
  // CHECK: Removed attr is null: 1

  // Add a large attribute to verify printing flags.
  int64_t eltsShape[] = {4};
  int32_t eltsData[] = {1, 2, 3, 4};
  aiirOperationSetDiscardableAttributeByName(
      operation, aiirStringRefCreateFromCString("elts"),
      aiirDenseElementsAttrInt32Get(
          aiirRankedTensorTypeGet(1, eltsShape, aiirIntegerTypeGet(ctx, 32),
                                  aiirAttributeGetNull()),
          4, eltsData));
  AiirOpPrintingFlags flags = aiirOpPrintingFlagsCreate();
  aiirOpPrintingFlagsElideLargeElementsAttrs(flags, 2);
  aiirOpPrintingFlagsPrintGenericOpForm(flags);
  aiirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/1, /*prettyForm=*/0);
  aiirOpPrintingFlagsUseLocalScope(flags);
  fprintf(stderr, "Op print with all flags: ");
  aiirOperationPrintWithFlags(operation, flags, printToStderr, NULL);
  fprintf(stderr, "\n");
  fprintf(stderr, "Op print with state: ");
  AiirAsmState state = aiirAsmStateCreateForOperation(parentOperation, flags);
  aiirOperationPrintWithState(operation, state, printToStderr, NULL);
  fprintf(stderr, "\n");
  // clang-format off
  // CHECK: Op print with all flags: %{{.*}} = "arith.constant"() <{value = 0 : index}> {elts = dense_resource<__elided__> : tensor<4xi32>} : () -> index loc(unknown)
  // clang-format on

  aiirOpPrintingFlagsDestroy(flags);
  flags = aiirOpPrintingFlagsCreate();
  aiirOpPrintingFlagsSkipRegions(flags);
  fprintf(stderr, "Op print with skip regions flag: ");
  aiirOperationPrintWithFlags(function, flags, printToStderr, NULL);
  fprintf(stderr, "\n");
  // clang-format off
  // CHECK: Op print with skip regions flag: func.func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>)
  // CHECK-NOT: constant
  // CHECK-NOT: return
  // clang-format on

  fprintf(stderr, "With state: |");
  aiirValuePrintAsOperand(value, state, printToStderr, NULL);
  // CHECK: With state: |%0|
  fprintf(stderr, "|\n");
  aiirAsmStateDestroy(state);

  aiirOpPrintingFlagsDestroy(flags);
}

static int constructAndTraverseIr(AiirContext ctx) {
  AiirLocation location = aiirLocationUnknownGet(ctx);

  AiirModule moduleOp = makeAndDumpAdd(ctx, location);
  AiirOperation module = aiirModuleGetOperation(moduleOp);
  assert(!aiirModuleIsNull(aiirModuleFromOperation(module)));

  int errcode = collectStats(module);
  if (errcode)
    return errcode;

  printFirstOfEach(ctx, module);

  aiirModuleDestroy(moduleOp);
  return 0;
}

/// Creates an operation with a region containing multiple blocks with
/// operations and dumps it. The blocks and operations are inserted using
/// block/operation-relative API and their final order is checked.
static void buildWithInsertionsAndPrint(AiirContext ctx) {
  AiirLocation loc = aiirLocationUnknownGet(ctx);
  aiirContextSetAllowUnregisteredDialects(ctx, true);

  AiirRegion owningRegion = aiirRegionCreate();
  AiirBlock nullBlock = aiirRegionGetFirstBlock(owningRegion);
  AiirOperationState state = aiirOperationStateGet(
      aiirStringRefCreateFromCString("insertion.order.test"), loc);
  aiirOperationStateAddOwnedRegions(&state, 1, &owningRegion);
  AiirOperation op = aiirOperationCreate(&state);
  AiirRegion region = aiirOperationGetRegion(op, 0);

  // Use integer types of different bitwidth as block arguments in order to
  // differentiate blocks.
  AiirType i1 = aiirIntegerTypeGet(ctx, 1);
  AiirType i2 = aiirIntegerTypeGet(ctx, 2);
  AiirType i3 = aiirIntegerTypeGet(ctx, 3);
  AiirType i4 = aiirIntegerTypeGet(ctx, 4);
  AiirType i5 = aiirIntegerTypeGet(ctx, 5);
  AiirBlock block1 = aiirBlockCreate(1, &i1, &loc);
  AiirBlock block2 = aiirBlockCreate(1, &i2, &loc);
  AiirBlock block3 = aiirBlockCreate(1, &i3, &loc);
  AiirBlock block4 = aiirBlockCreate(1, &i4, &loc);
  AiirBlock block5 = aiirBlockCreate(1, &i5, &loc);
  // Insert blocks so as to obtain the 1-2-3-4 order,
  aiirRegionInsertOwnedBlockBefore(region, nullBlock, block3);
  aiirRegionInsertOwnedBlockBefore(region, block3, block2);
  aiirRegionInsertOwnedBlockAfter(region, nullBlock, block1);
  aiirRegionInsertOwnedBlockAfter(region, block3, block4);
  aiirRegionInsertOwnedBlockBefore(region, block3, block5);

  AiirOperationState op1State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op1"), loc);
  AiirOperationState op2State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op2"), loc);
  AiirOperationState op3State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op3"), loc);
  AiirOperationState op4State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op4"), loc);
  AiirOperationState op5State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op5"), loc);
  AiirOperationState op6State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op6"), loc);
  AiirOperationState op7State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op7"), loc);
  AiirOperationState op8State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op8"), loc);
  AiirOperation op1 = aiirOperationCreate(&op1State);
  AiirOperation op2 = aiirOperationCreate(&op2State);
  AiirOperation op3 = aiirOperationCreate(&op3State);
  AiirOperation op4 = aiirOperationCreate(&op4State);
  AiirOperation op5 = aiirOperationCreate(&op5State);
  AiirOperation op6 = aiirOperationCreate(&op6State);
  AiirOperation op7 = aiirOperationCreate(&op7State);
  AiirOperation op8 = aiirOperationCreate(&op8State);

  // Insert operations in the first block so as to obtain the 1-2-3-4 order.
  AiirOperation nullOperation = aiirBlockGetFirstOperation(block1);
  assert(aiirOperationIsNull(nullOperation));
  aiirBlockInsertOwnedOperationBefore(block1, nullOperation, op3);
  aiirBlockInsertOwnedOperationBefore(block1, op3, op2);
  aiirBlockInsertOwnedOperationAfter(block1, nullOperation, op1);
  aiirBlockInsertOwnedOperationAfter(block1, op3, op4);

  // Append operations to the rest of blocks to make them non-empty and thus
  // printable.
  aiirBlockAppendOwnedOperation(block2, op5);
  aiirBlockAppendOwnedOperation(block3, op6);
  aiirBlockAppendOwnedOperation(block4, op7);
  aiirBlockAppendOwnedOperation(block5, op8);

  // Remove block5.
  aiirBlockDetach(block5);
  aiirBlockDestroy(block5);

  aiirOperationDump(op);
  aiirOperationDestroy(op);
  aiirContextSetAllowUnregisteredDialects(ctx, false);
  // clang-format off
  // CHECK-LABEL:  "insertion.order.test"
  // CHECK:      ^{{.*}}(%{{.*}}: i1
  // CHECK:        "dummy.op1"
  // CHECK-NEXT:   "dummy.op2"
  // CHECK-NEXT:   "dummy.op3"
  // CHECK-NEXT:   "dummy.op4"
  // CHECK:      ^{{.*}}(%{{.*}}: i2
  // CHECK:        "dummy.op5"
  // CHECK-NOT:  ^{{.*}}(%{{.*}}: i5
  // CHECK-NOT:    "dummy.op8"
  // CHECK:      ^{{.*}}(%{{.*}}: i3
  // CHECK:        "dummy.op6"
  // CHECK:      ^{{.*}}(%{{.*}}: i4
  // CHECK:        "dummy.op7"
  // clang-format on
}

/// Creates operations with type inference and tests various failure modes.
static int createOperationWithTypeInference(AiirContext ctx) {
  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirAttribute iAttr = aiirIntegerAttrGet(aiirIntegerTypeGet(ctx, 32), 4);

  // The shape.const_size op implements result type inference and is only used
  // for that reason.
  AiirOperationState state = aiirOperationStateGet(
      aiirStringRefCreateFromCString("shape.const_size"), loc);
  AiirNamedAttribute valueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")), iAttr);
  aiirOperationStateAddAttributes(&state, 1, &valueAttr);
  aiirOperationStateEnableResultTypeInference(&state);

  // Expect result type inference to succeed.
  AiirOperation op = aiirOperationCreate(&state);
  if (aiirOperationIsNull(op)) {
    fprintf(stderr, "ERROR: Result type inference unexpectedly failed");
    return 1;
  }

  // CHECK: RESULT_TYPE_INFERENCE: !shape.size
  fprintf(stderr, "RESULT_TYPE_INFERENCE: ");
  aiirTypeDump(aiirValueGetType(aiirOperationGetResult(op, 0)));
  fprintf(stderr, "\n");
  aiirOperationDestroy(op);
  return 0;
}

/// Dumps instances of all builtin types to check that C API works correctly.
/// Additionally, performs simple identity checks that a builtin type
/// constructed with C API can be inspected and has the expected type. The
/// latter achieves full coverage of C API for builtin types. Returns 0 on
/// success and a non-zero error code on failure.
static int printBuiltinTypes(AiirContext ctx) {
  // Integer types.
  AiirType i32 = aiirIntegerTypeGet(ctx, 32);
  AiirType si32 = aiirIntegerTypeSignedGet(ctx, 32);
  AiirType ui32 = aiirIntegerTypeUnsignedGet(ctx, 32);
  if (!aiirTypeIsAInteger(i32) || aiirTypeIsAF32(i32))
    return 1;
  if (!aiirTypeIsAInteger(si32) || !aiirIntegerTypeIsSigned(si32))
    return 2;
  if (!aiirTypeIsAInteger(ui32) || !aiirIntegerTypeIsUnsigned(ui32))
    return 3;
  if (aiirTypeEqual(i32, ui32) || aiirTypeEqual(i32, si32))
    return 4;
  if (aiirIntegerTypeGetWidth(i32) != aiirIntegerTypeGetWidth(si32))
    return 5;
  fprintf(stderr, "@types\n");
  aiirTypeDump(i32);
  fprintf(stderr, "\n");
  aiirTypeDump(si32);
  fprintf(stderr, "\n");
  aiirTypeDump(ui32);
  fprintf(stderr, "\n");
  // CHECK-LABEL: @types
  // CHECK: i32
  // CHECK: si32
  // CHECK: ui32

  // Index type.
  AiirType index = aiirIndexTypeGet(ctx);
  if (!aiirTypeIsAIndex(index))
    return 6;
  aiirTypeDump(index);
  fprintf(stderr, "\n");
  // CHECK: index

  // Floating-point types.
  AiirType bf16 = aiirBF16TypeGet(ctx);
  AiirType f16 = aiirF16TypeGet(ctx);
  AiirType f32 = aiirF32TypeGet(ctx);
  AiirType f64 = aiirF64TypeGet(ctx);
  if (!aiirTypeIsABF16(bf16))
    return 7;
  if (!aiirTypeIsAF16(f16))
    return 9;
  if (!aiirTypeIsAF32(f32))
    return 10;
  if (!aiirTypeIsAF64(f64))
    return 11;
  aiirTypeDump(bf16);
  fprintf(stderr, "\n");
  aiirTypeDump(f16);
  fprintf(stderr, "\n");
  aiirTypeDump(f32);
  fprintf(stderr, "\n");
  aiirTypeDump(f64);
  fprintf(stderr, "\n");
  // CHECK: bf16
  // CHECK: f16
  // CHECK: f32
  // CHECK: f64

  // None type.
  AiirType none = aiirNoneTypeGet(ctx);
  if (!aiirTypeIsANone(none))
    return 12;
  aiirTypeDump(none);
  fprintf(stderr, "\n");
  // CHECK: none

  // Complex type.
  AiirType cplx = aiirComplexTypeGet(f32);
  if (!aiirTypeIsAComplex(cplx) ||
      !aiirTypeEqual(aiirComplexTypeGetElementType(cplx), f32))
    return 13;
  aiirTypeDump(cplx);
  fprintf(stderr, "\n");
  // CHECK: complex<f32>

  // Vector (and Shaped) type. ShapedType is a common base class for vectors,
  // memrefs and tensors, one cannot create instances of this class so it is
  // tested on an instance of vector type.
  int64_t shape[] = {2, 3};
  AiirType vector =
      aiirVectorTypeGet(sizeof(shape) / sizeof(int64_t), shape, f32);
  if (!aiirTypeIsAVector(vector) || !aiirTypeIsAShaped(vector))
    return 14;
  if (!aiirTypeEqual(aiirShapedTypeGetElementType(vector), f32) ||
      !aiirShapedTypeHasRank(vector) || aiirShapedTypeGetRank(vector) != 2 ||
      aiirShapedTypeGetDimSize(vector, 0) != 2 ||
      aiirShapedTypeIsDynamicDim(vector, 0) ||
      aiirShapedTypeGetDimSize(vector, 1) != 3 ||
      !aiirShapedTypeHasStaticShape(vector))
    return 15;
  aiirTypeDump(vector);
  fprintf(stderr, "\n");
  // CHECK: vector<2x3xf32>

  // Scalable vector type.
  bool scalable[] = {false, true};
  AiirType scalableVector = aiirVectorTypeGetScalable(
      sizeof(shape) / sizeof(int64_t), shape, scalable, f32);
  if (!aiirTypeIsAVector(scalableVector))
    return 16;
  if (!aiirVectorTypeIsScalable(scalableVector) ||
      aiirVectorTypeIsDimScalable(scalableVector, 0) ||
      !aiirVectorTypeIsDimScalable(scalableVector, 1))
    return 17;
  aiirTypeDump(scalableVector);
  fprintf(stderr, "\n");
  // CHECK: vector<2x[3]xf32>

  // Ranked tensor type.
  AiirType rankedTensor = aiirRankedTensorTypeGet(
      sizeof(shape) / sizeof(int64_t), shape, f32, aiirAttributeGetNull());
  if (!aiirTypeIsATensor(rankedTensor) ||
      !aiirTypeIsARankedTensor(rankedTensor) ||
      !aiirAttributeIsNull(aiirRankedTensorTypeGetEncoding(rankedTensor)))
    return 18;
  aiirTypeDump(rankedTensor);
  fprintf(stderr, "\n");
  // CHECK: tensor<2x3xf32>

  // Unranked tensor type.
  AiirType unrankedTensor = aiirUnrankedTensorTypeGet(f32);
  if (!aiirTypeIsATensor(unrankedTensor) ||
      !aiirTypeIsAUnrankedTensor(unrankedTensor) ||
      aiirShapedTypeHasRank(unrankedTensor))
    return 19;
  aiirTypeDump(unrankedTensor);
  fprintf(stderr, "\n");
  // CHECK: tensor<*xf32>

  // MemRef type.
  AiirAttribute memSpace2 = aiirIntegerAttrGet(aiirIntegerTypeGet(ctx, 64), 2);
  AiirType memRef = aiirMemRefTypeContiguousGet(
      f32, sizeof(shape) / sizeof(int64_t), shape, memSpace2);
  if (!aiirTypeIsAMemRef(memRef) ||
      !aiirAttributeEqual(aiirMemRefTypeGetMemorySpace(memRef), memSpace2))
    return 20;
  aiirTypeDump(memRef);
  fprintf(stderr, "\n");
  // CHECK: memref<2x3xf32, 2>

  // Unranked MemRef type.
  AiirAttribute memSpace4 = aiirIntegerAttrGet(aiirIntegerTypeGet(ctx, 64), 4);
  AiirType unrankedMemRef = aiirUnrankedMemRefTypeGet(f32, memSpace4);
  if (!aiirTypeIsAUnrankedMemRef(unrankedMemRef) ||
      aiirTypeIsAMemRef(unrankedMemRef) ||
      !aiirAttributeEqual(aiirUnrankedMemrefGetMemorySpace(unrankedMemRef),
                          memSpace4))
    return 21;
  aiirTypeDump(unrankedMemRef);
  fprintf(stderr, "\n");
  // CHECK: memref<*xf32, 4>

  // Tuple type.
  AiirType types[] = {unrankedMemRef, f32};
  AiirType tuple = aiirTupleTypeGet(ctx, 2, types);
  if (!aiirTypeIsATuple(tuple) || aiirTupleTypeGetNumTypes(tuple) != 2 ||
      !aiirTypeEqual(aiirTupleTypeGetType(tuple, 0), unrankedMemRef) ||
      !aiirTypeEqual(aiirTupleTypeGetType(tuple, 1), f32))
    return 22;
  aiirTypeDump(tuple);
  fprintf(stderr, "\n");
  // CHECK: tuple<memref<*xf32, 4>, f32>

  // Function type.
  AiirType funcInputs[2] = {aiirIndexTypeGet(ctx), aiirIntegerTypeGet(ctx, 1)};
  AiirType funcResults[3] = {aiirIntegerTypeGet(ctx, 16),
                             aiirIntegerTypeGet(ctx, 32),
                             aiirIntegerTypeGet(ctx, 64)};
  AiirType funcType = aiirFunctionTypeGet(ctx, 2, funcInputs, 3, funcResults);
  if (aiirFunctionTypeGetNumInputs(funcType) != 2)
    return 23;
  if (aiirFunctionTypeGetNumResults(funcType) != 3)
    return 24;
  if (!aiirTypeEqual(funcInputs[0], aiirFunctionTypeGetInput(funcType, 0)) ||
      !aiirTypeEqual(funcInputs[1], aiirFunctionTypeGetInput(funcType, 1)))
    return 25;
  if (!aiirTypeEqual(funcResults[0], aiirFunctionTypeGetResult(funcType, 0)) ||
      !aiirTypeEqual(funcResults[1], aiirFunctionTypeGetResult(funcType, 1)) ||
      !aiirTypeEqual(funcResults[2], aiirFunctionTypeGetResult(funcType, 2)))
    return 26;
  aiirTypeDump(funcType);
  fprintf(stderr, "\n");
  // CHECK: (index, i1) -> (i16, i32, i64)

  // Opaque type.
  AiirStringRef namespace = aiirStringRefCreate("dialect", 7);
  AiirStringRef data = aiirStringRefCreate("type", 4);
  aiirContextSetAllowUnregisteredDialects(ctx, true);
  AiirType opaque = aiirOpaqueTypeGet(ctx, namespace, data);
  aiirContextSetAllowUnregisteredDialects(ctx, false);
  if (!aiirTypeIsAOpaque(opaque) ||
      !aiirStringRefEqual(aiirOpaqueTypeGetDialectNamespace(opaque),
                          namespace) ||
      !aiirStringRefEqual(aiirOpaqueTypeGetData(opaque), data))
    return 27;
  aiirTypeDump(opaque);
  fprintf(stderr, "\n");
  // CHECK: !dialect.type

  return 0;
}

void callbackSetFixedLengthString(const char *data, intptr_t len,
                                  void *userData) {
  strncpy(userData, data, len);
}

bool stringIsEqual(const char *lhs, AiirStringRef rhs) {
  if (strlen(lhs) != rhs.length) {
    return false;
  }
  return !strncmp(lhs, rhs.data, rhs.length);
}

int printBuiltinAttributes(AiirContext ctx) {
  AiirAttribute floating =
      aiirFloatAttrDoubleGet(ctx, aiirF64TypeGet(ctx), 2.0);
  if (!aiirAttributeIsAFloat(floating) ||
      fabs(aiirFloatAttrGetValueDouble(floating) - 2.0) > 1E-6)
    return 1;
  fprintf(stderr, "@attrs\n");
  aiirAttributeDump(floating);
  // CHECK-LABEL: @attrs
  // CHECK: 2.000000e+00 : f64

  // Exercise aiirAttributeGetType() just for the first one.
  AiirType floatingType = aiirAttributeGetType(floating);
  aiirTypeDump(floatingType);
  // CHECK: f64

  AiirAttribute integer = aiirIntegerAttrGet(aiirIntegerTypeGet(ctx, 32), 42);
  AiirAttribute signedInteger =
      aiirIntegerAttrGet(aiirIntegerTypeSignedGet(ctx, 8), -1);
  AiirAttribute unsignedInteger =
      aiirIntegerAttrGet(aiirIntegerTypeUnsignedGet(ctx, 8), 255);
  if (!aiirAttributeIsAInteger(integer) ||
      aiirIntegerAttrGetValueInt(integer) != 42 ||
      aiirIntegerAttrGetValueSInt(signedInteger) != -1 ||
      aiirIntegerAttrGetValueUInt(unsignedInteger) != 255)
    return 2;
  aiirAttributeDump(integer);
  aiirAttributeDump(signedInteger);
  aiirAttributeDump(unsignedInteger);
  // CHECK: 42 : i32
  // CHECK: -1 : si8
  // CHECK: 255 : ui8

  AiirAttribute boolean = aiirBoolAttrGet(ctx, 1);
  if (!aiirAttributeIsABool(boolean) || !aiirBoolAttrGetValue(boolean))
    return 3;
  aiirAttributeDump(boolean);
  // CHECK: true

  const char data[] = "abcdefghijklmnopqestuvwxyz";
  AiirAttribute opaque =
      aiirOpaqueAttrGet(ctx, aiirStringRefCreateFromCString("func"), 3, data,
                        aiirNoneTypeGet(ctx));
  if (!aiirAttributeIsAOpaque(opaque) ||
      !stringIsEqual("func", aiirOpaqueAttrGetDialectNamespace(opaque)))
    return 4;

  AiirStringRef opaqueData = aiirOpaqueAttrGetData(opaque);
  if (opaqueData.length != 3 ||
      strncmp(data, opaqueData.data, opaqueData.length))
    return 5;
  aiirAttributeDump(opaque);
  // CHECK: #func.abc

  AiirAttribute string =
      aiirStringAttrGet(ctx, aiirStringRefCreate(data + 3, 2));
  if (!aiirAttributeIsAString(string))
    return 6;

  AiirStringRef stringValue = aiirStringAttrGetValue(string);
  if (stringValue.length != 2 ||
      strncmp(data + 3, stringValue.data, stringValue.length))
    return 7;
  aiirAttributeDump(string);
  // CHECK: "de"

  AiirAttribute flatSymbolRef =
      aiirFlatSymbolRefAttrGet(ctx, aiirStringRefCreate(data + 5, 3));
  if (!aiirAttributeIsAFlatSymbolRef(flatSymbolRef))
    return 8;

  AiirStringRef flatSymbolRefValue =
      aiirFlatSymbolRefAttrGetValue(flatSymbolRef);
  if (flatSymbolRefValue.length != 3 ||
      strncmp(data + 5, flatSymbolRefValue.data, flatSymbolRefValue.length))
    return 9;
  aiirAttributeDump(flatSymbolRef);
  // CHECK: @fgh

  AiirAttribute symbols[] = {flatSymbolRef, flatSymbolRef};
  AiirAttribute symbolRef =
      aiirSymbolRefAttrGet(ctx, aiirStringRefCreate(data + 8, 2), 2, symbols);
  if (!aiirAttributeIsASymbolRef(symbolRef) ||
      aiirSymbolRefAttrGetNumNestedReferences(symbolRef) != 2 ||
      !aiirAttributeEqual(aiirSymbolRefAttrGetNestedReference(symbolRef, 0),
                          flatSymbolRef) ||
      !aiirAttributeEqual(aiirSymbolRefAttrGetNestedReference(symbolRef, 1),
                          flatSymbolRef))
    return 10;

  AiirStringRef symbolRefLeaf = aiirSymbolRefAttrGetLeafReference(symbolRef);
  AiirStringRef symbolRefRoot = aiirSymbolRefAttrGetRootReference(symbolRef);
  if (symbolRefLeaf.length != 3 ||
      strncmp(data + 5, symbolRefLeaf.data, symbolRefLeaf.length) ||
      symbolRefRoot.length != 2 ||
      strncmp(data + 8, symbolRefRoot.data, symbolRefRoot.length))
    return 11;
  aiirAttributeDump(symbolRef);
  // CHECK: @ij::@fgh::@fgh

  AiirAttribute type = aiirTypeAttrGet(aiirF32TypeGet(ctx));
  if (!aiirAttributeIsAType(type) ||
      !aiirTypeEqual(aiirF32TypeGet(ctx), aiirTypeAttrGetValue(type)))
    return 12;
  aiirAttributeDump(type);
  // CHECK: f32

  AiirAttribute unit = aiirUnitAttrGet(ctx);
  if (!aiirAttributeIsAUnit(unit))
    return 13;
  aiirAttributeDump(unit);
  // CHECK: unit

  int64_t shape[] = {1, 2};

  int bools[] = {0, 1};
  uint8_t uints8[] = {0u, 1u};
  int8_t ints8[] = {0, 1};
  uint16_t uints16[] = {0u, 1u};
  int16_t ints16[] = {0, 1};
  uint32_t uints32[] = {0u, 1u};
  int32_t ints32[] = {0, 1};
  uint64_t uints64[] = {0u, 1u};
  int64_t ints64[] = {0, 1};
  float floats[] = {0.0f, 1.0f};
  double doubles[] = {0.0, 1.0};
  uint16_t bf16s[] = {0x0, 0x3f80};
  uint16_t f16s[] = {0x0, 0x3c00};
  AiirAttribute encoding = aiirAttributeGetNull();
  AiirAttribute boolElements = aiirDenseElementsAttrBoolGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 1), encoding),
      2, bools);
  AiirAttribute uint8Elements = aiirDenseElementsAttrUInt8Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 8),
                              encoding),
      2, uints8);
  AiirAttribute int8Elements = aiirDenseElementsAttrInt8Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 8), encoding),
      2, ints8);
  AiirAttribute uint16Elements = aiirDenseElementsAttrUInt16Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 16),
                              encoding),
      2, uints16);
  AiirAttribute int16Elements = aiirDenseElementsAttrInt16Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 16), encoding),
      2, ints16);
  AiirAttribute uint32Elements = aiirDenseElementsAttrUInt32Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 32),
                              encoding),
      2, uints32);
  AiirAttribute int32Elements = aiirDenseElementsAttrInt32Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 32), encoding),
      2, ints32);
  AiirAttribute uint64Elements = aiirDenseElementsAttrUInt64Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 64),
                              encoding),
      2, uints64);
  AiirAttribute int64Elements = aiirDenseElementsAttrInt64Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 64), encoding),
      2, ints64);
  AiirAttribute floatElements = aiirDenseElementsAttrFloatGet(
      aiirRankedTensorTypeGet(2, shape, aiirF32TypeGet(ctx), encoding), 2,
      floats);
  AiirAttribute doubleElements = aiirDenseElementsAttrDoubleGet(
      aiirRankedTensorTypeGet(2, shape, aiirF64TypeGet(ctx), encoding), 2,
      doubles);
  AiirAttribute bf16Elements = aiirDenseElementsAttrBFloat16Get(
      aiirRankedTensorTypeGet(2, shape, aiirBF16TypeGet(ctx), encoding), 2,
      bf16s);
  AiirAttribute f16Elements = aiirDenseElementsAttrFloat16Get(
      aiirRankedTensorTypeGet(2, shape, aiirF16TypeGet(ctx), encoding), 2,
      f16s);

  if (!aiirAttributeIsADenseElements(boolElements) ||
      !aiirAttributeIsADenseElements(uint8Elements) ||
      !aiirAttributeIsADenseElements(int8Elements) ||
      !aiirAttributeIsADenseElements(uint32Elements) ||
      !aiirAttributeIsADenseElements(int32Elements) ||
      !aiirAttributeIsADenseElements(uint64Elements) ||
      !aiirAttributeIsADenseElements(int64Elements) ||
      !aiirAttributeIsADenseElements(floatElements) ||
      !aiirAttributeIsADenseElements(doubleElements) ||
      !aiirAttributeIsADenseElements(bf16Elements) ||
      !aiirAttributeIsADenseElements(f16Elements))
    return 14;

  if (aiirDenseElementsAttrGetBoolValue(boolElements, 1) != 1 ||
      aiirDenseElementsAttrGetUInt8Value(uint8Elements, 1) != 1 ||
      aiirDenseElementsAttrGetInt8Value(int8Elements, 1) != 1 ||
      aiirDenseElementsAttrGetUInt16Value(uint16Elements, 1) != 1 ||
      aiirDenseElementsAttrGetInt16Value(int16Elements, 1) != 1 ||
      aiirDenseElementsAttrGetUInt32Value(uint32Elements, 1) != 1 ||
      aiirDenseElementsAttrGetInt32Value(int32Elements, 1) != 1 ||
      aiirDenseElementsAttrGetUInt64Value(uint64Elements, 1) != 1 ||
      aiirDenseElementsAttrGetInt64Value(int64Elements, 1) != 1 ||
      fabsf(aiirDenseElementsAttrGetFloatValue(floatElements, 1) - 1.0f) >
          1E-6f ||
      fabs(aiirDenseElementsAttrGetDoubleValue(doubleElements, 1) - 1.0) > 1E-6)
    return 15;

  aiirAttributeDump(boolElements);
  aiirAttributeDump(uint8Elements);
  aiirAttributeDump(int8Elements);
  aiirAttributeDump(uint32Elements);
  aiirAttributeDump(int32Elements);
  aiirAttributeDump(uint64Elements);
  aiirAttributeDump(int64Elements);
  aiirAttributeDump(floatElements);
  aiirAttributeDump(doubleElements);
  aiirAttributeDump(bf16Elements);
  aiirAttributeDump(f16Elements);
  // CHECK: dense<{{\[}}[false, true]]> : tensor<1x2xi1>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui8>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi8>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui32>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi32>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui64>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi64>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf32>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf64>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xbf16>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf16>

  AiirAttribute splatBool = aiirDenseElementsAttrBoolSplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 1), encoding),
      1);
  AiirAttribute splatUInt8 = aiirDenseElementsAttrUInt8SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 8),
                              encoding),
      1);
  AiirAttribute splatInt8 = aiirDenseElementsAttrInt8SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 8), encoding),
      1);
  AiirAttribute splatUInt32 = aiirDenseElementsAttrUInt32SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 32),
                              encoding),
      1);
  AiirAttribute splatInt32 = aiirDenseElementsAttrInt32SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 32), encoding),
      1);
  AiirAttribute splatUInt64 = aiirDenseElementsAttrUInt64SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 64),
                              encoding),
      1);
  AiirAttribute splatInt64 = aiirDenseElementsAttrInt64SplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 64), encoding),
      1);
  AiirAttribute splatFloat = aiirDenseElementsAttrFloatSplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirF32TypeGet(ctx), encoding), 1.0f);
  AiirAttribute splatDouble = aiirDenseElementsAttrDoubleSplatGet(
      aiirRankedTensorTypeGet(2, shape, aiirF64TypeGet(ctx), encoding), 1.0);

  if (!aiirAttributeIsADenseElements(splatBool) ||
      !aiirDenseElementsAttrIsSplat(splatBool) ||
      !aiirAttributeIsADenseElements(splatUInt8) ||
      !aiirDenseElementsAttrIsSplat(splatUInt8) ||
      !aiirAttributeIsADenseElements(splatInt8) ||
      !aiirDenseElementsAttrIsSplat(splatInt8) ||
      !aiirAttributeIsADenseElements(splatUInt32) ||
      !aiirDenseElementsAttrIsSplat(splatUInt32) ||
      !aiirAttributeIsADenseElements(splatInt32) ||
      !aiirDenseElementsAttrIsSplat(splatInt32) ||
      !aiirAttributeIsADenseElements(splatUInt64) ||
      !aiirDenseElementsAttrIsSplat(splatUInt64) ||
      !aiirAttributeIsADenseElements(splatInt64) ||
      !aiirDenseElementsAttrIsSplat(splatInt64) ||
      !aiirAttributeIsADenseElements(splatFloat) ||
      !aiirDenseElementsAttrIsSplat(splatFloat) ||
      !aiirAttributeIsADenseElements(splatDouble) ||
      !aiirDenseElementsAttrIsSplat(splatDouble))
    return 16;

  if (aiirDenseElementsAttrGetBoolSplatValue(splatBool) != 1 ||
      aiirDenseElementsAttrGetUInt8SplatValue(splatUInt8) != 1 ||
      aiirDenseElementsAttrGetInt8SplatValue(splatInt8) != 1 ||
      aiirDenseElementsAttrGetUInt32SplatValue(splatUInt32) != 1 ||
      aiirDenseElementsAttrGetInt32SplatValue(splatInt32) != 1 ||
      aiirDenseElementsAttrGetUInt64SplatValue(splatUInt64) != 1 ||
      aiirDenseElementsAttrGetInt64SplatValue(splatInt64) != 1 ||
      fabsf(aiirDenseElementsAttrGetFloatSplatValue(splatFloat) - 1.0f) >
          1E-6f ||
      fabs(aiirDenseElementsAttrGetDoubleSplatValue(splatDouble) - 1.0) > 1E-6)
    return 17;

  const uint8_t *uint8RawData =
      (const uint8_t *)aiirDenseElementsAttrGetRawData(uint8Elements);
  const int8_t *int8RawData =
      (const int8_t *)aiirDenseElementsAttrGetRawData(int8Elements);
  const uint32_t *uint32RawData =
      (const uint32_t *)aiirDenseElementsAttrGetRawData(uint32Elements);
  const int32_t *int32RawData =
      (const int32_t *)aiirDenseElementsAttrGetRawData(int32Elements);
  const uint64_t *uint64RawData =
      (const uint64_t *)aiirDenseElementsAttrGetRawData(uint64Elements);
  const int64_t *int64RawData =
      (const int64_t *)aiirDenseElementsAttrGetRawData(int64Elements);
  const float *floatRawData =
      (const float *)aiirDenseElementsAttrGetRawData(floatElements);
  const double *doubleRawData =
      (const double *)aiirDenseElementsAttrGetRawData(doubleElements);
  const uint16_t *bf16RawData =
      (const uint16_t *)aiirDenseElementsAttrGetRawData(bf16Elements);
  const uint16_t *f16RawData =
      (const uint16_t *)aiirDenseElementsAttrGetRawData(f16Elements);
  if (uint8RawData[0] != 0u || uint8RawData[1] != 1u || int8RawData[0] != 0 ||
      int8RawData[1] != 1 || uint32RawData[0] != 0u || uint32RawData[1] != 1u ||
      int32RawData[0] != 0 || int32RawData[1] != 1 || uint64RawData[0] != 0u ||
      uint64RawData[1] != 1u || int64RawData[0] != 0 || int64RawData[1] != 1 ||
      floatRawData[0] != 0.0f || floatRawData[1] != 1.0f ||
      doubleRawData[0] != 0.0 || doubleRawData[1] != 1.0 ||
      bf16RawData[0] != 0 || bf16RawData[1] != 0x3f80 || f16RawData[0] != 0 ||
      f16RawData[1] != 0x3c00)
    return 18;

  aiirAttributeDump(splatBool);
  aiirAttributeDump(splatUInt8);
  aiirAttributeDump(splatInt8);
  aiirAttributeDump(splatUInt32);
  aiirAttributeDump(splatInt32);
  aiirAttributeDump(splatUInt64);
  aiirAttributeDump(splatInt64);
  aiirAttributeDump(splatFloat);
  aiirAttributeDump(splatDouble);
  // CHECK: dense<true> : tensor<1x2xi1>
  // CHECK: dense<1> : tensor<1x2xui8>
  // CHECK: dense<1> : tensor<1x2xi8>
  // CHECK: dense<1> : tensor<1x2xui32>
  // CHECK: dense<1> : tensor<1x2xi32>
  // CHECK: dense<1> : tensor<1x2xui64>
  // CHECK: dense<1> : tensor<1x2xi64>
  // CHECK: dense<1.000000e+00> : tensor<1x2xf32>
  // CHECK: dense<1.000000e+00> : tensor<1x2xf64>

  aiirAttributeDump(aiirElementsAttrGetValue(floatElements, 2, uints64));
  aiirAttributeDump(aiirElementsAttrGetValue(doubleElements, 2, uints64));
  aiirAttributeDump(aiirElementsAttrGetValue(bf16Elements, 2, uints64));
  aiirAttributeDump(aiirElementsAttrGetValue(f16Elements, 2, uints64));
  // CHECK: 1.000000e+00 : f32
  // CHECK: 1.000000e+00 : f64
  // CHECK: 1.000000e+00 : bf16
  // CHECK: 1.000000e+00 : f16

  int64_t indices[] = {0, 1};
  int64_t one = 1;
  AiirAttribute indicesAttr = aiirDenseElementsAttrInt64Get(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 64), encoding),
      2, indices);
  AiirAttribute valuesAttr = aiirDenseElementsAttrFloatGet(
      aiirRankedTensorTypeGet(1, &one, aiirF32TypeGet(ctx), encoding), 1,
      floats);
  AiirAttribute sparseAttr = aiirSparseElementsAttribute(
      aiirRankedTensorTypeGet(2, shape, aiirF32TypeGet(ctx), encoding),
      indicesAttr, valuesAttr);
  aiirAttributeDump(sparseAttr);
  // CHECK: sparse<{{\[}}[0, 1]], 0.000000e+00> : tensor<1x2xf32>

  AiirAttribute boolArray = aiirDenseBoolArrayGet(ctx, 2, bools);
  AiirAttribute int8Array = aiirDenseI8ArrayGet(ctx, 2, ints8);
  AiirAttribute int16Array = aiirDenseI16ArrayGet(ctx, 2, ints16);
  AiirAttribute int32Array = aiirDenseI32ArrayGet(ctx, 2, ints32);
  AiirAttribute int64Array = aiirDenseI64ArrayGet(ctx, 2, ints64);
  AiirAttribute floatArray = aiirDenseF32ArrayGet(ctx, 2, floats);
  AiirAttribute doubleArray = aiirDenseF64ArrayGet(ctx, 2, doubles);
  if (!aiirAttributeIsADenseBoolArray(boolArray) ||
      !aiirAttributeIsADenseI8Array(int8Array) ||
      !aiirAttributeIsADenseI16Array(int16Array) ||
      !aiirAttributeIsADenseI32Array(int32Array) ||
      !aiirAttributeIsADenseI64Array(int64Array) ||
      !aiirAttributeIsADenseF32Array(floatArray) ||
      !aiirAttributeIsADenseF64Array(doubleArray))
    return 19;

  if (aiirDenseArrayGetNumElements(boolArray) != 2 ||
      aiirDenseArrayGetNumElements(int8Array) != 2 ||
      aiirDenseArrayGetNumElements(int16Array) != 2 ||
      aiirDenseArrayGetNumElements(int32Array) != 2 ||
      aiirDenseArrayGetNumElements(int64Array) != 2 ||
      aiirDenseArrayGetNumElements(floatArray) != 2 ||
      aiirDenseArrayGetNumElements(doubleArray) != 2)
    return 20;

  if (aiirDenseBoolArrayGetElement(boolArray, 1) != 1 ||
      aiirDenseI8ArrayGetElement(int8Array, 1) != 1 ||
      aiirDenseI16ArrayGetElement(int16Array, 1) != 1 ||
      aiirDenseI32ArrayGetElement(int32Array, 1) != 1 ||
      aiirDenseI64ArrayGetElement(int64Array, 1) != 1 ||
      fabsf(aiirDenseF32ArrayGetElement(floatArray, 1) - 1.0f) > 1E-6f ||
      fabs(aiirDenseF64ArrayGetElement(doubleArray, 1) - 1.0) > 1E-6)
    return 21;

  int64_t layoutStrides[3] = {5, 7, 13};
  AiirAttribute stridedLayoutAttr =
      aiirStridedLayoutAttrGet(ctx, 42, 3, &layoutStrides[0]);

  // CHECK: strided<[5, 7, 13], offset: 42>
  aiirAttributeDump(stridedLayoutAttr);

  if (aiirStridedLayoutAttrGetOffset(stridedLayoutAttr) != 42 ||
      aiirStridedLayoutAttrGetNumStrides(stridedLayoutAttr) != 3 ||
      aiirStridedLayoutAttrGetStride(stridedLayoutAttr, 0) != 5 ||
      aiirStridedLayoutAttrGetStride(stridedLayoutAttr, 1) != 7 ||
      aiirStridedLayoutAttrGetStride(stridedLayoutAttr, 2) != 13)
    return 22;

  AiirAttribute uint8Blob = aiirUnmanagedDenseUInt8ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 8),
                              encoding),
      aiirStringRefCreateFromCString("resource_ui8"), 2, uints8);
  AiirAttribute uint16Blob = aiirUnmanagedDenseUInt16ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 16),
                              encoding),
      aiirStringRefCreateFromCString("resource_ui16"), 2, uints16);
  AiirAttribute uint32Blob = aiirUnmanagedDenseUInt32ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 32),
                              encoding),
      aiirStringRefCreateFromCString("resource_ui32"), 2, uints32);
  AiirAttribute uint64Blob = aiirUnmanagedDenseUInt64ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeUnsignedGet(ctx, 64),
                              encoding),
      aiirStringRefCreateFromCString("resource_ui64"), 2, uints64);
  AiirAttribute int8Blob = aiirUnmanagedDenseInt8ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 8), encoding),
      aiirStringRefCreateFromCString("resource_i8"), 2, ints8);
  AiirAttribute int16Blob = aiirUnmanagedDenseInt16ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 16), encoding),
      aiirStringRefCreateFromCString("resource_i16"), 2, ints16);
  AiirAttribute int32Blob = aiirUnmanagedDenseInt32ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 32), encoding),
      aiirStringRefCreateFromCString("resource_i32"), 2, ints32);
  AiirAttribute int64Blob = aiirUnmanagedDenseInt64ResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 64), encoding),
      aiirStringRefCreateFromCString("resource_i64"), 2, ints64);
  AiirAttribute floatsBlob = aiirUnmanagedDenseFloatResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirF32TypeGet(ctx), encoding),
      aiirStringRefCreateFromCString("resource_f32"), 2, floats);
  AiirAttribute doublesBlob = aiirUnmanagedDenseDoubleResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirF64TypeGet(ctx), encoding),
      aiirStringRefCreateFromCString("resource_f64"), 2, doubles);
  AiirAttribute blobBlob = aiirUnmanagedDenseResourceElementsAttrGet(
      aiirRankedTensorTypeGet(2, shape, aiirIntegerTypeGet(ctx, 64), encoding),
      aiirStringRefCreateFromCString("resource_i64_blob"), /*data=*/uints64,
      /*dataLength=*/sizeof(uints64),
      /*dataAlignment=*/_Alignof(uint64_t),
      /*dataIsMutable=*/false,
      /*deleter=*/reportResourceDelete,
      /*userData=*/(void *)&resourceI64BlobUserData);

  aiirAttributeDump(uint8Blob);
  aiirAttributeDump(uint16Blob);
  aiirAttributeDump(uint32Blob);
  aiirAttributeDump(uint64Blob);
  aiirAttributeDump(int8Blob);
  aiirAttributeDump(int16Blob);
  aiirAttributeDump(int32Blob);
  aiirAttributeDump(int64Blob);
  aiirAttributeDump(floatsBlob);
  aiirAttributeDump(doublesBlob);
  aiirAttributeDump(blobBlob);
  // CHECK: dense_resource<resource_ui8> : tensor<1x2xui8>
  // CHECK: dense_resource<resource_ui16> : tensor<1x2xui16>
  // CHECK: dense_resource<resource_ui32> : tensor<1x2xui32>
  // CHECK: dense_resource<resource_ui64> : tensor<1x2xui64>
  // CHECK: dense_resource<resource_i8> : tensor<1x2xi8>
  // CHECK: dense_resource<resource_i16> : tensor<1x2xi16>
  // CHECK: dense_resource<resource_i32> : tensor<1x2xi32>
  // CHECK: dense_resource<resource_i64> : tensor<1x2xi64>
  // CHECK: dense_resource<resource_f32> : tensor<1x2xf32>
  // CHECK: dense_resource<resource_f64> : tensor<1x2xf64>
  // CHECK: dense_resource<resource_i64_blob> : tensor<1x2xi64>

  if (aiirDenseUInt8ResourceElementsAttrGetValue(uint8Blob, 1) != 1 ||
      aiirDenseUInt16ResourceElementsAttrGetValue(uint16Blob, 1) != 1 ||
      aiirDenseUInt32ResourceElementsAttrGetValue(uint32Blob, 1) != 1 ||
      aiirDenseUInt64ResourceElementsAttrGetValue(uint64Blob, 1) != 1 ||
      aiirDenseInt8ResourceElementsAttrGetValue(int8Blob, 1) != 1 ||
      aiirDenseInt16ResourceElementsAttrGetValue(int16Blob, 1) != 1 ||
      aiirDenseInt32ResourceElementsAttrGetValue(int32Blob, 1) != 1 ||
      aiirDenseInt64ResourceElementsAttrGetValue(int64Blob, 1) != 1 ||
      fabsf(aiirDenseF32ArrayGetElement(floatArray, 1) - 1.0f) > 1E-6f ||
      fabsf(aiirDenseFloatResourceElementsAttrGetValue(floatsBlob, 1) - 1.0f) >
          1e-6 ||
      fabs(aiirDenseDoubleResourceElementsAttrGetValue(doublesBlob, 1) - 1.0f) >
          1e-6 ||
      aiirDenseUInt64ResourceElementsAttrGetValue(blobBlob, 1) != 1)
    return 23;

  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirAttribute locAttr = aiirLocationGetAttribute(loc);
  if (!aiirAttributeIsALocation(locAttr))
    return 24;

  return 0;
}

int printAffineMap(AiirContext ctx) {
  AiirAffineMap emptyAffineMap = aiirAffineMapEmptyGet(ctx);
  AiirAffineMap affineMap = aiirAffineMapZeroResultGet(ctx, 3, 2);
  AiirAffineMap constAffineMap = aiirAffineMapConstantGet(ctx, 2);
  AiirAffineMap multiDimIdentityAffineMap =
      aiirAffineMapMultiDimIdentityGet(ctx, 3);
  AiirAffineMap minorIdentityAffineMap =
      aiirAffineMapMinorIdentityGet(ctx, 3, 2);
  unsigned permutation[] = {1, 2, 0};
  AiirAffineMap permutationAffineMap = aiirAffineMapPermutationGet(
      ctx, sizeof(permutation) / sizeof(unsigned), permutation);

  fprintf(stderr, "@affineMap\n");
  aiirAffineMapDump(emptyAffineMap);
  aiirAffineMapDump(affineMap);
  aiirAffineMapDump(constAffineMap);
  aiirAffineMapDump(multiDimIdentityAffineMap);
  aiirAffineMapDump(minorIdentityAffineMap);
  aiirAffineMapDump(permutationAffineMap);
  // CHECK-LABEL: @affineMap
  // CHECK: () -> ()
  // CHECK: (d0, d1, d2)[s0, s1] -> ()
  // CHECK: () -> (2)
  // CHECK: (d0, d1, d2) -> (d0, d1, d2)
  // CHECK: (d0, d1, d2) -> (d1, d2)
  // CHECK: (d0, d1, d2) -> (d1, d2, d0)

  if (!aiirAffineMapIsIdentity(emptyAffineMap) ||
      aiirAffineMapIsIdentity(affineMap) ||
      aiirAffineMapIsIdentity(constAffineMap) ||
      !aiirAffineMapIsIdentity(multiDimIdentityAffineMap) ||
      aiirAffineMapIsIdentity(minorIdentityAffineMap) ||
      aiirAffineMapIsIdentity(permutationAffineMap))
    return 1;

  if (!aiirAffineMapIsMinorIdentity(emptyAffineMap) ||
      aiirAffineMapIsMinorIdentity(affineMap) ||
      !aiirAffineMapIsMinorIdentity(multiDimIdentityAffineMap) ||
      !aiirAffineMapIsMinorIdentity(minorIdentityAffineMap) ||
      aiirAffineMapIsMinorIdentity(permutationAffineMap))
    return 2;

  if (!aiirAffineMapIsEmpty(emptyAffineMap) ||
      aiirAffineMapIsEmpty(affineMap) || aiirAffineMapIsEmpty(constAffineMap) ||
      aiirAffineMapIsEmpty(multiDimIdentityAffineMap) ||
      aiirAffineMapIsEmpty(minorIdentityAffineMap) ||
      aiirAffineMapIsEmpty(permutationAffineMap))
    return 3;

  if (aiirAffineMapIsSingleConstant(emptyAffineMap) ||
      aiirAffineMapIsSingleConstant(affineMap) ||
      !aiirAffineMapIsSingleConstant(constAffineMap) ||
      aiirAffineMapIsSingleConstant(multiDimIdentityAffineMap) ||
      aiirAffineMapIsSingleConstant(minorIdentityAffineMap) ||
      aiirAffineMapIsSingleConstant(permutationAffineMap))
    return 4;

  if (aiirAffineMapGetSingleConstantResult(constAffineMap) != 2)
    return 5;

  if (aiirAffineMapGetNumDims(emptyAffineMap) != 0 ||
      aiirAffineMapGetNumDims(affineMap) != 3 ||
      aiirAffineMapGetNumDims(constAffineMap) != 0 ||
      aiirAffineMapGetNumDims(multiDimIdentityAffineMap) != 3 ||
      aiirAffineMapGetNumDims(minorIdentityAffineMap) != 3 ||
      aiirAffineMapGetNumDims(permutationAffineMap) != 3)
    return 6;

  if (aiirAffineMapGetNumSymbols(emptyAffineMap) != 0 ||
      aiirAffineMapGetNumSymbols(affineMap) != 2 ||
      aiirAffineMapGetNumSymbols(constAffineMap) != 0 ||
      aiirAffineMapGetNumSymbols(multiDimIdentityAffineMap) != 0 ||
      aiirAffineMapGetNumSymbols(minorIdentityAffineMap) != 0 ||
      aiirAffineMapGetNumSymbols(permutationAffineMap) != 0)
    return 7;

  if (aiirAffineMapGetNumResults(emptyAffineMap) != 0 ||
      aiirAffineMapGetNumResults(affineMap) != 0 ||
      aiirAffineMapGetNumResults(constAffineMap) != 1 ||
      aiirAffineMapGetNumResults(multiDimIdentityAffineMap) != 3 ||
      aiirAffineMapGetNumResults(minorIdentityAffineMap) != 2 ||
      aiirAffineMapGetNumResults(permutationAffineMap) != 3)
    return 8;

  if (aiirAffineMapGetNumInputs(emptyAffineMap) != 0 ||
      aiirAffineMapGetNumInputs(affineMap) != 5 ||
      aiirAffineMapGetNumInputs(constAffineMap) != 0 ||
      aiirAffineMapGetNumInputs(multiDimIdentityAffineMap) != 3 ||
      aiirAffineMapGetNumInputs(minorIdentityAffineMap) != 3 ||
      aiirAffineMapGetNumInputs(permutationAffineMap) != 3)
    return 9;

  if (!aiirAffineMapIsProjectedPermutation(emptyAffineMap) ||
      !aiirAffineMapIsPermutation(emptyAffineMap) ||
      aiirAffineMapIsProjectedPermutation(affineMap) ||
      aiirAffineMapIsPermutation(affineMap) ||
      aiirAffineMapIsProjectedPermutation(constAffineMap) ||
      aiirAffineMapIsPermutation(constAffineMap) ||
      !aiirAffineMapIsProjectedPermutation(multiDimIdentityAffineMap) ||
      !aiirAffineMapIsPermutation(multiDimIdentityAffineMap) ||
      !aiirAffineMapIsProjectedPermutation(minorIdentityAffineMap) ||
      aiirAffineMapIsPermutation(minorIdentityAffineMap) ||
      !aiirAffineMapIsProjectedPermutation(permutationAffineMap) ||
      !aiirAffineMapIsPermutation(permutationAffineMap))
    return 10;

  intptr_t sub[] = {1};

  AiirAffineMap subMap = aiirAffineMapGetSubMap(
      multiDimIdentityAffineMap, sizeof(sub) / sizeof(intptr_t), sub);
  AiirAffineMap majorSubMap =
      aiirAffineMapGetMajorSubMap(multiDimIdentityAffineMap, 1);
  AiirAffineMap minorSubMap =
      aiirAffineMapGetMinorSubMap(multiDimIdentityAffineMap, 1);

  aiirAffineMapDump(subMap);
  aiirAffineMapDump(majorSubMap);
  aiirAffineMapDump(minorSubMap);
  // CHECK: (d0, d1, d2) -> (d1)
  // CHECK: (d0, d1, d2) -> (d0)
  // CHECK: (d0, d1, d2) -> (d2)

  // CHECK: distinct[0]<"foo">
  aiirAttributeDump(aiirDistinctAttrCreate(
      aiirStringAttrGet(ctx, aiirStringRefCreateFromCString("foo"))));

  return 0;
}

int printAffineExpr(AiirContext ctx) {
  AiirAffineExpr affineDimExpr = aiirAffineDimExprGet(ctx, 5);
  AiirAffineExpr affineSymbolExpr = aiirAffineSymbolExprGet(ctx, 5);
  AiirAffineExpr affineConstantExpr = aiirAffineConstantExprGet(ctx, 5);
  AiirAffineExpr affineAddExpr =
      aiirAffineAddExprGet(affineDimExpr, affineSymbolExpr);
  AiirAffineExpr affineMulExpr =
      aiirAffineMulExprGet(affineDimExpr, affineSymbolExpr);
  AiirAffineExpr affineModExpr =
      aiirAffineModExprGet(affineDimExpr, affineSymbolExpr);
  AiirAffineExpr affineFloorDivExpr =
      aiirAffineFloorDivExprGet(affineDimExpr, affineSymbolExpr);
  AiirAffineExpr affineCeilDivExpr =
      aiirAffineCeilDivExprGet(affineDimExpr, affineSymbolExpr);

  // Tests aiirAffineExprDump.
  fprintf(stderr, "@affineExpr\n");
  aiirAffineExprDump(affineDimExpr);
  aiirAffineExprDump(affineSymbolExpr);
  aiirAffineExprDump(affineConstantExpr);
  aiirAffineExprDump(affineAddExpr);
  aiirAffineExprDump(affineMulExpr);
  aiirAffineExprDump(affineModExpr);
  aiirAffineExprDump(affineFloorDivExpr);
  aiirAffineExprDump(affineCeilDivExpr);
  // CHECK-LABEL: @affineExpr
  // CHECK: d5
  // CHECK: s5
  // CHECK: 5
  // CHECK: d5 + s5
  // CHECK: d5 * s5
  // CHECK: d5 mod s5
  // CHECK: d5 floordiv s5
  // CHECK: d5 ceildiv s5

  // Tests methods of affine binary operation expression, takes add expression
  // as an example.
  aiirAffineExprDump(aiirAffineBinaryOpExprGetLHS(affineAddExpr));
  aiirAffineExprDump(aiirAffineBinaryOpExprGetRHS(affineAddExpr));
  // CHECK: d5
  // CHECK: s5

  // Tests methods of affine dimension expression.
  if (aiirAffineDimExprGetPosition(affineDimExpr) != 5)
    return 1;

  // Tests methods of affine symbol expression.
  if (aiirAffineSymbolExprGetPosition(affineSymbolExpr) != 5)
    return 2;

  // Tests methods of affine constant expression.
  if (aiirAffineConstantExprGetValue(affineConstantExpr) != 5)
    return 3;

  // Tests methods of affine expression.
  if (aiirAffineExprIsSymbolicOrConstant(affineDimExpr) ||
      !aiirAffineExprIsSymbolicOrConstant(affineSymbolExpr) ||
      !aiirAffineExprIsSymbolicOrConstant(affineConstantExpr) ||
      aiirAffineExprIsSymbolicOrConstant(affineAddExpr) ||
      aiirAffineExprIsSymbolicOrConstant(affineMulExpr) ||
      aiirAffineExprIsSymbolicOrConstant(affineModExpr) ||
      aiirAffineExprIsSymbolicOrConstant(affineFloorDivExpr) ||
      aiirAffineExprIsSymbolicOrConstant(affineCeilDivExpr))
    return 4;

  if (!aiirAffineExprIsPureAffine(affineDimExpr) ||
      !aiirAffineExprIsPureAffine(affineSymbolExpr) ||
      !aiirAffineExprIsPureAffine(affineConstantExpr) ||
      !aiirAffineExprIsPureAffine(affineAddExpr) ||
      aiirAffineExprIsPureAffine(affineMulExpr) ||
      aiirAffineExprIsPureAffine(affineModExpr) ||
      aiirAffineExprIsPureAffine(affineFloorDivExpr) ||
      aiirAffineExprIsPureAffine(affineCeilDivExpr))
    return 5;

  if (aiirAffineExprGetLargestKnownDivisor(affineDimExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineSymbolExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineConstantExpr) != 5 ||
      aiirAffineExprGetLargestKnownDivisor(affineAddExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineMulExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineModExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineFloorDivExpr) != 1 ||
      aiirAffineExprGetLargestKnownDivisor(affineCeilDivExpr) != 1)
    return 6;

  if (!aiirAffineExprIsMultipleOf(affineDimExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineSymbolExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineConstantExpr, 5) ||
      !aiirAffineExprIsMultipleOf(affineAddExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineMulExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineModExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineFloorDivExpr, 1) ||
      !aiirAffineExprIsMultipleOf(affineCeilDivExpr, 1))
    return 7;

  if (!aiirAffineExprIsFunctionOfDim(affineDimExpr, 5) ||
      aiirAffineExprIsFunctionOfDim(affineSymbolExpr, 5) ||
      aiirAffineExprIsFunctionOfDim(affineConstantExpr, 5) ||
      !aiirAffineExprIsFunctionOfDim(affineAddExpr, 5) ||
      !aiirAffineExprIsFunctionOfDim(affineMulExpr, 5) ||
      !aiirAffineExprIsFunctionOfDim(affineModExpr, 5) ||
      !aiirAffineExprIsFunctionOfDim(affineFloorDivExpr, 5) ||
      !aiirAffineExprIsFunctionOfDim(affineCeilDivExpr, 5))
    return 8;

  // Tests 'IsA' methods of affine binary operation expression.
  if (!aiirAffineExprIsAAdd(affineAddExpr))
    return 9;

  if (!aiirAffineExprIsAMul(affineMulExpr))
    return 10;

  if (!aiirAffineExprIsAMod(affineModExpr))
    return 11;

  if (!aiirAffineExprIsAFloorDiv(affineFloorDivExpr))
    return 12;

  if (!aiirAffineExprIsACeilDiv(affineCeilDivExpr))
    return 13;

  if (!aiirAffineExprIsABinary(affineAddExpr))
    return 14;

  // Test other 'IsA' method on affine expressions.
  if (!aiirAffineExprIsAConstant(affineConstantExpr))
    return 15;

  if (!aiirAffineExprIsADim(affineDimExpr))
    return 16;

  if (!aiirAffineExprIsASymbol(affineSymbolExpr))
    return 17;

  // Test equality and nullity.
  AiirAffineExpr otherDimExpr = aiirAffineDimExprGet(ctx, 5);
  if (!aiirAffineExprEqual(affineDimExpr, otherDimExpr))
    return 18;

  if (aiirAffineExprIsNull(affineDimExpr))
    return 19;

  return 0;
}

int affineMapFromExprs(AiirContext ctx) {
  AiirAffineExpr affineDimExpr = aiirAffineDimExprGet(ctx, 0);
  AiirAffineExpr affineSymbolExpr = aiirAffineSymbolExprGet(ctx, 1);
  AiirAffineExpr exprs[] = {affineDimExpr, affineSymbolExpr};
  AiirAffineMap map = aiirAffineMapGet(ctx, 3, 3, 2, exprs);

  // CHECK-LABEL: @affineMapFromExprs
  fprintf(stderr, "@affineMapFromExprs");
  // CHECK: (d0, d1, d2)[s0, s1, s2] -> (d0, s1)
  aiirAffineMapDump(map);

  if (aiirAffineMapGetNumResults(map) != 2)
    return 1;

  if (!aiirAffineExprEqual(aiirAffineMapGetResult(map, 0), affineDimExpr))
    return 2;

  if (!aiirAffineExprEqual(aiirAffineMapGetResult(map, 1), affineSymbolExpr))
    return 3;

  AiirAffineExpr affineDim2Expr = aiirAffineDimExprGet(ctx, 1);
  AiirAffineExpr composed = aiirAffineExprCompose(affineDim2Expr, map);
  // CHECK: s1
  aiirAffineExprDump(composed);
  if (!aiirAffineExprEqual(composed, affineSymbolExpr))
    return 4;

  return 0;
}

int printIntegerSet(AiirContext ctx) {
  AiirIntegerSet emptySet = aiirIntegerSetEmptyGet(ctx, 2, 1);

  // CHECK-LABEL: @printIntegerSet
  fprintf(stderr, "@printIntegerSet");

  // CHECK: (d0, d1)[s0] : (1 == 0)
  aiirIntegerSetDump(emptySet);

  if (!aiirIntegerSetIsCanonicalEmpty(emptySet))
    return 1;

  AiirIntegerSet anotherEmptySet = aiirIntegerSetEmptyGet(ctx, 2, 1);
  if (!aiirIntegerSetEqual(emptySet, anotherEmptySet))
    return 2;

  // Construct a set constrained by:
  //   d0 - s0 == 0,
  //   d1 - 42 >= 0.
  AiirAffineExpr negOne = aiirAffineConstantExprGet(ctx, -1);
  AiirAffineExpr negFortyTwo = aiirAffineConstantExprGet(ctx, -42);
  AiirAffineExpr d0 = aiirAffineDimExprGet(ctx, 0);
  AiirAffineExpr d1 = aiirAffineDimExprGet(ctx, 1);
  AiirAffineExpr s0 = aiirAffineSymbolExprGet(ctx, 0);
  AiirAffineExpr negS0 = aiirAffineMulExprGet(negOne, s0);
  AiirAffineExpr d0minusS0 = aiirAffineAddExprGet(d0, negS0);
  AiirAffineExpr d1minus42 = aiirAffineAddExprGet(d1, negFortyTwo);
  AiirAffineExpr constraints[] = {d0minusS0, d1minus42};
  bool flags[] = {true, false};

  AiirIntegerSet set = aiirIntegerSetGet(ctx, 2, 1, 2, constraints, flags);
  // CHECK: (d0, d1)[s0] : (
  // CHECK-DAG: d0 - s0 == 0
  // CHECK-DAG: d1 - 42 >= 0
  aiirIntegerSetDump(set);

  // Transform d1 into s0.
  AiirAffineExpr s1 = aiirAffineSymbolExprGet(ctx, 1);
  AiirAffineExpr repl[] = {d0, s1};
  AiirIntegerSet replaced = aiirIntegerSetReplaceGet(set, repl, &s0, 1, 2);
  // CHECK: (d0)[s0, s1] : (
  // CHECK-DAG: d0 - s0 == 0
  // CHECK-DAG: s1 - 42 >= 0
  aiirIntegerSetDump(replaced);

  if (aiirIntegerSetGetNumDims(set) != 2)
    return 3;
  if (aiirIntegerSetGetNumDims(replaced) != 1)
    return 4;

  if (aiirIntegerSetGetNumSymbols(set) != 1)
    return 5;
  if (aiirIntegerSetGetNumSymbols(replaced) != 2)
    return 6;

  if (aiirIntegerSetGetNumInputs(set) != 3)
    return 7;

  if (aiirIntegerSetGetNumConstraints(set) != 2)
    return 8;

  if (aiirIntegerSetGetNumEqualities(set) != 1)
    return 9;

  if (aiirIntegerSetGetNumInequalities(set) != 1)
    return 10;

  AiirAffineExpr cstr1 = aiirIntegerSetGetConstraint(set, 0);
  AiirAffineExpr cstr2 = aiirIntegerSetGetConstraint(set, 1);
  bool isEq1 = aiirIntegerSetIsConstraintEq(set, 0);
  bool isEq2 = aiirIntegerSetIsConstraintEq(set, 1);
  if (!aiirAffineExprEqual(cstr1, isEq1 ? d0minusS0 : d1minus42))
    return 11;
  if (!aiirAffineExprEqual(cstr2, isEq2 ? d0minusS0 : d1minus42))
    return 12;

  return 0;
}

int registerOnlyStd(void) {
  AiirContext ctx = aiirContextCreate();
  // The built-in dialect is always loaded.
  if (aiirContextGetNumLoadedDialects(ctx) != 1)
    return 1;

  AiirDialectHandle stdHandle = aiirGetDialectHandle__func__();

  AiirDialect std = aiirContextGetOrLoadDialect(
      ctx, aiirDialectHandleGetNamespace(stdHandle));
  if (!aiirDialectIsNull(std))
    return 2;

  aiirDialectHandleRegisterDialect(stdHandle, ctx);

  std = aiirContextGetOrLoadDialect(ctx,
                                    aiirDialectHandleGetNamespace(stdHandle));
  if (aiirDialectIsNull(std))
    return 3;

  AiirDialect alsoStd = aiirDialectHandleLoadDialect(stdHandle, ctx);
  if (!aiirDialectEqual(std, alsoStd))
    return 4;

  AiirStringRef stdNs = aiirDialectGetNamespace(std);
  AiirStringRef alsoStdNs = aiirDialectHandleGetNamespace(stdHandle);
  if (stdNs.length != alsoStdNs.length ||
      strncmp(stdNs.data, alsoStdNs.data, stdNs.length))
    return 5;

  fprintf(stderr, "@registration\n");
  // CHECK-LABEL: @registration

  // CHECK: func.call is_registered: 1
  fprintf(stderr, "func.call is_registered: %d\n",
          aiirContextIsRegisteredOperation(
              ctx, aiirStringRefCreateFromCString("func.call")));

  // CHECK: func.not_existing_op is_registered: 0
  fprintf(stderr, "func.not_existing_op is_registered: %d\n",
          aiirContextIsRegisteredOperation(
              ctx, aiirStringRefCreateFromCString("func.not_existing_op")));

  // CHECK: not_existing_dialect.not_existing_op is_registered: 0
  fprintf(stderr, "not_existing_dialect.not_existing_op is_registered: %d\n",
          aiirContextIsRegisteredOperation(
              ctx, aiirStringRefCreateFromCString(
                       "not_existing_dialect.not_existing_op")));

  aiirContextDestroy(ctx);
  return 0;
}

/// Tests backreference APIs
static int testBackreferences(void) {
  fprintf(stderr, "@test_backreferences\n");

  AiirContext ctx = aiirContextCreate();
  aiirContextSetAllowUnregisteredDialects(ctx, true);
  AiirLocation loc = aiirLocationUnknownGet(ctx);

  AiirOperationState opState =
      aiirOperationStateGet(aiirStringRefCreateFromCString("invalid.op"), loc);
  AiirRegion region = aiirRegionCreate();
  AiirBlock block = aiirBlockCreate(0, NULL, NULL);
  aiirRegionAppendOwnedBlock(region, block);
  aiirOperationStateAddOwnedRegions(&opState, 1, &region);
  AiirOperation op = aiirOperationCreate(&opState);
  AiirIdentifier ident =
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("identifier"));

  if (!aiirContextEqual(ctx, aiirOperationGetContext(op))) {
    fprintf(stderr, "ERROR: Getting context from operation failed\n");
    return 1;
  }
  if (!aiirOperationEqual(op, aiirBlockGetParentOperation(block))) {
    fprintf(stderr, "ERROR: Getting parent operation from block failed\n");
    return 2;
  }
  if (!aiirContextEqual(ctx, aiirIdentifierGetContext(ident))) {
    fprintf(stderr, "ERROR: Getting context from identifier failed\n");
    return 3;
  }

  aiirOperationDestroy(op);
  aiirContextDestroy(ctx);

  // CHECK-LABEL: @test_backreferences
  return 0;
}

/// Tests operand APIs.
int testOperands(void) {
  fprintf(stderr, "@testOperands\n");
  // CHECK-LABEL: @testOperands

  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("arith"));
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("test"));
  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirType indexType = aiirIndexTypeGet(ctx);

  // Create some constants to use as operands.
  AiirAttribute indexZeroLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("0 : index"));
  AiirNamedAttribute indexZeroValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")),
      indexZeroLiteral);
  AiirOperationState constZeroState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), loc);
  aiirOperationStateAddResults(&constZeroState, 1, &indexType);
  aiirOperationStateAddAttributes(&constZeroState, 1, &indexZeroValueAttr);
  AiirOperation constZero = aiirOperationCreate(&constZeroState);
  AiirValue constZeroValue = aiirOperationGetResult(constZero, 0);

  AiirAttribute indexOneLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("1 : index"));
  AiirNamedAttribute indexOneValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")),
      indexOneLiteral);
  AiirOperationState constOneState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), loc);
  aiirOperationStateAddResults(&constOneState, 1, &indexType);
  aiirOperationStateAddAttributes(&constOneState, 1, &indexOneValueAttr);
  AiirOperation constOne = aiirOperationCreate(&constOneState);
  AiirValue constOneValue = aiirOperationGetResult(constOne, 0);

  // Create the operation under test.
  aiirContextSetAllowUnregisteredDialects(ctx, true);
  AiirOperationState opState =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op"), loc);
  AiirValue initialOperands[] = {constZeroValue};
  aiirOperationStateAddOperands(&opState, 1, initialOperands);
  AiirOperation op = aiirOperationCreate(&opState);

  // Test operand APIs.
  intptr_t numOperands = aiirOperationGetNumOperands(op);
  fprintf(stderr, "Num Operands: %" PRIdPTR "\n", numOperands);
  // CHECK: Num Operands: 1

  AiirValue opOperand1 = aiirOperationGetOperand(op, 0);
  fprintf(stderr, "Original operand: ");
  aiirValuePrint(opOperand1, printToStderr, NULL);
  // CHECK: Original operand: {{.+}} arith.constant 0 : index

  aiirOperationSetOperand(op, 0, constOneValue);
  AiirValue opOperand2 = aiirOperationGetOperand(op, 0);
  fprintf(stderr, "Updated operand: ");
  aiirValuePrint(opOperand2, printToStderr, NULL);
  // CHECK: Updated operand: {{.+}} arith.constant 1 : index

  // Test op operand APIs.
  AiirOpOperand use1 = aiirValueGetFirstUse(opOperand1);
  if (!aiirOpOperandIsNull(use1)) {
    fprintf(stderr, "ERROR: Use should be null\n");
    return 1;
  }

  AiirOpOperand use2 = aiirValueGetFirstUse(opOperand2);
  if (aiirOpOperandIsNull(use2)) {
    fprintf(stderr, "ERROR: Use should not be null\n");
    return 2;
  }

  fprintf(stderr, "Use owner: ");
  aiirOperationPrint(aiirOpOperandGetOwner(use2), printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Use owner: "dummy.op"

  fprintf(stderr, "Use operandNumber: %d\n",
          aiirOpOperandGetOperandNumber(use2));
  // CHECK: Use operandNumber: 0

  use2 = aiirOpOperandGetNextUse(use2);
  if (!aiirOpOperandIsNull(use2)) {
    fprintf(stderr, "ERROR: Next use should be null\n");
    return 3;
  }

  AiirOperationState op2State =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op2"), loc);
  AiirValue initialOperands2[] = {constOneValue};
  aiirOperationStateAddOperands(&op2State, 1, initialOperands2);
  AiirOperation op2 = aiirOperationCreate(&op2State);

  AiirOpOperand use3 = aiirValueGetFirstUse(constOneValue);
  fprintf(stderr, "First use owner: ");
  aiirOperationPrint(aiirOpOperandGetOwner(use3), printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: First use owner: "dummy.op2"

  use3 = aiirOpOperandGetNextUse(aiirValueGetFirstUse(constOneValue));
  fprintf(stderr, "Second use owner: ");
  aiirOperationPrint(aiirOpOperandGetOwner(use3), printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Second use owner: "dummy.op"

  AiirAttribute indexTwoLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("2 : index"));
  AiirNamedAttribute indexTwoValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, aiirStringRefCreateFromCString("value")),
      indexTwoLiteral);
  AiirOperationState constTwoState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), loc);
  aiirOperationStateAddResults(&constTwoState, 1, &indexType);
  aiirOperationStateAddAttributes(&constTwoState, 1, &indexTwoValueAttr);
  AiirOperation constTwo = aiirOperationCreate(&constTwoState);
  AiirValue constTwoValue = aiirOperationGetResult(constTwo, 0);

  aiirValueReplaceAllUsesOfWith(constOneValue, constTwoValue);

  use3 = aiirValueGetFirstUse(constOneValue);
  if (!aiirOpOperandIsNull(use3)) {
    fprintf(stderr, "ERROR: Use should be null\n");
    return 4;
  }

  AiirOpOperand use4 = aiirValueGetFirstUse(constTwoValue);
  fprintf(stderr, "First replacement use owner: ");
  aiirOperationPrint(aiirOpOperandGetOwner(use4), printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: First replacement use owner: "dummy.op"

  use4 = aiirOpOperandGetNextUse(aiirValueGetFirstUse(constTwoValue));
  fprintf(stderr, "Second replacement use owner: ");
  aiirOperationPrint(aiirOpOperandGetOwner(use4), printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Second replacement use owner: "dummy.op2"

  AiirOpOperand use5 = aiirValueGetFirstUse(constTwoValue);
  AiirOpOperand use6 = aiirOpOperandGetNextUse(use5);
  if (!aiirValueEqual(aiirOpOperandGetValue(use5),
                      aiirOpOperandGetValue(use6))) {
    fprintf(stderr,
            "ERROR: First and second operand should share the same value\n");
    return 5;
  }

  aiirOperationDestroy(op);
  aiirOperationDestroy(op2);
  aiirOperationDestroy(constZero);
  aiirOperationDestroy(constOne);
  aiirOperationDestroy(constTwo);
  aiirContextDestroy(ctx);

  return 0;
}

/// Tests clone APIs.
int testClone(void) {
  fprintf(stderr, "@testClone\n");
  // CHECK-LABEL: @testClone

  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);

  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("func"));
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("arith"));
  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirType indexType = aiirIndexTypeGet(ctx);
  AiirStringRef valueStringRef = aiirStringRefCreateFromCString("value");

  AiirAttribute indexZeroLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("0 : index"));
  AiirNamedAttribute indexZeroValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, valueStringRef), indexZeroLiteral);
  AiirOperationState constZeroState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), loc);
  aiirOperationStateAddResults(&constZeroState, 1, &indexType);
  aiirOperationStateAddAttributes(&constZeroState, 1, &indexZeroValueAttr);
  AiirOperation constZero = aiirOperationCreate(&constZeroState);

  AiirAttribute indexOneLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("1 : index"));
  AiirOperation constOne = aiirOperationClone(constZero);
  aiirOperationSetAttributeByName(constOne, valueStringRef, indexOneLiteral);

  aiirOperationPrint(constZero, printToStderr, NULL);
  aiirOperationPrint(constOne, printToStderr, NULL);
  // CHECK: arith.constant 0 : index
  // CHECK: arith.constant 1 : index

  aiirOperationDestroy(constZero);
  aiirOperationDestroy(constOne);
  aiirContextDestroy(ctx);
  return 0;
}

// Wraps a diagnostic into additional text we can match against.
AiirLogicalResult errorHandler(AiirDiagnostic diagnostic, void *userData) {
  fprintf(stderr, "processing diagnostic (userData: %" PRIdPTR ") <<\n",
          (intptr_t)userData);
  aiirDiagnosticPrint(diagnostic, printToStderr, NULL);
  fprintf(stderr, "\n");
  AiirLocation loc = aiirDiagnosticGetLocation(diagnostic);
  aiirLocationPrint(loc, printToStderr, NULL);
  assert(aiirDiagnosticGetNumNotes(diagnostic) == 0);
  fprintf(stderr, "\n>> end of diagnostic (userData: %" PRIdPTR ")\n",
          (intptr_t)userData);
  return aiirLogicalResultSuccess();
}

// Logs when the delete user data callback is called
static void deleteUserData(void *userData) {
  fprintf(stderr, "deleting user data (userData: %" PRIdPTR ")\n",
          (intptr_t)userData);
}

int testTypeID(AiirContext ctx) {
  fprintf(stderr, "@testTypeID\n");

  // Test getting and comparing type and attribute type ids.
  AiirType i32 = aiirIntegerTypeGet(ctx, 32);
  AiirTypeID i32ID = aiirTypeGetTypeID(i32);
  AiirType ui32 = aiirIntegerTypeUnsignedGet(ctx, 32);
  AiirTypeID ui32ID = aiirTypeGetTypeID(ui32);
  AiirType f32 = aiirF32TypeGet(ctx);
  AiirTypeID f32ID = aiirTypeGetTypeID(f32);
  AiirAttribute i32Attr = aiirIntegerAttrGet(i32, 1);
  AiirTypeID i32AttrID = aiirAttributeGetTypeID(i32Attr);

  if (aiirTypeIDIsNull(i32ID) || aiirTypeIDIsNull(ui32ID) ||
      aiirTypeIDIsNull(f32ID) || aiirTypeIDIsNull(i32AttrID)) {
    fprintf(stderr, "ERROR: Expected type ids to be present\n");
    return 1;
  }

  if (!aiirTypeIDEqual(i32ID, ui32ID) ||
      aiirTypeIDHashValue(i32ID) != aiirTypeIDHashValue(ui32ID)) {
    fprintf(
        stderr,
        "ERROR: Expected different integer types to have the same type id\n");
    return 2;
  }

  if (aiirTypeIDEqual(i32ID, f32ID)) {
    fprintf(stderr,
            "ERROR: Expected integer type id to not equal float type id\n");
    return 3;
  }

  if (aiirTypeIDEqual(i32ID, i32AttrID)) {
    fprintf(stderr, "ERROR: Expected integer type id to not equal integer "
                    "attribute type id\n");
    return 4;
  }

  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirType indexType = aiirIndexTypeGet(ctx);
  AiirStringRef valueStringRef = aiirStringRefCreateFromCString("value");

  // Create a registered operation, which should have a type id.
  AiirAttribute indexZeroLiteral =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString("0 : index"));
  AiirNamedAttribute indexZeroValueAttr = aiirNamedAttributeGet(
      aiirIdentifierGet(ctx, valueStringRef), indexZeroLiteral);
  AiirOperationState constZeroState = aiirOperationStateGet(
      aiirStringRefCreateFromCString("arith.constant"), loc);
  aiirOperationStateAddResults(&constZeroState, 1, &indexType);
  aiirOperationStateAddAttributes(&constZeroState, 1, &indexZeroValueAttr);
  AiirOperation constZero = aiirOperationCreate(&constZeroState);

  if (!aiirOperationVerify(constZero)) {
    fprintf(stderr, "ERROR: Expected operation to verify correctly\n");
    return 5;
  }

  if (aiirOperationIsNull(constZero)) {
    fprintf(stderr, "ERROR: Expected registered operation to be present\n");
    return 6;
  }

  AiirTypeID registeredOpID = aiirOperationGetTypeID(constZero);

  if (aiirTypeIDIsNull(registeredOpID)) {
    fprintf(stderr,
            "ERROR: Expected registered operation type id to be present\n");
    return 7;
  }

  // Create an unregistered operation, which should not have a type id.
  aiirContextSetAllowUnregisteredDialects(ctx, true);
  AiirOperationState opState =
      aiirOperationStateGet(aiirStringRefCreateFromCString("dummy.op"), loc);
  AiirOperation unregisteredOp = aiirOperationCreate(&opState);
  if (aiirOperationIsNull(unregisteredOp)) {
    fprintf(stderr, "ERROR: Expected unregistered operation to be present\n");
    return 8;
  }

  AiirTypeID unregisteredOpID = aiirOperationGetTypeID(unregisteredOp);

  if (!aiirTypeIDIsNull(unregisteredOpID)) {
    fprintf(stderr,
            "ERROR: Expected unregistered operation type id to be null\n");
    return 9;
  }

  aiirOperationDestroy(constZero);
  aiirOperationDestroy(unregisteredOp);

  return 0;
}

int testSymbolTable(AiirContext ctx) {
  fprintf(stderr, "@testSymbolTable\n");

  const char *moduleString = "func.func private @foo()"
                             "func.func private @bar()";
  const char *otherModuleString = "func.func private @qux()"
                                  "func.func private @foo()";

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirModule otherModule = aiirModuleCreateParse(
      ctx, aiirStringRefCreateFromCString(otherModuleString));

  AiirSymbolTable symbolTable =
      aiirSymbolTableCreate(aiirModuleGetOperation(module));

  AiirOperation funcFoo =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("foo"));
  if (aiirOperationIsNull(funcFoo))
    return 1;

  AiirOperation funcBar =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("bar"));
  if (aiirOperationEqual(funcFoo, funcBar))
    return 2;

  AiirOperation missing =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("qux"));
  if (!aiirOperationIsNull(missing))
    return 3;

  AiirBlock moduleBody = aiirModuleGetBody(module);
  AiirBlock otherModuleBody = aiirModuleGetBody(otherModule);
  AiirOperation operation = aiirBlockGetFirstOperation(otherModuleBody);
  aiirOperationRemoveFromParent(operation);
  aiirBlockAppendOwnedOperation(moduleBody, operation);

  // At this moment, the operation is still missing from the symbol table.
  AiirOperation stillMissing =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("qux"));
  if (!aiirOperationIsNull(stillMissing))
    return 4;

  // After it is added to the symbol table, and not only the operation with
  // which the table is associated, it can be looked up.
  aiirSymbolTableInsert(symbolTable, operation);
  AiirOperation funcQux =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("qux"));
  if (!aiirOperationEqual(operation, funcQux))
    return 5;

  // Erasing from the symbol table also removes the operation.
  aiirSymbolTableErase(symbolTable, funcBar);
  AiirOperation nowMissing =
      aiirSymbolTableLookup(symbolTable, aiirStringRefCreateFromCString("bar"));
  if (!aiirOperationIsNull(nowMissing))
    return 6;

  // Adding a symbol with the same name to the table should rename.
  AiirOperation duplicateNameOp = aiirBlockGetFirstOperation(otherModuleBody);
  aiirOperationRemoveFromParent(duplicateNameOp);
  aiirBlockAppendOwnedOperation(moduleBody, duplicateNameOp);
  AiirAttribute newName = aiirSymbolTableInsert(symbolTable, duplicateNameOp);
  AiirStringRef newNameStr = aiirStringAttrGetValue(newName);
  if (aiirStringRefEqual(newNameStr, aiirStringRefCreateFromCString("foo")))
    return 7;
  AiirAttribute updatedName = aiirOperationGetAttributeByName(
      duplicateNameOp, aiirSymbolTableGetSymbolAttributeName());
  if (!aiirAttributeEqual(updatedName, newName))
    return 8;

  aiirOperationDump(aiirModuleGetOperation(module));
  aiirOperationDump(aiirModuleGetOperation(otherModule));
  // clang-format off
  // CHECK-LABEL: @testSymbolTable
  // CHECK: module
  // CHECK:   func private @foo
  // CHECK:   func private @qux
  // CHECK:   func private @foo{{.+}}
  // CHECK: module
  // CHECK-NOT: @qux
  // CHECK-NOT: @foo
  // clang-format on

  aiirSymbolTableDestroy(symbolTable);
  aiirModuleDestroy(module);
  aiirModuleDestroy(otherModule);

  return 0;
}

typedef struct {
  const char *x;
} callBackData;

AiirWalkResult walkCallBack(AiirOperation op, void *rootOpVoid) {
  fprintf(stderr, "%s: %s\n", ((callBackData *)(rootOpVoid))->x,
          aiirIdentifierStr(aiirOperationGetName(op)).data);
  return AiirWalkResultAdvance;
}

AiirWalkResult walkCallBackTestWalkResult(AiirOperation op, void *rootOpVoid) {
  fprintf(stderr, "%s: %s\n", ((callBackData *)(rootOpVoid))->x,
          aiirIdentifierStr(aiirOperationGetName(op)).data);
  if (strcmp(aiirIdentifierStr(aiirOperationGetName(op)).data, "func.func") ==
      0)
    return AiirWalkResultSkip;
  if (strcmp(aiirIdentifierStr(aiirOperationGetName(op)).data, "arith.addi") ==
      0)
    return AiirWalkResultInterrupt;
  return AiirWalkResultAdvance;
}

int testOperationWalk(AiirContext ctx) {
  // CHECK-LABEL: @testOperationWalk
  fprintf(stderr, "@testOperationWalk\n");

  const char *moduleString = "module {\n"
                             "  func.func @foo() {\n"
                             "    %1 = arith.constant 10: i32\n"
                             "    arith.addi %1, %1: i32\n"
                             "    return\n"
                             "  }\n"
                             "  func.func @bar() {\n"
                             "    return\n"
                             "  }\n"
                             "}";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));

  callBackData data;
  data.x = "i love you";

  // CHECK-NEXT: i love you: arith.constant
  // CHECK-NEXT: i love you: arith.addi
  // CHECK-NEXT: i love you: func.return
  // CHECK-NEXT: i love you: func.func
  // CHECK-NEXT: i love you: func.return
  // CHECK-NEXT: i love you: func.func
  // CHECK-NEXT: i love you: builtin.module
  aiirOperationWalk(aiirModuleGetOperation(module), walkCallBack,
                    (void *)(&data), AiirWalkPostOrder);

  data.x = "i don't love you";
  // CHECK-NEXT: i don't love you: builtin.module
  // CHECK-NEXT: i don't love you: func.func
  // CHECK-NEXT: i don't love you: arith.constant
  // CHECK-NEXT: i don't love you: arith.addi
  // CHECK-NEXT: i don't love you: func.return
  // CHECK-NEXT: i don't love you: func.func
  // CHECK-NEXT: i don't love you: func.return
  aiirOperationWalk(aiirModuleGetOperation(module), walkCallBack,
                    (void *)(&data), AiirWalkPreOrder);

  data.x = "interrupt";
  // Interrupted at `arith.addi`
  // CHECK-NEXT: interrupt: arith.constant
  // CHECK-NEXT: interrupt: arith.addi
  aiirOperationWalk(aiirModuleGetOperation(module), walkCallBackTestWalkResult,
                    (void *)(&data), AiirWalkPostOrder);

  data.x = "skip";
  // Skip at `func.func`
  // CHECK-NEXT: skip: builtin.module
  // CHECK-NEXT: skip: func.func
  // CHECK-NEXT: skip: func.func
  aiirOperationWalk(aiirModuleGetOperation(module), walkCallBackTestWalkResult,
                    (void *)(&data), AiirWalkPreOrder);

  aiirModuleDestroy(module);
  return 0;
}

int testDialectRegistry(void) {
  fprintf(stderr, "@testDialectRegistry\n");

  AiirDialectRegistry registry = aiirDialectRegistryCreate();
  if (aiirDialectRegistryIsNull(registry)) {
    fprintf(stderr, "ERROR: Expected registry to be present\n");
    return 1;
  }

  AiirDialectHandle stdHandle = aiirGetDialectHandle__func__();
  aiirDialectHandleInsertDialect(stdHandle, registry);

  AiirContext ctx = aiirContextCreate();
  if (aiirContextGetNumRegisteredDialects(ctx) != 0) {
    fprintf(stderr,
            "ERROR: Expected no dialects to be registered to new context\n");
  }

  aiirContextAppendDialectRegistry(ctx, registry);
  if (aiirContextGetNumRegisteredDialects(ctx) != 1) {
    fprintf(stderr, "ERROR: Expected the dialect in the registry to be "
                    "registered to the context\n");
  }

  aiirContextDestroy(ctx);
  aiirDialectRegistryDestroy(registry);

  return 0;
}

void testExplicitThreadPools(void) {
  AiirLlvmThreadPool threadPool = aiirLlvmThreadPoolCreate();
  AiirDialectRegistry registry = aiirDialectRegistryCreate();
  aiirRegisterAllDialects(registry);
  AiirContext context =
      aiirContextCreateWithRegistry(registry, /*threadingEnabled=*/false);
  aiirContextSetThreadPool(context, threadPool);
  aiirContextDestroy(context);
  aiirDialectRegistryDestroy(registry);
  aiirLlvmThreadPoolDestroy(threadPool);
}

void testDiagnostics(void) {
  AiirContext ctx = aiirContextCreate();
  AiirDiagnosticHandlerID id = aiirContextAttachDiagnosticHandler(
      ctx, errorHandler, (void *)42, deleteUserData);
  fprintf(stderr, "@test_diagnostics\n");
  AiirLocation unknownLoc = aiirLocationUnknownGet(ctx);
  aiirEmitError(unknownLoc, "test diagnostics");
  AiirAttribute unknownAttr = aiirLocationGetAttribute(unknownLoc);
  AiirLocation unknownClone = aiirLocationFromAttribute(unknownAttr);
  aiirEmitError(unknownClone, "test clone");
  AiirLocation fileLineColLoc = aiirLocationFileLineColGet(
      ctx, aiirStringRefCreateFromCString("file.c"), 1, 2);
  aiirEmitError(fileLineColLoc, "test diagnostics");
  AiirLocation fileLineColRange = aiirLocationFileLineColRangeGet(
      ctx, aiirStringRefCreateFromCString("other-file.c"), 1, 2, 3, 4);
  aiirEmitError(fileLineColRange, "test diagnostics");
  AiirLocation callSiteLoc = aiirLocationCallSiteGet(
      aiirLocationFileLineColGet(
          ctx, aiirStringRefCreateFromCString("other-file.c"), 2, 3),
      fileLineColLoc);
  aiirEmitError(callSiteLoc, "test diagnostics");
  AiirLocation null = {0};
  AiirLocation nameLoc =
      aiirLocationNameGet(ctx, aiirStringRefCreateFromCString("named"), null);
  aiirEmitError(nameLoc, "test diagnostics");
  AiirLocation locs[2] = {nameLoc, callSiteLoc};
  AiirAttribute nullAttr = {0};
  AiirLocation fusedLoc = aiirLocationFusedGet(ctx, 2, locs, nullAttr);
  aiirEmitError(fusedLoc, "test diagnostics");
  aiirContextDetachDiagnosticHandler(ctx, id);
  aiirEmitError(unknownLoc, "more test diagnostics");
  // CHECK-LABEL: @test_diagnostics
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc(unknown)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test clone
  // CHECK:   loc(unknown)
  // CHECK: >> end of diagnostic (userData: 42)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc("file.c":1:2)
  // CHECK: >> end of diagnostic (userData: 42)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc("other-file.c":1:2 to 3:4)
  // CHECK: >> end of diagnostic (userData: 42)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc(callsite("other-file.c":2:3 at "file.c":1:2))
  // CHECK: >> end of diagnostic (userData: 42)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc("named")
  // CHECK: >> end of diagnostic (userData: 42)
  // CHECK: processing diagnostic (userData: 42) <<
  // CHECK:   test diagnostics
  // CHECK:   loc(fused["named", callsite("other-file.c":2:3 at "file.c":1:2)])
  // CHECK: deleting user data (userData: 42)
  // CHECK-NOT: processing diagnostic
  // CHECK:     more test diagnostics
  aiirContextDestroy(ctx);
}

int testBlockPredecessorsSuccessors(AiirContext ctx) {
  // CHECK-LABEL: @testBlockPredecessorsSuccessors
  fprintf(stderr, "@testBlockPredecessorsSuccessors\n");

  const char *moduleString = "module {\n"
                             "  func.func @test(%arg0: i32, %arg1: i16) {\n"
                             "    cf.br ^bb1(%arg1 : i16)\n"
                             "  ^bb1(%0: i16):  // pred: ^bb0\n"
                             "    cf.br ^bb2(%arg0 : i32)\n"
                             "  ^bb2(%1: i32):  // pred: ^bb1\n"
                             "    return\n"
                             "  }\n"
                             "}\n";

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));

  AiirOperation moduleOp = aiirModuleGetOperation(module);
  AiirRegion moduleRegion = aiirOperationGetRegion(moduleOp, 0);
  AiirBlock moduleBlock = aiirRegionGetFirstBlock(moduleRegion);
  AiirOperation function = aiirBlockGetFirstOperation(moduleBlock);
  AiirRegion funcRegion = aiirOperationGetRegion(function, 0);
  AiirBlock entryBlock = aiirRegionGetFirstBlock(funcRegion);
  AiirBlock middleBlock = aiirBlockGetNextInRegion(entryBlock);
  AiirBlock successorBlock = aiirBlockGetNextInRegion(middleBlock);

#define FPRINTF_OP(OP, FMT) fprintf(stderr, #OP ": " FMT "\n", OP)

  // CHECK: aiirBlockGetNumPredecessors(entryBlock): 0
  FPRINTF_OP(aiirBlockGetNumPredecessors(entryBlock), "%ld");

  // CHECK: aiirBlockGetNumSuccessors(entryBlock): 1
  FPRINTF_OP(aiirBlockGetNumSuccessors(entryBlock), "%ld");
  // CHECK: aiirBlockEqual(middleBlock, aiirBlockGetSuccessor(entryBlock, 0)): 1
  FPRINTF_OP(aiirBlockEqual(middleBlock, aiirBlockGetSuccessor(entryBlock, 0)),
             "%d");
  // CHECK: aiirBlockGetNumPredecessors(middleBlock): 1
  FPRINTF_OP(aiirBlockGetNumPredecessors(middleBlock), "%ld");
  // CHECK: aiirBlockEqual(entryBlock, aiirBlockGetPredecessor(middleBlock, 0))
  FPRINTF_OP(
      aiirBlockEqual(entryBlock, aiirBlockGetPredecessor(middleBlock, 0)),
      "%d");

  // CHECK: aiirBlockGetNumSuccessors(middleBlock): 1
  FPRINTF_OP(aiirBlockGetNumSuccessors(middleBlock), "%ld");
  // CHECK: BlockEqual(successorBlock, aiirBlockGetSuccessor(middleBlock, 0)): 1
  fprintf(
      stderr,
      "BlockEqual(successorBlock, aiirBlockGetSuccessor(middleBlock, 0)): %d\n",
      aiirBlockEqual(successorBlock, aiirBlockGetSuccessor(middleBlock, 0)));
  // CHECK: aiirBlockGetNumPredecessors(successorBlock): 1
  FPRINTF_OP(aiirBlockGetNumPredecessors(successorBlock), "%ld");
  // CHECK: Equal(middleBlock, aiirBlockGetPredecessor(successorBlock, 0)): 1
  fprintf(
      stderr,
      "Equal(middleBlock, aiirBlockGetPredecessor(successorBlock, 0)): %d\n",
      aiirBlockEqual(middleBlock, aiirBlockGetPredecessor(successorBlock, 0)));

  // CHECK: aiirBlockGetNumSuccessors(successorBlock): 0
  FPRINTF_OP(aiirBlockGetNumSuccessors(successorBlock), "%ld");

#undef FPRINTF_OP

  aiirModuleDestroy(module);

  return 0;
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  registerAllUpstreamDialects(ctx);
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("func"));
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("memref"));
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("shape"));
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("scf"));

  if (constructAndTraverseIr(ctx))
    return 1;
  buildWithInsertionsAndPrint(ctx);
  if (createOperationWithTypeInference(ctx))
    return 2;

  if (printBuiltinTypes(ctx))
    return 3;
  if (printBuiltinAttributes(ctx))
    return 4;
  if (printAffineMap(ctx))
    return 5;
  if (printAffineExpr(ctx))
    return 6;
  if (affineMapFromExprs(ctx))
    return 7;
  if (printIntegerSet(ctx))
    return 8;
  if (registerOnlyStd())
    return 9;
  if (testBackreferences())
    return 10;
  if (testOperands())
    return 11;
  if (testClone())
    return 12;
  if (testTypeID(ctx))
    return 13;
  if (testSymbolTable(ctx))
    return 14;
  if (testDialectRegistry())
    return 15;
  if (testOperationWalk(ctx))
    return 16;

  testExplicitThreadPools();
  testDiagnostics();

  if (testBlockPredecessorsSuccessors(ctx))
    return 17;

  // CHECK: DESTROY MAIN CONTEXT
  // CHECK: reportResourceDelete: resource_i64_blob
  fprintf(stderr, "DESTROY MAIN CONTEXT\n");
  aiirContextDestroy(ctx);

  return 0;
}

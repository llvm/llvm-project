//===- rewrite.c - Test of the rewriting C API ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-rewrite-test 2>&1 | FileCheck %s

#include "aiir-c/Rewrite.h"
#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

AiirOperation createOperationWithName(AiirContext ctx, const char *name) {
  AiirStringRef nameRef = aiirStringRefCreateFromCString(name);
  AiirLocation loc = aiirLocationUnknownGet(ctx);
  AiirOperationState state = aiirOperationStateGet(nameRef, loc);
  AiirType indexType = aiirIndexTypeGet(ctx);
  aiirOperationStateAddResults(&state, 1, &indexType);
  return aiirOperationCreate(&state);
}

void testInsertionPoint(AiirContext ctx) {
  // CHECK-LABEL: @testInsertionPoint
  fprintf(stderr, "@testInsertionPoint\n");

  const char *moduleString = "\"dialect.op1\"() : () -> ()\n";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);
  AiirOperation op1 = aiirBlockGetFirstOperation(body);

  // IRRewriter create
  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);

  // Insert before op
  aiirRewriterBaseSetInsertionPointBefore(rewriter, op1);
  AiirOperation op2 = createOperationWithName(ctx, "dialect.op2");
  aiirRewriterBaseInsert(rewriter, op2);

  // Insert after op
  aiirRewriterBaseSetInsertionPointAfter(rewriter, op2);
  AiirOperation op3 = createOperationWithName(ctx, "dialect.op3");
  aiirRewriterBaseInsert(rewriter, op3);
  AiirValue op3Res = aiirOperationGetResult(op3, 0);

  // Insert after value
  aiirRewriterBaseSetInsertionPointAfterValue(rewriter, op3Res);
  AiirOperation op4 = createOperationWithName(ctx, "dialect.op4");
  aiirRewriterBaseInsert(rewriter, op4);

  // Insert at beginning of block
  aiirRewriterBaseSetInsertionPointToStart(rewriter, body);
  AiirOperation op5 = createOperationWithName(ctx, "dialect.op5");
  aiirRewriterBaseInsert(rewriter, op5);

  // Insert at end of block
  aiirRewriterBaseSetInsertionPointToEnd(rewriter, body);
  AiirOperation op6 = createOperationWithName(ctx, "dialect.op6");
  aiirRewriterBaseInsert(rewriter, op6);

  // Get insertion blocks
  AiirBlock block1 = aiirRewriterBaseGetBlock(rewriter);
  AiirBlock block2 = aiirRewriterBaseGetInsertionBlock(rewriter);
  (void)block1;
  (void)block2;
  assert(body.ptr == block1.ptr);
  assert(body.ptr == block2.ptr);

  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %{{.*}} = "dialect.op5"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op2"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op3"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op4"() : () -> index
  // CHECK-NEXT:   "dialect.op1"() : () -> ()
  // CHECK-NEXT:   %{{.*}} = "dialect.op6"() : () -> index
  // CHECK-NEXT: }
  // clang-format on
  aiirOperationDump(op);

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testCreateBlock(AiirContext ctx) {
  // CHECK-LABEL: @testCreateBlock
  fprintf(stderr, "@testCreateBlock\n");

  const char *moduleString = "\"dialect.op1\"() ({^bb0:}) : () -> ()\n"
                             "\"dialect.op2\"() ({^bb0:}) : () -> ()\n";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  AiirOperation op1 = aiirBlockGetFirstOperation(body);
  AiirRegion region1 = aiirOperationGetRegion(op1, 0);
  AiirBlock block1 = aiirRegionGetFirstBlock(region1);

  AiirOperation op2 = aiirOperationGetNextInBlock(op1);
  AiirRegion region2 = aiirOperationGetRegion(op2, 0);
  AiirBlock block2 = aiirRegionGetFirstBlock(region2);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);

  // Create block before
  AiirType indexType = aiirIndexTypeGet(ctx);
  AiirLocation unknown = aiirLocationUnknownGet(ctx);
  aiirRewriterBaseCreateBlockBefore(rewriter, block1, 1, &indexType, &unknown);

  aiirRewriterBaseSetInsertionPointToEnd(rewriter, body);

  // Clone operation
  aiirRewriterBaseClone(rewriter, op1);

  // Clone without regions
  aiirRewriterBaseCloneWithoutRegions(rewriter, op1);

  // Clone region before
  aiirRewriterBaseCloneRegionBefore(rewriter, region1, block2);

  aiirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: "builtin.module"() ({
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // clang-format on

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testInlineRegionBlock(AiirContext ctx) {
  // CHECK-LABEL: @testInlineRegionBlock
  fprintf(stderr, "@testInlineRegionBlock\n");

  const char *moduleString =
      "\"dialect.op1\"() ({\n"
      "  ^bb0(%arg0: index):\n"
      "    \"dialect.op1_in1\"(%arg0) [^bb1] : (index) -> ()\n"
      "  ^bb1():\n"
      "    \"dialect.op1_in2\"() : () -> ()\n"
      "}) : () -> ()\n"
      "\"dialect.op2\"() ({^bb0:}) : () -> ()\n"
      "\"dialect.op3\"() ({\n"
      "  ^bb0(%arg0: index):\n"
      "    \"dialect.op3_in1\"(%arg0) : (index) -> ()\n"
      "  ^bb1():\n"
      "    %x = \"dialect.op3_in2\"() : () -> index\n"
      "    %y = \"dialect.op3_in3\"() : () -> index\n"
      "}) : () -> ()\n"
      "\"dialect.op4\"() ({\n"
      "  ^bb0():\n"
      "    \"dialect.op4_in1\"() : () -> index\n"
      "  ^bb1(%arg0: index):\n"
      "    \"dialect.op4_in2\"(%arg0) : (index) -> ()\n"
      "}) : () -> ()\n";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  AiirOperation op1 = aiirBlockGetFirstOperation(body);
  AiirRegion region1 = aiirOperationGetRegion(op1, 0);

  AiirOperation op2 = aiirOperationGetNextInBlock(op1);
  AiirRegion region2 = aiirOperationGetRegion(op2, 0);
  AiirBlock block2 = aiirRegionGetFirstBlock(region2);

  AiirOperation op3 = aiirOperationGetNextInBlock(op2);
  AiirRegion region3 = aiirOperationGetRegion(op3, 0);
  AiirBlock block3_1 = aiirRegionGetFirstBlock(region3);
  AiirBlock block3_2 = aiirBlockGetNextInRegion(block3_1);
  AiirOperation op3_in2 = aiirBlockGetFirstOperation(block3_2);
  AiirValue op3_in2_res = aiirOperationGetResult(op3_in2, 0);
  AiirOperation op3_in3 = aiirOperationGetNextInBlock(op3_in2);

  AiirOperation op4 = aiirOperationGetNextInBlock(op3);
  AiirRegion region4 = aiirOperationGetRegion(op4, 0);
  AiirBlock block4_1 = aiirRegionGetFirstBlock(region4);
  AiirOperation op4_in1 = aiirBlockGetFirstOperation(block4_1);
  AiirValue op4_in1_res = aiirOperationGetResult(op4_in1, 0);
  AiirBlock block4_2 = aiirBlockGetNextInRegion(block4_1);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);

  // Test these three functions
  aiirRewriterBaseInlineRegionBefore(rewriter, region1, block2);
  aiirRewriterBaseInlineBlockBefore(rewriter, block3_1, op3_in3, 1,
                                    &op3_in2_res);
  aiirRewriterBaseMergeBlocks(rewriter, block4_2, block4_1, 1, &op4_in1_res);

  aiirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: "builtin.module"() ({
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:     "dialect.op1_in1"(%{{.*}})[^[[bb:.*]]] : (index) -> ()
  // CHECK-NEXT:   ^[[bb]]:
  // CHECK-NEXT:     "dialect.op1_in2"() : () -> ()
  // CHECK-NEXT:   ^{{.*}}:  // no predecessors
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op3"() ({
  // CHECK-NEXT:     %{{.*}} = "dialect.op3_in2"() : () -> index
  // CHECK-NEXT:     "dialect.op3_in1"(%{{.*}}) : (index) -> ()
  // CHECK-NEXT:     %{{.*}} = "dialect.op3_in3"() : () -> index
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op4"() ({
  // CHECK-NEXT:     %{{.*}} = "dialect.op4_in1"() : () -> index
  // CHECK-NEXT:     "dialect.op4_in2"(%{{.*}}) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // clang-format on

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testReplaceOp(AiirContext ctx) {
  // CHECK-LABEL: @testReplaceOp
  fprintf(stderr, "@testReplaceOp\n");

  const char *moduleString =
      "%x, %y, %z = \"dialect.create_values\"() : () -> (index, index, index)\n"
      "%x_1, %y_1 = \"dialect.op1\"() : () -> (index, index)\n"
      "\"dialect.use_op1\"(%x_1, %y_1) : (index, index) -> ()\n"
      "%x_2, %y_2 = \"dialect.op2\"() : () -> (index, index)\n"
      "%x_3, %y_3 = \"dialect.op3\"() : () -> (index, index)\n"
      "\"dialect.use_op2\"(%x_2, %y_2) : (index, index) -> ()\n";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  // get a handle to all operations/values
  AiirOperation createValues = aiirBlockGetFirstOperation(body);
  AiirValue x = aiirOperationGetResult(createValues, 0);
  AiirValue z = aiirOperationGetResult(createValues, 2);
  AiirOperation op1 = aiirOperationGetNextInBlock(createValues);
  AiirOperation useOp1 = aiirOperationGetNextInBlock(op1);
  AiirOperation op2 = aiirOperationGetNextInBlock(useOp1);
  AiirOperation op3 = aiirOperationGetNextInBlock(op2);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);

  // Test replace op with values
  AiirValue xz[2] = {x, z};
  aiirRewriterBaseReplaceOpWithValues(rewriter, op1, 2, xz);

  // Test replace op with op
  aiirRewriterBaseReplaceOpWithOperation(rewriter, op2, op3);

  aiirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %[[res:.*]]:3 = "dialect.create_values"() : () -> (index, index, index)
  // CHECK-NEXT:   "dialect.use_op1"(%[[res]]#0, %[[res]]#2) : (index, index) -> ()
  // CHECK-NEXT:   %[[res2:.*]]:2 = "dialect.op3"() : () -> (index, index)
  // CHECK-NEXT:   "dialect.use_op2"(%[[res2]]#0, %[[res2]]#1) : (index, index) -> ()
  // CHECK-NEXT: }
  // clang-format on

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testErase(AiirContext ctx) {
  // CHECK-LABEL: @testErase
  fprintf(stderr, "@testErase\n");

  const char *moduleString = "\"dialect.op_to_erase\"() : () -> ()\n"
                             "\"dialect.op2\"() ({\n"
                             "^bb0():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "^block_to_erase():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "^bb1():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "}) : () -> ()\n";
  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  // get a handle to all operations/values
  AiirOperation opToErase = aiirBlockGetFirstOperation(body);
  AiirOperation op2 = aiirOperationGetNextInBlock(opToErase);
  AiirRegion op2Region = aiirOperationGetRegion(op2, 0);
  AiirBlock bb0 = aiirRegionGetFirstBlock(op2Region);
  AiirBlock blockToErase = aiirBlockGetNextInRegion(bb0);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);
  aiirRewriterBaseEraseOp(rewriter, opToErase);
  aiirRewriterBaseEraseBlock(rewriter, blockToErase);

  aiirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT: "dialect.op2"() ({
  // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
  // CHECK-NEXT: ^{{.*}}:
  // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // CHECK-NEXT: }

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testMove(AiirContext ctx) {
  // CHECK-LABEL: @testMove
  fprintf(stderr, "@testMove\n");

  const char *moduleString = "\"dialect.op1\"() : () -> ()\n"
                             "\"dialect.op2\"() ({\n"
                             "^bb0(%arg0: index):\n"
                             "  \"dialect.op2_1\"(%arg0) : (index) -> ()"
                             "^bb1(%arg1: index):\n"
                             "  \"dialect.op2_2\"(%arg1) : (index) -> ()"
                             "}) : () -> ()\n"
                             "\"dialect.op3\"() : () -> ()\n"
                             "\"dialect.op4\"() : () -> ()\n";

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  // get a handle to all operations/values
  AiirOperation op1 = aiirBlockGetFirstOperation(body);
  AiirOperation op2 = aiirOperationGetNextInBlock(op1);
  AiirOperation op3 = aiirOperationGetNextInBlock(op2);
  AiirOperation op4 = aiirOperationGetNextInBlock(op3);

  AiirRegion region2 = aiirOperationGetRegion(op2, 0);
  AiirBlock block0 = aiirRegionGetFirstBlock(region2);
  AiirBlock block1 = aiirBlockGetNextInRegion(block0);

  // Test move operations.
  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);
  aiirRewriterBaseMoveOpBefore(rewriter, op3, op1);
  aiirRewriterBaseMoveOpAfter(rewriter, op4, op1);
  aiirRewriterBaseMoveBlockBefore(rewriter, block1, block0);

  aiirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT:   "dialect.op3"() : () -> ()
  // CHECK-NEXT:   "dialect.op1"() : () -> ()
  // CHECK-NEXT:   "dialect.op4"() : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%[[arg0:.*]]: index):
  // CHECK-NEXT:     "dialect.op2_2"(%[[arg0]]) : (index) -> ()
  // CHECK-NEXT:   ^{{.*}}(%[[arg1:.*]]: index):  // no predecessors
  // CHECK-NEXT:     "dialect.op2_1"(%[[arg1]]) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testOpModification(AiirContext ctx) {
  // CHECK-LABEL: @testOpModification
  fprintf(stderr, "@testOpModification\n");

  const char *moduleString =
      "%x, %y = \"dialect.op1\"() : () -> (index, index)\n"
      "\"dialect.op2\"(%x) : (index) -> ()\n";

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  // get a handle to all operations/values
  AiirOperation op1 = aiirBlockGetFirstOperation(body);
  AiirValue y = aiirOperationGetResult(op1, 1);
  AiirOperation op2 = aiirOperationGetNextInBlock(op1);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);
  aiirRewriterBaseStartOpModification(rewriter, op1);
  aiirRewriterBaseCancelOpModification(rewriter, op1);

  aiirRewriterBaseStartOpModification(rewriter, op2);
  aiirOperationSetOperand(op2, 0, y);
  aiirRewriterBaseFinalizeOpModification(rewriter, op2);

  aiirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT: %[[xy:.*]]:2 = "dialect.op1"() : () -> (index, index)
  // CHECK-NEXT: "dialect.op2"(%[[xy]]#1) : (index) -> ()
  // CHECK-NEXT: }

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testReplaceUses(AiirContext ctx) {
  // CHECK-LABEL: @testReplaceUses
  fprintf(stderr, "@testReplaceUses\n");

  const char *moduleString =
      // Replace values with values
      "%x1, %y1, %z1 = \"dialect.op1\"() : () -> (index, index, index)\n"
      "%x2, %y2, %z2 = \"dialect.op2\"() : () -> (index, index, index)\n"
      "\"dialect.op1_uses\"(%x1, %y1, %z1) : (index, index, index) -> ()\n"
      // Replace op with values
      "%x3 = \"dialect.op3\"() : () -> index\n"
      "%x4 = \"dialect.op4\"() : () -> index\n"
      "\"dialect.op3_uses\"(%x3) : (index) -> ()\n"
      // Replace op with op
      "%x5 = \"dialect.op5\"() : () -> index\n"
      "%x6 = \"dialect.op6\"() : () -> index\n"
      "\"dialect.op5_uses\"(%x5) : (index) -> ()\n"
      // Replace op in block;
      "%x7 = \"dialect.op7\"() : () -> index\n"
      "%x8 = \"dialect.op8\"() : () -> index\n"
      "\"dialect.op9\"() ({\n"
      "^bb0:\n"
      "   \"dialect.op7_uses\"(%x7) : (index) -> ()\n"
      "}): () -> ()\n"
      "\"dialect.op7_uses\"(%x7) : (index) -> ()\n"
      // Replace value with value except in op
      "%x10 = \"dialect.op10\"() : () -> index\n"
      "%x11 = \"dialect.op11\"() : () -> index\n"
      "\"dialect.op10_uses\"(%x10) : (index) -> ()\n"
      "\"dialect.op10_uses\"(%x10) : (index) -> ()\n";

  AiirModule module =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(moduleString));
  AiirOperation op = aiirModuleGetOperation(module);
  AiirBlock body = aiirModuleGetBody(module);

  // get a handle to all operations/values
  AiirOperation op1 = aiirBlockGetFirstOperation(body);
  AiirValue x1 = aiirOperationGetResult(op1, 0);
  AiirValue y1 = aiirOperationGetResult(op1, 1);
  AiirValue z1 = aiirOperationGetResult(op1, 2);
  AiirOperation op2 = aiirOperationGetNextInBlock(op1);
  AiirValue x2 = aiirOperationGetResult(op2, 0);
  AiirValue y2 = aiirOperationGetResult(op2, 1);
  AiirValue z2 = aiirOperationGetResult(op2, 2);
  AiirOperation op1Uses = aiirOperationGetNextInBlock(op2);

  AiirOperation op3 = aiirOperationGetNextInBlock(op1Uses);
  AiirOperation op4 = aiirOperationGetNextInBlock(op3);
  AiirValue x4 = aiirOperationGetResult(op4, 0);
  AiirOperation op3Uses = aiirOperationGetNextInBlock(op4);

  AiirOperation op5 = aiirOperationGetNextInBlock(op3Uses);
  AiirOperation op6 = aiirOperationGetNextInBlock(op5);
  AiirOperation op5Uses = aiirOperationGetNextInBlock(op6);

  AiirOperation op7 = aiirOperationGetNextInBlock(op5Uses);
  AiirOperation op8 = aiirOperationGetNextInBlock(op7);
  AiirValue x8 = aiirOperationGetResult(op8, 0);
  AiirOperation op9 = aiirOperationGetNextInBlock(op8);
  AiirRegion region9 = aiirOperationGetRegion(op9, 0);
  AiirBlock block9 = aiirRegionGetFirstBlock(region9);
  AiirOperation op7Uses = aiirOperationGetNextInBlock(op9);

  AiirOperation op10 = aiirOperationGetNextInBlock(op7Uses);
  AiirValue x10 = aiirOperationGetResult(op10, 0);
  AiirOperation op11 = aiirOperationGetNextInBlock(op10);
  AiirValue x11 = aiirOperationGetResult(op11, 0);
  AiirOperation op10Uses1 = aiirOperationGetNextInBlock(op11);

  AiirRewriterBase rewriter = aiirIRRewriterCreate(ctx);

  // Replace values
  aiirRewriterBaseReplaceAllUsesWith(rewriter, x1, x2);
  AiirValue y1z1[2] = {y1, z1};
  AiirValue y2z2[2] = {y2, z2};
  aiirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, 2, y1z1, y2z2);

  // Replace op with values
  aiirRewriterBaseReplaceOpWithValues(rewriter, op3, 1, &x4);

  // Replace op with op
  aiirRewriterBaseReplaceOpWithOperation(rewriter, op5, op6);

  // Replace op with op in block
  aiirRewriterBaseReplaceOpUsesWithinBlock(rewriter, op7, 1, &x8, block9);

  // Replace value with value except in op
  aiirRewriterBaseReplaceAllUsesExcept(rewriter, x10, x11, op10Uses1);

  aiirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %{{.*}}:3 = "dialect.op1"() : () -> (index, index, index)
  // CHECK-NEXT:   %[[res2:.*]]:3 = "dialect.op2"() : () -> (index, index, index)
  // CHECK-NEXT:   "dialect.op1_uses"(%[[res2]]#0, %[[res2]]#1, %[[res2]]#2) : (index, index, index) -> ()
  // CHECK-NEXT:   %[[res4:.*]] = "dialect.op4"() : () -> index
  // CHECK-NEXT:   "dialect.op3_uses"(%[[res4]]) : (index) -> ()
  // CHECK-NEXT:   %[[res6:.*]] = "dialect.op6"() : () -> index
  // CHECK-NEXT:   "dialect.op5_uses"(%[[res6]]) : (index) -> ()
  // CHECK-NEXT:   %[[res7:.*]] = "dialect.op7"() : () -> index
  // CHECK-NEXT:   %[[res8:.*]] = "dialect.op8"() : () -> index
  // CHECK-NEXT:   "dialect.op9"() ({
  // CHECK-NEXT:     "dialect.op7_uses"(%[[res8]]) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op7_uses"(%[[res7]]) : (index) -> ()
  // CHECK-NEXT:   %[[res10:.*]] = "dialect.op10"() : () -> index
  // CHECK-NEXT:   %[[res11:.*]] = "dialect.op11"() : () -> index
  // CHECK-NEXT:   "dialect.op10_uses"(%[[res10]]) : (index) -> ()
  // CHECK-NEXT:   "dialect.op10_uses"(%[[res11]]) : (index) -> ()
  // CHECK-NEXT: }
  // clang-format on

  aiirIRRewriterDestroy(rewriter);
  aiirModuleDestroy(module);
}

void testGreedyRewriteDriverConfig(AiirContext ctx) {
  // CHECK-LABEL: @testGreedyRewriteDriverConfig
  fprintf(stderr, "@testGreedyRewriteDriverConfig\n");

  // Test config creation and destruction
  AiirGreedyRewriteDriverConfig config = aiirGreedyRewriteDriverConfigCreate();

  // Test all configuration setters
  aiirGreedyRewriteDriverConfigSetMaxIterations(config, 5);
  aiirGreedyRewriteDriverConfigSetMaxNumRewrites(config, 100);
  aiirGreedyRewriteDriverConfigSetUseTopDownTraversal(config, true);
  aiirGreedyRewriteDriverConfigEnableFolding(config, false);
  aiirGreedyRewriteDriverConfigSetStrictness(
      config, AIIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS);
  aiirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
      config, AIIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL);
  aiirGreedyRewriteDriverConfigEnableConstantCSE(config, false);

  // Test all configuration getters and verify values
  // CHECK: MaxIterations: 5
  fprintf(stderr, "MaxIterations: %" PRId64 "\n",
          aiirGreedyRewriteDriverConfigGetMaxIterations(config));
  // CHECK: MaxNumRewrites: 100
  fprintf(stderr, "MaxNumRewrites: %" PRId64 "\n",
          aiirGreedyRewriteDriverConfigGetMaxNumRewrites(config));
  // CHECK: UseTopDownTraversal: 1
  fprintf(stderr, "UseTopDownTraversal: %d\n",
          aiirGreedyRewriteDriverConfigGetUseTopDownTraversal(config));
  // CHECK: FoldingEnabled: 0
  fprintf(stderr, "FoldingEnabled: %d\n",
          aiirGreedyRewriteDriverConfigIsFoldingEnabled(config));
  // CHECK: Strictness: 2
  fprintf(stderr, "Strictness: %d\n",
          aiirGreedyRewriteDriverConfigGetStrictness(config));
  // CHECK: RegionSimplificationLevel: 1
  fprintf(stderr, "RegionSimplificationLevel: %d\n",
          aiirGreedyRewriteDriverConfigGetRegionSimplificationLevel(config));
  // CHECK: ConstantCSEEnabled: 0
  fprintf(stderr, "ConstantCSEEnabled: %d\n",
          aiirGreedyRewriteDriverConfigIsConstantCSEEnabled(config));

  // CHECK: Config test completed successfully
  fprintf(stderr, "Config test completed successfully\n");
  aiirGreedyRewriteDriverConfigDestroy(config);
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirContextSetAllowUnregisteredDialects(ctx, true);
  aiirContextGetOrLoadDialect(ctx, aiirStringRefCreateFromCString("builtin"));

  testInsertionPoint(ctx);
  testCreateBlock(ctx);
  testInlineRegionBlock(ctx);
  testReplaceOp(ctx);
  testErase(ctx);
  testMove(ctx);
  testOpModification(ctx);
  testReplaceUses(ctx);
  testGreedyRewriteDriverConfig(ctx);

  aiirContextDestroy(ctx);
  return 0;
}

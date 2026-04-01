//===- sparse_tensor.c - Test of sparse_tensor APIs -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aiir-capi-sparse-tensor-test 2>&1 | FileCheck %s

#include "aiir-c/Dialect/SparseTensor.h"
#include "aiir-c/IR.h"
#include "aiir-c/RegisterEverything.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CHECK-LABEL: testRoundtripEncoding()
static int testRoundtripEncoding(AiirContext ctx) {
  fprintf(stderr, "testRoundtripEncoding()\n");
  // clang-format off
  const char *originalAsm =
    "#sparse_tensor.encoding<{ "
    "map = [s0](d0, d1) -> (s0 : dense, d0 : compressed, d1 : compressed), "
    "posWidth = 32, crdWidth = 64, explicitVal = 1 : i64}>";
  // clang-format on
  AiirAttribute originalAttr =
      aiirAttributeParseGet(ctx, aiirStringRefCreateFromCString(originalAsm));
  // CHECK: isa: 1
  fprintf(stderr, "isa: %d\n",
          aiirAttributeIsASparseTensorEncodingAttr(originalAttr));
  AiirAffineMap dimToLvl =
      aiirSparseTensorEncodingAttrGetDimToLvl(originalAttr);
  // CHECK: (d0, d1)[s0] -> (s0, d0, d1)
  aiirAffineMapDump(dimToLvl);
  // CHECK: level_type: 65536
  // CHECK: level_type: 262144
  // CHECK: level_type: 262144
  AiirAffineMap lvlToDim =
      aiirSparseTensorEncodingAttrGetLvlToDim(originalAttr);
  int lvlRank = aiirSparseTensorEncodingGetLvlRank(originalAttr);
  AiirSparseTensorLevelType *lvlTypes =
      malloc(sizeof(AiirSparseTensorLevelType) * lvlRank);
  for (int l = 0; l < lvlRank; ++l) {
    lvlTypes[l] = aiirSparseTensorEncodingAttrGetLvlType(originalAttr, l);
    fprintf(stderr, "level_type: %" PRIu64 "\n", lvlTypes[l]);
  }
  // CHECK: posWidth: 32
  int posWidth = aiirSparseTensorEncodingAttrGetPosWidth(originalAttr);
  fprintf(stderr, "posWidth: %d\n", posWidth);
  // CHECK: crdWidth: 64
  int crdWidth = aiirSparseTensorEncodingAttrGetCrdWidth(originalAttr);
  fprintf(stderr, "crdWidth: %d\n", crdWidth);

  // CHECK: explicitVal: 1 : i64
  AiirAttribute explicitVal =
      aiirSparseTensorEncodingAttrGetExplicitVal(originalAttr);
  fprintf(stderr, "explicitVal: ");
  aiirAttributeDump(explicitVal);
  // CHECK: implicitVal: <<NULL ATTRIBUTE>>
  AiirAttribute implicitVal =
      aiirSparseTensorEncodingAttrGetImplicitVal(originalAttr);
  fprintf(stderr, "implicitVal: ");
  aiirAttributeDump(implicitVal);

  AiirAttribute newAttr = aiirSparseTensorEncodingAttrGet(
      ctx, lvlRank, lvlTypes, dimToLvl, lvlToDim, posWidth, crdWidth,
      explicitVal, implicitVal);
  aiirAttributeDump(newAttr); // For debugging filecheck output.
  // CHECK: equal: 1
  fprintf(stderr, "equal: %d\n", aiirAttributeEqual(originalAttr, newAttr));
  free(lvlTypes);
  return 0;
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleRegisterDialect(aiirGetDialectHandle__sparse_tensor__(),
                                   ctx);
  if (testRoundtripEncoding(ctx))
    return 1;

  aiirContextDestroy(ctx);
  return 0;
}

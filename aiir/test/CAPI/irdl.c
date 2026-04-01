//===- irdl.c - Test for the C bindings for IRDL registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: aiir-capi-irdl-test 2>&1 | FileCheck %s
 */

#include "aiir-c/Dialect/IRDL.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

const char irdlDialect[] = "\
  irdl.dialect @foo {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.results(baz: %i32)\
    }\
  }\
  irdl.dialect @bar {\
    irdl.operation @op {\
      %i32 = irdl.is i32\
      irdl.operands(baz: %i32)\
    }\
  }";

// CHECK:      module {
// CHECK-NEXT:   %[[RES:.*]] = "foo.op"() : () -> i32
// CHECK-NEXT:   "bar.op"(%[[RES]]) :  (i32) -> ()
// CHECK-NEXT: }
const char newDialectUsage[] = "\
  module {\
    %res = \"foo.op\"() : () -> i32\
    \"bar.op\"(%res) : (i32) -> ()\
  }";

void testVariadicityAttributes(AiirContext ctx) {
  AiirAttribute variadicitySingle =
      aiirIRDLVariadicityAttrGet(ctx, aiirStringRefCreateFromCString("single"));

  // CHECK: #irdl<variadicity single>
  aiirAttributeDump(variadicitySingle);

  AiirAttribute variadicityOptional = aiirIRDLVariadicityAttrGet(
      ctx, aiirStringRefCreateFromCString("optional"));

  // CHECK: #irdl<variadicity optional>
  aiirAttributeDump(variadicityOptional);

  AiirAttribute variadicityVariadic = aiirIRDLVariadicityAttrGet(
      ctx, aiirStringRefCreateFromCString("variadic"));

  // CHECK: #irdl<variadicity variadic>
  aiirAttributeDump(variadicityVariadic);

  AiirAttribute variadicities[] = {variadicitySingle, variadicityOptional,
                                   variadicityVariadic};
  AiirAttribute variadicityArray =
      aiirIRDLVariadicityArrayAttrGet(ctx, 3, variadicities);

  // CHECK: #irdl<variadicity_array[single, optional, variadic]>
  aiirAttributeDump(variadicityArray);
}

int main(void) {
  AiirContext ctx = aiirContextCreate();
  aiirDialectHandleLoadDialect(aiirGetDialectHandle__irdl__(), ctx);

  // Test loading an IRDL dialect and using it.
  AiirModule dialectDecl =
      aiirModuleCreateParse(ctx, aiirStringRefCreateFromCString(irdlDialect));

  aiirLoadIRDLDialects(dialectDecl);
  aiirModuleDestroy(dialectDecl);

  AiirModule usingModule = aiirModuleCreateParse(
      ctx, aiirStringRefCreateFromCString(newDialectUsage));

  aiirOperationDump(aiirModuleGetOperation(usingModule));

  aiirModuleDestroy(usingModule);

  // Test variadicity attributes.
  testVariadicityAttributes(ctx);

  aiirContextDestroy(ctx);
  return 0;
}

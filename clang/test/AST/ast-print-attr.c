// RUN: %clang_cc1 -ast-print -x objective-c++ -fms-extensions %s -o - | FileCheck %s

// CHECK: using A = __kindof id (*)[1];
using A = __kindof id (*)[1];

// CHECK: using B = int ** __ptr32 *[3];
using B = int ** __ptr32 *[3];

// FIXME: This is the wrong spelling for the attribute.
// FIXME: Too many parens here!
// CHECK: using C = int ((*))() __attribute__((cdecl));
using C = int (*)() [[gnu::cdecl]];

// CHECK: int fun_asm() asm("");
int fun_asm() asm("");
// CHECK: int var_asm asm("");
int var_asm asm("");

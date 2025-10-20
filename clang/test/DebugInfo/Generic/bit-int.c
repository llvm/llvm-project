// RUN: %clang_cc1 -x c++ %s -debug-info-kind=standalone -gno-column-info -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c   %s -debug-info-kind=standalone -gno-column-info -emit-llvm -o - | FileCheck %s

unsigned _BitInt(2) a;
_BitInt(2) b;

// CHECK: !DIBasicType(name: "_BitInt", size: 2, encoding: DW_ATE_signed)
// CHECK: !DIBasicType(name: "unsigned _BitInt", size: 2, encoding: DW_ATE_unsigned)

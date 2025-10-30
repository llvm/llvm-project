// RUN: %clang_cc1 -x c++ %s -debug-info-kind=standalone -gno-column-info -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c   %s -debug-info-kind=standalone -gno-column-info -emit-llvm -o - | FileCheck %s

unsigned _BitInt(17) a;
_BitInt(2) b;

// CHECK: !DIBasicType(name: "_BitInt(2)", size: 8, dataSize: 2, encoding: DW_ATE_signed)
// CHECK: !DIBasicType(name: "unsigned _BitInt(17)", size: 32,  dataSize: 17, encoding: DW_ATE_unsigned)

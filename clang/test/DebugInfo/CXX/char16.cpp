// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -debug-info-kind=limited %s -o -| FileCheck %s

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

// CHECK: !{{.*}} = !DIBasicType(name: "char16_t"
// CHECK-SAME:                   encoding: DW_ATE_UTF)

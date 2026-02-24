// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=standalone -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -debug-info-kind=standalone -emit-llvm -o - %s | FileCheck %s

// Verify DWARF debug info encoding for __int256_t and __uint256_t.

__int256_t s256;
__uint256_t u256;

// CHECK-DAG: !DIBasicType(name: "__int256", size: 256, encoding: DW_ATE_signed)
// CHECK-DAG: !DIBasicType(name: "unsigned __int256", size: 256, encoding: DW_ATE_unsigned)
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__int256_t"
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__uint256_t"

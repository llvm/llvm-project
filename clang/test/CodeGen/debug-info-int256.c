// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=standalone -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -debug-info-kind=standalone -emit-llvm -o - %s | FileCheck %s

// Verify DWARF debug info encoding for __int256_t and __uint256_t.

// Global variables
__int256_t s256;
__uint256_t u256;

// Function with __int256_t parameter and local variable
void f(__int256_t param) {
  __uint256_t local = (__uint256_t)param;
  (void)local;
}

// Basic type encoding
// CHECK-DAG: !DIBasicType(name: "__int256", size: 256, encoding: DW_ATE_signed)
// CHECK-DAG: !DIBasicType(name: "unsigned __int256", size: 256, encoding: DW_ATE_unsigned)

// Typedef encoding
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__int256_t"
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "__uint256_t"

// Function parameter and local variable
// CHECK-DAG: !DILocalVariable(name: "param", arg: 1,
// CHECK-DAG: !DILocalVariable(name: "local",

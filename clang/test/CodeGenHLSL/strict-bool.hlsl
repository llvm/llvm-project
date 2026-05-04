// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -O1 -load-bool-from-mem=truncate -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK-TRUNCATE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -O1 -load-bool-from-mem=nonzero -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK-NONZERO

typedef bool bool8_t __attribute__((ext_vector_type(8)));
extern bool8_t vec;

bool8_t getvec(void) {
    // CHECK-TRUNCATE: [[BOOL:%.+]] = load <8 x i32>
    // CHECK-TRUNCATE: trunc <8 x i32> [[BOOL]] to <8 x i1>

    // CHECK-NONZERO: [[BOOL:%.+]] = load <8 x i32>
    // CHECK-NONZERO: icmp ne <8 x i32> [[BOOL]], zeroinitializer
    return vec;
}

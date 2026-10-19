// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -O1 -load-bool-from-mem=truncate -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=CHECK,TRUNCATE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -O1 -load-bool-from-mem=nonzero -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -O0 -emit-llvm -o - %s | FileCheck %s

// TRUNCATE: -load-bool-from-mem=truncate ignored for HLSL (always nonzero)

typedef bool bool8_t __attribute__((ext_vector_type(8)));
extern bool8_t vec;

bool8_t getvec(void) {
    // CHECK: [[BOOL:%.+]] = load <8 x i32>
    // CHECK: icmp ne <8 x i32> [[BOOL]], zeroinitializer
    return vec;
}

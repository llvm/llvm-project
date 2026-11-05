// Verify that a riscv_rvv_vector_bits typedef records the LMUL of the sizeless
// data type, or the ratio of the sizeless mask type, it was written against.

// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64d \
// RUN:   -mvscale-min=2 -mvscale-max=2 -ast-dump=json %s \
// RUN:   | FileCheck %s --check-prefix=JSON

// VLEN=128.
typedef __rvv_int8mf8_t fixed_i8mf8 __attribute__((riscv_rvv_vector_bits(16)));
typedef __rvv_int16mf4_t fixed_i16mf4 __attribute__((riscv_rvv_vector_bits(32)));
typedef __rvv_int32mf2_t fixed_i32mf2 __attribute__((riscv_rvv_vector_bits(64)));
typedef __rvv_int32m1_t fixed_i32m1 __attribute__((riscv_rvv_vector_bits(128)));
typedef __rvv_int32m2_t fixed_i32m2 __attribute__((riscv_rvv_vector_bits(256)));
typedef __rvv_int64m4_t fixed_i64m4 __attribute__((riscv_rvv_vector_bits(512)));
typedef __rvv_int8m8_t fixed_i8m8 __attribute__((riscv_rvv_vector_bits(1024)));
typedef __rvv_bool1_t fixed_b1 __attribute__((riscv_rvv_vector_bits(128)));
typedef __rvv_bool8_t fixed_b8 __attribute__((riscv_rvv_vector_bits(16)));
typedef __rvv_bool32_t fixed_b32 __attribute__((riscv_rvv_vector_bits(4)));
typedef __rvv_bool64_t fixed_b64 __attribute__((riscv_rvv_vector_bits(2)));

// CHECK-LABEL: TypedefDecl {{.*}} fixed_i8mf8
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul mf8 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i16mf4
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul mf4 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32mf2
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul mf2 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32m1
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul m1 4
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32m2
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul m2 8
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i64m4
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul m4 8
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i8m8
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul m8 128
// CHECK-LABEL: TypedefDecl {{.*}} fixed_b1
// CHECK:       VectorType {{.*}} fixed-length rvv mask vector ratio 1 16
// CHECK-LABEL: TypedefDecl {{.*}} fixed_b8
// CHECK:       VectorType {{.*}} fixed-length rvv mask vector ratio 8 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_b32
// CHECK:       VectorType {{.*}} fixed-length rvv mask vector ratio 32 1
// CHECK-LABEL: TypedefDecl {{.*}} fixed_b64
// CHECK:       VectorType {{.*}} fixed-length rvv mask vector ratio 64 1

// JSON: "vectorKind": "fixed-length rvv data vector lmul mf8"
// JSON: "vectorKind": "fixed-length rvv data vector lmul mf4"
// JSON: "vectorKind": "fixed-length rvv data vector lmul mf2"
// JSON: "vectorKind": "fixed-length rvv data vector lmul m1"
// JSON: "vectorKind": "fixed-length rvv data vector lmul m2"
// JSON: "vectorKind": "fixed-length rvv data vector lmul m4"
// JSON: "vectorKind": "fixed-length rvv data vector lmul m8"
// JSON: "vectorKind": "fixed-length rvv mask vector ratio 1"
// JSON: "vectorKind": "fixed-length rvv mask vector ratio 8"
// JSON: "vectorKind": "fixed-length rvv mask vector ratio 32"
// JSON: "vectorKind": "fixed-length rvv mask vector ratio 64"

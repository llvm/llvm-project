// Verify that a riscv_rvv_vector_bits typedef of a data type records the LMUL
// of the sizeless type it was written against.

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

// CHECK-LABEL: TypedefDecl {{.*}} fixed_i8mf8
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 1/8 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i16mf4
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 1/4 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32mf2
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 1/2 2
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32m1
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 1 4
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i32m2
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 2 8
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i64m4
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 4 8
// CHECK-LABEL: TypedefDecl {{.*}} fixed_i8m8
// CHECK:       VectorType {{.*}} fixed-length rvv data vector lmul 8 128

// JSON: "vectorKind": "fixed-length rvv data vector lmul 1/8"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 1/4"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 1/2"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 1"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 2"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 4"
// JSON: "vectorKind": "fixed-length rvv data vector lmul 8"

// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +zve32x \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM-ZVE32X %s
// RUN: %clang_cc1 -std=c23 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s

#include <riscv_vector.h>

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @bar
vint32m1_t __attribute__((riscv_vector_cc)) bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call riscv_vector_cc <vscale x 2 x i32> @bar
[[riscv::vector_cc]] vint32m1_t bar(vint32m1_t input);
vint32m1_t test_vector_cc_attr2(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = bar(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: call <vscale x 2 x i32> @baz
vint32m1_t baz(vint32m1_t input);
vint32m1_t test_no_vector_cc_attr(vint32m1_t input, int32_t *base, size_t vl) {
  vint32m1_t val = __riscv_vle32_v_i32m1(base, vl);
  vint32m1_t ret = baz(input);
  __riscv_vse32_v_i32m1(base, val, vl);
  return ret;
}

// CHECK-LLVM: define dso_local void @test_vls_no_cc(i128 noundef %arg.coerce)
void test_vls_no_cc(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen(<vscale x 2 x i32> noundef %arg.coerce)
void __attribute__((riscv_vls_cc)) test_vls_default_abi_vlen(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_c23(<vscale x 2 x i32> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen_c23(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_unsupported_feature(<vscale x 8 x i8> noundef %arg.coerce)
void __attribute__((riscv_vls_cc)) test_vls_default_abi_vlen_unsupported_feature(__attribute__((vector_size(16))) _Float16 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_c23_unsupported_feature(<vscale x 8 x i8> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen_c23_unsupported_feature(__attribute__((vector_size(16))) _Float16 arg) {}

// CHECK-LLVM-ZVE32X: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_unsupported_feature_zve32x(<vscale x 8 x i8> noundef %arg.coerce)
void __attribute__((riscv_vls_cc)) test_vls_default_abi_vlen_unsupported_feature_zve32x(__attribute__((vector_size(16))) float arg) {}

// CHECK-LLVM-ZVE32X: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_c23_unsupported_feature_zve32x(<vscale x 8 x i8> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen_c23_unsupported_feature_zve32x(__attribute__((vector_size(16))) float arg) {}

// CHECK-LLVM-ZVE32X: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_unsupported_feature_no_zve64x(<vscale x 8 x i8> noundef %arg.coerce)
void __attribute__((riscv_vls_cc)) test_vls_default_abi_vlen_unsupported_feature_no_zve64x(__attribute__((vector_size(16))) uint64_t arg) {}

// CHECK-LLVM-ZVE32X: define dso_local riscv_vls_cc(128) void @test_vls_default_abi_vlen_c23_unsupported_feature_no_zve64x(<vscale x 8 x i8> noundef %arg.coerce)
[[riscv::vls_cc]] void test_vls_default_abi_vlen_c23_unsupported_feature_no_zve64x(__attribute__((vector_size(16))) uint64_t arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_vls_256_abi_vlen(<vscale x 1 x i32> noundef %arg.coerce)
void __attribute__((riscv_vls_cc(256))) test_vls_256_abi_vlen(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_vls_256_abi_vlen_c23(<vscale x 1 x i32> noundef %arg.coerce)
[[riscv::vls_cc(256)]] void test_vls_256_abi_vlen_c23(__attribute__((vector_size(16))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(1024) void @test_vls_least_element(<vscale x 1 x i32> noundef %arg.coerce)
void __attribute__((riscv_vls_cc(1024))) test_vls_least_element(__attribute__((vector_size(8))) int arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(1024) void @test_vls_least_element_c23(<vscale x 1 x i32> noundef %arg.coerce)
[[riscv::vls_cc(1024)]] void test_vls_least_element_c23(__attribute__((vector_size(8))) int arg) {}


struct st_i32x4 {
    __attribute__((vector_size(16))) int i32;
};

struct st_i32x4_arr1 {
    __attribute__((vector_size(16))) int i32[1];
};

struct st_i32x4_arr4 {
    __attribute__((vector_size(16))) int i32[4];
};

struct st_i32x4_arr8 {
    __attribute__((vector_size(16))) int i32[8];
};


struct st_i32x4x2 {
    __attribute__((vector_size(16))) int i32_1;
    __attribute__((vector_size(16))) int i32_2;
};

struct st_i32x8x2 {
    __attribute__((vector_size(32))) int i32_1;
    __attribute__((vector_size(32))) int i32_2;
};

struct st_i32x64x2 {
    __attribute__((vector_size(256))) int i32_1;
    __attribute__((vector_size(256))) int i32_2;
};

struct st_i32x4x3 {
    __attribute__((vector_size(16))) int i32_1;
    __attribute__((vector_size(16))) int i32_2;
    __attribute__((vector_size(16))) int i32_3;
};

struct st_i32x4x8 {
    __attribute__((vector_size(16))) int i32_1;
    __attribute__((vector_size(16))) int i32_2;
    __attribute__((vector_size(16))) int i32_3;
    __attribute__((vector_size(16))) int i32_4;
    __attribute__((vector_size(16))) int i32_5;
    __attribute__((vector_size(16))) int i32_6;
    __attribute__((vector_size(16))) int i32_7;
    __attribute__((vector_size(16))) int i32_8;
};

struct st_i32x4x9 {
    __attribute__((vector_size(16))) int i32_1;
    __attribute__((vector_size(16))) int i32_2;
    __attribute__((vector_size(16))) int i32_3;
    __attribute__((vector_size(16))) int i32_4;
    __attribute__((vector_size(16))) int i32_5;
    __attribute__((vector_size(16))) int i32_6;
    __attribute__((vector_size(16))) int i32_7;
    __attribute__((vector_size(16))) int i32_8;
    __attribute__((vector_size(16))) int i32_9;
};

typedef int __attribute__((vector_size(256))) int32x64_t;

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_too_large(ptr noundef %0)
void __attribute__((riscv_vls_cc)) test_too_large(int32x64_t arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_too_large_256(<vscale x 16 x i32> noundef %arg.coerce)
void __attribute__((riscv_vls_cc(256))) test_too_large_256(int32x64_t arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4(<vscale x 2 x i32> %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4(struct st_i32x4 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4_256(<vscale x 1 x i32> %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4_256(struct st_i32x4 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4_arr1(<vscale x 2 x i32> %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4_arr1(struct st_i32x4_arr1 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4_arr1_256(<vscale x 1 x i32> %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4_arr1_256(struct st_i32x4_arr1 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4_arr4(<vscale x 8 x i32> %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4_arr4(struct st_i32x4_arr4 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4_arr4_256(<vscale x 4 x i32> %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4_arr4_256(struct st_i32x4_arr4 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4_arr8(<vscale x 16 x i32> %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4_arr8(struct st_i32x4_arr8 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4_arr8_256(<vscale x 8 x i32> %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4_arr8_256(struct st_i32x4_arr8 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4x2(target("riscv.vector.tuple", <vscale x 8 x i8>, 2) %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4x2(struct st_i32x4x2 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4x2_256(target("riscv.vector.tuple", <vscale x 4 x i8>, 2) %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4x2_256(struct st_i32x4x2 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x8x2(target("riscv.vector.tuple", <vscale x 16 x i8>, 2) %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x8x2(struct st_i32x8x2 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x8x2_256(target("riscv.vector.tuple", <vscale x 8 x i8>, 2) %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x8x2_256(struct st_i32x8x2 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x64x2(ptr noundef %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x64x2(struct st_i32x64x2 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x64x2_256(ptr noundef %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x64x2_256(struct st_i32x64x2 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4x3(target("riscv.vector.tuple", <vscale x 8 x i8>, 3) %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4x3(struct st_i32x4x3 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4x3_256(target("riscv.vector.tuple", <vscale x 4 x i8>, 3) %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4x3_256(struct st_i32x4x3 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4x8(target("riscv.vector.tuple", <vscale x 8 x i8>, 8) %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4x8(struct st_i32x4x8 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4x8_256(target("riscv.vector.tuple", <vscale x 4 x i8>, 8) %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4x8_256(struct st_i32x4x8 arg) {}

// CHECK-LLVM: define dso_local riscv_vls_cc(128) void @test_st_i32x4x9(ptr noundef %arg)
void __attribute__((riscv_vls_cc)) test_st_i32x4x9(struct st_i32x4x9 arg) {}
// CHECK-LLVM: define dso_local riscv_vls_cc(256) void @test_st_i32x4x9_256(ptr noundef %arg)
void __attribute__((riscv_vls_cc(256))) test_st_i32x4x9_256(struct st_i32x4x9 arg) {}

// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -std=c++11 -triple riscv64-none-linux-gnu -target-feature +v \
// RUN:   -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s

#include <riscv_vector.h>

typedef vint32m1_t fixed_int32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

//===----------------------------------------------------------------------===//
// Eligible cases work in C++ the same as C
//===----------------------------------------------------------------------===//

struct st_1field { fixed_int32m1_t x; };
// CHECK-LABEL: @test_1field(
// CHECK-SAME: <vscale x 2 x i32>
extern "C" void test_1field(st_1field s) {}

struct st_2field { fixed_int32m1_t x; fixed_int32m1_t y; };
// CHECK-LABEL: @test_2field(
// CHECK-SAME: target("riscv.vector.tuple", <vscale x 8 x i8>, 2)
extern "C" void test_2field(st_2field s) {}

//===----------------------------------------------------------------------===//
// Ineligible: struct with base class → passed indirectly
//
// detectHomogeneousRVVFixedLengthStruct rejects any struct with base classes
// to avoid having to reason about their layout contribution.
//===----------------------------------------------------------------------===//

struct empty_base {};
struct derived_1field : empty_base { fixed_int32m1_t x; };
// CHECK-LABEL: @test_base_class(
// CHECK-SAME: ptr
extern "C" void test_base_class(derived_1field s) {}

//===----------------------------------------------------------------------===//
// Ineligible: non-trivial copy constructor → passed indirectly
//
// Handled by getRecordArgABI before detectHomogeneousRVVFixedLengthStruct
// is ever consulted.
//===----------------------------------------------------------------------===//

struct nontrivial_copy {
  fixed_int32m1_t x;
  nontrivial_copy(const nontrivial_copy &) {}
};
// CHECK-LABEL: @test_nontrivial_copy(
// CHECK-SAME: ptr
extern "C" void test_nontrivial_copy(nontrivial_copy s) {}

//===----------------------------------------------------------------------===//
// Ineligible: non-trivial destructor → passed indirectly
//
// Also handled by getRecordArgABI before reaching our function.
//===----------------------------------------------------------------------===//

struct nontrivial_dtor {
  fixed_int32m1_t x;
  ~nontrivial_dtor() {}
};
// CHECK-LABEL: @test_nontrivial_dtor(
// CHECK-SAME: ptr
extern "C" void test_nontrivial_dtor(nontrivial_dtor s) {}

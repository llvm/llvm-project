// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-unknown-linux-gnu \
// RUN:   -target-feature +cx16 -Wno-atomic-alignment \
// RUN:   | FileCheck --check-prefixes=CHECK,X64 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -Wno-atomic-alignment | FileCheck --check-prefixes=CHECK,AMDGCN %s

typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef int int2 __attribute__((ext_vector_type(2)));

// CHECK-LABEL: @test_load(
half2 test_load(half2 *p) {
  // CHECK: load atomic <2 x half>, ptr {{.*}} monotonic
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_store(
void test_store(half2 *p, half2 v) {
  // CHECK: store atomic <2 x half> {{.*}}, ptr {{.*}} monotonic
  __atomic_store_n(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_exchange(
half2 test_exchange(half2 *p, half2 v) {
  // CHECK: atomicrmw xchg ptr {{.*}}, <2 x half> {{.*}} monotonic
  return __atomic_exchange_n(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_fetch_add_half2(
half2 test_fetch_add_half2(half2 *p, half2 v) {
  // CHECK: atomicrmw fadd ptr {{.*}}, <2 x half> {{.*}} monotonic
  return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_fetch_add_bfloat2(
bfloat2 test_fetch_add_bfloat2(bfloat2 *p, bfloat2 v) {
  // CHECK: atomicrmw fadd ptr {{.*}}, <2 x bfloat> {{.*}} monotonic
  return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_add_fetch_float2(
float2 test_add_fetch_float2(float2 *p, float2 v) {
  // CHECK: atomicrmw fadd ptr {{.*}}, <2 x float> {{.*}} monotonic
  // CHECK: fadd <2 x float>
  return __atomic_add_fetch(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_fetch_fmaximum_float2(
float2 test_fetch_fmaximum_float2(float2 *p, float2 v) {
  // CHECK: atomicrmw fmaximum ptr {{.*}}, <2 x float> {{.*}} monotonic
  return __atomic_fetch_fmaximum(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_max_fetch_int2(
int2 test_max_fetch_int2(int2 *p, int2 v) {
  // CHECK: atomicrmw max ptr {{.*}}, <2 x i32> {{.*}} monotonic
  // CHECK: icmp sgt <2 x i32>
  // CHECK: select <2 x i1>
  return __atomic_max_fetch(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_nand_fetch_int2(
int2 test_nand_fetch_int2(int2 *p, int2 v) {
  // CHECK: atomicrmw nand ptr {{.*}}, <2 x i32> {{.*}} monotonic
  // CHECK: and <2 x i32>
  // CHECK: xor <2 x i32>
  return __atomic_nand_fetch(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_c11_fetch_add(
half2 test_c11_fetch_add(_Atomic(half2) *p, half2 v) {
  // CHECK: atomicrmw fadd ptr {{.*}}, <2 x half> {{.*}} monotonic
  return __c11_atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// CHECK-LABEL: @test_scoped_fetch_add(
half2 test_scoped_fetch_add(half2 *p, half2 v) {
  // X64: atomicrmw fadd ptr {{.*}}, <2 x half> {{.*}} monotonic
  // AMDGCN: atomicrmw fadd ptr {{.*}}, <2 x half> {{.*}} syncscope("agent") monotonic
  return __scoped_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                   __MEMORY_SCOPE_DEVICE);
}

// A vector whose size is not a power of two is still accessed as an integer.
// CHECK-LABEL: @test_load_float3(
float3 test_load_float3(float3 *p) {
  // CHECK: load atomic i128, ptr {{.*}} monotonic
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

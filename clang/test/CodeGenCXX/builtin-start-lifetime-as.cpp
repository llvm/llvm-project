// RUN: %clang_cc1 -std=c++23 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct Trivial { int x; float y; };

// CHECK-LABEL: define {{.*}}@_Z12test_trivialPv(
// CHECK-NOT: call ptr @llvm.launder.invariant.group
// CHECK: ret ptr
Trivial* test_trivial(void* p) {
  return __builtin_start_lifetime_as((Trivial*)p);
}

// CHECK-LABEL: define {{.*}}@_Z18test_array_trivialPv(
// CHECK-NOT: call ptr @llvm.launder.invariant.group
// CHECK: ret ptr
int (*test_array_trivial(void* p))[5] {
  return __builtin_start_lifetime_as((int(*)[5])p);
}

struct WithConst { const int x; };
struct WithRef { int& x; };

// CHECK-LABEL: define {{.*}}@_Z10test_constPv(
// CHECK-NOT: call ptr @llvm.launder.invariant.group
// CHECK: ret ptr
WithConst* test_const(void* p) {
  return __builtin_start_lifetime_as((WithConst*)p);
}

// CHECK-LABEL: define {{.*}}@_Z8test_refPv(
// CHECK-NOT: call ptr @llvm.launder.invariant.group
// CHECK: ret ptr
WithRef* test_ref(void* p) {
  return __builtin_start_lifetime_as((WithRef*)p);
}

// CHECK-LABEL: define {{.*}}@_Z20test_strict_flag_laxPv(
// CHECK-NOT: call ptr @llvm.launder.invariant.group
// CHECK: ret ptr
WithConst* test_strict_flag_lax(void* p) {
  // Pass 'false' to lax mode; CodeGen cleanly drops the second argument
  return __builtin_start_lifetime_as((WithConst*)p, false);
}

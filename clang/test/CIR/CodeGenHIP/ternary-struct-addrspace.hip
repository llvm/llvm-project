// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o - | FileCheck %s

// Verify that a ternary expression returning a struct parameter from a
// __device__ function produces a well-typed cir.ternary: both branch yields
// and the result must all carry the same pointer type (no address-space
// mismatch between the private-address-space alloca and the cast result).

struct S { int a; };

__attribute__((device)) bool cmp(S, S);

__attribute__((device)) S choose(const S x) {
  S y = x;
  return cmp(x, y) ? x : y;
}


// CHECK: %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!rec_S, target_address_space(5)>
// CHECK: %[[Y:.*]] = cir.alloca "y" {{.*}} : !cir.ptr<!rec_S, target_address_space(5)> loc(#loc18)

// CHECK: %[[Y_CAST:.*]] = cir.cast address_space %[[Y]] : !cir.ptr<!rec_S, target_address_space(5)> -> !cir.ptr<!rec_S>
// CHECK: %[[X_CAST:.*]] = cir.cast address_space %[[X]] : !cir.ptr<!rec_S, target_address_space(5)> -> !cir.ptr<!rec_S>

// CHECK: %[[TERNARY:.*]] = cir.ternary({{.*}}, true {
// CHECK:   cir.yield %[[X_CAST]] : !cir.ptr<!rec_S>
// CHECK: }, false {
// CHECK:   cir.yield %[[Y_CAST]] : !cir.ptr<!rec_S>
// CHECK: }) : (!cir.bool) -> !cir.ptr<!rec_S>

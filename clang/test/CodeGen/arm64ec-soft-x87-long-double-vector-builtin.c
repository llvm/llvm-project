// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s -DOPTNONE | FileCheck %s --check-prefix=OPTNONE

// Element-wise floating-point builtins work on the vector values themselves,
// which stay real `<2 x x86_fp80>` in the IR; the backend splits and softens
// the lanes.
typedef long double V __attribute__((vector_size(32)));

#ifdef OPTNONE
// `optnone` does not change the representation.
__attribute__((optnone)) V reject(V a) {
  return __builtin_elementwise_abs(a);
}
// OPTNONE-LABEL: define {{.*}} @reject(
// OPTNONE: call <2 x x86_fp80> @llvm.fabs.v2f80
#else
V reject(V a, V b, V c) {
  V abs_a = __builtin_elementwise_abs(a);
  return __builtin_elementwise_fma(abs_a, b, c);
}
// CHECK-LABEL: define {{.*}} @reject(
// CHECK: call <2 x x86_fp80> @llvm.fabs.v2f80
// CHECK: call <2 x x86_fp80> @llvm.fma.v2f80
#endif

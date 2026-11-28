// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx77 -emit-llvm -fcuda-is-device -o - %s \
// RUN:   | FileCheck %s
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_60 \
// RUN:   -target-feature +ptx77 -emit-llvm -fcuda-is-device -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_70 \
// RUN:   -target-feature +ptx76 -emit-llvm -fcuda-is-device -o - %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR

// CHECK-LABEL: define{{.*}} @_Z4testPKv(
// CHECK: call i1 @llvm.nvvm.isspacep.grid_const(ptr
__attribute__((device)) unsigned int test(const void *p) {
  return __nvvm_isspacep_grid_const(p);
}

// ERROR: '__nvvm_isspacep_grid_const' needs target feature

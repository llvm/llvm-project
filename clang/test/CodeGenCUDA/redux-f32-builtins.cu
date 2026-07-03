// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda" "-target-feature" "+ptx86" "-target-cpu" "sm_100a" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 "-triple" "nvptx64-nvidia-cuda" "-target-feature" "+ptx86" "-target-cpu" "sm_100a" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

// CHECK: define{{.*}} void @_Z6kernelPf(ptr noundef %out_f)
__attribute__((global)) void kernel(float* out_f) {
  float a = 3.0;
  int i = 0;

  out_f[i++] = __nvvm_redux_sync_fmin(a, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin

  out_f[i++] = __nvvm_redux_sync_fmin_abs(a, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.abs

  out_f[i++] = __nvvm_redux_sync_fmin_NaN(a, 0xF0);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.NaN

  out_f[i++] = __nvvm_redux_sync_fmin_abs_NaN(a, 0x0F);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmin.abs.NaN

  out_f[i++] = __nvvm_redux_sync_fmax(a, 0xFF);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax

  out_f[i++] = __nvvm_redux_sync_fmax_abs(a, 0x01);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.abs

  out_f[i++] = __nvvm_redux_sync_fmax_NaN(a, 0xF1);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.NaN

  out_f[i++] = __nvvm_redux_sync_fmax_abs_NaN(a, 0x10);
  // CHECK: call contract float @llvm.nvvm.redux.sync.fmax.abs.NaN

  // CHECK: ret void
}

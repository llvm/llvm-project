// RUN: %clang_cc1 -no-opaque-pointers "-triple" "nvptx-nvidia-cuda" "-target-feature" "+ptx70" "-target-cpu" "sm_80" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers "-triple" "nvptx64-nvidia-cuda" "-target-feature" "+ptx70" "-target-cpu" "sm_80" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

// CHECK: define{{.*}} void @_Z6kernelPi(i32* noundef %out)
__attribute__((global)) void kernel(int *out) {
  int a = 1;
  unsigned int b = 5;
  int i = 0;

  out[i++] = __nvvm_redux_sync_add(a, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.add

  out[i++] = __nvvm_redux_sync_add(b, 0x01);
  // CHECK: call i32 @llvm.nvvm.redux.sync.add

  out[i++] = __nvvm_redux_sync_min(a, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.min

  out[i++] = __nvvm_redux_sync_umin(b, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.umin

  out[i++] = __nvvm_redux_sync_max(a, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.max

  out[i++] = __nvvm_redux_sync_umax(b, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.umax

  out[i++] = __nvvm_redux_sync_and(a, 0xF0);
  // CHECK: call i32 @llvm.nvvm.redux.sync.and

  out[i++] = __nvvm_redux_sync_and(b, 0x0F);
  // CHECK: call i32 @llvm.nvvm.redux.sync.and

  out[i++] = __nvvm_redux_sync_xor(a, 0x10);
  // CHECK: call i32 @llvm.nvvm.redux.sync.xor

  out[i++] = __nvvm_redux_sync_xor(b, 0x01);
  // CHECK: call i32 @llvm.nvvm.redux.sync.xor

  out[i++] = __nvvm_redux_sync_or(a, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.or

  out[i++] = __nvvm_redux_sync_or(b, 0xFF);
  // CHECK: call i32 @llvm.nvvm.redux.sync.or

  // CHECK: ret void
}

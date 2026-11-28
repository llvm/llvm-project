// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=DEVICE,SPIRV
// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=DEVICE,NVPTX
// RUN: %clang_cc1 -fsycl-is-device -triple amdgcn-amd-amdhsa \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=DEVICE,AMDGPU
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -disable-llvm-passes -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST

// Check that on the device all calls to __builtin_sycl_dynamic_local_memory
// refer to a single external zero-length array in local memory, and that on
// the host the global is not emitted.

// DEVICE: @__sycl_dynamic_local_memory = external addrspace(3) global [0 x i8]
// DEVICE-NOT: @__sycl_dynamic_local_memory{{.*}} = external

// HOST-NOT: __sycl_dynamic_local_memory

using LocalPtr = unsigned char [[clang::sycl_local]] *;

[[clang::sycl_external]] LocalPtr get() {
  return __builtin_sycl_dynamic_local_memory();
}
// DEVICE-LABEL: define {{.*}}ptr addrspace(3) @_Z3getv()
// DEVICE: ret ptr addrspace(3) @__sycl_dynamic_local_memory

[[clang::sycl_external]] void casts() {
  // DEVICE-LABEL: define {{.*}}void @_Z5castsv()

  LocalPtr Local = __builtin_sycl_dynamic_local_memory();
  // DEVICE: store ptr addrspace(3) @__sycl_dynamic_local_memory, ptr {{.*}}%Local

  float [[clang::sycl_local]] *LocalFloat =
      reinterpret_cast<float [[clang::sycl_local]] *>(
          __builtin_sycl_dynamic_local_memory());
  // DEVICE: store ptr addrspace(3) @__sycl_dynamic_local_memory, ptr {{.*}}%LocalFloat

  unsigned char *Generic =
      static_cast<unsigned char *>(__builtin_sycl_dynamic_local_memory());
  // SPIRV: store ptr addrspace(4) addrspacecast (ptr addrspace(3) @__sycl_dynamic_local_memory to ptr addrspace(4)), ptr addrspace(4) %Generic
  // NVPTX: store ptr addrspacecast (ptr addrspace(3) @__sycl_dynamic_local_memory to ptr), ptr %Generic
  // AMDGPU: store ptr addrspacecast (ptr addrspace(3) @__sycl_dynamic_local_memory to ptr), ptr %Generic
}

#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 -fclangir \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -std=c++11 \
// RUN:            -target-cpu gfx1100 -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct BufferResourceHolder {
  int x;
  __amdgpu_buffer_rsrc_t r;
};

__device__ void consume_buffer(__amdgpu_buffer_rsrc_t);
__device__ __amdgpu_buffer_rsrc_t make_resource();

// CIR-LABEL: cir.func {{.*}}test_buffer_rsrc_passthrough
// CIR-SAME: !cir.ptr<!void, target_address_space(8)>
// CIR-SAME: -> !cir.ptr<!void, target_address_space(8)>
// LLVM-LABEL: define{{.*}} ptr addrspace(8) @{{.*}}test_buffer_rsrc_passthrough
__device__ __amdgpu_buffer_rsrc_t
test_buffer_rsrc_passthrough(__amdgpu_buffer_rsrc_t rsrc) {
  return rsrc;
}

// CIR-LABEL: cir.func {{.*}}test_buffer_rsrc_load
// CIR: cir.load {{.*}} : !cir.ptr<!cir.ptr<!void, target_address_space(8)>>, !cir.ptr<!void, target_address_space(8)>
// LLVM-LABEL: define{{.*}} ptr addrspace(8) @{{.*}}test_buffer_rsrc_load
__device__ __amdgpu_buffer_rsrc_t
test_buffer_rsrc_load(__amdgpu_buffer_rsrc_t *p) {
  return *p;
}

// CIR-LABEL: cir.func {{.*}}test_buffer_rsrc_store
// CIR: cir.store{{.*}} : !cir.ptr<!void, target_address_space(8)>,
// LLVM-LABEL: define{{.*}}@{{.*}}test_buffer_rsrc_store
__device__ void
test_buffer_rsrc_store(__amdgpu_buffer_rsrc_t *p, __amdgpu_buffer_rsrc_t rsrc) {
  *p = rsrc;
}

// CIR-LABEL: cir.func {{.*}}test_struct_member
// CIR: cir.get_member {{.*}} {name = "r"} {{.*}} -> !cir.ptr<!cir.ptr<!void, target_address_space(8)>>
// CIR: cir.load {{.*}} : !cir.ptr<!cir.ptr<!void, target_address_space(8)>>, !cir.ptr<!void, target_address_space(8)>
// LLVM-LABEL: define{{.*}} ptr addrspace(8) @{{.*}}test_struct_member
__device__ __amdgpu_buffer_rsrc_t test_struct_member(BufferResourceHolder *a) {
  return a->r;
}

// CIR-LABEL: cir.func {{.*}}test_pass_by_value
// CIR: cir.call {{.*}}consume_buffer{{.*}}!cir.ptr<!void, target_address_space(8)>
// LLVM-LABEL: define{{.*}}@{{.*}}test_pass_by_value
// LLVM: call void @{{.*}}consume_buffer{{.*}}(ptr addrspace(8)
__device__ void test_pass_by_value(__amdgpu_buffer_rsrc_t rsrc) {
  consume_buffer(rsrc);
}

// CIR-LABEL: cir.func {{.*}}test_call_returns_resource
// CIR: cir.call {{.*}}make_resource{{.*}} -> !cir.ptr<!void, target_address_space(8)>
// LLVM-LABEL: define{{.*}} ptr addrspace(8) @{{.*}}test_call_returns_resource
__device__ __amdgpu_buffer_rsrc_t test_call_returns_resource() {
  return make_resource();
}

// CIR-LABEL: cir.func {{.*}}test_return_struct
// CIR: cir.get_member {{.*}} {name = "r"} {{.*}} -> !cir.ptr<!cir.ptr<!void, target_address_space(8)>>
// LLVM-LABEL: define{{.*}}@{{.*}}test_return_struct
__device__ BufferResourceHolder test_return_struct(__amdgpu_buffer_rsrc_t rsrc) {
  BufferResourceHolder a;
  a.x = 0;
  a.r = rsrc;
  return a;
}

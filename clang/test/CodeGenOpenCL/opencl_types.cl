// RUN: %clang_cc1 -cl-std=CL2.0 %s -triple "spir-unknown-unknown" -emit-llvm -o - -O0 | FileCheck %s --check-prefixes=CHECK-COM,CHECK-SPIR
// RUN: %clang_cc1 -cl-std=CL2.0 %s -triple "amdgcn--amdhsa" -emit-llvm -o - -O0 | FileCheck %s --check-prefixes=CHECK-COM,CHECK-AMDGCN

#define CLK_ADDRESS_CLAMP_TO_EDGE       2
#define CLK_NORMALIZED_COORDS_TRUE      1
#define CLK_FILTER_NEAREST              0x10
#define CLK_FILTER_LINEAR               0x20

constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE|CLK_NORMALIZED_COORDS_TRUE|CLK_FILTER_NEAREST;
// CHECK-COM-NOT: constant i32

void fnc1(image1d_t img) {}
// CHECK-SPIR: @fnc1(%opencl.image1d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1(%opencl.image1d_ro_t addrspace(4)*

void fnc1arr(image1d_array_t img) {}
// CHECK-SPIR: @fnc1arr(%opencl.image1d_array_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1arr(%opencl.image1d_array_ro_t addrspace(4)*

void fnc1buff(image1d_buffer_t img) {}
// CHECK-SPIR: @fnc1buff(%opencl.image1d_buffer_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1buff(%opencl.image1d_buffer_ro_t addrspace(4)*

void fnc2(image2d_t img) {}
// CHECK-SPIR: @fnc2(%opencl.image2d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc2(%opencl.image2d_ro_t addrspace(4)*

void fnc2arr(image2d_array_t img) {}
// CHECK-SPIR: @fnc2arr(%opencl.image2d_array_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc2arr(%opencl.image2d_array_ro_t addrspace(4)*

void fnc3(image3d_t img) {}
// CHECK-SPIR: @fnc3(%opencl.image3d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc3(%opencl.image3d_ro_t addrspace(4)*

void fnc4smp(sampler_t s) {}
// CHECK-SPIR-LABEL: define {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(2)*
// CHECK-AMDGCN-LABEL: define {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(4)*

kernel void foo(image1d_t img) {
  sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE|CLK_NORMALIZED_COORDS_TRUE|CLK_FILTER_LINEAR;
  // CHECK-SPIR: alloca %opencl.sampler_t addrspace(2)*
  // CHECK-AMDGCN: alloca %opencl.sampler_t addrspace(4)*
  event_t evt;
  // CHECK-SPIR: alloca %opencl.event_t*
  // CHECK-AMDGCN: alloca %opencl.event_t addrspace(5)*
  clk_event_t clk_evt;
  // CHECK-SPIR: alloca %opencl.clk_event_t*
  // CHECK-AMDGCN: alloca %opencl.clk_event_t addrspace(1)*
  queue_t queue;
  // CHECK-SPIR: alloca %opencl.queue_t*
  // CHECK-AMDGCN: alloca %opencl.queue_t addrspace(1)*
  reserve_id_t rid;
  // CHECK-SPIR: alloca %opencl.reserve_id_t*
  // CHECK-AMDGCN: alloca %opencl.reserve_id_t addrspace(1)*
  // CHECK-SPIR: store %opencl.sampler_t addrspace(2)*
  // CHECK-AMDGCN: store %opencl.sampler_t addrspace(4)*
  fnc4smp(smp);
  // CHECK-SPIR: call {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(2)*
  // CHECK-AMDGCN: call {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(4)*
  fnc4smp(glb_smp);
  // CHECK-SPIR: call {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(2)*
  // CHECK-AMDGCN: call {{.*}}void @fnc4smp(%opencl.sampler_t addrspace(4)*
}

kernel void foo_ro_pipe(read_only pipe int p) {}
// CHECK-SPIR: @foo_ro_pipe(%opencl.pipe_ro_t addrspace(1)* %p)
// CHECK_AMDGCN: @foo_ro_pipe(%opencl.pipe_ro_t addrspace(1)* %p)

kernel void foo_wo_pipe(write_only pipe int p) {}
// CHECK-SPIR: @foo_wo_pipe(%opencl.pipe_wo_t addrspace(1)* %p)
// CHECK_AMDGCN: @foo_wo_pipe(%opencl.pipe_wo_t addrspace(1)* %p)

void __attribute__((overloadable)) bad1(image1d_t b, image2d_t c, image2d_t d) {}
// CHECK-SPIR-LABEL: @{{_Z4bad114ocl_image1d_ro14ocl_image2d_roS0_|"\\01\?bad1@@\$\$J0YAXPAUocl_image1d_ro@@PAUocl_image2d_ro@@1@Z"}}
// CHECK-AMDGCN-LABEL: @{{_Z4bad114ocl_image1d_ro14ocl_image2d_roS0_|"\\01\?bad1@@\$\$J0YAXPAUocl_image1d_ro@@PAUocl_image2d_ro@@1@Z"}}(%opencl.image1d_ro_t addrspace(4)*{{.*}}%opencl.image2d_ro_t addrspace(4)*{{.*}}%opencl.image2d_ro_t addrspace(4)*{{.*}})

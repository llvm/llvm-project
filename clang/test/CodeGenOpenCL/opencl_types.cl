// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - -O0 | FileCheck %s --check-prefix=CHECK-SPIR
// RUN: %clang_cc1 %s -triple "amdgcn--amdhsa" -emit-llvm -o - -O0 | FileCheck %s --check-prefix=CHECK-AMDGCN

constant sampler_t glb_smp = 7;
// CHECK-SPIR: constant i32 7
// CHECK-AMDGCN: addrspace(2) constant i32 7

void fnc1(image1d_t img) {}
// CHECK-SPIR: @fnc1(%opencl.image1d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1(%opencl.image1d_ro_t addrspace(2)*

void fnc1arr(image1d_array_t img) {}
// CHECK-SPIR: @fnc1arr(%opencl.image1d_array_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1arr(%opencl.image1d_array_ro_t addrspace(2)*

void fnc1buff(image1d_buffer_t img) {}
// CHECK-SPIR: @fnc1buff(%opencl.image1d_buffer_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc1buff(%opencl.image1d_buffer_ro_t addrspace(2)*

void fnc2(image2d_t img) {}
// CHECK-SPIR: @fnc2(%opencl.image2d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc2(%opencl.image2d_ro_t addrspace(2)*

void fnc2arr(image2d_array_t img) {}
// CHECK-SPIR: @fnc2arr(%opencl.image2d_array_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc2arr(%opencl.image2d_array_ro_t addrspace(2)*

void fnc3(image3d_t img) {}
// CHECK-SPIR: @fnc3(%opencl.image3d_ro_t addrspace(1)*
// CHECK-AMDGCN: @fnc3(%opencl.image3d_ro_t addrspace(2)*

void fnc4smp(sampler_t s) {}
// CHECK-SPIR-LABEL: define {{.*}}void @fnc4smp(i32

kernel void foo(image1d_t img) {
  sampler_t smp = 5;
  // CHECK-SPIR: alloca i32
  event_t evt;
  // CHECK-SPIR: alloca %opencl.event_t*
  // CHECK-SPIR: store i32 5,
  fnc4smp(smp);
  // CHECK-SPIR: call {{.*}}void @fnc4smp(i32
  fnc4smp(glb_smp);
  // CHECK-SPIR: call {{.*}}void @fnc4smp(i32
}

void __attribute__((overloadable)) bad1(image1d_t b, image2d_t c, image2d_t d) {}
// CHECK-SPIR-LABEL: @{{_Z4bad114ocl_image1d_ro14ocl_image2d_roS0_|"\\01\?bad1@@\$\$J0YAXPAUocl_image1d_ro@@PAUocl_image2d_ro@@1@Z"}}
// CHECK-AMDGCN-LABEL: @{{_Z4bad114ocl_image1d_ro14ocl_image2d_roS0_|"\\01\?bad1@@\$\$J0YAXPAUocl_image1d_ro@@PAUocl_image2d_ro@@1@Z"}}(%opencl.image1d_ro_t addrspace(2)*{{.*}}%opencl.image2d_ro_t addrspace(2)*{{.*}}%opencl.image2d_ro_t addrspace(2)*{{.*}})

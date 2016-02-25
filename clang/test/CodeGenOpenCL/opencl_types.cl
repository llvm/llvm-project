// RUN: %clang_cc1 %s -emit-llvm -o - -O0 | FileCheck %s

constant sampler_t glb_smp = 7;
// CHECK: constant i32 7

void fnc1(image1d_t img) {}
// CHECK: @fnc1(%opencl.image1d_t*

void fnc1arr(image1d_array_t img) {}
// CHECK: @fnc1arr(%opencl.image1d_array_t*

void fnc1buff(image1d_buffer_t img) {}
// CHECK: @fnc1buff(%opencl.image1d_buffer_t*

void fnc2(image2d_t img) {}
// CHECK: @fnc2(%opencl.image2d_t*

void fnc2arr(image2d_array_t img) {}
// CHECK: @fnc2arr(%opencl.image2d_array_t*

void fnc3(image3d_t img) {}
// CHECK: @fnc3(%opencl.image3d_t*

void fnc4smp(sampler_t s) {}
// CHECK-LABEL: define {{.*}}void @fnc4smp(i32

kernel void foo(image1d_t img) {
	sampler_t smp = 5;
// CHECK: alloca i32
	event_t evt;
// CHECK: alloca %opencl.event_t*
// CHECK: store i32 5,
  fnc4smp(smp);
// CHECK: call {{.*}}void @fnc4smp(i32
  fnc4smp(glb_smp);
// CHECK: call {{.*}}void @fnc4smp(i32
}

void __attribute__((overloadable)) bad1(image1d_t b, image2d_t c, image2d_t d) {}
// CHECK-LABEL: @{{_Z4bad111ocl_image1d11ocl_image2dS0_|"\\01\?bad1@@\$\$J0YAXPAUocl_image1d@@PAUocl_image2d@@1@Z"}}

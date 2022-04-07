// RUN: %clang_cc1 -no-opaque-pointers %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - -cl-std=clc++ | FileCheck %s

__attribute__((overloadable)) void read_image(read_only image1d_t img_ro);
__attribute__((overloadable)) void read_image(write_only image1d_t img_wo);

kernel void test_read_image(read_only image1d_t img_ro, write_only image1d_t img_wo) {
  // CHECK: call void @_Z10read_image14ocl_image1d_ro(%opencl.image1d_ro_t* %{{[0-9]+}})
  read_image(img_ro);
  // CHECK: call void @_Z10read_image14ocl_image1d_wo(%opencl.image1d_wo_t* %{{[0-9]+}})
  read_image(img_wo);
}

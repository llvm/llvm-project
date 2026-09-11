// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL30-CIR %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++ -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX-CIR %s

// CL30-CIR: cir.lang = #cir.lang<opencl_c>
// CLCXX-CIR: cir.lang = #cir.lang<opencl_cxx>

__kernel void lang_marker(__global int *out) {
  out[0] = 1;
}

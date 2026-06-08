// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL12-CIR %s
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL30-CIR %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++ -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX10-CIR %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++2021 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX2021-CIR %s

// CL12-CIR: cir.cl.version = #cir.cl.version<1, 2>
// CL30-CIR: cir.cl.version = #cir.cl.version<3, 0>
// CLCXX10-CIR-DAG: cir.cl.cxx.version = #cir.cl.version<1, 0>
// CLCXX10-CIR-DAG: cir.cl.version = #cir.cl.version<2, 0>
// CLCXX2021-CIR-DAG: cir.cl.cxx.version = #cir.cl.version<2021, 0>
// CLCXX2021-CIR-DAG: cir.cl.version = #cir.cl.version<3, 0>

__kernel void version_marker(__global int *out) {
  out[0] = 1;
}

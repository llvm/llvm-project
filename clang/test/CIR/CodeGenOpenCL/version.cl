// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL12-CIR %s
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL30-CIR %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++ -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX10-CIR %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++2021 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX2021-CIR %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL12-LLVM %s
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CL30-LLVM %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++ -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX10-LLVM %s
// RUN: %clang_cc1 -x clcpp -cl-std=CLC++2021 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=CLCXX2021-LLVM %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-cir -triple spir-unknown-unknown %s -o - | FileCheck --check-prefix=SPIR12-CIR %s
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-cir -triple spir64-unknown-unknown %s -o - | FileCheck --check-prefix=SPIR30-CIR %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-llvm -triple spir-unknown-unknown %s -o - | FileCheck --check-prefix=SPIR12-LLVM %s
// RUN: %clang_cc1 -cl-std=CL3.0 -fclangir -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck --check-prefix=SPIR30-LLVM %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=SPIRV-CIR %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o - | FileCheck --check-prefix=SPIRV-LLVM %s

// CL12-CIR: cir.cl.version = #cir.cl.version<1, 2>
// CL30-CIR: cir.cl.version = #cir.cl.version<3, 0>
// CLCXX10-CIR-DAG: cir.cl.cxx.version = #cir.cl.version<1, 0>
// CLCXX10-CIR-DAG: cir.cl.version = #cir.cl.version<2, 0>
// CLCXX2021-CIR-DAG: cir.cl.cxx.version = #cir.cl.version<2021, 0>
// CLCXX2021-CIR-DAG: cir.cl.version = #cir.cl.version<3, 0>
// CL12-LLVM-DAG: !opencl.ocl.version = !{[[CL12_VERSION:![0-9]+]]}
// CL12-LLVM-DAG: [[CL12_VERSION]] = !{i32 1, i32 2}
// CL30-LLVM-DAG: !opencl.ocl.version = !{[[CL30_VERSION:![0-9]+]]}
// CL30-LLVM-DAG: [[CL30_VERSION]] = !{i32 3, i32 0}
// CLCXX10-LLVM-DAG: !opencl.ocl.version = !{[[CLCXX10_VERSION:![0-9]+]]}
// CLCXX10-LLVM-DAG: !opencl.cxx.version = !{[[CLCXX10_CXX_VERSION:![0-9]+]]}
// CLCXX10-LLVM-DAG: [[CLCXX10_VERSION]] = !{i32 2, i32 0}
// CLCXX10-LLVM-DAG: [[CLCXX10_CXX_VERSION]] = !{i32 1, i32 0}
// CLCXX2021-LLVM-DAG: !opencl.ocl.version = !{[[CLCXX2021_VERSION:![0-9]+]]}
// CLCXX2021-LLVM-DAG: !opencl.cxx.version = !{[[CLCXX2021_CXX_VERSION:![0-9]+]]}
// CLCXX2021-LLVM-DAG: [[CLCXX2021_VERSION]] = !{i32 3, i32 0}
// CLCXX2021-LLVM-DAG: [[CLCXX2021_CXX_VERSION]] = !{i32 2021, i32 0}
// SPIR12-CIR: cir.cl.spir.version = #cir.cl.version<1, 2>
// SPIR30-CIR: cir.cl.spir.version = #cir.cl.version<3, 0>
// SPIR12-LLVM-DAG: !opencl.spir.version = !{[[SPIR12_VERSION:![0-9]+]]}
// SPIR12-LLVM-DAG: [[SPIR12_VERSION]] = !{i32 1, i32 2}
// SPIR30-LLVM-DAG: !opencl.spir.version = !{[[SPIR30_VERSION:![0-9]+]]}
// SPIR30-LLVM-DAG: [[SPIR30_VERSION]] = !{i32 3, i32 0}
// SPIRV-CIR-NOT: cir.cl.spir.version
// SPIRV-LLVM-NOT: !opencl.spir.version

__kernel void version_marker(__global int *out) {
  out[0] = 1;
}

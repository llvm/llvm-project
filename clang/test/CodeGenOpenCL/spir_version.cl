// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - | FileCheck %s --check-prefix=CL10
// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - -cl-std=CL1.2 | FileCheck %s --check-prefix=CL12
// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - -cl-std=CL2.0 | FileCheck %s --check-prefix=CL20
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - | FileCheck %s --check-prefix=CL10
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - -cl-std=CL1.2 | FileCheck %s --check-prefix=CL12
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - -cl-std=CL2.0 | FileCheck %s --check-prefix=CL20
kernel void foo() {}
// CL10: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CL10: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CL10: [[SPIR]] = !{i32 2, i32 0}
// CL10: [[OCL]] = !{i32 1, i32 0}
// CL12: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CL12: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CL12: [[SPIR]] = !{i32 2, i32 0}
// CL12: [[OCL]] = !{i32 1, i32 2}
// CL20: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CL20: !opencl.ocl.version = !{[[SPIR:![0-9]+]]}
// CL20: [[SPIR]] = !{i32 2, i32 0}

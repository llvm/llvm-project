// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SPIR-CL10
// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - -cl-std=CL1.2 | FileCheck %s --check-prefix=CHECK-SPIR-CL12
// RUN: %clang_cc1 %s -triple "spir-unknown-unknown" -emit-llvm -o - -cl-std=CL2.0 | FileCheck %s --check-prefix=CHECK-SPIR-CL20
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SPIR-CL10
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - -cl-std=CL1.2 | FileCheck %s --check-prefix=CHECK-SPIR-CL12
// RUN: %clang_cc1 %s -triple "spir64-unknown-unknown" -emit-llvm -o - -cl-std=CL2.0 | FileCheck %s --check-prefix=CHECK-SPIR-CL20

// RUN: %clang_cc1 %s -triple "amdgcn--amdhsa" -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-AMDGCN-CL10
// RUN: %clang_cc1 %s -triple "amdgcn--amdhsa" -emit-llvm -o - -cl-std=CL1.2 | FileCheck %s --check-prefix=CHECK-AMDGCN-CL12
// RUN: %clang_cc1 %s -triple "amdgcn--amdhsa" -emit-llvm -o - -cl-std=CL2.0 | FileCheck %s --check-prefix=CHECK-AMDGCN-CL20

kernel void foo() {}
kernel void bar() {}

// CHECK-SPIR-CL10-DAG: !opencl.spir.version = !{[[SPIR:![0-9]+]]}
// CHECK-SPIR-CL10-DAG: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK-SPIR-CL10-DAG: [[SPIR]] = !{i32 1, i32 2}
// CHECK-SPIR-CL10-DAG: [[OCL]] = !{i32 1, i32 0}
// CHECK-SPIR-CL12-DAG: !opencl.spir.version = !{[[VER:![0-9]+]]}
// CHECK-SPIR-CL12-DAG: !opencl.ocl.version = !{[[VER]]}
// CHECK-SPIR-CL12: [[VER]] = !{i32 1, i32 2}

// CHECK-SPIR-CL20-DAG: !opencl.spir.version = !{[[VER:![0-9]+]]}
// CHECK-SPIR-CL20-DAG: !opencl.ocl.version = !{[[VER]]}
// CHECK-SPIR-CL20: [[VER]] = !{i32 2, i32 0}

// CHECK-AMDGCN-CL10-NOT: !opencl.spir.version
// CHECK-AMDGCN-CL10: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK-AMDGCN-CL10: [[OCL]] = !{i32 1, i32 0}
// CHECK-AMDGCN-CL12-NOT: !opencl.spir.version
// CHECK-AMDGCN-CL12: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK-AMDGCN-CL12: [[OCL]] = !{i32 1, i32 2}
// CHECK-AMDGCN-CL20-NOT: !opencl.spir.version
// CHECK-AMDGCN-CL20: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK-AMDGCN-CL20: [[OCL]] = !{i32 2, i32 0}

// Create module flag for code object version.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -o - %s | FileCheck %s -check-prefix=V4

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=2 -o - %s | FileCheck -check-prefix=V2 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=3 -o - %s | FileCheck -check-prefix=V3 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=4 -o - %s | FileCheck -check-prefix=V4 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=5 -o - %s | FileCheck -check-prefix=V5 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=none -o - %s | FileCheck %s -check-prefix=NONE

// RUN: not %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=4.1 -o - %s 2>&1| FileCheck %s -check-prefix=INV

// V2: !{{.*}} = !{i32 1, !"amdgpu_code_object_version", i32 200}
// V3: !{{.*}} = !{i32 1, !"amdgpu_code_object_version", i32 300}
// V4: !{{.*}} = !{i32 1, !"amdgpu_code_object_version", i32 400}
// V5: !{{.*}} = !{i32 1, !"amdgpu_code_object_version", i32 500}
// NONE-NOT: !{{.*}} = !{i32 1, !"amdgpu_code_object_version",
// INV: error: invalid value '4.1' in '-mcode-object-version=4.1'

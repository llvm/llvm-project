// Create module flag for printf kind.

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -o - %s | FileCheck %s -check-prefix=HOSTCALL

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -mprintf-kind=hostcall -o - %s | FileCheck %s -check-prefix=HOSTCALL

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -mprintf-kind=buffered -o - %s | FileCheck -check-prefix=BUFFERED %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN: -mprintf-kind=none -o - %s 2>&1| FileCheck %s -check-prefix=NONE

// HOSTCALL: !{{.*}} = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
// BUFFERED: !{{.*}} = !{i32 1, !"amdgpu_printf_kind", !"buffered"}
// NONE: !{{.*}} = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}

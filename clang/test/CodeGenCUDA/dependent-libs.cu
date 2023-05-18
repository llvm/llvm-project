// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - -fcuda-is-device -x hip %s | FileCheck --check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - -x hip %s | FileCheck --check-prefix=HOST %s

// DEV-NOT: llvm.dependent-libraries
// HOST: llvm.dependent-libraries
#pragma comment(lib, "libabc")

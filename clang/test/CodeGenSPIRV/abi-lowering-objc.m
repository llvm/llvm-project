// Check that the LLVM ABI library classifies Objective-C object pointers the
// same way as Clang's own ABI lowering. They are address space 0 pointers, so
// a kernel argument is only coerced where that is the default address space.

// RUN: %clang_cc1 -triple spirv64 -fobjc-runtime=gnustep-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,PRIV
// RUN: %clang_cc1 -triple spirv64 -fobjc-runtime=gnustep-2.0 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,PRIV
// RUN: %clang_cc1 -triple spirv64-intel -fobjc-runtime=gnustep-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GEN
// RUN: %clang_cc1 -triple spirv64-intel -fobjc-runtime=gnustep-2.0 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GEN

// CHECK: define spir_func ptr @pass_id(ptr noundef %x)
id pass_id(id x) { return x; }

// PRIV: define spir_kernel void @k_id(ptr addrspace(1) noundef %x.coerce)
// GEN:  define spir_kernel void @k_id(ptr noundef %x)
__attribute__((device_kernel)) void k_id(id x) {}

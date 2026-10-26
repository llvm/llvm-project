// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=GPR64
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -cl-std=CL2.0 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=GPR64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=ILP32
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -cl-std=CL2.0 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=ILP32 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=GPR64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -cl-std=CL2.0 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=GPR64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=GPR64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -cl-std=CL2.0 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=GPR64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -cl-std=CL2.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=GPR64
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -cl-std=CL2.0 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=GPR64 --implicit-check-not="not yet implemented"

// OpenCL pointer pointees have an address-space qualifier. On AArch64 those
// language address spaces lower to address space 0, and a record of them is
// coerced to an integer.

typedef struct {
  __global int *p;
} GlobalPtr;
void arg_global_ptr(GlobalPtr s) {}
// GPR64: define{{.*}} void @arg_global_ptr(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_global_ptr(i32 %{{.*}})

typedef struct {
  __local int *p;
} LocalPtr;
void arg_local_ptr(LocalPtr s) {}
// GPR64: define{{.*}} void @arg_local_ptr(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_local_ptr(i32 %{{.*}})

typedef struct {
  __constant int *p;
} ConstantPtr;
void arg_constant_ptr(ConstantPtr s) {}
// GPR64: define{{.*}} void @arg_constant_ptr(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_constant_ptr(i32 %{{.*}})

typedef struct {
  __private int *p;
} PrivatePtr;
void arg_private_ptr(PrivatePtr s) {}
// GPR64: define{{.*}} void @arg_private_ptr(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_private_ptr(i32 %{{.*}})

typedef struct {
  __generic int *p;
} GenericPtr;
void arg_generic_ptr(GenericPtr s) {}
// GPR64: define{{.*}} void @arg_generic_ptr(i64 %{{.*}})
// ILP32: define{{.*}} void @arg_generic_ptr(i32 %{{.*}})

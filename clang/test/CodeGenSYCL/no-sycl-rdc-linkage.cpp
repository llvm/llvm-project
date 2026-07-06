// RUN: %clang_cc1 -fsycl-is-device -triple spirv64 -fgpu-rdc -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,RDC %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64 -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK,NORDC %s

// This test verifies the linkage of SYCL device symbols.
//  * In RDC mode non-exported device functions get linkonce_odr
//    linkage so they can be deduplicated across translation units.
//  * In non-RDC mode they get internal linkage.
//  * Functions marked [[clang::sycl_external]] stay externally visible in both modes.

// External linkage prints as nothing in LLVM IR, so the 'exported' check
// asserts by omission.
// CHECK-DAG: define{{ }}spir_func noundef i32 @_Z8exportedi

// RDC-DAG:   define linkonce_odr spir_func noundef i32 @_Z6helperi
// NORDC-DAG: define internal spir_func noundef i32 @_Z6helperi

int helper(int x) { return x * 2; }

[[clang::sycl_external]] int exported(int x) { return helper(x); }

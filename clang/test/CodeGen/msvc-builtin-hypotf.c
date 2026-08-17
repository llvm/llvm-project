// Test that calling builtin math library functions directly (without including standard headers
// like <math.h>) on Windows MSVC targets correctly resolves MSVC-specific CRT symbol names
// and DLL import storage classes.
//
// 1. On Windows MSVC (including x86, x64, ARM, and ARM64), __builtin_hypotf must be resolved with
//    an underscore prefix (i.e. _hypotf) because it is not exported without a prefix in the MSVC CRT.
// 2. Under dynamic CRT configurations (/MD or -D_DLL), standard library functions reside
//    in runtime DLLs. Thus, these freestanding builtin calls must inherit dllimport attributes
//    and have their dso_local attributes cleared to avoid unresolved external symbol errors
//    at link time.
// 3. Under static CRT configurations (/MT), these functions should not have dllimport
//    attributes, but MSVC targets still require the underscore prefix mapping for _hypotf.

// RUN: %clang_cc1 -triple i686-pc-windows-msvc -fdeclspec -D_DLL -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X86-DLL
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -D_DLL -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-DLL
// RUN: %clang_cc1 -triple thumbv7-pc-windows-msvc -fdeclspec -D_DLL -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-DLL
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fdeclspec -D_DLL -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-DLL
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X86-STATIC
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-STATIC
// RUN: %clang_cc1 -triple thumbv7-pc-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-STATIC
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-X64-STATIC


float test_hypotf(float x, float y) {
  return __builtin_hypotf(x, y);
}

// CHECK-X86-DLL: declare dllimport {{.*}}float @_hypotf(float noundef, float noundef)
// CHECK-X64-DLL: declare dllimport {{.*}}float @_hypotf(float noundef, float noundef)
// CHECK-X86-STATIC: declare dso_local {{.*}}float @_hypotf(float noundef, float noundef)
// CHECK-X64-STATIC: declare dso_local {{.*}}float @_hypotf(float noundef, float noundef)




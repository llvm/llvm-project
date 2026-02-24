// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=MS

// Verify Itanium C++ name mangling for __int256_t / __uint256_t.
// These use vendor-extended type mangling since there are no standard
// single-letter codes for 256-bit integers (unlike 'n'/'o' for 128-bit).

// Verify Microsoft C++ name mangling for __int256_t / __uint256_t.
// These use $$_L / $$_M (extending the _L / _M pattern for __int128).

// ITANIUM-LABEL: define{{.*}} void @_Z3f01u7__int256u8__uint256
// MS-LABEL: define{{.*}} void @"?f01@@YAX$$_L$$_M@Z"
void f01(__int256_t, __uint256_t) {}

// ITANIUM-LABEL: define{{.*}} void @_Z3f02no
// MS-LABEL: define{{.*}} void @"?f02@@YAX_L_M@Z"
void f02(__int128_t, __uint128_t) {}

// Overloading: __int256_t vs __int128_t should produce different manglings
// ITANIUM-LABEL: define{{.*}} void @_Z3f03n
// MS-LABEL: define{{.*}} void @"?f03@@YAX_L@Z"
void f03(__int128_t) {}
// ITANIUM-LABEL: define{{.*}} void @_Z3f03u7__int256
// MS-LABEL: define{{.*}} void @"?f03@@YAX$$_L@Z"
void f03(__int256_t) {}

// ITANIUM-LABEL: define{{.*}} void @_Z3f04o
// MS-LABEL: define{{.*}} void @"?f04@@YAX_M@Z"
void f04(__uint128_t) {}
// ITANIUM-LABEL: define{{.*}} void @_Z3f04u8__uint256
// MS-LABEL: define{{.*}} void @"?f04@@YAX$$_M@Z"
void f04(__uint256_t) {}

// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32 --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies return types in
// the same way that Clang does without the library.

// The AArch64 support in the ABI library is a work in progress. New test cases
// will be added here as the types are implemented. Unimplemented cases will
// report a warning if the ABI library is used.

void ret_void() {}
// CHECK: define{{.*}} void @ret_void

_Bool ret_bool() { return 1; }
// AAPCS: define{{.*}} i1 @ret_bool
// DARWIN: define{{.*}} zeroext i1 @ret_bool

char ret_char() { return 'a'; }
// AAPCS: define{{.*}} i8 @ret_char
// DARWIN: define{{.*}} signext i8 @ret_char

short ret_short() { return 1; }
// AAPCS: define{{.*}} i16 @ret_short
// DARWIN: define{{.*}} signext i16 @ret_short

unsigned short ret_ushort() { return 1; }
// AAPCS: define{{.*}} i16 @ret_ushort
// DARWIN: define{{.*}} zeroext i16 @ret_ushort

int ret_int() { return 1; }
// CHECK: define{{.*}} i32 @ret_int

unsigned int ret_uint() { return 1; }
// CHECK: define{{.*}} i32 @ret_uint

long int ret_long() { return 1; }
// LONG64: define{{.*}} i64 @ret_long
// LONG32: define{{.*}} i32 @ret_long

_Float16 ret_float16() { return (_Float16)1.0f; }
// CHECK: define{{.*}} half @ret_float16

__fp16 ret_fp16() { return (__fp16)1.0f; }
// CHECK: define{{.*}} half @ret_fp16

float ret_float() { return 1.0f; }
// CHECK: define{{.*}} float @ret_float

double ret_double() { return 1.0; }
// CHECK: define{{.*}} double @ret_double

int gi;
int* ret_int_ptr() { return &gi; }
// CHECK: define{{.*}} ptr @ret_int_ptr

void* ret_void_ptr() { return &gi; }
// CHECK: define{{.*}} ptr @ret_void_ptr

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
fx2x2_t ret_matrix() { return (fx2x2_t){1.0f, 2.0f, 3.0f, 4.0f}; }
// CHECK: define{{.*}} <4 x float> @ret_matrix

_BitInt(7) ret_bitint7(void) { return 0; }
// AAPCS: define{{.*}} i7 @ret_bitint7
// DARWIN: define{{.*}} signext i7 @ret_bitint7

unsigned _BitInt(7) ret_ubitint7(void) { return 0; }
// AAPCS: define{{.*}} i7 @ret_ubitint7
// DARWIN: define{{.*}} zeroext i7 @ret_ubitint7

_BitInt(65) ret_bitint65(void) { return 0; }
// CHECK: define{{.*}} i65 @ret_bitint65

_BitInt(128) ret_bitint128(void) { return 0; }
// CHECK: define{{.*}} i128 @ret_bitint128

_BitInt(129) ret_bitint129(void) { return 0; }
// CHECK: define{{.*}} void @ret_bitint129(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}})

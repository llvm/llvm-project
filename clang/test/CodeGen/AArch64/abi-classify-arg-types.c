// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies argument types in
// the same way that Clang does without the library.

// The AArch64 support in the ABI library is a work in progress. New test cases
// will be added here as the types are implemented. Unimplemented cases will
// report a warning if the ABI library is used.

void arg_void(void) {
}
// CHECK: define{{.*}} void @arg_void()

void arg_bool(_Bool b) {}
// AAPCS: define{{.*}} void @arg_bool(i1 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_bool(i1 noundef zeroext %{{.*}})

void arg_char(char c) {}
// AAPCS: define{{.*}} void @arg_char(i8 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_char(i8 noundef signext %{{.*}})

void arg_short(short s) {}
// AAPCS: define{{.*}} void @arg_short(i16 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_short(i16 noundef signext %{{.*}})

void arg_ushort(unsigned short us) {}
// AAPCS: define{{.*}} void @arg_ushort(i16 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_ushort(i16 noundef zeroext %{{.*}})

void arg_int(int i) {}
// CHECK: define{{.*}} void @arg_int(i32 noundef %{{.*}})

void arg_uint(unsigned int ui) {}
// CHECK: define{{.*}} void @arg_uint(i32 noundef %{{.*}})

void arg_long(long int li) {}
// CHECK: define{{.*}} void @arg_long(i64 noundef %{{.*}})

void arg_float16(_Float16 f16) {}
// CHECK: define{{.*}} void @arg_float16(half noundef %{{.*}})

void arg_fp16(__fp16 f16) {}
// CHECK: define{{.*}} void @arg_fp16(half noundef %{{.*}})

void arg_float(float f) {}
// CHECK: define{{.*}} void @arg_float(float noundef %{{.*}})

void arg_double(double d) {}
// CHECK: define{{.*}} void @arg_double(double noundef %{{.*}})

int gi;
void arg_int_ptr(int* pi) {}
// CHECK: define{{.*}} void @arg_int_ptr(ptr noundef %{{.*}})

void arg_void_ptr(void* pv) {}
// CHECK: define{{.*}} void @arg_void_ptr(ptr noundef %{{.*}})

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
void arg_matrix(fx2x2_t m) {}
// CHECK: define{{.*}} void @arg_matrix(<4 x float> noundef %{{.*}})

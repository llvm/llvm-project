// RUN: %clang_cc1 -triple x86_64-linux-android -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=A64
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=G64
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=P64
// RUN: %clang_cc1 -triple i686-linux-android -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=A32
// RUN: %clang_cc1 -triple i686-linux-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=G32
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=P32

// Check mangled name of long double.
// Android's gcc and llvm use fp128 for long double.
void test(long, float, double, long double, long double _Complex) { }
// A64:  define void @_Z4testlfdgCg(i64, float, double, fp128, { fp128, fp128 }*
// G64:  define void @_Z4testlfdeCe(i64, float, double, x86_fp80, { x86_fp80, x86_fp80 }*
// P64:  define void @_Z4testlfdgCg(i64, float, double, ppc_fp128, ppc_fp128 {{.*}}, ppc_fp128
// A32:  define void @_Z4testlfdeCe(i32, float, double, double, { double, double }*
// G32:  define void @_Z4testlfdeCe(i32, float, double, x86_fp80, { x86_fp80, x86_fp80 }*
// P32:  define void @_Z4testlfdgCg(i32, float, double, ppc_fp128, { ppc_fp128, ppc_fp128 }*

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
// RUN: %clang_cc1 -triple x86_64-nacl -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=N64

// Check mangled name of long double.
// Android's gcc and llvm use fp128 for long double.
// NaCl uses double format for long double, but still has separate overloads.
void test(long, float, double, long double, long double _Complex) { }
// A64:  define{{.*}} void @_Z4testlfdgCg(i64 noundef %0, float noundef %1, double noundef %2, fp128 noundef %3, ptr
// G64:  define{{.*}} void @_Z4testlfdeCe(i64 noundef %0, float noundef %1, double noundef %2, x86_fp80 noundef %3, ptr
// P64:  define{{.*}} void @_Z4testlfdgCg(i64 noundef %0, float noundef %1, double noundef %2, ppc_fp128 noundef %3, ppc_fp128 {{.*}}, ppc_fp128
// A32:  define{{.*}} void @_Z4testlfdeCe(i32 noundef %0, float noundef %1, double noundef %2, double noundef %3, ptr
// G32:  define{{.*}} void @_Z4testlfdeCe(i32 noundef %0, float noundef %1, double noundef %2, x86_fp80 noundef %3, ptr
// P32:  define{{.*}} void @_Z4testlfdgCg(i32 noundef %0, float noundef %1, double noundef %2, ppc_fp128 noundef %3, ptr
// N64: define{{.*}} void @_Z4testlfdeCe(i32 noundef %0, float noundef %1, double noundef %2, double noundef %3, double noundef {{.*}}, double

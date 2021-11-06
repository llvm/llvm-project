// RUN: %clang_cc1 -triple powerpc64-linux-musl -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=FP64 %s
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s -mlong-double-64 | \
// RUN:   FileCheck --check-prefix=FP64 %s

// musl defaults to -mlong-double-64, so -mlong-double-128 is needed to make
// -mabi=ieeelongdouble effective.
// RUN: %clang_cc1 -triple powerpc64-linux-musl -emit-llvm -o - %s -mlong-double-128 \
// RUN:   -mabi=ieeelongdouble | FileCheck --check-prefix=FP128 %s
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s \
// RUN:   -mabi=ieeelongdouble | FileCheck --check-prefix=FP128 %s

// IBM extended double is the default.
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=IBM128 %s
// RUN: %clang_cc1 -triple powerpc64-linux-musl -emit-llvm -o - -mlong-double-128 %s | \
// RUN:   FileCheck --check-prefix=IBM128 %s

// Check IBM-quad and IEEE-quad macros are defined.
// RUN: %clang -E -dM -ffreestanding -target powerpc64le-linux-gnu %s \
// RUN:   -mabi=ibmlongdouble | FileCheck -check-prefix=CHECK-DEF-IBM128 %s
// RUN: %clang -E -dM -ffreestanding -target powerpc64le-linux-gnu %s \
// RUN:   -mabi=ieeelongdouble | FileCheck -check-prefix=CHECK-DEF-IEEE128 %s
// RUN: %clang -E -dM -ffreestanding -target powerpc64le-linux-gnu %s \
// RUN:   -mlong-double-64 | FileCheck -check-prefix=CHECK-DEF-F64 %s

// CHECK-DEF-IBM128: #define __LONG_DOUBLE_128__
// CHECK-DEF-IBM128: #define __LONG_DOUBLE_IBM128__
// CHECK-DEF-IEEE128: #define __LONG_DOUBLE_128__
// CHECK-DEF-IEEE128: #define __LONG_DOUBLE_IEEE128__
// CHECK-DEF-F64-NOT: #define __LONG_DOUBLE_128__

long double x = 0;
int size = sizeof(x);

// FP64: @x ={{.*}} global double {{.*}}, align 8
// FP64: @size ={{.*}} global i32 8
// FP128: @x ={{.*}} global fp128 {{.*}}, align 16
// FP128: @size ={{.*}} global i32 16
// IBM128: @x ={{.*}} global ppc_fp128 {{.*}}, align 16
// IBM128: @size ={{.*}} global i32 16

long double foo(long double d) { return d; }

// FP64: double @_Z3fooe(double %d)
// FP128: fp128 @_Z3foou9__ieee128(fp128 %d)
// IBM128: ppc_fp128 @_Z3foog(ppc_fp128 %d)

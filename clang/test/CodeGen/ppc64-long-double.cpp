// RUN: %clang_cc1 -triple powerpc64-linux-musl -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=FP64 %s
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s -mlong-double-64 | \
// RUN:   FileCheck --check-prefix=FP64 %s

// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefix=IBM128 %s
// RUN: %clang_cc1 -triple powerpc64-linux-musl -emit-llvm -o - -mlong-double-128 %s | \
// RUN:   FileCheck --check-prefix=IBM128 %s

long double x = 0;
int size = sizeof(x);

// FP64: @x = global double {{.*}}, align 8
// FP64: @size = global i32 8
// IBM128: @x = global ppc_fp128 {{.*}}, align 16
// IBM128: @size = global i32 16

long double foo(long double d) { return d; }

// FP64: double @_Z3fooe(double %d)
// IBM128: ppc_fp128 @_Z3foog(ppc_fp128 %d)

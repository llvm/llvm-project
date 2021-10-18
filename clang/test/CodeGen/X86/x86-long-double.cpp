// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686 | \
// RUN:   FileCheck --check-prefixes=FP80,FP80-ELF32 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin | \
// RUN:   FileCheck --check-prefixes=FP80,FP80-DARWIN %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64 | \
// RUN:   FileCheck --check-prefixes=FP80,FP80-ELF64 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin | \
// RUN:   FileCheck --check-prefixes=FP80,FP80-DARWIN %s

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686 -mlong-double-64 | \
// RUN:   FileCheck --check-prefixes=FP64,FP64-X32 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin -mlong-double-64 | \
// RUN:   FileCheck --check-prefixes=FP64,FP64-X32 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64 -mlong-double-64 | \
// RUN:   FileCheck --check-prefixes=FP64,FP64-X64 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin -mlong-double-64 | \
// RUN:   FileCheck --check-prefixes=FP64,FP64-X64 %s

// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686 -mlong-double-128 | \
// RUN:   FileCheck --check-prefix=FP128 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin -mlong-double-128 | \
// RUN:   FileCheck --check-prefix=FP128 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64 -mlong-double-128 | \
// RUN:   FileCheck --check-prefix=FP128 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin -mlong-double-128 | \
// RUN:   FileCheck --check-prefix=FP128 %s

// Check -malign-double increases the alignment from 4 to 8 on x86-32.
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686 -mlong-double-64 \
// RUN:   -malign-double | FileCheck --check-prefixes=FP64,FP64-X64 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64 -mlong-double-64 \
// RUN:   -malign-double | FileCheck --check-prefixes=FP64,FP64-X64 %s

long double x = 0;
int size = sizeof(x);

// FP80-ELF32: @x ={{.*}} global x86_fp80 {{.*}}, align 4
// FP80-ELF32: @size ={{.*}} global i32 12
// FP80-ELF64: @x ={{.*}} global x86_fp80 {{.*}}, align 16
// FP80-ELF64: @size ={{.*}} global i32 16
// FP80-DARWIN: @x ={{.*}} global x86_fp80 {{.*}}, align 16
// FP80-DARWIN: @size ={{.*}} global i32 16

// FP64-X32: @x ={{.*}} global double {{.*}}, align 4
// FP64-X32: @size ={{.*}} global i32 8
// FP64-X64: @x ={{.*}} global double {{.*}}, align 8
// FP64-X64: @size ={{.*}} global i32 8

// FP128: @x ={{.*}} global fp128 {{.*}}, align 16
// FP128: @size ={{.*}} global i32 16

long double foo(long double d) { return d; }

// FP64: double @_Z3fooe(double %d)
// FP80: x86_fp80 @_Z3fooe(x86_fp80 %d)
// FP128: fp128 @_Z3foog(fp128 %d)

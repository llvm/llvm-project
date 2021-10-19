// RUN: %clang_cc1 -triple powerpc-unknown-freebsd -emit-llvm -o - %s| FileCheck -check-prefix=SVR4-CHECK %s
// RUN: %clang_cc1 -triple powerpc-unknown-netbsd -emit-llvm -o - %s| FileCheck -check-prefix=SVR4-CHECK %s
// RUN: %clang_cc1 -triple powerpc-unknown-openbsd -emit-llvm -o - %s| FileCheck -check-prefix=SVR4-CHECK %s

#include <stdarg.h>

int va_list_size = sizeof(va_list);
// SVR4-CHECK: va_list_size ={{.*}} global i32 12, align 4
int long_double_size = sizeof(long double);
// SVR4-CHECK: long_double_size ={{.*}} global i32 8, align 4
int double_size = sizeof(double);
// SVR4-CHECK: double_size ={{.*}} global i32 8, align 4

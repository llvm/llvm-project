// RUN: %clang_cc1 -fsyntax-only --show-includes -triple i686-pc-win32 \
// RUN:  -isystem %S/Inputs/ms-crt -fms-compatibility-version=17.00 %s \
// RUN:  | FileCheck %s

#include <stddef.h>
// CHECK: including file:{{.*}}stddef.h
// CHECK: including file:{{.*}}corecrt.h
#include <stdarg.h>
// CHECK: including file:{{.*}}stdarg.h
// CHECK: including file:{{.*}}vcruntime.h

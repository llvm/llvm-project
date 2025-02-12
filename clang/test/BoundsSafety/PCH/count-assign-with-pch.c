// Test without pch.
// RUN: %clang_cc1 -fbounds-safety -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -fbounds-safety -emit-pch -o %t %s
// RUN: %clang_cc1 -fbounds-safety -include-pch %t %s -ast-dump-all 2>&1 | FileCheck %s
// expected-no-diagnostics
#include <ptrcheck.h>

#ifndef HEADER
#define HEADER
int foo(int *__counted_by(len) ptr, int len) {
    ptr = ptr;
    len = 10;
    return ptr[len-1];
}
#else

int main() {
    int arr[10] = {0};
    return foo(arr, 10);
}

// CHECK: |-FunctionDecl {{.*}} imported used foo 'int (int *__single __counted_by(len), int)'
// CHECK: | |-ParmVarDecl [[PTR_DECL:0x[a-z0-9]*]] {{.*}} imported used ptr 'int *__single __counted_by(len)':'int *__single'
// CHECK: | |-ParmVarDecl {{.*}} imported used len 'int'
// CHECK: | | `-DependerDeclsAttr {{.*}} Implicit [[PTR_DECL]] 0

#endif

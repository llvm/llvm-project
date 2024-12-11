
// Test without pch.
// RUN: %clang_cc1 -fbounds-safety -include %s -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -fbounds-safety -emit-pch -o %t %s
// RUN: %clang_cc1 -fbounds-safety -include-pch %t -verify -ast-dump-all %s 2>&1 | FileCheck %s
// expected-no-diagnostics
#include <ptrcheck.h>

#ifndef HEADER
#define HEADER
typedef struct {
    int len;
    int *__counted_by(len) ptr;
} S;
int foo() {
    int arr[10];
    S s;
    s.len = 10;
    s.ptr = arr;
    return s.ptr[9];
}
#else

int main() {
    return foo();
}

// CHECK: |-RecordDecl [[STRUCT_DEF:0x[a-z0-9]*]] {{.*}} imported <undeserialized declarations> struct definition
// CHECK: | |-FieldDecl {{.*}} imported referenced len 'int'
// CHECK: | | `-DependerDeclsAttr {{.*}} Implicit [[FIELD_DECL_PTR:0x[a-z0-9]*]] 0
// CHECK: | `-FieldDecl [[FIELD_DECL_PTR]] {{.*}} imported referenced ptr 'int *__single __counted_by(len)':'int *__single'
// CHECK: |-TypedefDecl {{.*}} imported referenced S 'struct S':'S'
// CHECK: | `-ElaboratedType {{.*}} 'struct S' sugar imported
// CHECK: |   `-RecordType {{.*}} 'S' imported
// CHECK: |     `-Record [[STRUCT_DEF]]

#endif

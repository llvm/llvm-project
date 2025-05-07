

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>
#include <stddef.h>

struct Foo {
    int len;
    int fam[__counted_by((len - offsetof(struct Foo, fam)) / sizeof(int))];
};

// CHECK:      -RecordDecl {{.*}} struct Foo definition
// CHECK-NEXT:  |-FieldDecl {{.*}} referenced len 'int'
// CHECK-NEXT:  | `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:  `-FieldDecl {{.*}} fam 'int[__counted_by((len - 4UL) / 4UL)]':'int[]'

struct Bar {
    int len;
    int fam[__counted_by((len - sizeof(struct Foo)) / sizeof(int))];
};

// CHECK:      -RecordDecl {{.*}} struct Bar definition
// CHECK-NEXT:  |-FieldDecl {{.*}} referenced len 'int'
// CHECK-NEXT:  | `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:  `-FieldDecl {{.*}} fam 'int[__counted_by((len - 4UL) / 4UL)]':'int[]'

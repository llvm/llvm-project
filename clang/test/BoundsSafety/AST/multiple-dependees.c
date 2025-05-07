

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fexperimental-bounds-safety-attributes -x c %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fexperimental-bounds-safety-attributes -x c++ %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fexperimental-bounds-safety-attributes -x objective-c %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fexperimental-bounds-safety-attributes -x objective-c++ %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct T {
    int cnt1;
    int cnt2;
    int *__counted_by(cnt1 * 3 + cnt2 + 2) ptr;
};

// CHECK: RecordDecl {{.*}} struct T definition
// CHECK: |-FieldDecl {{.*}} referenced cnt1 'int'
// CHECK: | `-DependerDeclsAttr {{.*}} Implicit [[FIELD_PTR:0x[0-9a-f]+]] 0
// CHECK: |-FieldDecl {{.*}} referenced cnt2 'int'
// CHECK: | `-DependerDeclsAttr {{.*}} Implicit [[FIELD_PTR]] 0
// CHECK: `-FieldDecl [[FIELD_PTR]] {{.*}} ptr 'int *{{.*}}__counted_by(cnt1 * 3 + cnt2 + 2)':'int *{{.*}}'

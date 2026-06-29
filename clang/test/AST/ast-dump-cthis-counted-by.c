// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -ast-dump-all /dev/null | FileCheck %s

// Verifying that CThisExpr generation works for sturct fields
struct Packet {
    int size;
    int *data __attribute__((counted_by(size)));
};

// CHECK: RecordDecl {{.*}} struct Packet definition
// CHECK: FieldDecl {{.*}} size 'int'
// CHECK: FieldDecl {{.*}} data 'int * __counted_by(size)':'int *'

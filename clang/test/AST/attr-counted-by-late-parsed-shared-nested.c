// RUN: %clang_cc1 -fexperimental-late-parse-attributes -verify -ast-dump %s | FileCheck %s

// A late-parsed count in the declaration specifiers is shared by the
// declarators. One that nests it rejects it for itself only; the others keep it
// as their own count, completed with the record.

#define __counted_by(f)  __attribute__((counted_by(f)))

typedef int *int_ptr;

// CHECK-LABEL: RecordDecl {{.*}} struct own_first definition
// CHECK: FieldDecl {{.*}} a 'int_ptr __counted_by(n)':'int *'
// CHECK: FieldDecl {{.*}} p 'int_ptr *'
struct own_first {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int_ptr __counted_by(n) a, *p;
};

// CHECK-LABEL: RecordDecl {{.*}} struct nested_first definition
// CHECK: FieldDecl {{.*}} p 'int_ptr *'
// CHECK: FieldDecl {{.*}} a 'int_ptr __counted_by(n)':'int *'
struct nested_first {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int_ptr __counted_by(n) *p, a;
};

// A function's return type nests it too.
// CHECK-LABEL: RecordDecl {{.*}} struct return_type definition
// CHECK: FieldDecl {{.*}} a 'int_ptr __counted_by(n)':'int *'
// CHECK: FieldDecl {{.*}} b 'int_ptr (*)(void)'
struct return_type {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int_ptr __counted_by(n) a, (*b)(void);
};

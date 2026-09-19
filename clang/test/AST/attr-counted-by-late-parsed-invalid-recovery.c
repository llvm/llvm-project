// RUN: %clang_cc1 -fexperimental-late-parse-attributes -verify %s -ast-dump | FileCheck %s

// On the late-parsed path the CountAttributedType is built before its count
// argument is parsed, so a rejected argument is only discovered at completion,
// when the node is already embedded in the field's type. Rather than strip a
// (possibly nested) node -- which would force the enclosing types to be rebuilt
// -- the node is kept and completed in place with the raw argument, and the
// field is marked invalid. Consumers bail on such a count (see
// FieldDecl::findCountedByField). This applies whether the argument is unusable
// (a parse failure, or a non-declaration-reference such as `sizeof(...)`) or is
// a valid reference in an invalid position (a union member).
//
// A nested counted_by is the exception: it is diagnosed and dropped while the
// declarator is built -- before the enclosing pointer/array wraps the node, so
// the drop needs no rebuild -- leaving the field its plain wrapped type, exactly
// as on the eager path.

#define __counted_by(f)  __attribute__((counted_by(f)))

// Non-declaration-reference argument: the node is kept with the raw argument as
// its count and the field is marked invalid.
struct bad_count_expr {
  int n;
  int *__counted_by(sizeof(int)) p; // expected-error {{'counted_by' argument must be a simple declaration reference}}
};
// CHECK-LABEL: struct bad_count_expr definition
// CHECK: FieldDecl {{.*}} invalid p 'int * __counted_by(sizeof(int))':'int *'

// Valid reference in an invalid position: also kept with the raw argument and
// the field marked invalid.
union valid_ref_bad_position {
  int n;
  int *__counted_by(n) p; // expected-error {{'counted_by' cannot be applied to a union member}}
};
// CHECK-LABEL: union valid_ref_bad_position definition
// CHECK: FieldDecl {{.*}} invalid p 'int * __counted_by(n)':'int *'

// Nested under another pointer: diagnosed and dropped while the declarator is
// built, so the field keeps its plain wrapped type (no CountAttributedType) and
// stays valid -- exactly as on the eager path.
struct nested_under_pointer {
  int n;
  int *__counted_by(n) *pp; // expected-error {{'counted_by' attribute on nested pointer type is not allowed}}
};
// CHECK-LABEL: struct nested_under_pointer definition
// CHECK: FieldDecl {{.*}} pp 'int **'{{$}}

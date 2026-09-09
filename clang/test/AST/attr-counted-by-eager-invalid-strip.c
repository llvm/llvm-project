// RUN: %clang_cc1 -verify %s -ast-dump | FileCheck %s

// On the eager (non -fexperimental-late-parse-attributes) path a
// counted_by-family CountAttributedType is built during type processing, before
// the FieldDecl-dependent checks run in ActOnFields. When those checks reject
// the attribute, the CountAttributedType must be stripped so the field keeps
// its plain wrapped type -- matching the pre-refactor behavior, which built the
// type only after the check passed. A surviving bogus CountAttributedType would
// otherwise flow downstream. This test pins the stripped type; the diagnostics
// themselves are covered elsewhere.

#define __counted_by(f)  __attribute__((counted_by(f)))

union invalid_union_member {
  int n;
  int *__counted_by(n) p; // expected-error {{'counted_by' cannot be applied to a union member}}
};
// CHECK-LABEL: union invalid_union_member definition
// CHECK: FieldDecl {{.*}} p 'int *'{{$}}

struct invalid_non_fam_array {
  int n;
  int arr[10] __counted_by(n); // expected-error {{'counted_by' on arrays only applies to C99 flexible array members}}
  int last;
};
// CHECK-LABEL: struct invalid_non_fam_array definition
// CHECK: FieldDecl {{.*}} arr 'int[10]'{{$}}

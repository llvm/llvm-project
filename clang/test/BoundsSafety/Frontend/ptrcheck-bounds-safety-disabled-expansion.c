

// RUN: %clang_cc1 -dump-tokens %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

int HasPtrCheck = __has_ptrcheck;
// CHECK-LABEL: identifier 'HasPtrCheck'
// CHECK: numeric_constant '0'

struct Foo {
// CHECK-LABEL: identifier 'Foo'
// CHECK-NEXT: l_brace
  int *__single f0;
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: identifier 'f0'
// CHECK-LABEL: semi

  int *__unsafe_indexable f1;
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: identifier 'f1'
// CHECK-LABEL: semi

  int *__counted_by(10) f2;
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: identifier 'f2'
// CHECK-LABEL: semi

  int *__sized_by(10) f3;
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: identifier 'f3'
// CHECK-LABEL: semi
};
// CHECK-LABEL: r_brace
// CHECK: semi

__ptrcheck_abi_assume_single();
// CHECK-NEXT: semi
__ptrcheck_abi_assume_unsafe_indexable();
// CHECK-NEXT: semi

void code(void) {
// CHECK-LABEL: l_brace
    __unsafe_forge_single(int *, 100);
// CHECK-NEXT: l_paren
// CHECK-NEXT: l_paren
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: r_paren
// CHECK-NEXT: l_paren
// CHECK: numeric_constant '100'
// CHECK-NEXT: r_paren
// CHECK-NEXT: r_paren

// CHECK-LABEL: semi
    __unsafe_forge_bidi_indexable(int *, 200, 300);
// CHECK-NEXT: l_paren
// CHECK-NEXT: l_paren
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: r_paren
// CHECK-NEXT: l_paren
// CHECK: numeric_constant '200'
// CHECK-NEXT: r_paren
// CHECK-NEXT: r_paren

// CHECK-LABEL: semi
    __unsafe_forge_terminated_by(int *, 200, 300);
// CHECK-NEXT: l_paren
// CHECK-NEXT: l_paren
// CHECK-NEXT: 'int'
// CHECK-NEXT: star
// CHECK-NEXT: r_paren
// CHECK-NEXT: l_paren
// CHECK: numeric_constant '200'
// CHECK-NEXT: r_paren
// CHECK-NEXT: r_paren
}
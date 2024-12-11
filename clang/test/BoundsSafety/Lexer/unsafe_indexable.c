

// RUN: %clang_cc1 -dump-tokens -fbounds-safety %s 2>&1 | FileCheck %s

struct Foo {
  int *__attribute__((unsafe_indexable)) foo;
  // CHECK: int 'int'
  // CHECK: star '*'
  // CHECK: __attribute '__attribute__'
  // CHECK: l_paren '('
  // CHECK: l_paren '('
  // CHECK: identifier 'unsafe_indexable'
  // CHECK: r_paren ')'
  // CHECK: r_paren ')'
  // CHECK: identifier 'foo'
  // CHECK: semi ';'
}

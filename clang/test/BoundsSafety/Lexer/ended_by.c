

// RUN: %clang_cc1 -dump-tokens -fbounds-safety %s 2>&1 | FileCheck %s

struct Foo {
  int *__attribute__((ended_by(end))) start;
  int *end;
  // CHECK: int 'int'
  // CHECK: star '*'
  // CHECK: __attribute '__attribute__'
  // CHECK: l_paren '('
  // CHECK: l_paren '('
  // CHECK: identifier 'ended_by'
  // CHECK: l_paren '('
  // CHECK: identifier 'end'
  // CHECK: r_paren ')'
  // CHECK: r_paren ')'
  // CHECK: r_paren ')'
  // CHECK: identifier 'start'
  // CHECK: semi ';'
  // CHECK: int 'int'
  // CHECK: star '*'
  // CHECK: identifier 'end'
  // CHECK: semi ';

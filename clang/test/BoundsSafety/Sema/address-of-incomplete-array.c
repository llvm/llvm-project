
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

typedef unsigned long long size_t;
extern size_t count;
extern int glob[__attribute__((counted_by(count)))];
extern int glob2[__attribute__((sized_by(count)))];
// expected-error@-1 {{'sized_by' cannot apply to arrays: use 'counted_by' instead}}

void foo(int *__attribute__((counted_by(cnt))), size_t cnt);

void bar(void) {
  foo((int *)&glob, count);
  // expected-error@-1 {{cannot take address of incomplete __counted_by array}}
  // expected-note@-2 {{remove '&' to get address as 'int *' instead of 'int (*)[__counted_by(count)]' (aka 'int (*)[]')}}
}

void bar2(void) {
  foo((int *)&glob2, count);
  // expected-warning@-1 {{count value is not statically known: passing 'int *__single' to parameter of type 'int *__single __counted_by(cnt)' (aka 'int *__single') is invalid for any count other than 0 or 1}}
  // expected-note@-2 {{count passed here}}
}

// RUN: %clang_cc1 -fsyntax-only -verify %s

int callee0() __attribute__((not_tail_called,always_inline)); // expected-error{{'not_tail_called' and 'always_inline' attributes are not compatible}} \
// expected-note{{conflicting attribute is here}}
int callee1() __attribute__((always_inline,not_tail_called)); // expected-error{{'always_inline' and 'not_tail_called' attributes are not compatible}} \
// expected-note{{conflicting attribute is here}}

int foo(int a) {
  return a ? callee0() : callee1();
}

int g0 __attribute__((not_tail_called)); // expected-warning {{'not_tail_called' attribute only applies to functions}}

int foo2(int a) __attribute__((not_tail_called("abc"))); // expected-error {{'not_tail_called' attribute takes no arguments}}

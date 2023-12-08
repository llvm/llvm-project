// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

void f(void);

void g(void) {
  __builtin_call_with_static_chain(f(), f);
  __builtin_call_with_static_chain(f, f); // expected-error {{first argument to __builtin_call_with_static_chain must be a non-member call expression}}
  __builtin_call_with_static_chain(^{}(), f); // expected-error {{first argument to __builtin_call_with_static_chain must not be a block call}}
  __builtin_call_with_static_chain(__builtin_unreachable(), f); // expected-error {{first argument to __builtin_call_with_static_chain must not be a builtin call}}
  __builtin_call_with_static_chain(f(), 42); // expected-error {{second argument to __builtin_call_with_static_chain must be of pointer type}}
}

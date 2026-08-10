// RUN: %clang_cc1 -fsyntax-only -verify %s

int foo(void) __attribute__((__hot__));
int bar(void) __attribute__((__cold__));

int var1 __attribute__((__cold__)); // expected-warning{{'__cold__' attribute only applies to functions}}
int var2 __attribute__((__hot__)); // expected-warning{{'__hot__' attribute only applies to functions}}

int qux(void) __attribute__((__hot__)) __attribute__((__cold__)); // expected-error{{'__cold__' and 'hot' attributes are not compatible}} \
// expected-note{{conflicting attribute is here}}
int baz(void) __attribute__((__cold__)) __attribute__((__hot__)); // expected-error{{'__hot__' and 'cold' attributes are not compatible}} \
// expected-note{{conflicting attribute is here}}

__attribute__((cold)) void test1(void); // expected-note{{conflicting attribute is here}}
__attribute__((hot)) void test1(void); // expected-error{{'hot' and 'cold' attributes are not compatible}}

__attribute__((hot)) void test2(void); // expected-note{{conflicting attribute is here}}
__attribute__((cold)) void test2(void); // expected-error{{'cold' and 'hot' attributes are not compatible}}

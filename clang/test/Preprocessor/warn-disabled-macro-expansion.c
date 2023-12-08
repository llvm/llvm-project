// RUN: %clang_cc1 %s -E -Wdisabled-macro-expansion -verify

#define p p

#define a b
#define b a

#define f(a) a

#define g(b) a

#define h(x) i(x)
#define i(y) i(y)

#define c(x) x(0)

#define y(x) y
#define z(x) (z)(x)

p // no warning

a // expected-warning {{recursive macro}}

f(2)

g(3) // expected-warning {{recursive macro}}

h(0) // expected-warning {{recursive macro}}

c(c) // expected-warning {{recursive macro}}

y(5) // expected-warning {{recursive macro}}

z(z) // ok


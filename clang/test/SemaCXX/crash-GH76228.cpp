// RUN: %clang_cc1 -std=c++20 -verify %s
// Check we don't crash on incomplete members and bases when handling parenthesized initialization.
class incomplete; // expected-note@-0 3  {{forward declaration of 'incomplete'}}
struct foo {
  int a;
  incomplete b;
  // expected-error@-1 {{incomplete type}}
};
foo a1(0);

struct one_int {
    int a;
};
struct bar : one_int, incomplete {};
// expected-error@-1 {{incomplete type}}
bar a2(0);

incomplete a3[3](1,2,3);
// expected-error@-1 {{incomplete type}}

struct qux : foo {
};
qux a4(0);

struct fred {
    foo a[3];
};
fred a5(0);

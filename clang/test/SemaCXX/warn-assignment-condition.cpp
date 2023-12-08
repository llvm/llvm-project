// RUN: %clang_cc1 -fsyntax-only -Wparentheses -verify %s

struct A {
  int foo();
  friend A operator+(const A&, const A&);
  A operator|=(const A&);
  operator bool();
};

void test() {
  int x, *p;
  A a, b;

  // With scalars.
  if (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  if ((x = 7)) {}
  do {
  } while (x = 7); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  do {
  } while ((x = 7));
  while (x = 7) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}

  while ((x = 7)) {}
  for (; x = 7; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  for (; (x = 7); ) {}

  if (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  if ((p = p)) {}
  do {
  } while (p = p); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  do {
  } while ((p = p));
  while (p = p) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  while ((p = p)) {}
  for (; p = p; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  for (; (p = p); ) {}

  // Initializing variables (shouldn't warn).
  if (int y = x) {}
  while (int y = x) {}
  if (A y = a) {}
  while (A y = a) {}

  // With temporaries.
  if (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  if ((x = (b+b).foo())) {}
  do {
  } while (x = (b+b).foo()); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  do {
  } while ((x = (b+b).foo()));
  while (x = (b+b).foo()) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  while ((x = (b+b).foo())) {}
  for (; x = (b+b).foo(); ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  for (; (x = (b+b).foo()); ) {}

  // With a user-defined operator.
  if (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  if ((a = b + b)) {}
  do {
  } while (a = b + b); // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  do {
  } while ((a = b + b));
  while (a = b + b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  while ((a = b + b)) {}
  for (; a = b + b; ) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  for (; (a = b + b); ) {}

  // Compound assignments.
  if (x |= 2) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '!=' to turn this compound assignment into an inequality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}

  if (a |= b) {} // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                // expected-note{{use '!=' to turn this compound assignment into an inequality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}

  if ((x == 5)) {} // expected-warning {{equality comparison with extraneous parentheses}} \
                   // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                   // expected-note {{remove extraneous parentheses around the comparison to silence this warning}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wparentheses-equality"
  if ((x == 5)) {} // no-warning
#pragma clang diagnostic pop

  if ((5 == x)) {}

#define EQ(x,y) ((x) == (y))
  if (EQ(x, 5)) {}
#undef EQ
}

void (*fn)();

void test2() {
    if ((fn == test2)) {} // expected-warning {{equality comparison with extraneous parentheses}} \
                          // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                          // expected-note {{remove extraneous parentheses around the comparison to silence this warning}}
    if ((test2 == fn)) {}
}

namespace rdar9027658 {
template <typename T>
void f(T t) {
    if ((t.g == 3)) { } // expected-warning {{equality comparison with extraneous parentheses}} \
                         // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                         // expected-note {{remove extraneous parentheses around the comparison to silence this warning}}
}

struct S { int g; };
void test() {
  f(S()); // expected-note {{in instantiation}}
}
}

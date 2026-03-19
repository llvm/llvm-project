// RUN: %clang_cc1 -fsyntax-only -Wparentheses -std=c++2a -verify %s

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

namespace GH101863 {
void t1(auto... args) {
  if (((args == 0) or ...)) { }
}

template <typename... Args>
void t2(Args... args) {
    if (((args == 0) or ...)) { }
}

void t3(auto... args) {
  if ((... && (args == 0))) { }
}

void t4(auto... a, auto... b) {
  if (((a == 0) or ...) && ((b == 0) or ...)) { }
}

void t5(auto... args) {
  if ((((args == 0) or ...))) { }
}

void t6(auto a, auto... b) {
    static_assert(__is_same_as(decltype((a)), int&));
    static_assert(__is_same_as(decltype(((b), ...)), int&));
};

void t7(auto... args) {
  if ((((args == 0)) or ...)) { } // expected-warning {{equality comparison with extraneous parentheses}} \
                                  // expected-note {{use '=' to turn this equality comparison into an assignment}} \
                                  // expected-note {{remove extraneous parentheses around the comparison to silence this warning}}
}

void test() {
  t1(0, 1);
  t2<>();
  t3(1, 2, 3);
  t3(0, 1);
  t4(0, 1);
  t5(0, 1);
  t6(0, 0);
  t7(0); // expected-note {{in instantiation of function template specialization 'GH101863::t7<int>' requested here}}
}
}

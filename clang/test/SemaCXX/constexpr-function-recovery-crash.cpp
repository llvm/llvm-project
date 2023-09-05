// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -fcxx-exceptions -verify

// verify no value-dependent-assertion crash in constexpr function body and no
// bogus diagnostics.
class Foo {
  constexpr Foo() {
    while (invalid()) {} // expected-error {{use of undeclared identifier}}
    if (invalid()) {} // expected-error {{use of undeclared identifier}}
  }
};

constexpr void test1() {
  while (invalid()) {} // expected-error {{use of undeclared identifier}}
  if (invalid()) {} // expected-error {{use of undeclared identifier}}
}

struct A {
  int *p = new int(invalid()); // expected-error {{use of undeclared identifier}}
  constexpr ~A() { delete p; }
};
constexpr int test2() {
  A a;
  return 1;
}

constexpr int test3() {
  return invalid(); // expected-error {{use of undeclared identifier}}
}

constexpr int test4() {
  if (invalid()) // expected-error {{use of undeclared identifier}}
    return 1;
  else
    return 0;
}

constexpr int test5() { // expected-error {{constexpr function never produce}}
  for (;; a++); // expected-error {{use of undeclared identifier}}  \
                   expected-note {{constexpr evaluation hit maximum step limit; possible infinite loop?}}
  return 1;
}

constexpr int test6() { // expected-error {{constexpr function never produce}}
  int n = 0;
  switch (n) {
    for (;; a++) { // expected-error {{use of undeclared identifier}}
    case 0:; // expected-note {{constexpr evaluation hit maximum step limit; possible infinite loop?}}
    }
  }
  return 0;
}

constexpr bool test7() {
  for (int n = 0; ; invalid()) { if (n == 1) return true; } // expected-error {{use of undeclared identifier}}
  throw "bad";
}

constexpr void test8() {
  do {}  while (invalid()); // expected-error {{use of undeclared identifier}}
  throw "bad";
}

template<int x> constexpr int f(int y) { // expected-note {{candidate template ignored}}
  return x * y;
}
constexpr int test9(int x) {
  return f<1>(f<x>(1)); // expected-error {{no matching function for call to 'f'}}
}

constexpr int test10() { return undef(); } // expected-error {{use of undeclared identifier 'undef'}}
static_assert(test10() <= 1, "should not crash"); // expected-error {{static assertion expression is not an integral constant expression}}

struct X {} array[] = {undef()}; // expected-error {{use of undeclared identifier 'undef'}}
constexpr void test11() {
  for (X& e : array) {}
}

constexpr int test12() { return "wrong"; } // expected-error {{cannot initialize return object of type 'int'}}
constexpr int force12 = test12();          // expected-error {{must be initialized by a constant}}

#define TEST_EVALUATE(Name, X)         \
  constexpr int testEvaluate##Name() { \
    X return 0;                        \
  }                                    \
  constexpr int forceEvaluate##Name = testEvaluate##Name()
// Check that a variety of broken loops don't crash constant evaluation.
// We're not checking specific recovery here so don't assert diagnostics.
TEST_EVALUATE(Switch, switch (!!){});              // expected-error + {{}}
TEST_EVALUATE(SwitchInit, switch (auto x = !!){}); // expected-error + {{}}
TEST_EVALUATE(SwitchCondValDep, switch (invalid_value) { default: break; });    // expected-error + {{}}
TEST_EVALUATE(For, for (!!){}); // expected-error + {{}}
                                // FIXME: should bail out instead of looping.
                                // expected-note@-2 + {{infinite loop}}
                                // expected-note@-3 {{in call}}
TEST_EVALUATE(ForRange, for (auto x : !!){}); // expected-error + {{}}
TEST_EVALUATE(While, while (!!){});           // expected-error + {{}}
TEST_EVALUATE(DoWhile, do {} while (!!););    // expected-error + {{}}
TEST_EVALUATE(DoWhileCond, do {} while (some_cond < 10););    // expected-error {{use of undeclared identifier}}  \
                                                              // expected-error {{constexpr variable 'forceEvaluateDoWhileCond' must be initialized by a constant expression}}
TEST_EVALUATE(If, if (!!){};);                // expected-error + {{}}
TEST_EVALUATE(IfInit, if (auto x = !!; 1){};);// expected-error + {{}}
TEST_EVALUATE(ForInit, for (!!;;){};);// expected-error + {{}}
                                      // expected-note@-1 + {{infinite loop}}
                                      // expected-note@-2 {{in call}}
TEST_EVALUATE(ForCond, for (; !!;){};);// expected-error + {{}}
TEST_EVALUATE(ForInc, for (;; !!){};);// expected-error + {{}}
                                      // expected-note@-1 + {{infinite loop}}
                                      // expected-note@-2 {{in call}}

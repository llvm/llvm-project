// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s

// Test for C++26 erroneous behavior in constant expressions (P2795R5)
// Reading an uninitialized/erroneous value in a constant expression is an error.

// Direct read of default-initialized variable
constexpr int test1() {
  int x;        // default-initialized, has erroneous value
  return x;     // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val1 = test1();  // expected-error {{constexpr variable 'val1' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test1()'}}

// Reading member with erroneous value
struct S {
  int x;
  constexpr S() {}  // x has erroneous value
};

constexpr int test2() {
  S s;
  return s.x;   // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val2 = test2();  // expected-error {{constexpr variable 'val2' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test2()'}}

// [[indeterminate]] in constexpr - also an error
constexpr int test3() {
  [[indeterminate]] int x;  // x has indeterminate value (UB in general, error in constexpr)
  return x;                 // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val3 = test3();  // expected-error {{constexpr variable 'val3' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test3()'}}

// Proper initialization is fine
constexpr int test4() {
  int x = 42;
  return x;
}
constexpr int val4 = test4();  // OK

// Array with erroneous elements
constexpr int test5() {
  int arr[3];  // elements have erroneous values
  return arr[0];  // expected-note {{read of uninitialized object is not allowed in a constant expression}}
}
constexpr int val5 = test5();  // expected-error {{constexpr variable 'val5' must be initialized by a constant expression}} \
                               // expected-note {{in call to 'test5()'}}

// Partial initialization - uninitialized portion is erroneous
constexpr int test6() {
  int arr[3] = {1};  // arr[1] and arr[2] are zero-initialized, not erroneous
  return arr[1];     // OK - zero-initialized
}
constexpr int val6 = test6();  // OK, val6 == 0

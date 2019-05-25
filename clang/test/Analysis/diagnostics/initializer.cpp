// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-output=text \
// RUN:   -verify %s

namespace note_on_skipped_vbases {
struct A {
  int x;
  A() : x(0) {} // expected-note{{The value 0 is assigned to 'c.x'}}
  A(int x) : x(x) {}
};

struct B : virtual A {
  int y;
  // This note appears only once, when this constructor is called from C.
  // When this constructor is called from D, this note is still correct but
  // it doesn't appear because it's pruned out because it's irrelevant to the
  // bug report.
  B(): // expected-note{{Virtual base initialization skipped because it has already been handled by the most derived class}}
    A(1),
    y(1 / x) // expected-warning{{Division by zero}}
             // expected-note@-1{{Division by zero}}
  {}
};

struct C : B {
  C(): // expected-note{{Calling default constructor for 'A'}}
       // expected-note@-1{{Returning from default constructor for 'A'}}
    B() // expected-note{{Calling default constructor for 'B'}}
  {}
};

void test_note() {
  C c; // expected-note{{Calling default constructor for 'C'}}
}

struct D: B {
  D() : A(1), B() {}
};

void test_prunability() {
  D d;
  1 / 0; // expected-warning{{Division by zero}}
         // expected-note@-1{{Division by zero}}
}
} // namespace note_on_skipped_vbases

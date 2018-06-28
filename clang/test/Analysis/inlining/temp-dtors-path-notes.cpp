// RUN: %clang_analyze_cc1 -analyze -analyzer-checker core -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -analyzer-output=text -verify %s

namespace test_simple_temporary {
class C {
  int x;

public:
  C(int x): x(x) {} // expected-note{{The value 0 is assigned to field 'x'}}
  ~C() { x = 1 / x; } // expected-warning{{Division by zero}}
                      // expected-note@-1{{Division by zero}}
};

void test() {
  C(0); // expected-note   {{Passing the value 0 via 1st parameter 'x'}}
        // expected-note@-1{{Calling constructor for 'C'}}
        // expected-note@-2{{Returning from constructor for 'C'}}
        // expected-note@-3{{Calling '~C'}}
}
} // end namespace test_simple_temporary

namespace test_lifetime_extended_temporary {
class C {
  int x;

public:
  C(int x): x(x) {} // expected-note{{The value 0 is assigned to field 'x'}}
  void nop() const {}
  ~C() { x = 1 / x; } // expected-warning{{Division by zero}}
                      // expected-note@-1{{Division by zero}}
};

void test(int coin) {
  // We'd divide by zero in the automatic destructor for variable 'c'.
  const C &c = coin ? C(1) : C(0); // expected-note   {{Assuming 'coin' is 0}}
                                   // expected-note@-1{{'?' condition is false}}
                                   // expected-note@-2{{Passing the value 0 via 1st parameter 'x'}}
                                   // expected-note@-3{{Calling constructor for 'C'}}
                                   // expected-note@-4{{Returning from constructor for 'C'}}
  c.nop();
} // expected-note{{Calling '~C'}}
} // end namespace test_lifetime_extended_temporary

namespace test_bug_after_dtor {
int glob;

class C {
public:
  C() { glob += 1; }
  ~C() { glob -= 2; } // expected-note{{The value 0 is assigned to 'glob'}}
};

void test() {
  glob = 1;
  C(); // expected-note   {{Calling '~C'}}
       // expected-note@-1{{Returning from '~C'}}
  glob = 1 / glob; // expected-warning{{Division by zero}}
                   // expected-note@-1{{Division by zero}}
}
} // end namespace test_bug_after_dtor

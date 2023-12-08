// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check that "::new" and "::delete" in member initializer list are diagnosed
// correctly and don't lead to infinite loop on parsing.

// Error: X() (initializer on non-constructor), "::new" is skipped.
void f1() : X() ::new{}; // expected-error{{only constructors take base initializers}}

// Errors: first "::delete" and initializer on non-constructor, others skipped.
void f2() : ::delete, ::new, X() ::new ::delete{} // expected-error{{expected class member or base class name}}
                                                  // expected-error@-1{{only constructors take base initializers}}

// Errors: the '::' token, "::delete" and initializer on non-constructor, others skipped.
void f3() : ::, ::delete X(), ::new {}; // expected-error2{{expected class member or base class name}}
                                        // expected-error@-1{{only constructors take base initializers}}

template <class T>
struct Base1 {
  T x1;
  Base1(T a1) : x1(a1) {}
};

template <class T>
struct Base2 {
  T x2;
  Base2(T a2) : x2(a2) {}
};

struct S : public Base1<int>, public Base2<float> {
  int x;

  // 1-st initializer is correct (just missing ','), 2-nd incorrect, skip other.
  S() : ::Base1<int>(0) ::new, ::Base2<float>(1.0) ::delete x(2) {} // expected-error{{expected class member or base class name}}
                                                                    // expected-error@-1{{missing ',' between base or member initializers}}

  // 1-st and 2-nd are correct, errors: '::' and "::new", others skipped.
  S(int a) : Base1<int>(a), ::Base2<float>(1.0), ::, // expected-error{{expected class member or base class name}}
             ::new, ! ::delete, ::Base2<() x(3) {}   // expected-error{{expected class member or base class name}}

  // All initializers are correct, nothing to skip, diagnose 2 missing commas.
  S(const S &) : Base1<int>(0) ::Base2<float>(1.0) x(2) {} // expected-error2{{missing ',' between base or member initializers}}
};

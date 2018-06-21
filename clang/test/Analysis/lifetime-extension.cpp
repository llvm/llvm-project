// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=false -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -DTEMPORARIES -verify %s

void clang_analyzer_eval(bool);

namespace pr17001_call_wrong_destructor {
bool x;
struct A {
  int *a;
  A() {}
  ~A() {}
};
struct B : public A {
  B() {}
  ~B() { x = true; }
};

void f() {
  {
    const A &a = B();
  }
  clang_analyzer_eval(x); // expected-warning{{TRUE}}
}
} // end namespace pr17001_call_wrong_destructor

namespace pr19539_crash_on_destroying_an_integer {
struct A {
  int i;
  int j[2];
  A() : i(1) {
    j[0] = 2;
    j[1] = 3;
  }
  ~A() {}
};

void f() {
  const int &x = A().i; // no-crash
  const int &y = A().j[1]; // no-crash
  const int &z = (A().j[1], A().j[0]); // no-crash

  // FIXME: All of these should be TRUE, but constructors aren't inlined.
  clang_analyzer_eval(x == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(y == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(z == 2); // expected-warning{{UNKNOWN}}
}
} // end namespace pr19539_crash_on_destroying_an_integer

namespace maintain_original_object_address_on_lifetime_extension {
class C {
  C **after, **before;

public:
  bool x;

  C(bool x, C **after, C **before) : x(x), after(after), before(before) {
    *before = this;
  }

  // Don't track copies in our tests.
  C(const C &c) : x(c.x), after(nullptr), before(nullptr) {}

  ~C() { if (after) *after = this; }

  operator bool() const { return x; }
};

void f1() {
  C *after, *before;
  {
    const C &c = C(true, &after, &before);
  }
  clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
}

void f2() {
  C *after, *before;
  C c = C(1, &after, &before);
  clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
}

void f3(bool coin) {
  C *after, *before;
  {
    const C &c = coin ? C(true, &after, &before) : C(false, &after, &before);
  }
  clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
}

void f4(bool coin) {
  C *after, *before;
  {
    // no-crash
    const C &c = C(coin, &after, &before) ?: C(false, &after, &before);
  }
  // FIXME: Add support for lifetime extension through binary conditional
  // operator. Ideally also add support for the binary conditional operator in
  // C++. Because for now it calls the constructor for the condition twice.
  if (coin) {
    clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
  // expected-warning@-2{{The left operand of '==' is a garbage value}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
  } else {
    clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
    // Seems to work at the moment, but also seems accidental.
    // Feel free to break.
  // expected-warning@-4{{TRUE}}
#else
  // expected-warning@-6{{UNKNOWN}}
#endif
  }
}

void f5() {
  C *after, *before;
  {
    const bool &x = C(true, &after, &before).x; // no-crash
  }
  // FIXME: Should be TRUE. Should not warn about garbage value.
  clang_analyzer_eval(after == before); // expected-warning{{UNKNOWN}}
}
} // end namespace maintain_original_object_address_on_lifetime_extension

namespace maintain_original_object_address_on_move {
class C {
  int *x;

public:
  C() : x(nullptr) {}
  C(int *x) : x(x) {}
  C(const C &c) = delete;
  C(C &&c) : x(c.x) { c.x = nullptr; }
  C &operator=(C &&c) {
    x = c.x;
    c.x = nullptr;
    return *this;
  }
  ~C() {
    // This was triggering the division by zero warning in f1() and f2():
    // Because move-elision materialization was incorrectly causing the object
    // to be relocated from one address to another before move, but destructor
    // was operating on the old address, it was still thinking that 'x' is set.
    if (x)
      *x = 0;
  }
};

void f1() {
  int x = 1;
  // &x is replaced with nullptr in move-constructor before the temporary dies.
  C c = C(&x);
  // Hence x was not set to 0 yet.
  1 / x; // no-warning
}
void f2() {
  int x = 1;
  C c;
  // &x is replaced with nullptr in move-assignment before the temporary dies.
  c = C(&x);
  // Hence x was not set to 0 yet.
  1 / x; // no-warning
}
} // end namespace maintain_original_object_address_on_move

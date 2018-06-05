// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=false -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -DTEMPORARIES -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++17 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -DTEMPORARIES %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=false -DMOVES -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++11 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -DTEMPORARIES -DMOVES -verify %s
// RUN: %clang_analyze_cc1 -Wno-unused -std=c++17 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true -DTEMPORARIES -DMOVES %s

// Note: The C++17 run-lines don't -verify yet - it is a no-crash test.

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

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

  clang_analyzer_eval(x == 1);
  clang_analyzer_eval(y == 3);
  clang_analyzer_eval(z == 2);
#ifdef TEMPORARIES
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
#else
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
#endif
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

  static C make(C **after, C **before) { return C(false, after, before); }
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
  clang_analyzer_eval(after == before);
#ifdef TEMPORARIES
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
}

struct A { // A is an aggregate.
  const C &c;
};

void f6() {
  C *after, *before;
  {
    A a{C(true, &after, &before)};
  }
  // FIXME: Should be TRUE. Should not warn about garbage value.
  clang_analyzer_eval(after == before); // expected-warning{{UNKNOWN}}
}

void f7() {
  C *after, *before;
  {
    A a = {C(true, &after, &before)};
  }
  // FIXME: Should be TRUE. Should not warn about garbage value.
  clang_analyzer_eval(after == before); // expected-warning{{UNKNOWN}}
}

void f8() {
  C *after, *before;
  {
    A a[2] = {C(false, nullptr, nullptr), C(true, &after, &before)};
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

namespace maintain_address_of_copies {
class C;

struct AddressVector {
  C *buf[10];
  int len;

  AddressVector() : len(0) {}

  void push(C *c) {
    buf[len] = c;
    ++len;
  }
};

class C {
  AddressVector &v;

public:
  C(AddressVector &v) : v(v) { v.push(this); }
  ~C() { v.push(this); }

#ifdef MOVES
  C(C &&c) : v(c.v) { v.push(this); }
#endif

  // Note how return-statements prefer move-constructors when available.
  C(const C &c) : v(c.v) {
#ifdef MOVES
    clang_analyzer_checkInlined(false); // no-warning
#else
    v.push(this);
#endif
  } // no-warning

  static C make(AddressVector &v) { return C(v); }
};

void f1() {
  AddressVector v;
  {
    C c = C(v);
  }
  // 0. Create the original temporary and lifetime-extend it into variable 'c'
  //    construction argument.
  // 1. Construct variable 'c' (elidable copy/move).
  // 2. Destroy the temporary.
  // 3. Destroy variable 'c'.
  clang_analyzer_eval(v.len == 4);
  clang_analyzer_eval(v.buf[0] == v.buf[2]);
  clang_analyzer_eval(v.buf[1] == v.buf[3]);
#ifdef TEMPORARIES
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
#else
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
#endif
}

void f2() {
  AddressVector v;
  {
    const C &c = C::make(v);
  }
  // 0. Construct the original temporary within make(),
  // 1. Construct the return value of make() (elidable copy/move) and
  //    lifetime-extend it via reference 'c',
  // 2. Destroy the temporary within make(),
  // 3. Destroy the temporary lifetime-extended by 'c'.
  clang_analyzer_eval(v.len == 4);
  clang_analyzer_eval(v.buf[0] == v.buf[2]);
  clang_analyzer_eval(v.buf[1] == v.buf[3]);
#ifdef TEMPORARIES
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
#else
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
#endif
}

void f3() {
  AddressVector v;
  {
    C &&c = C::make(v);
  }
  // 0. Construct the original temporary within make(),
  // 1. Construct the return value of make() (elidable copy/move) and
  //    lifetime-extend it via reference 'c',
  // 2. Destroy the temporary within make(),
  // 3. Destroy the temporary lifetime-extended by 'c'.
  clang_analyzer_eval(v.len == 4);
  clang_analyzer_eval(v.buf[0] == v.buf[2]);
  clang_analyzer_eval(v.buf[1] == v.buf[3]);
#ifdef TEMPORARIES
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
  // expected-warning@-4{{TRUE}}
#else
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
  // expected-warning@-8{{UNKNOWN}}
#endif
}

C doubleMake(AddressVector &v) {
  return C::make(v);
}

void f4() {
  AddressVector v;
  {
    C c = doubleMake(v);
  }
  // 0. Construct the original temporary within make(),
  // 1. Construct the return value of make() (elidable copy/move) and
  //    lifetime-extend it into the return value constructor argument within
  //    doubleMake(),
  // 2. Destroy the temporary within make(),
  // 3. Construct the return value of doubleMake() (elidable copy/move) and
  //    lifetime-extend it into the variable 'c' constructor argument,
  // 4. Destroy the return value of make(),
  // 5. Construct variable 'c' (elidable copy/move),
  // 6. Destroy the return value of doubleMake(),
  // 7. Destroy variable 'c'.
  clang_analyzer_eval(v.len == 8);
  clang_analyzer_eval(v.buf[0] == v.buf[2]);
  clang_analyzer_eval(v.buf[1] == v.buf[4]);
  clang_analyzer_eval(v.buf[3] == v.buf[6]);
  clang_analyzer_eval(v.buf[5] == v.buf[7]);
#ifdef TEMPORARIES
  // expected-warning@-6{{TRUE}}
  // expected-warning@-6{{TRUE}}
  // expected-warning@-6{{TRUE}}
  // expected-warning@-6{{TRUE}}
  // expected-warning@-6{{TRUE}}
#else
  // expected-warning@-12{{UNKNOWN}}
  // expected-warning@-12{{UNKNOWN}}
  // expected-warning@-12{{UNKNOWN}}
  // expected-warning@-12{{UNKNOWN}}
  // expected-warning@-12{{UNKNOWN}}
#endif
}
} // end namespace maintain_address_of_copies

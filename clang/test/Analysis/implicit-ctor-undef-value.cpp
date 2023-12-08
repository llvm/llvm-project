// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s

namespace implicit_constructor {
struct S {
public:
  S() {}
  S(const S &) {}
};

// Warning is in a weird position because the body of the constructor is
// missing. Specify which field is being assigned.
class C { // expected-warning{{Value assigned to field 'y' in implicit constructor is garbage or undefined}}
          // expected-note@-1{{Value assigned to field 'y' in implicit constructor is garbage or undefined}}
  int x, y;
  S s;

public:
  C(): x(0) {}
};

void test() {
  C c1;
  C c2(c1); // expected-note{{Calling implicit copy constructor for 'C'}}
}
} // end namespace implicit_constructor


namespace explicit_constructor {
class C {
  int x, y;

public:
  C(): x(0) {}
  // It is not necessary to specify which field is being assigned to.
  C(const C &c):
    x(c.x),
    y(c.y) // expected-warning{{Assigned value is garbage or undefined}}
           // expected-note@-1{{Assigned value is garbage or undefined}}
  {}
};

void test() {
  C c1;
  C c2(c1); // expected-note{{Calling copy constructor for 'C'}}
}
} // end namespace explicit_constructor


namespace base_class_constructor {
struct S {
public:
  S() {}
  S(const S &) {}
};

class C { // expected-warning{{Value assigned to field 'y' in implicit constructor is garbage or undefined}}
          // expected-note@-1{{Value assigned to field 'y' in implicit constructor is garbage or undefined}}
  int x, y;
  S s;

public:
  C(): x(0) {}
};

class D: public C {
public:
  D(): C() {}
};

void test() {
  D d1;
  D d2(d1); // expected-note   {{Calling implicit copy constructor for 'D'}}
            // expected-note@-1{{Calling implicit copy constructor for 'C'}}
}
} // end namespace base_class_constructor

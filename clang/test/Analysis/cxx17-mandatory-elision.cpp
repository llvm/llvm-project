// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

template <typename T> struct AddressVector {
  T *buf[10];
  int len;

  AddressVector() : len(0) {}

  void push(T *t) {
    buf[len] = t;
    ++len;
  }
};

class ClassWithoutDestructor {
  AddressVector<ClassWithoutDestructor> &v;

public:
  ClassWithoutDestructor(AddressVector<ClassWithoutDestructor> &v) : v(v) {
    v.push(this);
  }

  ClassWithoutDestructor(ClassWithoutDestructor &&c) : v(c.v) { v.push(this); }
  ClassWithoutDestructor(const ClassWithoutDestructor &c) : v(c.v) {
    v.push(this);
  }
};

ClassWithoutDestructor make1(AddressVector<ClassWithoutDestructor> &v) {
  return ClassWithoutDestructor(v);
}
ClassWithoutDestructor make2(AddressVector<ClassWithoutDestructor> &v) {
  return make1(v);
}
ClassWithoutDestructor make3(AddressVector<ClassWithoutDestructor> &v) {
  return make2(v);
}

void testMultipleReturns() {
  AddressVector<ClassWithoutDestructor> v;
  ClassWithoutDestructor c = make3(v);

#if __cplusplus >= 201703L
  // FIXME: Both should be TRUE.
  clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == &c); // expected-warning{{FALSE}}
#else
  clang_analyzer_eval(v.len == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] != v.buf[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] != v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[2] != v.buf[3]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] != v.buf[4]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[4] == &c); // expected-warning{{TRUE}}
#endif
}

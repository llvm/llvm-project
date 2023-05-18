// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 \
// RUN:    -analyzer-config eagerly-assume=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 \
// RUN:    -analyzer-config eagerly-assume=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 \
// RUN:    -analyzer-config elide-constructors=false -DNO_ELIDE_FLAG              \
// RUN:    -analyzer-config eagerly-assume=false -verify=expected,no-elide %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 \
// RUN:    -analyzer-config elide-constructors=false                              \
// RUN:    -analyzer-config eagerly-assume=false -verify %s

// Copy elision always occurs in C++17, otherwise it's under
// an on-by-default flag.
#if __cplusplus >= 201703L
  #define ELIDE 1
#else
  #ifndef NO_ELIDE_FLAG
    #define ELIDE 1
  #endif
#endif

void clang_analyzer_eval(bool);
void clang_analyzer_dump(int);

namespace variable_functional_cast_crash {

struct A {
  A(int) {}
};

void foo() {
  A a = A(0);
}

struct B {
  A a;
  B(): a(A(0)) {}
};

} // namespace variable_functional_cast_crash


namespace ctor_initializer {

struct S {
  int x, y, z;
};

struct T {
  S s;
  int w;
  T(int w): s(), w(w) {}
};

class C {
  T t;
public:
  C() : t(T(4)) {
    S s = {1, 2, 3};
    t.s = s;
    // FIXME: Should be TRUE regardless of copy elision.
    clang_analyzer_eval(t.w == 4);
#ifdef ELIDE
    // expected-warning@-2{{TRUE}}
#else
    // expected-warning@-4{{UNKNOWN}}
#endif
  }
};


struct A {
  int x;
  A(): x(0) {}
  ~A() {}
};

struct B {
  A a;
  B() : a(A()) {}
};

void foo() {
  B b;
  clang_analyzer_eval(b.a.x == 0); // expected-warning{{TRUE}}
}

} // namespace ctor_initializer


namespace elision_on_ternary_op_branches {
class C1 {
  int x;
public:
  C1(int x): x(x) {}
  int getX() const { return x; }
  ~C1();
};

class C2 {
  int x;
  int y;
public:
  C2(int x, int y): x(x), y(y) {}
  int getX() const { return x; }
  int getY() const { return y; }
  ~C2();
};

void foo(int coin) {
  C1 c1 = coin ? C1(1) : C1(2);
  if (coin) {
    clang_analyzer_eval(c1.getX() == 1); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(c1.getX() == 2); // expected-warning{{TRUE}}
  }
  C2 c2 = coin ? C2(3, 4) : C2(5, 6);
  if (coin) {
    clang_analyzer_eval(c2.getX() == 3); // expected-warning{{TRUE}}
    clang_analyzer_eval(c2.getY() == 4); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(c2.getX() == 5); // expected-warning{{TRUE}}
    clang_analyzer_eval(c2.getY() == 6); // expected-warning{{TRUE}}
  }
}
} // namespace elision_on_ternary_op_branches


namespace address_vector_tests {

template <typename T> struct AddressVector {
  T *buf[20];
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
    push();
  }

  ClassWithoutDestructor(ClassWithoutDestructor &&c) : v(c.v) { push(); }
  ClassWithoutDestructor(const ClassWithoutDestructor &c) : v(c.v) { push(); }

  void push() { v.push(this); }
};

ClassWithoutDestructor make1(AddressVector<ClassWithoutDestructor> &v) {
  return ClassWithoutDestructor(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithoutDestructor' is still \
referred to by the stack variable 'v' upon returning to the caller}}
}
ClassWithoutDestructor make2(AddressVector<ClassWithoutDestructor> &v) {
  return make1(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithoutDestructor' is still \
referred to by the stack variable 'v' upon returning to the caller}}
}
ClassWithoutDestructor make3(AddressVector<ClassWithoutDestructor> &v) {
  return make2(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithoutDestructor' is still \
referred to by the stack variable 'v' upon returning to the caller}}
}

void testMultipleReturns() {
  AddressVector<ClassWithoutDestructor> v;
  ClassWithoutDestructor c = make3(v);

#if ELIDE
  clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == &c); // expected-warning{{TRUE}}
#else
  clang_analyzer_eval(v.len == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] != v.buf[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] != v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[2] != v.buf[3]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] != v.buf[4]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[4] == &c); // expected-warning{{TRUE}}
#endif
}

void consume(ClassWithoutDestructor c) {
  c.push();
  // expected-warning@-1 {{Address of stack memory associated with local \
variable 'c' is still referred to by the stack variable 'v' upon returning \
to the caller}}
}

void testArgumentConstructorWithoutDestructor() {
  AddressVector<ClassWithoutDestructor> v;

  consume(make3(v));

#if ELIDE
  clang_analyzer_eval(v.len == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
#else
  clang_analyzer_eval(v.len == 6); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] != v.buf[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] != v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[2] != v.buf[3]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] != v.buf[4]); // expected-warning{{TRUE}}
  // We forced a push() in consume(), let's see if the address here matches
  // the address during construction.
  clang_analyzer_eval(v.buf[4] == v.buf[5]); // expected-warning{{TRUE}}
#endif
}

class ClassWithDestructor {
  AddressVector<ClassWithDestructor> &v;

public:
  ClassWithDestructor(AddressVector<ClassWithDestructor> &v) : v(v) {
    push();
  }

  ClassWithDestructor(ClassWithDestructor &&c) : v(c.v) { push(); }
  ClassWithDestructor(const ClassWithDestructor &c) : v(c.v) { push(); }

  ~ClassWithDestructor() { push(); }

  void push() { v.push(this); }
};

void testVariable() {
  AddressVector<ClassWithDestructor> v;
  {
    ClassWithDestructor c = ClassWithDestructor(v);
    // Check if the last destructor is an automatic destructor.
    // A temporary destructor would have fired by now.
#if ELIDE
    clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
#else
    clang_analyzer_eval(v.len == 3); // expected-warning{{TRUE}}
#endif
  }
#if ELIDE
  // 0. Construct the variable.
  // 1. Destroy the variable.
  clang_analyzer_eval(v.len == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
#else
  // 0. Construct the temporary.
  // 1. Construct the variable.
  // 2. Destroy the temporary.
  // 3. Destroy the variable.
  clang_analyzer_eval(v.len == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] == v.buf[3]); // expected-warning{{TRUE}}
#endif
}

struct TestCtorInitializer {
  ClassWithDestructor c;
  TestCtorInitializer(AddressVector<ClassWithDestructor> &v)
    : c(ClassWithDestructor(v)) {}
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithDestructor' is still referred \
to by the stack variable 'v' upon returning to the caller}}
};

void testCtorInitializer() {
  AddressVector<ClassWithDestructor> v;
  {
    TestCtorInitializer t(v);
    // Check if the last destructor is an automatic destructor.
    // A temporary destructor would have fired by now.
#if ELIDE
    clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
#else
    clang_analyzer_eval(v.len == 3); // expected-warning{{TRUE}}
#endif
  }
#if ELIDE
  // 0. Construct the member variable.
  // 1. Destroy the member variable.
  clang_analyzer_eval(v.len == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
#else
  // 0. Construct the temporary.
  // 1. Construct the member variable.
  // 2. Destroy the temporary.
  // 3. Destroy the member variable.
  clang_analyzer_eval(v.len == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] == v.buf[3]); // expected-warning{{TRUE}}
#endif
}


ClassWithDestructor make1(AddressVector<ClassWithDestructor> &v) {
  return ClassWithDestructor(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithDestructor' is still referred \
to by the stack variable 'v' upon returning to the caller}}
}
ClassWithDestructor make2(AddressVector<ClassWithDestructor> &v) {
  return make1(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithDestructor' is still referred \
to by the stack variable 'v' upon returning to the caller}}
}
ClassWithDestructor make3(AddressVector<ClassWithDestructor> &v) {
  return make2(v);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'ClassWithDestructor' is still referred \
to by the stack variable 'v' upon returning to the caller}}
}

void testMultipleReturnsWithDestructors() {
  AddressVector<ClassWithDestructor> v;
  {
    ClassWithDestructor c = make3(v);
    // Check if the last destructor is an automatic destructor.
    // A temporary destructor would have fired by now.
#if ELIDE
    clang_analyzer_eval(v.len == 1); // expected-warning{{TRUE}}
#else
    clang_analyzer_eval(v.len == 9); // expected-warning{{TRUE}}
#endif
  }

#if ELIDE
  // 0. Construct the variable. Yes, constructor in make1() constructs
  //    the variable 'c'.
  // 1. Destroy the variable.
  clang_analyzer_eval(v.len == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
#else
  // 0. Construct the temporary in make1().
  // 1. Construct the temporary in make2().
  // 2. Destroy the temporary in make1().
  // 3. Construct the temporary in make3().
  // 4. Destroy the temporary in make2().
  // 5. Construct the temporary here.
  // 6. Destroy the temporary in make3().
  // 7. Construct the variable.
  // 8. Destroy the temporary here.
  // 9. Destroy the variable.
  clang_analyzer_eval(v.len == 10); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] == v.buf[4]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] == v.buf[6]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[5] == v.buf[8]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[7] == v.buf[9]); // expected-warning{{TRUE}}
#endif
}

void consume(ClassWithDestructor c) {
  c.push();
  // expected-warning@-1 {{Address of stack memory associated with local \
variable 'c' is still referred to by the stack variable 'v' upon returning \
to the caller}}
}

void testArgumentConstructorWithDestructor() {
  AddressVector<ClassWithDestructor> v;

  consume(make3(v));

#if ELIDE
  // 0. Construct the argument.
  // 1. Forced push() in consume().
  // 2. Destroy the argument.
  clang_analyzer_eval(v.len == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[1]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] == v.buf[2]); // expected-warning{{TRUE}}
#else
  // 0. Construct the temporary in make1().
  // 1. Construct the temporary in make2().
  // 2. Destroy the temporary in make1().
  // 3. Construct the temporary in make3().
  // 4. Destroy the temporary in make2().
  // 5. Construct the temporary here.
  // 6. Destroy the temporary in make3().
  // 7. Construct the argument.
  // 8. Forced push() in consume().
  // 9. Destroy the argument. Notice the reverse order!
  // 10. Destroy the temporary here.
  clang_analyzer_eval(v.len == 11); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[0] == v.buf[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[1] == v.buf[4]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[3] == v.buf[6]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[5] == v.buf[10]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[7] == v.buf[8]); // expected-warning{{TRUE}}
  clang_analyzer_eval(v.buf[8] == v.buf[9]); // expected-warning{{TRUE}}
#endif
}

struct Foo {
  Foo(Foo **q) {
    *q = this;
  }
};

Foo make1(Foo **r) {
  return Foo(r);
  // no-elide-warning@-1 {{Address of stack memory associated with temporary \
object of type 'Foo' is still referred to by the stack \
variable 'z' upon returning to the caller}}
}

void test_copy_elision() {
  Foo *z;
  // If the copy elided, 'z' points to 'tmp', otherwise it's a dangling pointer.
  Foo tmp = make1(&z);
  (void)tmp;
}

} // namespace address_vector_tests

namespace arg_directly_from_return_in_loop {

struct Result {
  int value;
};

Result create() {
  return Result{10};
}

int accessValue(Result r) {
  return r.value;
}

void test() {
  for (int i = 0; i < 3; ++i) {
    int v = accessValue(create());
    if (i == 0) {
      clang_analyzer_dump(v); // expected-warning {{10 S32b}}
    } else {
      clang_analyzer_dump(v); // expected-warning {{10 S32b}}
                              // was {{reg_${{[0-9]+}}<int r.value> }} for C++11
    }
  }
}

} // namespace arg_directly_from_return_in_loop

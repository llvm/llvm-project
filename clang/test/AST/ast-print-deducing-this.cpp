// RUN: %clang_cc1 -std=c++23 -ast-print %s | FileCheck %s --match-full-lines

struct S {
  void f(this const S &);
  template <typename> void g(this const S &);
};

struct Ptr1 {
  const S *operator->() const;
};

struct Ptr2 {
  const S *operator->(this const Ptr2 &);
};

// FIXME: Should output the syntax of calling member functions.
void h(S s, S *ptr, Ptr1 ptr1, Ptr2 ptr2) {
  s.f();
  // CHECK: f(s);
  s.S::f();
  // CHECK: S::f(s);
  s.g<S>();
  // CHECK: g<S>(s);
  s.template g<S>();
  // CHECK: template g<S>(s);
  s.S::g<S>();
  // CHECK: S::g<S>(s);
  s.S::template g<S>();
  // CHECK: S::template g<S>(s);

  ptr->f();
  // CHECK: f(*ptr);
  ptr->S::f();
  // CHECK: S::f(*ptr);
  ptr->g<S>();
  // CHECK: g<S>(*ptr);
  ptr->template g<S>();
  // CHECK: template g<S>(*ptr);
  ptr->S::g<S>();
  // CHECK: S::g<S>(*ptr);
  ptr->S::template g<S>();
  // CHECK: S::template g<S>(*ptr);

  ptr1->f();
  // CHECK: f(*ptr1);
  ptr1->S::f();
  // CHECK: S::f(*ptr1);
  ptr1->g<S>();
  // CHECK: g<S>(*ptr1);
  ptr1->template g<S>();
  // CHECK: template g<S>(*ptr1);
  ptr1->S::g<S>();
  // CHECK: S::g<S>(*ptr1);
  ptr1->S::template g<S>();
  // CHECK: S::template g<S>(*ptr1);

  ptr2->f();
  // CHECK: f(*ptr2);
  ptr2->S::f();
  // CHECK: S::f(*ptr2);
  ptr2->g<S>();
  // CHECK: g<S>(*ptr2);
  ptr2->template g<S>();
  // CHECK: template g<S>(*ptr2);
  ptr2->S::g<S>();
  // CHECK: S::g<S>(*ptr2);
  ptr2->S::template g<S>();
  // CHECK: S::template g<S>(*ptr2);
}

// RUN: %clang_cc1 -verify -pedantic %s -std=c++98
// RUN: %clang_cc1 -verify -pedantic %s -std=c++11

template<typename T> struct atomic {
  _Atomic(T) value; // expected-warning {{'_Atomic' is a C11 extension}}

  void f() _Atomic; // expected-error {{expected ';' at end of declaration list}}
};

template<typename T> struct user {
  struct inner { char n[sizeof(T)]; };
  atomic<inner> i;
};

user<int> u;

// Test overloading behavior of atomics.
struct A { };

int &ovl1(_Atomic(int)); // expected-warning {{'_Atomic' is a C11 extension}}
int &ovl1(_Atomic int);  // expected-warning {{'_Atomic' is a C11 extension}} // ok, redeclaration
long &ovl1(_Atomic(long)); // expected-warning {{'_Atomic' is a C11 extension}}
float &ovl1(_Atomic(float)); // expected-warning {{'_Atomic' is a C11 extension}}
double &ovl1(_Atomic(A const *const *)); // expected-warning {{'_Atomic' is a C11 extension}}
double &ovl1(A const *const *_Atomic); // expected-warning {{'_Atomic' is a C11 extension}}
short &ovl1(_Atomic(A **)); // expected-warning {{'_Atomic' is a C11 extension}}

void test_overloading(int i, float f, _Atomic(int) ai, _Atomic(float) af, // expected-warning 2 {{'_Atomic' is a C11 extension}}
                      long l, _Atomic(long) al, A const *const *acc, // expected-warning {{'_Atomic' is a C11 extension}}
                      A const ** ac, A **a) {
  int& ir1 = ovl1(i);
  int& ir2 = ovl1(ai);
  long& lr1 = ovl1(l);
  long& lr2 = ovl1(al);
  float &fr1 = ovl1(f);
  float &fr2 = ovl1(af);
  double &dr1 = ovl1(acc);
  double &dr2 = ovl1(ac);
  short &sr1 = ovl1(a);
}

typedef int (A::*fp)() _Atomic; // expected-error {{expected ';' after top level declarator}} expected-warning {{does not declare anything}} \
                                // expected-warning {{'_Atomic' is a C11 extension}}

typedef _Atomic(int(A::*)) atomic_mem_ptr_to_int; // expected-warning {{'_Atomic' is a C11 extension}}
typedef int(A::*_Atomic atomic_mem_ptr_to_int); // expected-warning {{'_Atomic' is a C11 extension}}

typedef _Atomic(int)(A::*mem_ptr_to_atomic_int); // expected-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic int(A::*mem_ptr_to_atomic_int); // expected-warning {{'_Atomic' is a C11 extension}}

typedef _Atomic(int)&atomic_int_ref; // expected-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic int &atomic_int_ref; // expected-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic atomic_int_ref atomic_int_ref; // expected-warning {{'_Atomic' qualifier on reference type 'atomic_int_ref' (aka '_Atomic(int) &') has no effect}} \
                                               // expected-warning {{'_Atomic' is a C11 extension}}

typedef int &_Atomic atomic_reference_to_int; // expected-error {{'_Atomic' qualifier may not be applied to a reference}} \
                                              // expected-warning {{'_Atomic' is a C11 extension}}
typedef _Atomic(int &) atomic_reference_to_int; // expected-error {{_Atomic cannot be applied to reference type 'int &'}} \
                                                // expected-warning {{'_Atomic' is a C11 extension}}

struct S {
  _Atomic union { int n; }; // expected-warning {{anonymous union cannot be '_Atomic'}} \
                            // expected-warning {{'_Atomic' is a C11 extension}}
};

namespace copy_init {
  struct X {
    X(int);
    int n;
  };
  _Atomic(X) y = X(0); // expected-warning {{'_Atomic' is a C11 extension}}
  _Atomic(X) z(X(0)); // expected-warning {{'_Atomic' is a C11 extension}}
  void f() { y = X(0); }

  _Atomic(X) e1(0); // expected-error {{cannot initialize}} \
                    // expected-warning {{'_Atomic' is a C11 extension}}
#if __cplusplus >= 201103L
  _Atomic(X) e2{0}; // expected-error {{illegal initializer}} \
                    // expected-warning {{'_Atomic' is a C11 extension}}
  _Atomic(X) a{X(0)}; // expected-warning {{'_Atomic' is a C11 extension}}
  // FIXME: This does not seem like the right answer.
  _Atomic(int) e3{0}; // expected-error {{illegal initializer}} \
                      // expected-warning {{'_Atomic' is a C11 extension}}
#endif

  struct Y {
    _Atomic(X) a; // expected-warning {{'_Atomic' is a C11 extension}}
    _Atomic(int) b; // expected-warning {{'_Atomic' is a C11 extension}}
  };
  Y y1 = { X(0), 4 };
  Y y2 = { 0, 4 }; // expected-error {{cannot initialize}}

  // FIXME: It's not really clear if we should allow these. Generally, C++11
  // allows extraneous braces around initializers. We should at least give the
  // same answer in all these cases:
  Y y3 = { X(0), { 4 } }; // expected-error {{illegal initializer type}}
  Y y4 = { { X(0) }, 4 };
  _Atomic(int) ai = { 4 }; // expected-error {{illegal initializer type}} \
                           // expected-warning {{'_Atomic' is a C11 extension}}
  _Atomic(X) ax = { X(0) }; // expected-warning {{'_Atomic' is a C11 extension}}
}

bool PR21836(_Atomic(int) *x) { // expected-warning {{'_Atomic' is a C11 extension}}
    return *x;
}

namespace non_trivially_copyable {
  struct S {
    ~S() {}
  };
  _Atomic S s;  // expected-error {{_Atomic cannot be applied to type 'S' which is not trivially copyable}} \
                // expected-warning {{'_Atomic' is a C11 extension}}
}

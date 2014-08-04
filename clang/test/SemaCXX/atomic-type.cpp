// RUN: %clang_cc1 -verify -pedantic %s

template<typename T> struct atomic {
  _Atomic(T) value;

  void f() _Atomic; // expected-error {{expected ';' at end of declaration list}}
};

template<typename T> struct user {
  struct inner { char n[sizeof(T)]; };
  atomic<inner> i;
};

user<int> u;

// Test overloading behavior of atomics.
struct A { };

int &ovl1(_Atomic(int));
int &ovl1(_Atomic int); // ok, redeclaration
long &ovl1(_Atomic(long));
float &ovl1(_Atomic(float));
double &ovl1(_Atomic(A const *const *));
double &ovl1(A const *const *_Atomic);
short &ovl1(_Atomic(A **));

void test_overloading(int i, float f, _Atomic(int) ai, _Atomic(float) af,
                      long l, _Atomic(long) al, A const *const *acc,
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

typedef int (A::*fp)() _Atomic; // expected-error {{expected ';' after top level declarator}} expected-warning {{does not declare anything}}

typedef _Atomic(int(A::*)) atomic_mem_ptr_to_int;
typedef int(A::*_Atomic atomic_mem_ptr_to_int);

typedef _Atomic(int)(A::*mem_ptr_to_atomic_int);
typedef _Atomic int(A::*mem_ptr_to_atomic_int);

typedef _Atomic(int)&atomic_int_ref;
typedef _Atomic int &atomic_int_ref;
typedef _Atomic atomic_int_ref atomic_int_ref; // expected-warning {{'_Atomic' qualifier on reference type 'atomic_int_ref' (aka '_Atomic(int) &') has no effect}}

typedef int &_Atomic atomic_reference_to_int; // expected-error {{'_Atomic' qualifier may not be applied to a reference}}
typedef _Atomic(int &) atomic_reference_to_int; // expected-error {{_Atomic cannot be applied to reference type 'int &'}}

struct S {
  _Atomic union { int n; }; // expected-warning {{anonymous union cannot be '_Atomic'}}
};

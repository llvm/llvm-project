// RUN: %clang_cc1 -verify %s

template<typename T> struct atomic {
  _Atomic(T) value;
};

template<typename T> struct user {
  struct inner { char n[sizeof(T)]; };
  atomic<inner> i;
};

user<int> u;

// Test overloading behavior of atomics.
struct A { };

int &ovl1(_Atomic(int));
long &ovl1(_Atomic(long));
float &ovl1(_Atomic(float));
double &ovl1(_Atomic(A const *const *));
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

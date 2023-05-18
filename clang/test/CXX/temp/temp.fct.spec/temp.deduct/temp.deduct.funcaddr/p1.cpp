// RUN: %clang_cc1 -fsyntax-only %s

template<typename T>
  T f0(T, int);

void test_f0() {
  int (*f0a)(int, int) = f0;
  int (*f0b)(int, int) = &f0;
  float (*f0c)(float, int) = &f0;
}

template<typename T> T f1(T, int);
template<typename T> T f1(T);

void test_f1() {
  float (*f1a)(float, int) = f1;
  float (*f1b)(float, int) = &f1;
  float (*f1c)(float) = f1;
  float (*f1d)(float) = (f1);
  float (*f1e)(float) = &f1;
  float (*f1f)(float) = (&f1);
}

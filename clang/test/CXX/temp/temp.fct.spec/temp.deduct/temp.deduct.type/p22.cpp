// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// If the original function parameter associated with A is a function
// parameter pack and the function parameter associated with P is not
// a function parameter pack, then template argument deduction fails.
template<class ... Args> int& f(Args ... args); 
template<class T1, class ... Args> float& f(T1 a1, Args ... args); 
template<class T1, class T2> double& f(T1 a1, T2 a2);

void test_f() {
  int &ir1 = f();
  float &fr1 = f(1, 2, 3);
  double &dr1 = f(1, 2);
}

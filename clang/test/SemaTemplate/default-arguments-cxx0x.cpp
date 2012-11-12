// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

// Test default template arguments for function templates.
template<typename T = int>
void f0();

template<typename T>
void f0();

void g0() {
  f0(); // okay!
} 

template<typename T, int N = T::value>
int &f1(T);

float &f1(...);

struct HasValue {
  static const int value = 17;
};

void g1() {
  float &fr = f1(15);
  int &ir = f1(HasValue());
}

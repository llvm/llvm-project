// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// The result of the expression const_cast<T>(v) is of type T. If T is
// an lvalue reference to object type, the result is an lvalue; if T
// is an rvalue reference to object type, the result is an xvalue;.

unsigned int f(int);

template<typename T> T& lvalue();
template<typename T> T&& xvalue();
template<typename T> T prvalue();

void test_classification(const int *ptr) {
  int *ptr0 = const_cast<int *&&>(ptr);
  int *ptr1 = const_cast<int *&&>(xvalue<const int*>());
  int *ptr2 = const_cast<int *&&>(prvalue<const int*>());
}

#ifndef TEMPLATE_CLASS_TEST_H // comment 1
#define TEMPLATE_CLASS_TEST_H

template <typename T>
class A {
 public:
  void f();
  void g();
  template <typename U> void h();
  template <typename U> void k();
  static int b;
  static int c;
};

template <typename T>
void A<T>::f() {}

template <typename T>
template <typename U>
void A<T>::h() {}

template <typename T>
int A<T>::b = 2;

class B {
 public:
  void f();
};

#endif // TEMPLATE_CLASS_TEST_H

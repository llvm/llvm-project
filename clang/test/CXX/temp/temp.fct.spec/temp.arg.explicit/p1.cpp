// RUN: %clang_cc1 -fsyntax-only %s

template<typename T> struct A { };

template<typename T> T make();
template<typename T> T make2(const T&);

void test_make() {
  int& ir0 = make<int&>();
  A<int> a0 = make< A<int> >();
  A<int> a1 = make2< A<int> >(A<int>());
}

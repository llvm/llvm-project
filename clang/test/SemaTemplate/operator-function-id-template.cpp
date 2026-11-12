// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T>
struct A { 
  template<typename U> A<T> operator+(U);
};

template<int Value, typename T> bool operator==(A<T>, A<T>);

template<> bool operator==<0>(A<int>, A<int>);

bool test_qualified_id(A<int> ai) {
  return ::operator==<0, int>(ai, ai);
}

void test_op(A<int> a, int i) {
  const A<int> &air = a.operator+<int>(i);
}

template<typename T>
void test_op_template(A<T> at, T x) {
  const A<T> &atr = at.template operator+<T>(x);
  const A<T> &atr2 = at.A::template operator+<T>(x);
  // FIXME: unrelated template-name instantiation issue
  //  const A<T> &atr3 = at.template A<T>::template operator+<T>(x);
}

template void test_op_template<float>(A<float>, float);

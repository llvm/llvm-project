// RUN: %clang_cc1 -fsyntax-only -Wunused-parameter -verify -std=c++11 %s
template<typename T>
struct X {
  T f0(T x);
  T f1(T x);
  T f2(T);
  template<typename U> U f3(U x);
  template<typename U> U f4(U x);
  template<typename U> U f5(U);
};

template<typename T> T X<T>::f0(T x) { return x; }
template<typename T> T X<T>::f1(T) { return T(); }
template<typename T> T X<T>::f2(T x) { return T(); } // expected-warning{{unused parameter 'x'}}
template<typename T> template<typename U> U X<T>::f3(U x) { return x; }
template<typename T> template<typename U> U X<T>::f4(U) { return U(); }
template<typename T> template<typename U> U X<T>::f5(U x) { return U(); } // expected-warning{{unused parameter 'x'}}

void test_X(X<int> &x, int i) {
  x.f0(i);
  x.f1(i);
  x.f2(i);
  x.f3(i);
  x.f4(i);
  x.f5(i);
}

// Make sure both parameters aren't considered unused.
template <typename... T>
static int test_pack(T... t, T... s)
{
  auto l = [&t...]() { return sizeof...(s); };
  return l();
}

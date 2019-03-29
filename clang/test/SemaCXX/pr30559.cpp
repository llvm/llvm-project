// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s

template < bool, class > struct A {};
template < class, int > void f () {};
template < class T, int >
decltype (f < T, 1 >) f (T t, typename A < t == 0, int >::type) {};

struct B {};

int main ()
{
  f < B, 0 >;
  return 0;
}

template <typename T>
auto foo(T x) -> decltype((x == nullptr), *x) {
  return *x;
}

void bar() {
  foo(new int);
}

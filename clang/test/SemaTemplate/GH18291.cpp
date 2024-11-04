// RUN: %clang_cc1 -std=c++23 -verify %s

namespace t1 {
template<bool> struct enable_if { typedef void type; };
template <class T> class Foo {};
template <class X> constexpr bool check() { return true; }
template <class X, class Enable = void> struct Bar {};

template<class X> void func(Bar<X, typename enable_if<check<X>()>::type>) {}
// expected-note@-1 {{candidate function}}

template<class T> void func(Bar<Foo<T>>) {}
// expected-note@-1 {{candidate function}}

void g() {
  func(Bar<Foo<int>>()); // expected-error {{call to 'func' is ambiguous}}
}
} // namespace t1

namespace t2 {
template <bool> struct enable_if;
template <> struct enable_if<true> {
  typedef int type;
};
struct pair {
  template <int = 0> pair(int);
  template <class _U2, enable_if<__is_constructible(int &, _U2)>::type = 0>
  pair(_U2 &&);
};
int test_test_i;
void test() { pair{test_test_i}; }
} // namespace t2

// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template <class T>
struct only
{
    only(T) {}

    template <class U>
    only(U)
    {
        static_assert(sizeof(U) == 0, "expected type failure");
    }
};

auto f() -> int
{
    return 0;
}

auto g(); // expected-error{{return without trailing return type}}

int h() -> int; // expected-error{{trailing return type must specify return type 'auto', not 'int'}}

int i();
auto i() -> int;
int i() {}

using T = auto (int) -> auto (*)(char) -> void; // expected-note {{previous}}
using T = void; // expected-error {{type alias redefinition with different types ('void' vs 'auto (int) -> auto (*)(char) -> void')}}

using U = auto (int) -> auto (*)(char) -> void;
using U = void (*(int))(char); // ok

int x;

template <class T>
auto i(T x) -> decltype(x)
{
    return x;
}

only<double> p1 = i(1.0);

template <class T>
struct X
{
    auto f(T x) -> T { return x; }

    template <class U>
    auto g(T x, U y) -> decltype(x + y)
    {
        return x + y;
    }

  template<typename U>
  struct nested {
    template <class V>
    auto h(T x, U y, V z) -> decltype(x + y + z)
    {
        return x + y + z;
    }
  };

  template<typename U>
  nested<U> get_nested();
};

X<int> xx;
only<int> p2 = xx.f(0L);
only<double> p3 = xx.g(0L, 1.0);
only<double> p4 = xx.get_nested<double>().h(0L, 1.0, 3.14f);

namespace PR12053 {
  template <typename T>
  auto f1(T t) -> decltype(f1(t)) {} // expected-note{{candidate template ignored}}
  
  void test_f1() {
    f1(0); // expected-error{{no matching function for call to 'f1'}}
  }
  
  template <typename T>
  auto f2(T t) -> decltype(f2(&t)) {} // expected-note{{candidate template ignored}}
  
  void test_f2() {
    f2(0); // expected-error{{no matching function for call to 'f2'}}
  }
}

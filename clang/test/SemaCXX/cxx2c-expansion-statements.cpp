// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify
namespace std {
template <typename T>
struct initializer_list {
  const T* a;
  const T* b;
  initializer_list(T* a, T* b): a{a}, b{b} {}
};
}

struct S {
  int x;
  constexpr S(int x) : x{x} {}
};

void g(int);
template <int n> constexpr int tg() { return n; }

void f1() {
  template for (auto x : {}) static_assert(false, "discarded");
  template for (constexpr auto x : {}) static_assert(false, "discarded");
  template for (auto x : {1}) g(x);
  template for (auto x : {1, 2, 3}) g(x);
  template for (constexpr auto x : {1}) g(x);
  template for (constexpr auto x : {1, 2, 3}) g(x);
  template for (constexpr auto x : {1}) tg<x>();
  template for (constexpr auto x : {1, 2, 3})
    static_assert(tg<x>());

  template for (int x : {1, 2, 3}) g(x);
  template for (S x : {1, 2, 3}) g(x.x);
  template for (constexpr S x : {1, 2, 3}) tg<x.x>();

  template for (int x : {"1", S(1), {1, 2}}) { // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'const char[2]'}} \
                                                  expected-error {{no viable conversion from 'S' to 'int'}} \
                                                  expected-error {{excess elements in scalar initializer}} \
                                                  expected-note 3 {{in instantiation of expansion statement requested here}}
    g(x);
  }

  template for (constexpr auto x : {1, 2, 3, 4}) { // expected-note 3 {{in instantiation of expansion statement requested here}}
    static_assert(tg<x>() == 4); // expected-error 3 {{static assertion failed due to requirement 'tg<x>() == 4'}} \
                                    expected-note {{expression evaluates to '1 == 4'}} \
                                    expected-note {{expression evaluates to '2 == 4'}} \
                                    expected-note {{expression evaluates to '3 == 4'}}
  }


  template for (constexpr auto x : {1, 2}) { // expected-note 2 {{in instantiation of expansion statement requested here}}
    static_assert(false, "not discarded"); // expected-error 2 {{static assertion failed: not discarded}}
  }
}

template <typename T>
void t1() {
  template for (T x : {}) g(x);
  template for (constexpr T x : {}) g(x);
  template for (auto x : {}) g(x);
  template for (constexpr auto x : {}) g(x);
  template for (T x : {1, 2}) g(x);
  template for (T x : {T(1), T(2)}) g(x);
  template for (auto x : {T(1), T(2)}) g(x);
  template for (constexpr T x : {T(1), T(2)}) static_assert(tg<x>());
  template for (constexpr auto x : {T(1), T(2)}) static_assert(tg<x>());
}

template <typename U>
struct s1 {
  template <typename T>
  void tf() {
      template for (T x : {}) g(x);
      template for (constexpr T x : {}) g(x);
      template for (U x : {}) g(x);
      template for (constexpr U x : {}) g(x);
      template for (auto x : {}) g(x);
      template for (constexpr auto x : {}) g(x);
      template for (T x : {1, 2}) g(x);
      template for (U x : {1, 2}) g(x);
      template for (U x : {T(1), T(2)}) g(x);
      template for (T x : {U(1), U(2)}) g(x);
      template for (auto x : {T(1), T(2)}) g(x);
      template for (auto x : {U(1), T(2)}) g(x);
      template for (constexpr U x : {T(1), T(2)}) static_assert(tg<x>());
      template for (constexpr T x : {U(1), U(2)}) static_assert(tg<x>());
      template for (constexpr auto x : {T(1), U(2)}) static_assert(tg<x>());
    }
};

template <typename T>
void t2() {
  template for (T x : {}) g(x);
}

void f2() {
  t1<int>();
  t1<long>();
  s1<long>().tf<long>();
  s1<int>().tf<int>();
  s1<int>().tf<long>();
  s1<long>().tf<int>();
  t2<S>();
  t2<S[1231]>();
  t2<S***>();
}

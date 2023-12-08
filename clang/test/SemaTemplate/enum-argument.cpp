// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

enum Enum { val = 1 };
template <Enum v> struct C {
  typedef C<v> Self;
};
template struct C<val>;

template<typename T>
struct get_size {
  static const unsigned value = sizeof(T);
};

template<typename T>
struct X0 {
  enum {
    Val1 = get_size<T>::value,
    Val2,
    SumOfValues = Val1 + Val2
  };
};

X0<int> x0i;

namespace rdar8020920 {
  template<typename T>
  struct X {
    enum { e0 = 32 };

    unsigned long long bitfield : e0;

    void f(int j) {
      bitfield + j;
    }
  };
}

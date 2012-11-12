// RUN: %clang_cc1 -fsyntax-only %s

// PR4607
template <class T> struct X {};

template <> struct X<char>
{
  static char* g();
};

template <class T> struct X2 {};

template <class U>
struct X2<U*> {
  static void f() {
    X<U>::g();
  }
};

void a(char *a, char *b) {X2<char*>::f();}

namespace WonkyAccess {
  template<typename T>
  struct X {
    int m;
  };

  template<typename U>
  class Y;

  template<typename U>
  struct Y<U*> : X<U> { };

  template<>
  struct Y<float*> : X<float> { };

  int f(Y<int*> y, Y<float*> y2) {
    return y.m + y2.m;
  }
}

// <rdar://problem/9169404>
namespace rdar9169404 {
  template<typename T, T N> struct X { };
  template<bool C> struct X<bool, C> {
    typedef int type;
  };

  X<bool, -1>::type value;
}

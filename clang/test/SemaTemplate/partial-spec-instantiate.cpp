// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
#if __cplusplus >= 201103L
  // expected-error@-2 {{non-type template argument evaluates to -1, which cannot be narrowed to type 'bool'}}
#else
  // expected-no-diagnostics
#endif
}

// rdar://problem/39524996
namespace rdar39524996 {
  template <typename T, typename U>
  struct enable_if_not_same
  {
    typedef void type;
  };
  template <typename T>
  struct enable_if_not_same<T, T>;

  template <typename T>
  struct Wrapper {
    // Assertion triggered on trying to set twice the same partial specialization
    // enable_if_not_same<int, int>
    template <class U>
    Wrapper(const Wrapper<U>& other,
            typename enable_if_not_same<U, T>::type* = 0) {}

    explicit Wrapper(int i) {}
  };

  template <class T>
  struct Container {
    // It is important that the struct has implicit copy and move constructors.
    Container() : x() {}

    template <class U>
    Container(const Container<U>& other) : x(static_cast<T>(other.x)) {}

    // Implicit constructors are member-wise, so the field triggers instantiation
    // of T constructors and we instantiate all of them for overloading purposes.
    T x;
  };

  void takesWrapperInContainer(const Container< Wrapper<int> >& c);
  void test() {
    // Type mismatch triggers initialization with conversion which requires
    // implicit constructors to be instantiated.
    Container<int> c;
    takesWrapperInContainer(c);
  }
}

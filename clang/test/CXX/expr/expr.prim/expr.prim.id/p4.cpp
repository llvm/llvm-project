// RUN:  %clang_cc1 -std=c++2a -verify %s

namespace functions
{
  template<typename T>
  struct S {
  static void foo(int) requires false {}
  // expected-note@-1 3{{because 'false' evaluated to false}}
  static void bar(int) requires true {}
  };

  void a(int);
  void a(double);

  void baz() {
    S<int>::foo(1); // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    S<int>::bar(1);
    void (*p1)(int) = S<int>::foo; // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    void (*p3)(int) = S<int>::bar;
    decltype(S<int>::foo)* a1 = nullptr; // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    decltype(S<int>::bar)* a2 = nullptr;
  }
}

namespace methods
{
  template<typename T>
  struct A {
    static void foo(int) requires (sizeof(T) == 1) {} // expected-note 3{{because 'sizeof(char[2]) == 1' (2 == 1) evaluated to false}}
    static void bar(int) requires (sizeof(T) == 2) {} // expected-note 3{{because 'sizeof(char) == 2' (1 == 2) evaluated to false}}
    // Make sure the function body is not instantiated before constraints are checked.
    static auto baz(int) requires (sizeof(T) == 2) { return T::foo(); } // expected-note{{because 'sizeof(char) == 2' (1 == 2) evaluated to false}}
  };

  void baz() {
    A<char>::foo(1);
    A<char>::bar(1); // expected-error{{invalid reference to function 'bar': constraints not satisfied}}
    A<char>::baz(1); // expected-error{{invalid reference to function 'baz': constraints not satisfied}}
    A<char[2]>::foo(1); // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    A<char[2]>::bar(1);
    void (*p1)(int) = A<char>::foo;
    void (*p2)(int) = A<char>::bar; // expected-error{{invalid reference to function 'bar': constraints not satisfied}}
    void (*p3)(int) = A<char[2]>::foo; // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    void (*p4)(int) = A<char[2]>::bar;
    decltype(A<char>::foo)* a1 = nullptr;
    decltype(A<char>::bar)* a2 = nullptr; // expected-error{{invalid reference to function 'bar': constraints not satisfied}}
    decltype(A<char[2]>::foo)* a3 = nullptr; // expected-error{{invalid reference to function 'foo': constraints not satisfied}}
    decltype(A<char[2]>::bar)* a4 = nullptr;
  }
}

namespace operators
{
  template<typename T>
  struct A {
    A<T> operator-(A<T> b) requires (sizeof(T) == 1) { return b; } // expected-note{{because 'sizeof(int) == 1' (4 == 1) evaluated to false}}
  };

  void baz() {
    auto* x = &A<int>::operator-; // expected-error{{invalid reference to function 'operator-': constraints not satisfied}}
    auto y = &A<char>::operator-;
  }
}

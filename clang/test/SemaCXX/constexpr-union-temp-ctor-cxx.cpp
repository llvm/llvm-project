// RUN: %clang_cc1 -std=c++14 -verify -fcxx-exceptions -Werror=c++14-extensions -Werror=c++20-extensions %s

template <class> struct C {
    union {
      int i;
    };
    constexpr C() {} // expected-error {{constexpr union constructor that does not initialize any member is a C++20 extension}}
};
constexpr C<int> c;

template <class> class D {
    union {
      int i;
    };
public:
    constexpr D() {} // expected-error {{constexpr union constructor that does not initialize any member is a C++20 extension}}
};
constexpr D<int> d;

template<typename T>
struct Foo {
    union {
      int i;
    };
    constexpr Foo(int a): i(a){}
};
constexpr Foo<int> f(5);

// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wconversion -verify %s

struct Test {
    int a = 0;
    int b = 42;
};

template <Test t>
struct A {
    static constexpr auto a = t.a;
    static constexpr auto b = t.b;
};

template <auto N>
struct Auto {};

template <typename T, T elem>
struct Explicit{};

struct L {};
struct M {};

struct Constructor {
    Constructor(L) {}; // expected-note {{here}}
    constexpr Constructor(M){};
};

template < Test = {} >
struct DefaultParam1{};

template < Test = {1, 2} >
struct DefaultParam2{};

template < Test = {. b = 5} >
struct DefaultParam3{};

void test() {
    static_assert(A<{}>::a == 0);
    static_assert(A<{}>::b == 42);
    static_assert(A<{.a = 3}>::a == 3);
    static_assert(A<{.b = 4}>::b == 4);

    Auto<{0}> a; // expected-error {{cannot deduce type of initializer list}}

    int notconst = 0; // expected-note {{declared here}}
    A<{notconst}> _; // expected-error {{non-type template argument is not a constant expression}} \
                     // expected-note  {{read of non-const variable 'notconst' is not allowed in a constant expression}}


    Explicit<Constructor, {L{}}> err; // expected-error {{non-type template argument is not a constant expression}} \
                                      // expected-note {{non-constexpr constructor 'Constructor' cannot be used in a constant expression}}
    Explicit<Constructor, {M{}}> ok;


    DefaultParam1<> d1;
    DefaultParam2<> d2;
    DefaultParam3<> d3;
}

template<auto n> struct B { /* ... */ };
template<int i> struct C { /* ... */ };
C<{ 42 }> c1;  // expected-warning {{braces around scalar initializer}}

struct J1 {
  J1 *self=this;
};
B<J1{}> j1;  // expected-error {{pointer to temporary object is not allowed in a template argument}}

struct J2 {
  J2 *self=this;
  constexpr J2() {}
  constexpr J2(const J2&) {}
};
B<J2{}> j2;  // expected-error {{pointer to temporary object is not allowed in a template argument}}


namespace GH58434 {

template<int>
void f();

void test() {
  f<{42}>();
}

}

namespace GH73666 {

template<class T, int I>
struct A {
    T x[I];
};

template< class T, class... U >
A( T, U... ) -> A<T, 1 + sizeof...(U)>;

template<A a> void foo() { }

void bar() {
    foo<{1}>();
}

}

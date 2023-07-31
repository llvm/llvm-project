// RUN: %clang_cc1 -std=c++2a -Wno-unused-value %s -verify
// RUN: %clang_cc1 -std=c++2b -Wno-unused-value %s -verify

consteval int id(int i) { return i; }
constexpr char id(char c) { return c; }

template <typename T>
constexpr int f(T t) { // expected-note {{declared here}}
    return t + id(t);  // expected-note 2{{'f<int>' is an immediate function because its body contains a call to a consteval function 'id' and that call is not a constant expression}}
}

namespace examples {

auto a = &f<char>; // ok, f<char> is not an immediate function
auto b = &f<int>;  // expected-error {{cannot take address of immediate function 'f<int>' outside of an immediate invocation}}

static_assert(f(3) == 6); // ok

template <typename T>
constexpr int g(T t) {    // g<int> is not an immediate function
    return t + id(42);    // because id(42) is already a constant
}

template <typename T, typename F>
constexpr bool is_not(T t, F f) {
    return not f(t);
}

consteval bool is_even(int i) { return i % 2 == 0; }

static_assert(is_not(5, is_even));

int x = 0; // expected-note {{declared here}}

template <typename T>
constexpr T h(T t = id(x)) { // expected-note {{read of non-const variable 'x' is not allowed in a constant expression}} \
                             // expected-note {{'hh<int>' is an immediate function because its body contains a call to a consteval function 'id' and that call is not a constant expression}}
    return t;
}

template <typename T>
constexpr T hh() {           // hh<int> is an immediate function
    [[maybe_unused]] auto x = h<T>();
    return h<T>();
}

int i = hh<int>(); // expected-error {{call to immediate function 'examples::hh<int>' is not a constant expression}} \
                   // expected-note {{in call to 'hh()'}}

struct A {
  int x;
  int y = id(x);
};

template <typename T>
constexpr int k(int) {
  return A(42).y;
}

}

namespace nested {

template <typename T>
constexpr int fdupe(T t) {
    return id(t);
}

struct a {
  constexpr a(int) { }
};

a aa(fdupe<int>((f<int>(7))));

template <typename T>
constexpr int foo(T t);     // expected-note {{declared here}}

a bb(f<int>(foo<int>(7))); // expected-error{{call to immediate function 'f<int>' is not a constant expression}} \
                           // expected-note{{undefined function 'foo<int>' cannot be used in a constant expression}}

}

namespace e2{
template <typename T>
constexpr int f(T t);
auto a = &f<char>;
auto b = &f<int>;
}

namespace forward_declare_constexpr{
template <typename T>
constexpr int f(T t);

auto a = &f<char>;
auto b = &f<int>;

template <typename T>
constexpr int f(T t) {
    return id(0);
}
}

namespace forward_declare_consteval{
template <typename T>
constexpr int f(T t);  // expected-note {{'f<int>' defined here}}

auto a = &f<char>;
auto b = &f<int>; // expected-error {{immediate function 'f<int>' used before it is defined}} \
                  // expected-note {{in instantiation of function template specialization}}

template <typename T>
constexpr int f(T t) {
    return id(t); // expected-note {{'f<int>' is an immediate function because its body contains a call to a consteval function 'id' and that call is not a constant expression}}
}
}

namespace constructors {
consteval int f(int) {
  return 0;
}
struct S {
  constexpr S(auto i) {
    f(i);
  }
};
constexpr void g(auto i) {
  [[maybe_unused]] S s{i};
}
void test() {
  g(0);
}
}

namespace aggregate {
consteval int f(int);
struct S{
    int a = 0;
    int b = f(a);
};

constexpr bool test(auto i) {
    S s{i};
    return s.b == 2 *i;
}
consteval int f(int i) {
    return 2 * i;
}

void test() {
    static_assert(test(42));
}

}

namespace ConstevalConstructor{
int x = 0; // expected-note {{declared here}}
struct S {
    consteval S(int) {};
};
constexpr int g(auto t) {
    S s(t); // expected-note {{'g<int>' is an immediate function because its body contains a call to a consteval constructor 'S' and that call is not a constant expression}}
    return 0;
}
int i = g(x); // expected-error {{call to immediate function 'ConstevalConstructor::g<int>' is not a constant expression}} \
              // expected-note {{read of non-const variable 'x' is not allowed in a constant expression}}
}



namespace Aggregate {
consteval int f(int); // expected-note {{declared here}}
struct S {
  int x = f(42); // expected-note {{undefined function 'f' cannot be used in a constant expression}} \
                 // expected-note {{'immediate<int>' is an immediate function because its body contains a call to a consteval function 'f' and that call is not a constant expression}}
};

constexpr S immediate(auto) {
    return S{};
}

void test_runtime() {
    (void)immediate(0); // expected-error {{call to immediate function 'Aggregate::immediate<int>' is not a constant expression}} \
                        // expected-note {{in call to 'immediate(0)'}}
}
consteval int f(int i) {
    return i;
}
consteval void test() {
    constexpr S s = immediate(0);
    static_assert(s.x == 42);
}
}



namespace GH63742 {
void side_effect(); // expected-note  {{declared here}}
consteval int f(int x) {
    if (!x) side_effect(); // expected-note {{non-constexpr function 'side_effect' cannot be used in a constant expression}}
    return x;
}
struct SS {
  int y = f(1); // Ok
  int x = f(0); // expected-error {{call to consteval function 'GH63742::f' is not a constant expression}} \
                // expected-note  {{declared here}} \
                // expected-note  {{in call to 'f(0)'}}
  SS();
};
SS::SS(){} // expected-note {{in the default initializer of 'x'}}

consteval int f2(int x) {
    if (!__builtin_is_constant_evaluated()) side_effect();
    return x;
}
struct S2 {
    int x = f2(0);
    constexpr S2();
};

constexpr S2::S2(){}
S2 s = {};
constinit S2 s2 = {};

struct S3 {
    int x = f2(0);
    S3();
};
S3::S3(){}

}

namespace Defaulted {
consteval int f(int x);
struct SS {
  int x = f(0);
  SS() = default;
};
}

namespace DefaultedUse{
consteval int f(int x);  // expected-note {{declared here}}
struct SS {
  int a = sizeof(f(0)); // Ok
  int x = f(0); // expected-note {{undefined function 'f' cannot be used in a constant expression}}

  SS() = default; // expected-note {{'SS' is an immediate constructor because the default initializer of 'x' contains a call to a consteval function 'f' and that call is not a constant expression}}
};

void test() {
    [[maybe_unused]] SS s; // expected-error {{call to immediate function 'DefaultedUse::SS::SS' is not a constant expression}} \
                           //  expected-note {{in call to 'SS()'}}
}
}

namespace UserDefinedConstructors {
consteval int f(int x) {
    return x;
}
extern int NonConst; // expected-note 2{{declared here}}

struct ConstevalCtr {
    int y;
    int x = f(y);
    consteval ConstevalCtr(int yy)
    : y(f(yy)) {}
};

ConstevalCtr c1(1);
ConstevalCtr c2(NonConst);
// expected-error@-1 {{call to consteval function 'UserDefinedConstructors::ConstevalCtr::ConstevalCtr' is not a constant expression}} \
// expected-note@-1 {{read of non-const variable 'NonConst' is not allowed in a constant expression}}

struct ImmediateEscalating {
    int y;
    int x = f(y);
    template<typename T>
    constexpr ImmediateEscalating(T yy) // expected-note {{ImmediateEscalating<int>' is an immediate constructor because the initializer of 'y' contains a call to a consteval function 'f' and that call is not a constant expression}}
    : y(f(yy)) {}
};

ImmediateEscalating c3(1);
ImmediateEscalating c4(NonConst);
// expected-error@-1 {{call to immediate function 'UserDefinedConstructors::ImmediateEscalating::ImmediateEscalating<int>' is not a constant expression}} \
// expected-note@-1 {{read of non-const variable 'NonConst' is not allowed in a constant expression}}


struct NonEscalating {
    int y;
    int x = f(this->y); // expected-error {{call to consteval function 'UserDefinedConstructors::f' is not a constant expression}} \
                        // expected-note  {{declared here}} \
                        // expected-note  {{use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}}
    constexpr NonEscalating(int yy) : y(yy) {} // expected-note {{in the default initializer of 'x'}}
};
NonEscalating s = {1};

}

namespace AggregateInit {

consteval int f(int x) {
    return x;
}

struct S {
    int i;
    int j = f(i);
};

constexpr S  test(auto) {
    return {};
}

S s = test(0);

}

namespace GlobalAggregateInit {

consteval int f(int x) {
    return x;
}

struct S {
    int i;
    int j = f(i); // expected-error {{call to consteval function 'GlobalAggregateInit::f' is not a constant expression}} \
                  // expected-note {{implicit use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}} \
                  // expected-note {{declared here}}
};

S s(0); // expected-note {{in the default initializer of 'j'}}

}

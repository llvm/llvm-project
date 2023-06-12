// RUN: %clang_cc1 -std=c++2a -emit-llvm-only -Wno-unused-value %s -verify
// RUN: %clang_cc1 -std=c++2b -emit-llvm-only -Wno-unused-value %s -verify

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
                             // expected-note 2{{'hh<int>' is an immediate function because its body contains a call to a consteval function 'id' and that call is not a constant expression}}
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

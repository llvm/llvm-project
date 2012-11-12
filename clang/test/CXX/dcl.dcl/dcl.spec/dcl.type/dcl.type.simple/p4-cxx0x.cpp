// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

const int&& foo();
int i;
struct A { double x; };
const A* a = new A();

static_assert(is_same<decltype(foo()), const int&&>::value, "");
static_assert(is_same<decltype(i), int>::value, "");
static_assert(is_same<decltype(a->x), double>::value, "");
static_assert(is_same<decltype((a->x)), const double&>::value, "");
static_assert(is_same<decltype(static_cast<int&&>(i)), int&&>::value, "");

int f0(int); // expected-note{{possible target}}
float f0(float); // expected-note{{possible target}}

decltype(f0) f0_a; // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}

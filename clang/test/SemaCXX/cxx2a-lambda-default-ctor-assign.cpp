// RUN: %clang_cc1 -std=c++2a -verify %s

auto a = []{};
decltype(a) a2;
void f(decltype(a) x, decltype(a) y) {
  x = y;
  x = static_cast<decltype(a)&&>(y);
}

template<typename T>
struct X {
  constexpr X() { T::error; } // expected-error {{'::'}}
  X(const X&);
  constexpr X &operator=(const X&) { T::error; } // expected-error {{'::'}}
  constexpr X &operator=(X&&) { T::error; } // expected-error {{'::'}}
};
extern X<int> x;
auto b = [x = x]{}; // expected-note 3{{in instantiation of}}
decltype(b) b2; // expected-note {{in implicit default constructor}}
void f(decltype(b) x, decltype(b) y) {
  x = y; // expected-note {{in implicit copy assignment}}
  x = static_cast<decltype(b)&&>(y); // expected-note {{in implicit move assignment}}
}

struct Y {
  Y() = delete; // expected-note {{deleted}}
  Y(const Y&);
  Y &operator=(const Y&) = delete; // expected-note 2{{deleted}}
  Y &operator=(Y&&) = delete;
};
extern Y y;
auto c = [y = y]{}; // expected-note 3{{deleted because}}
decltype(c) c2; // expected-error {{deleted}}
void f(decltype(c) x, decltype(c) y) {
  x = y; // expected-error {{deleted}}
  x = static_cast<decltype(c)&&>(y); // expected-error {{deleted}}
}

// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

struct S { int a; double b; };
S getS();

void test() {
  auto [x]; // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}}
  auto &[a, b]; // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}}
  const auto &[p, q]; // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}}
  auto &&[r, s]; // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}}

  auto [c, d] e = getS(); // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}} \
                           // expected-error {{expected ';' at end of declaration}}
}

template <typename T>
void test_template(T) {
  auto [x, y]; // expected-error {{requires an initializer; expected '=', '(', or a braced initializer list}}
}

void test_valid() {
  S s = {1, 2.0};
  auto [a, b] = s;
  auto &[c, d] = s;
  auto &&[e, f] = getS();
}

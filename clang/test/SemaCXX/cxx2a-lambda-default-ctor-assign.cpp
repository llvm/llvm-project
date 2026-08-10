// RUN: %clang_cc1 -std=c++2a -verify %s

void no_capture() {
  auto x = [] {};
  decltype(x) y;
  x = x;
  x = static_cast<decltype(x)&&>(x);
}

void capture_default(int i) {
  auto x = [=] {}; // expected-note 2{{candidate constructor}} expected-note 2{{lambda expression begins here}}
  decltype(x) y; // expected-error {{no matching constructor}}
  x = x; // expected-error {{cannot be assigned}}
  x = static_cast<decltype(x)&&>(x); // expected-error {{cannot be assigned}}
}

void explicit_capture(int i) {
  auto x = [i] {}; // expected-note 2{{candidate constructor}} expected-note 2{{lambda expression begins here}}
  decltype(x) y; // expected-error {{no matching constructor}}
  x = x; // expected-error {{cannot be assigned}}
  x = static_cast<decltype(x)&&>(x); // expected-error {{cannot be assigned}}
}

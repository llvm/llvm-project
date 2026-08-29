// RUN: %clang_cc1 -analyze -std=c++23 -analyzer-checker=core.DivideZero -verify %s

int implicit_capture_by_value() {
  int d = 0;
  auto lam = [d]() { return 1 / d; }; // expected-warning {{Division by zero}}
  return lam(); 
}

int explicit_rvalue_self_capture_by_reference() {
  int d = 0;
  auto lam = [&d](this auto &&self) { return 1 / d; }; // expected-warning {{Division by zero}}
  return lam(); 
}

int gh218708_explicit_rvalue_self() {
  int d = 0;
  auto lam = [d](this auto &&self) { return 1 / d; }; // expected-warning {{Division by zero}}
  return lam(); 
}

int gh218708_explicit_lvalue_self() {
  int d = 0;
  auto lam = [d](this auto &self) { return 1 / d; }; // expected-warning {{Division by zero}}
  return lam(); 
}

int gh218708_explicit_by_value_self() {
  int d = 0;
  auto lam = [d](this auto self) { return 1 / d; }; // expected-warning {{Division by zero}}
  return lam();
}

int explicit_rvalue_no_error() {
  int d = 5;
  auto lam = [d](this auto &&self) { return 1 / d; }; // 'd' is non-zero so there is no division by zero error. 
  return lam(); 
}

int explicit_by_value_no_error() {
  int d = 9;
  auto lam = [d](this auto self) { return 1 / d; }; // 'd' is non-zero so there is no division by zero error.
  return lam();
}

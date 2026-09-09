// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=core,cplusplus.Move -verify %s

#include "Inputs/system-header-simulator-cxx.h"

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
  auto lam = [d](this auto &&self) { return 1 / d; }; // no-warning 
  return lam(); 
}

int explicit_by_value_no_error() {
  int d = 9;
  auto lam = [d](this auto self) { return 1 / d; }; // no-warning 
  return lam();
}

auto by_val() {
  std::vector<int> v;
  auto lam = [v](this auto self) {
    auto res = std::move(v);
    return res;
  };

  lam();
  lam();
}

auto by_lval() {
  std::vector<int> v;
  auto lam = [v](this auto &self) {
    auto res = std::move(v); // expected-warning {{Moved-from object '' of type 'std::vector' is moved}}
    return res;
  };

  lam();
  lam();
}

auto by_rval() {
  std::vector<int> v;
  auto lam = [v](this auto &&self) {
    auto res = std::move(v); // expected-warning {{Moved-from object '' of type 'std::vector' is moved}}
    return res;
  };

  std::move(lam)();
  std::move(lam)();
}

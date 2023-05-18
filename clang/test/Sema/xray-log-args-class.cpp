// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c++11 -x c++

class Class {
  [[clang::xray_always_instrument, clang::xray_log_args(1)]] void Method();
  [[clang::xray_log_args(-1)]] void Invalid(); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}
  [[clang::xray_log_args("invalid")]] void InvalidStringArg(); // expected-error {{'xray_log_args'}}
};

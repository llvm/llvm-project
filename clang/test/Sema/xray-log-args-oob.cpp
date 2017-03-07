// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c++11 -x c++
void foo [[clang::xray_log_args(1)]] (int);
struct [[clang::xray_log_args(1)]] a { int x; }; // expected-warning {{'xray_log_args' attribute only applies to functions and methods}}

void fop [[clang::xray_log_args(1)]] (); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

void foq [[clang::xray_log_args(-1)]] (); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

void fos [[clang::xray_log_args(0)]] (); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

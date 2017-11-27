// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c11
void foo(int) __attribute__((xray_log_args(1)));
struct __attribute__((xray_log_args(1))) a { int x; }; // expected-warning {{'xray_log_args' attribute only applies to functions and Objective-C methods}}

void fop() __attribute__((xray_log_args(1))); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

void foq() __attribute__((xray_log_args(-1))); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

void fos() __attribute__((xray_log_args(0))); // expected-error {{'xray_log_args' attribute parameter 1 is out of bounds}}

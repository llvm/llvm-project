// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin11 -fobjc-arc -verify %s

# 1 "<command line>"
# 1 "/System/Library/Frameworks/Foundation.framework/Headers/Foundation.h" 1 3
id * foo(void); // expected-note {{declaration uses type that is ill-formed in ARC}}

# 1 "arc-unavailable-system-function.m" 2
void ret(void) {
  foo(); // expected-error {{'foo' is unavailable in ARC}}
}



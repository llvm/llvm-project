// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin11 -fobjc-arc -verify %s
// rdar://10186625

# 1 "<command line>"
# 1 "/System/Library/Frameworks/Foundation.framework/Headers/Foundation.h" 1 3
id * foo(); // expected-note {{function has been explicitly marked unavailable here}}

# 1 "arc-unavailable-system-function.m" 2
void ret() {
  foo(); // expected-error {{'foo' is unavailable: this system declaration uses an unsupported type}}
}



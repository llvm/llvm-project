// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify %s
// expected-no-diagnostics

namespace Name {
int __attribute((target_version("default"))) foo() { return 0; }
}

namespace Name {
int __attribute((target_version("sve"))) foo() { return 1; }
}

int bar() { return Name::foo(); }

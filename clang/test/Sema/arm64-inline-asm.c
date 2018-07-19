// RUN: %clang_cc1 -triple arm64-apple-ios7.1 -fsyntax-only -verify %s

void foo() {
  asm volatile("USE(%0)" :: "z"(0LL));
  asm volatile("USE(%x0)" :: "z"(0LL));
  asm volatile("USE(%w0)" :: "z"(0));

  asm volatile("USE(%0)" :: "z"(0)); // expected-warning {{value size does not match register size specified by the constraint and modifier}} expected-note {{use constraint modifier "w"}}
}

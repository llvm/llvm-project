// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 %s -triple riscv64 -verify

__attribute__((norelax)) int var; // expected-warning {{'norelax' attribute only applies to functions and methods}}

__attribute__((norelax)) void func() {}
__attribute__((norelax(1))) void func_invalid(); // expected-error {{'norelax' attribute takes no arguments}}

[[riscv::norelax]] int var2; // expected-warning {{'norelax' attribute only applies to functions and methods}}

[[riscv::norelax]] void func2() {}
[[riscv::norelax(1)]] void func_invalid2(); // expected-error {{'norelax' attribute takes no arguments}}

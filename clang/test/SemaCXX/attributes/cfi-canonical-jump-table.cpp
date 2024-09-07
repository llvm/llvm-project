// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsyntax-only -verify %s

__attribute__((cfi_canonical_jump_table)) void fdecl();

__attribute__((cfi_canonical_jump_table)) void f() {}

struct S {
  __attribute__((cfi_canonical_jump_table)) void f() {}
};

__attribute__((cfi_canonical_jump_table)) int i; // expected-error {{'cfi_canonical_jump_table' attribute only applies to functions}}

// RUN: %clang_cc1 %s -std=c23 -triple riscv64 -target-feature +v -verify

__attribute__((riscv_vector_cc)) int var; // expected-warning {{'riscv_vector_cc' only applies to function types; type here is 'int'}}

__attribute__((riscv_vector_cc)) void func();
__attribute__((riscv_vector_cc(1))) void func_invalid(); // expected-error {{'riscv_vector_cc' attribute takes no arguments}}

void test_no_attribute(int); // expected-note {{previous declaration is here}}
void __attribute__((riscv_vector_cc)) test_no_attribute(int x) { } // expected-error {{function declared 'riscv_vector_cc' here was previously declared without calling convention}}

[[riscv::vector_cc]] int var2; // expected-warning {{'vector_cc' only applies to function types; type here is 'int'}}

[[riscv::vector_cc]] void func2();
[[riscv::vector_cc(1)]] void func_invalid2(); // expected-error {{'vector_cc' attribute takes no arguments}}

void test_no_attribute2(int); // expected-note {{previous declaration is here}}
[[riscv::vector_cc]] void test_no_attribute2(int x) { } // expected-error {{function declared 'riscv_vector_cc' here was previously declared without calling convention}}

__attribute__((riscv_vls_cc)) int var_vls; // expected-warning {{'riscv_vls_cc' only applies to function types; type here is 'int'}}

__attribute__((riscv_vls_cc)) void func_vls();
__attribute__((riscv_vls_cc(1))) void func_vls_invalid(); // expected-error {{argument value 1 is outside the valid range [32, 65536]}}
__attribute__((riscv_vls_cc(129))) void func_vls_invalid(); // expected-error {{argument should be a power of 2}}

void test_vls_no_attribute(int); // expected-note {{previous declaration is here}}
void __attribute__((riscv_vls_cc)) test_vls_no_attribute(int x) { } // expected-error {{function declared 'riscv_vls_cc(128)' here was previously declared without calling convention}}

[[riscv::vls_cc]] int var2_vls; // expected-warning {{'vls_cc' only applies to function types; type here is 'int'}}

[[riscv::vls_cc]] void func2_vls();
[[riscv::vls_cc(1)]] void func_vls_invalid2(); // expected-error {{argument value 1 is outside the valid range [32, 65536]}}
[[riscv::vls_cc(129)]] void func_vls_invalid2(); // expected-error {{argument should be a power of 2}}

void test_vls_no_attribute2(int); // expected-note {{previous declaration is here}}
[[riscv::vls_cc]] void test_vls_no_attribute2(int x) { } // expected-error {{function declared 'riscv_vls_cc(128)' here was previously declared without calling convention}}

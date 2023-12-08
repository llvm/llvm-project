// RUN: %clang_cc1 -triple thumbv8m.base-none-eabi -mcmse -verify %s

extern "C" void foo() __attribute__((cmse_nonsecure_entry)) {}

void bar() __attribute__((cmse_nonsecure_entry)) {} // expected-error{{function type with 'cmse_nonsecure_entry' attribute must have C linkage}}

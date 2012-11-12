// RUN: %clang_cc1 %s -verify -fsyntax-only

int a __attribute__((naked)); // expected-warning {{'naked' attribute only applies to functions}}

__attribute__((naked)) int t0(void) { 
  __asm__ volatile("mov r0, #0");
}

void t1() __attribute__((naked));

void t2() __attribute__((naked(2))); // expected-error {{attribute takes no arguments}}


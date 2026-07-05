// RUN: %clang_cc1 %s -verify -fsyntax-only -triple i686-pc-linux

int a __attribute__((naked)); // expected-warning {{'naked' attribute only applies to functions}}

__attribute__((naked)) int t0(void) {
  __asm__ volatile("mov r0, #0");
}

void t1(void) __attribute__((naked));

void t2(void) __attribute__((naked(2))); // expected-error {{'naked' attribute takes no arguments}}

__attribute__((naked)) int t3(void) { // expected-note{{attribute is here}}
  return 42; // expected-error{{non-ASM statement in naked function is not supported}}
}

__attribute__((naked)) int t4(void) {
  asm("movl $42, %eax");
  asm("retl");
}

__attribute__((naked)) int t5(int x) {
  asm("movl x, %eax");
  asm("retl");
}

__attribute__((naked)) void t6(void) {
  ;
}

__attribute__((naked)) void t7(void) {
  asm("movl $42, %eax");
  ;
}

extern int x, y;

__attribute__((naked)) void t8(int z) { // expected-note{{attribute is here}}
  __asm__ ("movl $42, %1"
           : "=r"(x),
             "=r"(z) // expected-error{{parameter references not allowed in naked functions}}
           );
}

__attribute__((naked)) void t9(int z) { // expected-note{{attribute is here}}
  __asm__ ("movl %eax, %1"
           : : "r"(x),
               "r"(z) // expected-error{{parameter references not allowed in naked functions}}
           );
}

__attribute__((naked)) void t10(void) {  // expected-note{{attribute is here}}
  int a; // expected-error{{non-ASM statement in naked function is not supported}}
}

__attribute__((naked)) void t11(void) {  // expected-note{{attribute is here}}
  register int a asm("eax") = x; // expected-error{{non-ASM statement in naked function is not supported}}
}

__attribute__((naked)) void t12(void) {  // expected-note{{attribute is here}}
  register int a asm("eax"), b asm("ebx") = x; // expected-error{{non-ASM statement in naked function is not supported}}
}

__attribute__((naked)) void t13(void) {
  register int a asm("eax");
  register int b asm("ebx"), c asm("ecx");
}


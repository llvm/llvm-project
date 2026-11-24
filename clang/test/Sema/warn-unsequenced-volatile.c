// RUN: %clang_cc1 -fsyntax-only -verify=volatile,mod -Wunsequenced %s
// RUN: %clang_cc1 -fsyntax-only -verify=volatile -Wno-unsequenced -Wunsequenced-volatile %s
// RUN: %clang_cc1 -fsyntax-only -verify=mod -Wunsequenced -Wno-unsequenced-volatile %s

volatile int v;

void test(void) {
  int x = 0;
  x = v + v; // volatile-warning {{unsequenced accesses to volatile qualified 'v'}}
  x = x++ + x++; // mod-warning {{multiple unsequenced modifications to 'x'}}
}

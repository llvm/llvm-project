// RUN: %clang_cc1 -x c++ %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target
#pragma map(func2, "FUNC2A")

extern "C" void func2(int);
#pragma map(func2, "FUNC2") // expected-warning {{#pragma map has conflicting asm label; not applied to 'func2'}}

void func2(double);

void call_func2() {
  func2(1);
}


extern "C" void f1() asm("FUNC1");
#pragma map(f1, "F1") // expected-warning {{#pragma map has conflicting asm label; not applied to 'f1'}}

void call_f1() {
  f1();
}

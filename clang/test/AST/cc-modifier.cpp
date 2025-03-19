// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  %s -verify
// expected-no-diagnostics

void func();
void func2();

bool func3() {                                                                                               
  __asm__("%cc0 = %c1" : : "X"(func), "X"(func2));
  return func2 == func;
}                                                                                                                    

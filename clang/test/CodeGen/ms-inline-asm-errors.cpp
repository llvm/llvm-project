// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -x c++ %s -triple i386-apple-darwin10 -std=c++11 -fasm-blocks -verify

class A {
public:
  void foo(int a)   {}
  void foo(float a) {}
};


void t_fail() {
	__asm {
		mov ecx, [eax]A.foo // expected-error {{Unable to lookup field reference!}}
	}
}

// RUN: %clang_cc1 -triple arm64ec-windows-msvc -emit-llvm -o - %s -verify

// ARM64EC doesn't support generating __vectorcall calls... but __vectorcall
// function types need to be distinct from __cdecl function types to support
// compiling the STL. Make sure we only diagnose constructs that actually
// require generating code.
void __vectorcall f1();
void f2(void __vectorcall p()) {}
void f2(void p()) {}
void __vectorcall (*f3)();
void __vectorcall f4(); // expected-error {{__vectorcall}}
void __vectorcall f5() { // expected-error {{__vectorcall}}
  f4(); // expected-error{{__vectorcall}}
}

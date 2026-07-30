// RUN: %clang_cc1 %s -verify -fsyntax-only -triple arm-none-linux
// RUN: %clang_cc1 %s -verify -fsyntax-only -fms-compatibility -DDECLSPEC -triple i686-pc-win32

class Foo {
  void bar();
  static void bar2();
  unsigned v;
  static unsigned s;
};

void __attribute__((naked)) Foo::bar() { // expected-note{{attribute is here}}
  asm("mov r2, %0" : : "r"(v)); // expected-error{{'this' pointer references not allowed in naked functions}}
}

void __attribute__((naked)) Foo::bar2() {
  asm("mov r2, %0" : : "r"(s)); // static member reference is OK
}

struct Bar {
#ifdef DECLSPEC
  __declspec(naked) void func1(); // expected-error {{'naked' attribute only applies to non-member functions}}
  __declspec(naked) static void func2(); // expected-error {{'naked' attribute only applies to non-member functions}}
#endif
  __attribute__((naked)) void func3(); // OK
  __attribute__((naked)) static void func4(); // OK
};

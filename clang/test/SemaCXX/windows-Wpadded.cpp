// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsyntax-only -verify -Wpadded %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify -Wpadded %s

struct __attribute__((ms_struct)) Foo { // expected-warning {{padding size of 'Foo' with 3 bytes to alignment boundary}}
  int b : 1;
  char a; // expected-warning {{padding struct 'Foo' with 31 bits to align 'a'}}
};

struct __attribute__((ms_struct)) AlignedStruct { // expected-warning {{padding size of 'AlignedStruct' with 4 bytes to alignment boundary}}
  char c;
  alignas(8) int i; // expected-warning {{padding struct 'AlignedStruct' with 7 bytes to align 'i'}}
};


struct Base {
  int b;
};

struct Derived : public Base { // expected-warning {{padding size of 'Derived' with 3 bytes to alignment boundary}}
  char c;
};

union __attribute__((ms_struct)) Union {
  char c;
  long long u;
};

struct __attribute__((ms_struct)) StructWithUnion { // expected-warning {{padding size of 'StructWithUnion' with 6 bytes to alignment boundary}}
  char c;
  int : 0;
  Union t; // expected-warning {{padding struct 'StructWithUnion' with 7 bytes to align 't'}}
  short i;
};

int main() {
  Foo f;
  AlignedStruct a;
  Derived d;
  StructWithUnion swu;
}

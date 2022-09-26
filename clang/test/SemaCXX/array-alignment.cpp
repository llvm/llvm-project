// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef char __attribute__((aligned(2))) AlignedChar;
typedef AlignedChar arrayType0[4]; // expected-error {{size of array element}}

struct __attribute__((aligned(8))) AlignedStruct {
  int m0;
};

struct __attribute__((packed)) PackedStruct {
  char m0;
  int i0;
};

typedef PackedStruct AlignedPackedStruct __attribute__((aligned(4)));
typedef AlignedPackedStruct arrayType1[4]; // expected-error {{(5 bytes) isn't a multiple of its alignment (4 bytes)}}

AlignedChar a0[1]; // expected-error {{size of array element}}
AlignedStruct a1[1];
AlignedPackedStruct a2[1]; // expected-error {{size of array element}}

struct S {
  AlignedChar m0[1]; // expected-error {{size of array element}}
  AlignedStruct m1[1];
  AlignedPackedStruct m2[1]; // expected-error {{size of array element}}
};

void test(char *p) {
  auto p0 = (AlignedChar(*)[1])p;    // expected-error {{size of array element}}
  auto r0 = (AlignedChar(&)[1])(*p); // expected-error {{size of array element}}
  auto p1 = new AlignedChar[1];      // expected-error {{size of array element}}
  auto p2 = (AlignedStruct(*)[1])p;
  auto p3 = new AlignedStruct[1];
  auto p4 = (AlignedPackedStruct(*)[1])p; // expected-error {{size of array element}}
  auto p5 = new AlignedPackedStruct[1]; // expected-error {{size of array element}}
}

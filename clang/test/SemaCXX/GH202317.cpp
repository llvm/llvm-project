// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -Wconversion %s

typedef __attribute__((__ext_vector_type__(4))) char32_t vf4;
typedef __attribute__((__ext_vector_type__(4))) int vi4;

vi4 foo(vf4 &V) { return V.xyzw < V.x; }

void same_element_type(vf4 &V, char32_t u32) {
  vf4 v = u32;
  v = V.x;
  (void)(V.xyzw == u32);
  (void)(u32 < V.xyzw);
}

void different_element_type(vf4 &V, char8_t u8, char16_t u16, char32_t u32) {
  (void)(V.xyzw < u8); // expected-warning {{implicit conversion from 'char8_t' to 'vf4'}}
  vf4 v = u8;          // expected-warning {{implicit conversion from 'char8_t' to 'vf4'}}

  char16_t c16 = u32; // expected-warning {{implicit conversion from 'char32_t' to 'char16_t' may lose precision and change the meaning of the represented code unit}}
  char32_t c32 = u8;  // expected-warning {{implicit conversion from 'char8_t' to 'char32_t' may change the meaning of the represented code unit}}
  char32_t c32b = u16;
}

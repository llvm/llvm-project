// RUN: %clang_cc1 -fsyntax-only -verify -Wsign-conversion -std=c++11 %s

unsigned int test() {
  short foo;
  return foo; // expected-warning {{implicit conversion changes signedness}}

}

unsigned int test3() {
  // For a non-defined enum, use the underlying type.
  enum u8 : signed char;
  u8 foo{static_cast<u8>(0)};
  return foo; // expected-warning {{implicit conversion changes signedness}}

}
unsigned int test2() {
  // For a non-defined enum, use the underlying type.
  enum u8 : unsigned char;
  u8 foo{static_cast<u8>(0)};
  return foo;
}

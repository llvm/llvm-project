// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -emit-llvm -verify -o - %s

typedef struct { // expected-error {{special function __default_constructor_8_s8 for non-trivial C struct has incorrect type}}
  int i;
  id f1;
} StrongSmall;

int __default_constructor_8_s8(double a) {
  return 0;
}

void testIncorrectFunctionType(void) {
  StrongSmall x;
}

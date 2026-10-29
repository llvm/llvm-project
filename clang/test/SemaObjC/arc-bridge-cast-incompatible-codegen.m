// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm-only -fobjc-arc -fblocks -verify %s

typedef const struct __CFArray *CFArrayRef;

// A __bridge cast between two CF pointer types is invalid. clang used to crash
// in CodeGen when compiling the following code.
void test(CFArrayRef arr) {
  if ((__bridge CFArrayRef)arr) {} // expected-error {{incompatible types casting 'CFArrayRef' (aka 'const struct __CFArray *') to 'CFArrayRef' with a __bridge cast}}
}

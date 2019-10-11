// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wxor-used-as-pow %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wxor-used-as-pow %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#define FOOBAR(x, y) (x * y)
#define XOR(x, y) (x ^ y)
#define TWO 2
#define TEN 10
#define IOP 64
#define TWO_ULL 2ULL
#define EPSILON 10 ^ -300
#define ALPHA_OFFSET 3
#define EXP 3

#define flexor 7

#ifdef __cplusplus
constexpr long long operator"" _xor(unsigned long long v) { return v; }

constexpr long long operator"" _0b(unsigned long long v) { return v; }
constexpr long long operator"" _0X(unsigned long long v) { return v; }
#else
#define xor ^ // iso646.h
#endif

void test(unsigned a, unsigned b) {
  unsigned res;
  res = a ^ 5;
  res = 2 ^ b;
  res = a ^ b;
  res = 2 ^ -1;
  res = 2 ^ 0; // expected-warning {{result of '2 ^ 0' is 2; did you mean '1 << 0' (1)?}}
               // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:14}:"1"
               // expected-note@-2 {{replace expression with '0x2 ^ 0' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 1; // expected-warning {{result of '2 ^ 1' is 3; did you mean '1 << 1' (2)?}}
               // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:14}:"1 << 1"
               // expected-note@-2 {{replace expression with '0x2 ^ 1' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 2; // expected-warning {{result of '2 ^ 2' is 0; did you mean '1 << 2' (4)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:14}:"1 << 2"
  // expected-note@-2 {{replace expression with '0x2 ^ 2' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 8; // expected-warning {{result of '2 ^ 8' is 10; did you mean '1 << 8' (256)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:14}:"1 << 8"
  // expected-note@-2 {{replace expression with '0x2 ^ 8' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ +8; // expected-warning {{result of '2 ^ +8' is 10; did you mean '1 << +8' (256)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1 << +8"
  // expected-note@-2 {{replace expression with '0x2 ^ +8' or use 'xor' instead of '^' to silence this warning}}
  res = TWO ^ 8; // expected-warning {{result of 'TWO ^ 8' is 10; did you mean '1 << 8' (256)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:16}:"1 << 8"
  // expected-note@-2 {{replace expression with '0x2 ^ 8' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 16; // expected-warning {{result of '2 ^ 16' is 18; did you mean '1 << 16' (65536)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1 << 16"
  // expected-note@-2 {{replace expression with '0x2 ^ 16' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ TEN; // expected-warning {{result of '2 ^ TEN' is 8; did you mean '1 << TEN' (1024)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:16}:"1 << TEN"
  // expected-note@-2 {{replace expression with '0x2 ^ TEN' or use 'xor' instead of '^' to silence this warning}}
  res = res + (2 ^ ALPHA_OFFSET); // expected-warning {{result of '2 ^ ALPHA_OFFSET' is 1; did you mean '1 << ALPHA_OFFSET' (8)?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:32}:"1 << ALPHA_OFFSET"
  // expected-note@-2 {{replace expression with '0x2 ^ ALPHA_OFFSET' or use 'xor' instead of '^' to silence this warning}}
  res = 0x2 ^ 16;
  res = 2 xor 16;

  res = 2 ^ 0x4;
  res = 2 ^ 04;
  res = 0x2 ^ 10;
  res = 0X2 ^ 10;
  res = 02 ^ 10;
  res = FOOBAR(2, 16);
  res = 0b10 ^ 16;
  res = 0B10 ^ 16;
  res = 2 ^ 0b100;
  res = XOR(2, 16);
  unsigned char two = 2;
  res = two ^ 16;
  res = TWO_ULL ^ 16;
  res = 2 ^ 32; // expected-warning {{result of '2 ^ 32' is 34; did you mean '1LL << 32'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1LL << 32"
  // expected-note@-2 {{replace expression with '0x2 ^ 32' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 64; // expected-warning {{result of '2 ^ 64' is 66; did you mean exponentiation?}}
  // expected-note@-1 {{replace expression with '0x2 ^ 64' or use 'xor' instead of '^' to silence this warning}}
  res = 2 ^ 65;

  res = EPSILON;
  res = 10 ^ 0; // expected-warning {{result of '10 ^ 0' is 10; did you mean '1e0'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1e0"
  // expected-note@-2 {{replace expression with '0xA ^ 0' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ 1; // expected-warning {{result of '10 ^ 1' is 11; did you mean '1e1'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1e1"
  // expected-note@-2 {{replace expression with '0xA ^ 1' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ 2; // expected-warning {{result of '10 ^ 2' is 8; did you mean '1e2'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1e2"
  // expected-note@-2 {{replace expression with '0xA ^ 2' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ 4; // expected-warning {{result of '10 ^ 4' is 14; did you mean '1e4'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:15}:"1e4"
  // expected-note@-2 {{replace expression with '0xA ^ 4' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ +4; // expected-warning {{result of '10 ^ +4' is 14; did you mean '1e4'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:16}:"1e4"
  // expected-note@-2 {{replace expression with '0xA ^ +4' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ 10; // expected-warning {{result of '10 ^ 10' is 0; did you mean '1e10'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:16}:"1e10"
  // expected-note@-2 {{replace expression with '0xA ^ 10' or use 'xor' instead of '^' to silence this warning}}
  res = TEN ^ 10; // expected-warning {{result of 'TEN ^ 10' is 0; did you mean '1e10'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:17}:"1e10"
  // expected-note@-2 {{replace expression with '0xA ^ 10' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ TEN; // expected-warning {{result of '10 ^ TEN' is 0; did you mean '1e10'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:17}:"1e10"
  // expected-note@-2 {{replace expression with '0xA ^ TEN' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ 100; // expected-warning {{result of '10 ^ 100' is 110; did you mean '1e100'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:17}:"1e100"
  // expected-note@-2 {{replace expression with '0xA ^ 100' or use 'xor' instead of '^' to silence this warning}}
  res = 0xA ^ 10;
  res = 10 ^ -EXP; // expected-warning {{result of '10 ^ -EXP' is -9; did you mean '1e-3'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:18}:"1e-3"
  // expected-note@-2 {{replace expression with '0xA ^ -EXP' or use 'xor' instead of '^' to silence this warning}}
  res = 10 ^ +EXP; // expected-warning {{result of '10 ^ +EXP' is 9; did you mean '1e3'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:18}:"1e3"
  // expected-note@-2 {{replace expression with '0xA ^ +EXP' or use 'xor' instead of '^' to silence this warning}}
  res = 10 xor 10;
#ifdef __cplusplus
  res = 10 ^ 5_xor;
  res = 10_xor ^ 5;
  res = 10 ^ 5_0b;
  res = 10_0X ^ 5;
  res = 2 ^ 2'000;
  res = 2 ^ 0b0110'1001;
  res = 10 ^ 2'000;
#else
#undef xor
  res = 10 ^ 1; // expected-warning {{result of '10 ^ 1' is 11; did you mean '1e1'?}}
  // expected-note@-1 {{replace expression with '0xA ^ 1' to silence this warning}}
  res = 2 ^ 1; // expected-warning {{result of '2 ^ 1' is 3; did you mean '1 << 1' (2)?}}
  // expected-note@-1 {{replace expression with '0x2 ^ 1' to silence this warning}}
#endif
}

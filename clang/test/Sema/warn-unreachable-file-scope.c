// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef unsigned char u8;

u8 a1 = (0 ? 0xffff : 0xff);
u8 a2 = (1 ? 0xffff : 0xff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}
u8 a3 = (1 ? 0xff : 0xffff);
u8 a4 = (0 ? 0xff : 0xffff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}

unsigned long long b1 = 1 ? 0 : 1ULL << 64;
unsigned long long b2 = 0 ? 0 : 1ULL << 64; // expected-warning {{shift count >= width of type}}
unsigned long long b3 = 1 ? 1ULL << 64 : 0; // expected-warning {{shift count >= width of type}}

#define M(n) (((n) == 64) ? ~0ULL : ((1ULL << (n)) - 1))
unsigned long long c1 = M(64);
unsigned long long c2 = M(32);

static u8 d1 = (0 ? 0xffff : 0xff);
static u8 d2 = (1 ? 0xffff : 0xff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}

int a = 1 ? 6 : (1,2);
int b = 0 ? 6 : (1,2); // expected-warning {{left operand of comma operator has no effect}}

void f(void) {
  u8 e1 = (0 ? 0xffff : 0xff);
  u8 e2 = (1 ? 0xffff : 0xff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}

  unsigned long long e3 = 1 ? 0 : 1ULL << 64;
  unsigned long long e4 = 0 ? 0 : 1ULL << 64; // expected-warning {{shift count >= width of type}}
}

void statics(void) {
  static u8 f1 = (0 ? 0xffff : 0xff);
  static u8 f2 = (1 ? 0xffff : 0xff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}
  static u8 f3 = (1 ? 0xff : 0xffff);
  static u8 f4 = (0 ? 0xff : 0xffff); // expected-warning {{implicit conversion from 'int' to 'u8' (aka 'unsigned char') changes value from 65535 to 255}}
}

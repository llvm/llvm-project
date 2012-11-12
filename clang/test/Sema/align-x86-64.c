// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5599

void frob(void *);

void foo(void) {
  float x[4];
  char y[__alignof__(x) == 16 ? 1 : -1];
  frob(y);
}

// PR5637

typedef __attribute__((aligned(16))) struct {
  unsigned long long w[3];
} UINT192;

UINT192 ten2mk192M[] = {
    {{0xcddd6e04c0592104ULL, 0x0fcf80dc33721d53ULL, 0xa7c5ac471b478423ULL}},
    {{0xcddd6e04c0592104ULL, 0x0fcf80dc33721d53ULL, 0xa7c5ac471b478423ULL}},
    {{0xcddd6e04c0592104ULL, 0x0fcf80dc33721d53ULL, 0xa7c5ac471b478423ULL}}
};

short chk1[sizeof(ten2mk192M) == 80 ? 1 : -1];

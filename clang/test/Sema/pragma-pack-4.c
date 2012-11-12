// RUN: %clang_cc1 -triple i686-apple-darwin9 %s -fsyntax-only -verify
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -fsyntax-only -verify
// expected-no-diagnostics

// rdar://problem/7095436
#pragma pack(4)

struct s0 {
  long long a __attribute__((aligned(8)));
  long long b __attribute__((aligned(8)));
  unsigned int c __attribute__((aligned(8)));
  int d[12];
};

struct s1 {
  int a[15];
  struct s0 b;
};

int arr0[((sizeof(struct s1) % 64) == 0) ? 1 : -1];

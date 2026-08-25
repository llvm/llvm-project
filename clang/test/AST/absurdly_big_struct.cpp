// RUN: %clang_cc1 -fsyntax-only -verify=bit64 %s -triple x86_64-linux-gnu
// RUN: %clang_cc1 -fsyntax-only -verify=bit32 %s -triple armv7-unknown-linux-gnueabi

struct a { // bit64-error {{structure 'a' is too large, which exceeds maximum allowed size of 1152921504606846976 bytes}}
  char x[1ull<<60]; // bit32-error {{array is too large}}
  char x2[1ull<<60]; // bit32-error {{array is too large}}
};

a z[1];
long long x() { return sizeof(a); }
long long x2() { return sizeof(a::x); }
long long x3() { return sizeof(a::x2); }
long long x4() { return sizeof(z); }

// On 32-bit architectures, the struct size must be below (1 << 32).
struct b { // bit32-error {{structure 'b' is too large, which exceeds maximum allowed size of 4294967296 bytes}}
  char c[0xFFFFFFFF];
  char c2[1];
};

long long y() { return sizeof(b); }


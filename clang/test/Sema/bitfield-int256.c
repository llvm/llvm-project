// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-gnu %s

// Test __int256 bitfield support.

struct S1 {
  __int256 x : 200;
  __int256 y : 56;
};

_Static_assert(sizeof(struct S1) == 32, "S1 should be 32 bytes");

struct S2 {
  char a;
  __int256 x : 100;
};

struct S3 {
  unsigned __int256 x : 256; // Full width bitfield
};

_Static_assert(sizeof(struct S3) == 32, "S3 should be 32 bytes");

struct S4 {
  __int256 x : 1; // Single bit signed
  unsigned __int256 y : 1; // Single bit unsigned
};

// Test bitfield access
int test_bitfield(void) {
  struct S1 s = {};
  s.x = 42;
  s.y = -1;
  return (int)s.x + (int)s.y;
}

// Test zero-width bitfield
struct S5 {
  __int256 : 0; // Zero-width bitfield for alignment
  int x;
};

// expected-no-diagnostics

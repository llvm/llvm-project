// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=i686-apple-darwin9
// expected-no-diagnostics

#define CHECK_SIZE(kind, name, size) extern int name##1[sizeof(kind name) == size ? 1 : -1];
#define CHECK_ALIGN(kind, name, size) extern int name##2[__alignof(kind name) == size ? 1 : -1];

// Zero-width bit-fields
struct a {char x; int : 0; char y;};
CHECK_SIZE(struct, a, 5)
CHECK_ALIGN(struct, a, 1)

union b {char x; int : 0; char y;};
CHECK_SIZE(union, b, 1)
CHECK_ALIGN(union, b, 1)

// Unnamed bit-field align
struct c {char x; int : 20;};
CHECK_SIZE(struct, c, 4)
CHECK_ALIGN(struct, c, 1)

union d {char x; int : 20;};
CHECK_SIZE(union, d, 3)
CHECK_ALIGN(union, d, 1)

// Bit-field packing
struct __attribute__((packed)) e {int x : 4, y : 30, z : 30;};
CHECK_SIZE(struct, e, 8)
CHECK_ALIGN(struct, e, 1)

// Alignment on bit-fields
struct f {__attribute((aligned(8))) int x : 30, y : 30, z : 30;};
CHECK_SIZE(struct, f, 24)
CHECK_ALIGN(struct, f, 8)

// Large structure (overflows i32, in bits).
struct s0 {
  char a[0x32100000];
  int x:30, y:30;
};

CHECK_SIZE(struct, s0, 0x32100008)
CHECK_ALIGN(struct, s0, 4)


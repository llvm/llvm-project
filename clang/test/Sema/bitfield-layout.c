// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=i686-apple-darwin9
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=arm-linux-gnueabihf
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=aarch64-linux-gnu
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=x86_64-pc-linux-gnu
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=x86_64-scei-ps4
// expected-no-diagnostics
#include <stddef.h>

#define CHECK_SIZE(kind, name, size) \
  extern int name##_1[sizeof(kind name) == size ? 1 : -1];
#define CHECK_ALIGN(kind, name, size) \
  extern int name##_2[__alignof(kind name) == size ? 1 : -1];
#define CHECK_OFFSET(kind, name, member, offset) \
  extern int name##_3[offsetof(kind name, member) == offset ? 1 : -1];

// Zero-width bit-fields
struct a {char x; int : 0; char y;};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(struct, a, 8)
CHECK_ALIGN(struct, a, 4)
#else
CHECK_SIZE(struct, a, 5)
CHECK_ALIGN(struct, a, 1)
#endif

// Zero-width bit-fields with packed
struct __attribute__((packed)) a2 { short x : 9; char : 0; int y : 17; };
CHECK_SIZE(struct, a2, 5)
CHECK_ALIGN(struct, a2, 1)

// Zero-width bit-fields at the end of packed struct
struct __attribute__((packed)) a3 { short x : 9; int : 0; };
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(struct, a3, 4)
CHECK_ALIGN(struct, a3, 4)
#else
CHECK_SIZE(struct, a3, 4)
CHECK_ALIGN(struct, a3, 1)
#endif

// For comparison, non-zero-width bit-fields at the end of packed struct
struct __attribute__((packed)) a4 { short x : 9; int : 1; };
CHECK_SIZE(struct, a4, 2)
CHECK_ALIGN(struct, a4, 1)

union b {char x; int : 0; char y;};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(union, b, 4)
CHECK_ALIGN(union, b, 4)
#else
CHECK_SIZE(union, b, 1)
CHECK_ALIGN(union, b, 1)
#endif

// Unnamed bit-field align
struct c {char x; int : 20;};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(struct, c, 4)
CHECK_ALIGN(struct, c, 4)
#else
CHECK_SIZE(struct, c, 4)
CHECK_ALIGN(struct, c, 1)
#endif

union d {char x; int : 20;};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(union, d, 4)
CHECK_ALIGN(union, d, 4)
#else
CHECK_SIZE(union, d, 3)
CHECK_ALIGN(union, d, 1)
#endif

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

// Bit-field with explicit align bigger than normal.
struct g0 {
  char a;
  __attribute__((aligned(16))) int b : 1;
  char c;
};

#if defined(__PS4__)
CHECK_SIZE(struct, g0, 16);
CHECK_ALIGN(struct, g0, 16);
CHECK_OFFSET(struct, g0, c, 2);
#else
CHECK_SIZE(struct, g0, 32);
CHECK_ALIGN(struct, g0, 16);
CHECK_OFFSET(struct, g0, c, 17);
#endif

// Bit-field with explicit align smaller than normal.
struct g1 {
  char a;
  __attribute__((aligned(2))) int b : 1;
  char c;
};

CHECK_SIZE(struct, g1, 4);
CHECK_ALIGN(struct, g1, 4);
#if defined(__PS4__)
CHECK_OFFSET(struct, g1, c, 2);
#else
CHECK_OFFSET(struct, g1, c, 3);
#endif

// Same as above but without explicit align.
struct g2 {
  char a;
  int b : 1;
  char c;
};

CHECK_SIZE(struct, g2, 4);
CHECK_ALIGN(struct, g2, 4);
CHECK_OFFSET(struct, g2, c, 2);

// Explicit attribute align on bit-field has precedence over packed attribute
// applied too the struct.
struct __attribute__((packed)) g3 {
  char a;
  __attribute__((aligned(16))) int b : 1;
  char c;
};

CHECK_ALIGN(struct, g3, 16);
#if defined(__PS4__)
CHECK_SIZE(struct, g3, 16);
CHECK_OFFSET(struct, g3, c, 2);
#else
CHECK_SIZE(struct, g3, 32);
CHECK_OFFSET(struct, g3, c, 17);
#endif

struct __attribute__((packed)) g4 {
  char a;
  __attribute__((aligned(2))) int b : 1;
  char c;
};

CHECK_SIZE(struct, g4, 4);
CHECK_ALIGN(struct, g4, 2);
#if defined(__PS4__)
CHECK_OFFSET(struct, g4, c, 2);
#else
CHECK_OFFSET(struct, g4, c, 3);
#endif

struct g5 {
  char : 1;
  __attribute__((aligned(1))) int n : 24;
};
CHECK_SIZE(struct, g5, 4);
CHECK_ALIGN(struct, g5, 4);

struct __attribute__((packed)) g6 {
  char : 1;
  __attribute__((aligned(1))) int n : 24;
};
CHECK_SIZE(struct, g6, 4);
CHECK_ALIGN(struct, g6, 1);

struct g7 {
  char : 1;
  __attribute__((aligned(1))) int n : 25;
};
#if defined(__PS4__)
CHECK_SIZE(struct, g7, 4);
#else
CHECK_SIZE(struct, g7, 8);
#endif
CHECK_ALIGN(struct, g7, 4);

struct __attribute__((packed)) g8 {
  char : 1;
  __attribute__((aligned(1))) int n : 25;
};
#if defined(__PS4__)
CHECK_SIZE(struct, g8, 4);
#else
CHECK_SIZE(struct, g8, 5);
#endif
CHECK_ALIGN(struct, g8, 1);

struct g9 {
  __attribute__((aligned(1))) char a : 2, b : 2, c : 2, d : 2, e : 2;
  int i;
};
#if defined(__PS4__)
CHECK_SIZE(struct, g9, 8);
#else
CHECK_SIZE(struct, g9, 12);
#endif
CHECK_ALIGN(struct, g9, 4);

struct __attribute__((packed)) g10 {
  __attribute__((aligned(1))) char a : 2, b : 2, c : 2, d : 2, e : 2;
  int i;
};
#if defined(__PS4__)
CHECK_SIZE(struct, g10, 6);
#else
CHECK_SIZE(struct, g10, 9);
#endif
CHECK_ALIGN(struct, g10, 1);

struct g11 {
  char a;
  __attribute__((aligned(1))) long long b : 62;
  char c;
};
#if defined(__arm__) || defined(__aarch64__) || defined(__x86_64__)
CHECK_SIZE(struct, g11, 24);
CHECK_ALIGN(struct, g11, 8);
CHECK_OFFSET(struct, g11, c, 16);
#else
CHECK_SIZE(struct, g11, 16);
CHECK_ALIGN(struct, g11, 4);
CHECK_OFFSET(struct, g11, c, 12);
#endif

struct __attribute__((packed)) g12 {
  char a;
  __attribute__((aligned(1))) long long b : 62;
  char c;
};
CHECK_SIZE(struct, g12, 10);
CHECK_ALIGN(struct, g12, 1);
CHECK_OFFSET(struct, g12, c, 9);

struct g13 {
  char a;
  __attribute__((aligned(1))) long long : 0;
  char c;
};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(struct, g13, 16);
CHECK_ALIGN(struct, g13, 8);
CHECK_OFFSET(struct, g13, c, 8);
#elif (__x86_64__)
CHECK_SIZE(struct, g13, 9);
CHECK_ALIGN(struct, g13, 1);
CHECK_OFFSET(struct, g13, c, 8);
#else
CHECK_SIZE(struct, g13, 5);
CHECK_ALIGN(struct, g13, 1);
CHECK_OFFSET(struct, g13, c, 4);
#endif

struct __attribute__((packed)) g14 {
  char a;
  __attribute__((aligned(1))) long long : 0;
  char c;
};
#if defined(__arm__) || defined(__aarch64__)
CHECK_SIZE(struct, g14, 16);
CHECK_ALIGN(struct, g14, 8);
CHECK_OFFSET(struct, g14, c, 8);
#elif (__x86_64__)
CHECK_SIZE(struct, g14, 9);
CHECK_ALIGN(struct, g14, 1);
CHECK_OFFSET(struct, g14, c, 8);
#else
CHECK_SIZE(struct, g14, 5);
CHECK_ALIGN(struct, g14, 1);
CHECK_OFFSET(struct, g14, c, 4);
#endif

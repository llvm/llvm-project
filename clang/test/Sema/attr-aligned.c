// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

int x __attribute__((aligned(3))); // expected-error {{requested alignment is not a power of 2}}
int y __attribute__((aligned(1 << 29))); // expected-error {{requested alignment must be 268435456 bytes or smaller}}

// PR3254
short g0[3] __attribute__((aligned));
short g0_chk[__alignof__(g0) == 16 ? 1 : -1]; 

// <rdar://problem/6840045>
typedef char ueber_aligned_char __attribute__((aligned(8)));

struct struct_with_ueber_char {
  ueber_aligned_char c;
};

char a = 0;

char a0[__alignof__(ueber_aligned_char) == 8? 1 : -1] = { 0 };
char a1[__alignof__(struct struct_with_ueber_char) == 8? 1 : -1] = { 0 };
char a2[__alignof__(a) == 1? : -1] = { 0 };
char a3[sizeof(a) == 1? : -1] = { 0 };

typedef long long __attribute__((aligned(1))) underaligned_longlong;
char a4[__alignof__(underaligned_longlong) == 1 ?: -1] = {0};

typedef long long __attribute__((aligned(1))) underaligned_complex_longlong;
char a5[__alignof__(underaligned_complex_longlong) == 1 ?: -1] = {0};

// rdar://problem/8335865
int b __attribute__((aligned(2)));
char b1[__alignof__(b) == 2 ?: -1] = {0};

struct C { int member __attribute__((aligned(2))); } c;
char c1[__alignof__(c) == 4 ?: -1] = {0};
char c2[__alignof__(c.member) == 4 ?: -1] = {0};

struct D { int member __attribute__((aligned(2))) __attribute__((packed)); } d;
char d1[__alignof__(d) == 2 ?: -1] = {0};
char d2[__alignof__(d.member) == 2 ?: -1] = {0};

struct E { int member __attribute__((aligned(2))); } __attribute__((packed));
struct E e;
char e1[__alignof__(e) == 2 ?: -1] = {0};
char e2[__alignof__(e.member) == 2 ?: -1] = {0};

typedef char overaligned_char __attribute__((aligned(16)));
typedef overaligned_char array_with_overaligned_char[11];
typedef char array_with_align_attr[11] __attribute__((aligned(16)));

char f0[__alignof__(array_with_overaligned_char) == 16 ? 1 : -1] = { 0 };
char f1[__alignof__(array_with_align_attr) == 16 ? 1 : -1] = { 0 };
array_with_overaligned_char F2;
char f2[__alignof__(F2) == 16 ? 1 : -1] = { 0 };
array_with_align_attr F3;
char f3[__alignof__(F3) == 16 ? 1 : -1] = { 0 };

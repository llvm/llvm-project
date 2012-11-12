// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

int x __attribute__((aligned(3))); // expected-error {{requested alignment is not a power of 2}}

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

// rdar://problem/8335865
int b __attribute__((aligned(2)));
char b1[__alignof__(b) == 2 ?: -1] = {0};

struct C { int member __attribute__((aligned(2))); } c;
char c1[__alignof__(c) == 4 ?: -1] = {0};
char c2[__alignof__(c.member) == 4 ?: -1] = {0};

struct D { int member __attribute__((aligned(2))) __attribute__((packed)); } d;
char d1[__alignof__(d) == 2 ?: -1] = {0};
char d2[__alignof__(d.member) == 2 ?: -1] = {0};

struct E { int member __attribute__((align(2))); } __attribute__((packed));
struct E e;
char e1[__alignof__(e) == 2 ?: -1] = {0};
char e2[__alignof__(e.member) == 2 ?: -1] = {0};

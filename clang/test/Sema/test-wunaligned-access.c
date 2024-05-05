// RUN: %clang_cc1 %s -triple=armv7-none-none-eabi -verify -Wunaligned-access -emit-llvm -o %t
// REQUIRES: arm-registered-target
//
// This test suite tests the warning triggered by the -Wunaligned-access option.
// The warning occurs when a struct or other type of record contains a field
// that is itself a record. The outer record must be a packed structure, while
// while the inner record must be unpacked. This is the fundamental condition
// for the warning to be triggered. Some of these tests may have three layers.
//
// The command line option -fsyntax-only is not used as Clang needs to be
// forced to layout the structs used in this test.
// The triple in the command line above is used for the assumptions about
// size and alignment of types.

// Set 1
struct T1 {
  char a;
  int b;
};

struct __attribute__((packed)) U1 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U1' is less aligned than 'struct T1' and is usually due to 'struct U1' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U2 {
  char a;
  struct T1 b __attribute__((aligned(2))); // expected-warning {{field b within 'struct U2' is less aligned than 'struct T1' and is usually due to 'struct U2' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U3 {
  char a;
  struct T1 b __attribute__((aligned(4)));
  int c;
};

struct __attribute__((aligned(2))) U4 {
  char a;
  struct T1 b;
  int c;
};

struct U5 {
  char a;
  struct T1 b;
  int c;
};

struct U6 {
  char a;
  int b;
  struct T1 c __attribute__((aligned(2)));
};

struct __attribute__((packed)) U7 {
  short a;
  short b;
  char c;
  struct T1 d; // expected-warning {{field d within 'struct U7' is less aligned than 'struct T1' and is usually due to 'struct U7' being packed, which can lead to unaligned accesses}}
};

struct U8 {
  short a;
  short b;
  char c;
  struct T1 d;
};

struct __attribute__((packed)) U9 {
  short a;
  short b;
  char c;
  struct T1 d __attribute__((aligned(4)));
};

struct __attribute__((packed)) U10 {
  short a;
  short b;
  char c;
  struct T1 d __attribute__((aligned(2))); // expected-warning {{field d within 'struct U10' is less aligned than 'struct T1' and is usually due to 'struct U10' being packed, which can lead to unaligned accesses}}
};

struct __attribute__((aligned(2))) U11 {
  short a;
  short b;
  char c;
  struct T1 d;
};

// Set 2
#pragma pack(push, 1)

struct U12 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U12' is less aligned than 'struct T1' and is usually due to 'struct U12' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U13 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U13' is less aligned than 'struct T1' and is usually due to 'struct U13' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U14 {
  char a;
  struct T1 b __attribute__((aligned(4))); // expected-warning {{field b within 'struct U14' is less aligned than 'struct T1' and is usually due to 'struct U14' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(2))) U15 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U15' is less aligned than 'struct T1' and is usually due to 'struct U15' being packed, which can lead to unaligned accesses}}
  int c;
};

struct U16 {
  char a;
  char b;
  short c;
  struct T1 d;
};

struct U17 {
  char a;
  char b;
  short c;
  struct T1 d __attribute__((aligned(4)));
};

struct __attribute__((packed)) U18 {
  char a;
  short b;
  struct T1 c __attribute__((aligned(4))); // expected-warning {{field c within 'struct U18' is less aligned than 'struct T1' and is usually due to 'struct U18' being packed, which can lead to unaligned accesses}}
};

struct __attribute__((aligned(4))) U19 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U19' is less aligned than 'struct T1' and is usually due to 'struct U19' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(4))) U20 {
  char a[4];
  struct T1 b;
  int c;
};

struct U21 {
  char a;
  short c;
  struct T1 d; // expected-warning {{field d within 'struct U21' is less aligned than 'struct T1' and is usually due to 'struct U21' being packed, which can lead to unaligned accesses}}
};

struct U22 {
  char a;
  short c;
  struct T1 d __attribute__((aligned(4))); // expected-warning {{field d within 'struct U22' is less aligned than 'struct T1' and is usually due to 'struct U22' being packed, which can lead to unaligned accesses}}
};

#pragma pack(pop)

// Set 3
#pragma pack(push, 2)

struct __attribute__((packed)) U23 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U23' is less aligned than 'struct T1' and is usually due to 'struct U23' being packed, which can lead to unaligned accesses}}
  int c;
};

struct U24 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct U24' is less aligned than 'struct T1' and is usually due to 'struct U24' being packed, which can lead to unaligned accesses}}
  int c;
};

struct U25 {
  char a;
  char b;
  short c;
  struct T1 d;
};

struct U26 {
  char a;
  char b;
  short c;
  struct T1 d;
};

#pragma pack(pop)

// Set 4

struct __attribute__((packed)) T2 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct T2' is less aligned than 'struct T1' and is usually due to 'struct T2' being packed, which can lead to unaligned accesses}}
};

struct T3 {
  char a;
  struct T1 b;
};

struct __attribute__((packed)) U27 {
  char a;
  struct T2 b;
  int c;
};

struct U28 {
  char a;
  char _p[2];
  struct T2 b;
  int c;
};

struct U29 {
  char a;
  struct T3 b;
  int c;
};

struct __attribute__((packed)) U30 {
  char a;
  struct T3 b; // expected-warning {{field b within 'struct U30' is less aligned than 'struct T3' and is usually due to 'struct U30' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U31 {
  char a;
  struct T2 b __attribute__((aligned(4)));
};

struct __attribute__((packed)) U32 {
  char a;
  char b;
  char c;
  char d;
  struct T3 e;
};

struct __attribute__((packed)) U33 {
  char a;
  char b;
  char c;
  char d;
  struct T2 e __attribute__((aligned(4)));
};

struct __attribute__((packed)) U34 {
  char a;
  struct T1 b __attribute__((packed)); // expected-warning {{field b within 'struct U34' is less aligned than 'struct T1' and is usually due to 'struct U34' being packed, which can lead to unaligned accesses}}
  struct T2 c;
};

struct __attribute__((packed)) U35 {
  char a;
  struct T4 {
    char b;
    struct T1 c;
  } d; // expected-warning {{field d within 'struct U35' is less aligned than 'struct T4' and is usually due to 'struct U35' being packed, which can lead to unaligned accesses}}
};

// Set 5

#pragma pack(push, 1)
struct T5 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct T5' is less aligned than 'struct T1' and is usually due to 'struct T5' being packed, which can lead to unaligned accesses}}
};
#pragma pack(pop)

#pragma pack(push, 1)
struct U36 {
  char a;
  struct T5 b;
  int c;
};

struct U37 {
  char a;
  struct T3 b; // expected-warning {{field b within 'struct U37' is less aligned than 'struct T3' and is usually due to 'struct U37' being packed, which can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)
struct U38 {
  char a;
  struct T5 b __attribute__((aligned(4)));
  int c;
};

#pragma pack(push, 1)

#pragma pack(push, 4)
struct U39 {
  char a;
  struct T5 b;
  int c;
};
#pragma pack(pop)

#pragma pack(pop)

// Set 6

struct __attribute__((packed)) A1 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct A1' is less aligned than 'struct T1' and is usually due to 'struct A1' being packed, which can lead to unaligned accesses}}
};

struct A2 {
  char a;
  struct T1 b;
};

struct __attribute__((packed)) A3 {
  char a;
  struct T1 b __attribute__((aligned(4)));
};

#pragma pack(push, 1)
struct A4 {
  char a;
  struct T1 b; // expected-warning {{field b within 'struct A4' is less aligned than 'struct T1' and is usually due to 'struct A4' being packed, which can lead to unaligned accesses}}
};

struct A5 {
  char a;
  struct T1 b __attribute__((aligned(4))); // expected-warning {{field b within 'struct A5' is less aligned than 'struct T1' and is usually due to 'struct A5' being packed, which can lead to unaligned accesses}}
};
#pragma pack(pop)

struct __attribute__((packed)) A6 {
  struct T1 a;
};

struct A7 {
  char a;
  struct T1 b __attribute__((packed));
};

struct A8 {
  char a;
  char b;
  short c;
  struct T1 d;
};

struct A9 {
  char a;
  struct T2 b;
};

struct A10 {
  char a;
  struct T2 b __attribute__((aligned(4)));
};

struct __attribute__((packed)) A11 {
  char a;
  struct T2 b;
};

struct __attribute__((packed)) U40 {
  char a;
  struct A1 b;
  int c;
};

struct __attribute__((packed)) U41 {
  char a;
  struct A3 b; // expected-warning {{field b within 'struct U41' is less aligned than 'struct A3' and is usually due to 'struct U41' being packed, which can lead to unaligned accesses}}
  int c;
};

#pragma pack(push, 1)
struct U42 {
  char a;
  struct A1 b;
  int c;
};
#pragma pack(pop)

struct __attribute__((packed)) U43 {
  char a;
  struct A9 b;
  int c;
};

struct __attribute__((packed)) U44 {
  char a;
  struct A10 b; // expected-warning {{field b within 'struct U44' is less aligned than 'struct A10' and is usually due to 'struct U44' being packed, which can lead to unaligned accesses}}
  int c;
};

#pragma pack(push, 1)

struct U45 {
  char a;
  struct A10 b; // expected-warning {{field b within 'struct U45' is less aligned than 'struct A10' and is usually due to 'struct U45' being packed, which can lead to unaligned accesses}}
  int c;
};

#pragma pack(pop)

struct __attribute__((packed)) U46 {
  char a;
  struct A2 b; // expected-warning {{field b within 'struct U46' is less aligned than 'struct A2' and is usually due to 'struct U46' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U47 {
  char a;
  struct A8 b; // expected-warning {{field b within 'struct U47' is less aligned than 'struct A8' and is usually due to 'struct U47' being packed, which can lead to unaligned accesses}}
  int c;
};

#pragma pack(push, 1)
struct U48 {
  char a;
  struct A8 b; // expected-warning {{field b within 'struct U48' is less aligned than 'struct A8' and is usually due to 'struct U48' being packed, which can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)

struct U49 {
  char a;
  struct A11 b;
  int c;
};

struct U50 {
  char a;
  struct A1 b;
  int c;
};

struct U51 {
  char a;
  struct A5 b;
  int c;
};

struct __attribute__((packed)) U52 {
  char a;
  struct A6 b;
};

struct U53 {
  char a;
  struct A4 b;
};

struct U54 {
  char b;
  struct A7 c;
};

struct U1 s1;
struct U2 s2;
struct U3 s3;
struct U4 s4;
struct U5 s5;
struct U6 s6;
struct U7 s7;
struct U8 s8;
struct U9 s9;
struct U10 s10;
struct U11 s11;
struct U12 s12;
struct U13 s13;
struct U14 s14;
struct U15 s15;
struct U16 s16;
struct U17 s17;
struct U18 s18;
struct U19 s19;
struct U20 s20;
struct U21 s21;
struct U22 s22;
struct U23 s23;
struct U24 s24;
struct U25 s25;
struct U26 s26;
struct U27 s27;
struct U28 s28;
struct U29 s29;
struct U30 s30;
struct U31 s31;
struct U32 s32;
struct U33 s33;
struct U34 s34;
struct U35 s35;
struct U36 s36;
struct U37 s37;
struct U38 s38;
struct U39 s39;
struct U40 s40;
struct U41 s41;
struct U42 s42;
struct U43 s43;
struct U44 s44;
struct U45 s45;
struct U46 s46;
struct U47 s47;
struct U48 s48;
struct U49 s49;
struct U50 s50;
struct U51 s51;
struct U52 s52;
struct U53 s53;
struct U54 s54;

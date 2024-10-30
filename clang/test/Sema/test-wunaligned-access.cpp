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

// Packed-Unpacked Tests (No Pragma)

struct T1 {
  char a;
  int b;
};

struct __attribute__((packed)) U1 {
  char a;
  T1 b; // expected-warning {{field b within 'U1' is less aligned than 'T1' and is usually due to 'U1' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U2 {
  char a;
  T1 b __attribute__((aligned(4)));
  int c;
};

struct __attribute__((packed)) U3 {
  char a;
  char b;
  short c;
  T1 d;
};

struct __attribute__((packed)) U4 {
  T1 a;
  int b;
};

struct __attribute__((aligned(4), packed)) U5 {
  char a;
  T1 b; // expected-warning {{field b within 'U5' is less aligned than 'T1' and is usually due to 'U5' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(4), packed)) U6 {
  char a;
  char b;
  short c;
  T1 d;
};

// Packed-Unpacked Tests with Pragma

#pragma pack(push, 1)

struct __attribute__((packed)) U7 {
  char a;
  T1 b; // expected-warning {{field b within 'U7' is less aligned than 'T1' and is usually due to 'U7' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U8 {
  char a;
  T1 b __attribute__((aligned(4))); // expected-warning {{field b within 'U8' is less aligned than 'T1' and is usually due to 'U8' being packed, which can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(4))) U9 {
  char a;
  T1 b; // expected-warning {{field b within 'U9' is less aligned than 'T1' and is usually due to 'U9' being packed, which can lead to unaligned accesses}}
  int c;
};

struct U10 {
  char a;
  T1 b; // expected-warning {{field b within 'U10' is less aligned than 'T1' and is usually due to 'U10' being packed, which can lead to unaligned accesses}}
  int c;
};

#pragma pack(pop)

// Packed-Packed Tests

struct __attribute__((packed)) T2 {
  char a;
  int b;
};

struct __attribute__((packed)) U11 {
  char a;
  T2 b;
  int c;
};

#pragma pack(push, 1)
struct U12 {
  char a;
  T2 b;
  int c;
};
#pragma pack(pop)

// Unpacked-Packed Tests

struct U13 {
  char a;
  T2 b;
  int c;
};

struct U14 {
  char a;
  T2 b __attribute__((aligned(4)));
  int c;
};

// Unpacked-Unpacked Test

struct T3 {
  char a;
  int b;
};

struct U15 {
  char a;
  T3 b;
  int c;
};

// Packed-Packed-Unpacked Test (No pragma)

struct __attribute__((packed)) A1 {
  char a;
  T1 b; // expected-warning {{field b within 'A1' is less aligned than 'T1' and is usually due to 'A1' being packed, which can lead to unaligned accesses}}
};

struct __attribute__((packed)) U16 {
  char a;
  A1 b;
  int c;
};

struct __attribute__((packed)) A2 {
  char a;
  T1 b __attribute__((aligned(4)));
};

struct __attribute__((packed)) U17 {
  char a;
  A2 b; // expected-warning {{field b within 'U17' is less aligned than 'A2' and is usually due to 'U17' being packed, which can lead to unaligned accesses}}
  int c;
};

// Packed-Unpacked-Packed tests

struct A3 {
  char a;
  T2 b;
};

struct __attribute__((packed)) U18 {
  char a;
  A3 b;
  int c;
};

struct A4 {
  char a;
  T2 b;
  int c;
};

#pragma pack(push, 1)
struct U19 {
  char a;
  A4 b; // expected-warning {{field b within 'U19' is less aligned than 'A4' and is usually due to 'U19' being packed, which can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)

// Packed-Unpacked-Unpacked tests

struct A5 {
  char a;
  T1 b;
};

struct __attribute__((packed)) U20 {
  char a;
  A5 b; // expected-warning {{field b within 'U20' is less aligned than 'A5' and is usually due to 'U20' being packed, which can lead to unaligned accesses}}
  int c;
};

struct A6 {
  char a;
  T1 b;
};

#pragma pack(push, 1)
struct U21 {
  char a;
  A6 b; // expected-warning {{field b within 'U21' is less aligned than 'A6' and is usually due to 'U21' being packed, which can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)

// Unpacked-Packed-Packed test

struct __attribute__((packed)) A7 {
  char a;
  T2 b;
};

struct U22 {
  char a;
  A7 b;
  int c;
};

// Unpacked-Packed-Unpacked tests

struct __attribute__((packed)) A8 {
  char a;
  T1 b; // expected-warning {{field b within 'A8' is less aligned than 'T1' and is usually due to 'A8' being packed, which can lead to unaligned accesses}}
};

struct U23 {
  char a;
  A8 b;
  int c;
};

struct __attribute__((packed)) A9 {
  char a;
  T1 b __attribute__((aligned(4)));
};

struct U24 {
  char a;
  A9 b;
  int c;
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
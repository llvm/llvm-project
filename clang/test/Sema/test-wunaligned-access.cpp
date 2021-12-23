// RUN: %clang_cc1 %s -verify -fsyntax-only -Wunaligned-access

// Packed-Unpacked Tests (No Pragma)

struct T1 {
  char a;
  int b;
};

struct __attribute__((packed)) U1 // Causes warning
{
  char a;
  T1 b; // expected-warning {{field b within its parent 'U1' has an alignment greater than its parent this may be caused by 'U1' being packed and can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U2 // No warning
{
  char a;
  T1 b __attribute__((aligned(4)));
  int c;
};

struct __attribute__((packed)) U3 // No warning
{
  char a;
  char b;
  short c;
  T1 d;
};

struct __attribute__((packed)) U4 // No warning
{
  T1 a;
  int b;
};

struct __attribute__((aligned(4), packed)) U5 // Causes warning
{
  char a;
  T1 b; // expected-warning {{field b within its parent 'U5' has an alignment greater than its parent this may be caused by 'U5' being packed and can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(4), packed)) U6 // No warning
{
  char a;
  char b;
  short c;
  T1 d;
};

// Packed-Unpacked Tests with Pragma

#pragma pack(push, 1)

struct __attribute__((packed)) U7 // Causes warning
{
  char a;
  T1 b; // expected-warning {{field b within its parent 'U7' has an alignment greater than its parent this may be caused by 'U7' being packed and can lead to unaligned accesses}}
  int c;
};

struct __attribute__((packed)) U8 {
  char a;
  T1 b __attribute__((aligned(4))); // expected-warning {{field b within its parent 'U8' has an alignment greater than its parent this may be caused by 'U8' being packed and can lead to unaligned accesses}}
  int c;
};

struct __attribute__((aligned(4))) U9 {
  char a;
  T1 b; // expected-warning {{field b within its parent 'U9' has an alignment greater than its parent this may be caused by 'U9' being packed and can lead to unaligned accesses}}
  int c;
};

struct U10 {
  char a;
  T1 b; // expected-warning {{field b within its parent 'U10' has an alignment greater than its parent this may be caused by 'U10' being packed and can lead to unaligned accesses}}
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
struct U12 // No warning
{
  char a;
  T2 b;
  int c;
};
#pragma pack(pop)

// Unpacked-Packed Tests

struct U13 // No warning
{
  char a;
  T2 b;
  int c;
};

struct U14 // No warning
{
  char a;
  T2 b __attribute__((aligned(4)));
  int c;
};

// Unpacked-Unpacked Test

struct T3 {
  char a;
  int b;
};

struct U15 // No warning
{
  char a;
  T3 b;
  int c;
};

// Packed-Packed-Unpacked Test (No pragma)

struct __attribute__((packed)) A1 {
  char a;
  T1 b; // expected-warning {{field b within its parent 'A1' has an alignment greater than its parent this may be caused by 'A1' being packed and can lead to unaligned accesses}}
};

struct __attribute__((packed)) U16 // No warning
{
  char a;
  A1 b;
  int c;
};

struct __attribute__((packed)) A2 // No warning
{
  char a;
  T1 b __attribute__((aligned(4)));
};

struct __attribute__((packed)) U17 // Caused warning
{
  char a;
  A2 b; // expected-warning {{field b within its parent 'U17' has an alignment greater than its parent this may be caused by 'U17' being packed and can lead to unaligned accesses}}
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
struct U19 // Caused warning
{
  char a;
  A4 b; // expected-warning {{field b within its parent 'U19' has an alignment greater than its parent this may be caused by 'U19' being packed and can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)

// Packed-Unpacked-Unpacked tests

struct A5 {
  char a;
  T1 b;
};

struct __attribute__((packed)) U20 // Caused warning
{
  char a;
  A5 b; // expected-warning {{field b within its parent 'U20' has an alignment greater than its parent this may be caused by 'U20' being packed and can lead to unaligned accesses}}
  int c;
};

struct A6 {
  char a;
  T1 b;
};

#pragma pack(push, 1)
struct U21 // Caused warning
{
  char a;
  A6 b; // expected-warning {{field b within its parent 'U21' has an alignment greater than its parent this may be caused by 'U21' being packed and can lead to unaligned accesses}}
  int c;
};
#pragma pack(pop)

// Unpacked-Packed-Packed test

struct __attribute__((packed)) A7 // No warning
{
  char a;
  T2 b;
};

struct U22 // No warning
{
  char a;
  A7 b;
  int c;
};

// Unpacked-Packed-Unpacked tests

struct __attribute__((packed)) A8 // Should cause warning
{
  char a;
  T1 b; // expected-warning {{field b within its parent 'A8' has an alignment greater than its parent this may be caused by 'A8' being packed and can lead to unaligned accesses}}
};

struct U23 // No warning
{
  char a;
  A8 b;
  int c;
};

struct __attribute__((packed)) A9 // No warning
{
  char a;
  T1 b __attribute__((aligned(4)));
};

struct U24 // No warning
{
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
// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-w64-windows-gnu %s
// RUN: %clang_cc1 -emit-llvm-only -triple i686-w64-windows-gnu %s
// RUN: %clang_cc1 -emit-llvm-only -triple i386-apple-darwin9 %s

// Regression test: when #pragma pack(N) with N < alignof(int) is combined
// with ms_struct layout and a zero-width bitfield, the bitfield storage unit
// could extend past the zero-width bitfield boundary, overlapping the next
// member's storage and triggering the "Bitfield access unit is not clipped"
// assertion in checkBitfieldClipping().

// Minimal case: pack(1) + char + bitfield + :0 + bitfield
#pragma pack(1)
struct S0 {
  char f1;
  unsigned f3 : 1;
  unsigned : 0;
  unsigned f4 : 1;
} __attribute__((__ms_struct__));
struct S0 s0;

// Variation: pack(1) with more complex bitfield layout (original CSmith case)
struct S1 {
  signed f0 : 28;
  const signed char f1;
  const volatile signed f2 : 25;
  const unsigned f3 : 23;
  unsigned : 0;
  unsigned f4 : 12;
  volatile unsigned f5 : 29;
} __attribute__((__ms_struct__));
struct S1 s1;

// Variation: pack(2) with short-sized first member
#pragma pack(2)
struct S2 {
  short f1;
  unsigned f3 : 1;
  unsigned : 0;
  unsigned f4 : 1;
} __attribute__((__ms_struct__));
struct S2 s2;

// Variation: pack(1), multiple zero-width bitfields
#pragma pack(1)
struct S3 {
  char f1;
  unsigned f3 : 1;
  unsigned : 0;
  unsigned f4 : 1;
  unsigned : 0;
  unsigned f5 : 1;
} __attribute__((__ms_struct__));
struct S3 s3;

// Variation: pack(1), zero-width of different type
struct S4 {
  char f1;
  unsigned f3 : 1;
  short : 0;
  unsigned f4 : 1;
} __attribute__((__ms_struct__));
struct S4 s4;

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// echo -en 'Im_so_cute&pretty_:)' > crash
//
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// Force noinline, as this test might be interesting for experimenting with
// data flow tracing approach started in https://reviews.llvm.org/D46666.
__attribute__((noinline))
int func1(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 15 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func2(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 80 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func3(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func4(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) ^ a3;
  if ( v > 44 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func5(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 72 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func6(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 72 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func7(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 43 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func8(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func9(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func10(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 83 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func11(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 117 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func12(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func13(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 80 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func14(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func15(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 116 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func16(uint8_t a1) {
  char v = a1 >> 5;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func17(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func18(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 28 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func19(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 18 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func20(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func21(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = (((a1 ^ a2))) & a3;
  if ( v > 108 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func22(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func23(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 7 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func24(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 25 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func25(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func26(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 41 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func27(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 14 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func28(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func29(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func30(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func31(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 45 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func32(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func33(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func34(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 95 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func35(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 12 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func36(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 121 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func37(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func38(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 61 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func39(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func40(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 125 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func41(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func42(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = (((a1 ^ a2))) & a3;
  if ( v > 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func43(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func44(uint8_t a1) {
  char v = a1 >> 5;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func45(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func46(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 106 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func47(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 33 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func48(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func49(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 58 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func50(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 42 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func51(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 46 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func52(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func53(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func54(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 23 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func55(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 17 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func56(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func57(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func58(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 102 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func59(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 49 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func60(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 26 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func61(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 55 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func62(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func63(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func64(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 34 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func65(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func66(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 4 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func67(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 50 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func68(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 37 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func69(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func70(uint8_t a1) {
  char v = a1 << 6;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func71(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 85 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func72(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func73(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 30 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func74(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func75(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v <= 59 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func76(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func77(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 30 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func78(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 32 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func79(uint8_t a1) {
  char v = 16 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func80(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 ^ a2)) | a3;
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func81(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 120 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func82(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 81 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func83(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 119 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func84(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func85(uint8_t a1) {
  char v = 2 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func86(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func87(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 84 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func88(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func89(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func90(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 60 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func91(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 13 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func92(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 38 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func93(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func94(uint8_t a1) {
  char v = 16 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func95(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func96(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func97(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func98(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 102 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func99(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 96 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func100(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 ^ a2)) | a3;
  if ( v != 127 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func101(uint8_t a1) {
  char v = 4 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func102(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 43 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func103(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 95 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func104(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = (((a1 ^ a2))) & a3;
  if ( v <= 2 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func105(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 65 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func106(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 24 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func107(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func108(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func109(uint8_t a1) {
  char v = 2 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func110(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 101 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func111(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v <= 121 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func112(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 40 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func113(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 50 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func114(uint8_t a1) {
  char v = a1 << 6;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func115(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 12 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func116(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func117(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 79 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func118(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func119(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 44 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func120(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v <= 28 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func121(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 93 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func122(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 40 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func123(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func124(uint8_t a1) {
  char v = a1 >> 5;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func125(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func126(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func127(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 8 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func128(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func129(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 3 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func130(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 102 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func131(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 68 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func132(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 73 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func133(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 68 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func134(uint8_t a1) {
  char v = 16 * a1;
  if ( v > 125 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func135(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 79 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func136(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 6 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func137(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func138(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func139(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func140(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 74 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func141(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func142(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 89 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func143(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 46 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func144(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 29 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func145(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 77 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func146(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 12 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func147(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func148(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 27 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func149(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func150(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 122 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func151(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 3 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func152(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 56 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func153(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 3 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func154(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 43 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func155(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func156(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func157(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func158(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func159(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 88 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func160(uint8_t a1) {
  char v = ~a1;
  if ( v > 33 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func161(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 46 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func162(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func163(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v <= 9 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func164(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 96 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func165(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func166(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func167(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 91 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func168(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func169(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 32 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func170(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 32 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func171(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func172(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func173(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func174(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func175(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 32 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func176(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 61 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func177(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 33 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func178(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func179(uint8_t a1) {
  char v = ~a1;
  if ( v > 64 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func180(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 95 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func181(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func182(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 113 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func183(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 41 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func184(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func185(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func186(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func187(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 43 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func188(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 57 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func189(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func190(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func191(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 92 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func192(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func193(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func194(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 20 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func195(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 82 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func196(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 117 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func197(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 50 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func198(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func199(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v == 127 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func200(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func201(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func202(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 56 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func203(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 95 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func204(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func205(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 ^ a2)) | a3;
  if ( v > 95 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func206(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 78 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func207(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 7 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func208(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 123 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func209(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func210(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 101 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func211(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 61 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func212(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 73 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func213(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 34 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func214(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func215(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 5 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func216(uint8_t a1) {
  char v = ~a1;
  if ( v > 85 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func217(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 113 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func218(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 61 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func219(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func220(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 106 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func221(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func222(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 84 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func223(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 81 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func224(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func225(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 49 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func226(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func227(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func228(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 81 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func229(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 41 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func230(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 82 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func231(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 84 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func232(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 34 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func233(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func234(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func235(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 73 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func236(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 12 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func237(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 9 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func238(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 42 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func239(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 44 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func240(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 14 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func241(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func242(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 74 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func243(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 102 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func244(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func245(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 87 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func246(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 29 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func247(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 51 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func248(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 74 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func249(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func250(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 56 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func251(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 11 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func252(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func253(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 22 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func254(uint8_t a1, uint8_t a2, uint8_t a3) {
  char v = ((a1 & a2)) | a3;
  if ( v > 122 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func255(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 74 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func256(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func257(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func258(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 102 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func259(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 74 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func260(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 27 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func261(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 58 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func262(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 77 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func263(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 3 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func264(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 13 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func265(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func266(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 39 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func267(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v == 127 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func268(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 66 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func269(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func270(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func271(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 122 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func272(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 65 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func273(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 120 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func274(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 83 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func275(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 99 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func276(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func277(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 42 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func278(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func279(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 110 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func280(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 92 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func281(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 59 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func282(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func283(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func284(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func285(uint8_t a1) {
  char v = ~a1;
  if ( v > 17 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func286(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func287(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 78 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func288(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func289(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 90 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func290(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 78 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func291(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 30 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func292(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func293(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func294(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func295(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 17 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func296(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 86 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func297(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 120 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func298(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 46 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func299(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func300(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 5 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func301(uint8_t a1) {
  char v = ~a1;
  if ( v > 17 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func302(uint8_t a1) {
  char v = ~a1;
  if ( v > 113 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func303(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func304(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 73 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func305(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 60 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func306(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 119 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func307(uint8_t a1) {
  char v = ~a1;
  if ( v > 21 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func308(uint8_t a1) {
  char v = ~a1;
  if ( v > 107 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func309(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 44 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func310(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 57 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func311(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 59 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func312(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func313(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func314(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 58 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func315(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func316(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 101 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func317(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 99 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func318(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 78 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func319(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 16 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func320(uint8_t a1) {
  char v = ~a1;
  if ( v > 10 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func321(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func322(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 3 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func323(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func324(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func325(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func326(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func327(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 101 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func328(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 18 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func329(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func330(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func331(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func332(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func333(uint8_t a1) {
  char v = 16 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func334(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 38 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func335(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func336(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func337(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func338(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func339(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 47 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func340(uint8_t a1) {
  char v = a1 >> 5;
  if ( v <= 0 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func341(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func342(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func343(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 58 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func344(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 91 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func345(uint8_t a1) {
  char v = 16 * a1;
  if ( v <= 72 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func346(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func347(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func348(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func349(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 57 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func350(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func351(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 99 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func352(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 63 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func353(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 81 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func354(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func355(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func356(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func357(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 72 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func358(uint8_t a1) {
  char v = 16 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func359(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 110 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func360(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func361(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 68 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func362(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 91 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func363(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func364(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 99 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func365(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 40 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func366(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v <= 31 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func367(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func368(uint8_t a1) {
  char v = a1 >> 5;
  if ( v > 96 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func369(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func370(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 42 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func371(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 118 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func372(uint8_t a1) {
  char v = (char)a1 >> 1;
  if ( v > 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func373(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func374(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func375(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 64 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func376(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 110 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func377(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 104 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func378(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v > 112 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func379(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 62 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func380(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 48 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func381(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 58 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func382(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 104 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func383(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 50 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func384(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 38 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func385(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 85 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func386(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 18 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func387(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 97 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func388(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func389(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 26 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func390(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 67 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func391(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func392(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v > 50 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func393(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v <= 22 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func394(uint8_t a1) {
  char v = 4 * a1;
  if ( v <= 103 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func395(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 38 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func396(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 52 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func397(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 17 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func398(uint8_t a1) {
  char v = 4 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func399(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 92 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func400(uint8_t a1, uint8_t a2) {
  char v = (a1 & a2);
  if ( v <= 55 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func401(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 81 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func402(uint8_t a1) {
  char v = 4 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func403(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 94 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func404(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func405(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func406(uint8_t a1, uint8_t a2) {
  char v = (a1 ^ a2);
  if ( v > 101 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func407(uint8_t a1) {
  char v = 4 * a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func408(uint8_t a1, uint8_t a2) {
  char v = a1 | a2;
  if ( v <= 44 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func409(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

__attribute__((noinline))
int func410(uint8_t a1) {
  char v = ~a1;
  if ( v > 1 )
    return 0;
  return 1;
}

int api(const uint8_t *data, size_t size) {
  if (size != 20) return 0;

  if (func1(data[0], data[1]) == 0)
    return 0;
  if (func2(data[0], data[1]) == 0)
    return 0;
  if (func3(data[18], data[1]) == 0)
    return 0;
  if (func4(data[7], data[4], data[0]) == 0)
    return 0;
  if (func5(data[0], data[2]) == 0)
    return 0;
  if (func6(data[11], data[17]) == 0)
    return 0;
  if (func7(data[0], data[13]) == 0)
    return 0;
  if (func8(data[13], data[10]) == 0)
    return 0;
  if (func9(data[11], data[16]) == 0)
    return 0;
  if (func10(data[10], data[8]) == 0)
    return 0;
  if (func11(data[19], data[5]) == 0)
    return 0;
  if (func12(data[0], data[1]) == 0)
    return 0;
  if (func13(data[17], data[3]) == 0)
    return 0;
  if (func14(data[14]) == 0)
    return 0;
  if (func15(data[13], data[15]) == 0)
    return 0;
  if (func16(data[0]) == 0)
    return 0;
  if (func17(data[19]) == 0)
    return 0;
  if (func18(data[1], data[11]) == 0)
    return 0;
  if (func19(data[12], data[15]) == 0)
    return 0;
  if (func20(data[13], data[1]) == 0)
    return 0;
  if (func21(data[10], data[19], data[12]) == 0)
    return 0;
  if (func22(data[6]) == 0)
    return 0;
  if (func23(data[1], data[9]) == 0)
    return 0;
  if (func24(data[16]) == 0)
    return 0;
  if (func25(data[6]) == 0)
    return 0;
  if (func26(data[4], data[12]) == 0)
    return 0;
  if (func27(data[16]) == 0)
    return 0;
  if (func28(data[14]) == 0)
    return 0;
  if (func29(data[0]) == 0)
    return 0;
  if (func30(data[19]) == 0)
    return 0;
  if (func31(data[0], data[1]) == 0)
    return 0;
  if (func32(data[0], data[1]) == 0)
    return 0;
  if (func33(data[14]) == 0)
    return 0;
  if (func34(data[0], data[19]) == 0)
    return 0;
  if (func35(data[0]) == 0)
    return 0;
  if (func36(data[16], data[7]) == 0)
    return 0;
  if (func37(data[19]) == 0)
    return 0;
  if (func38(data[15], data[3]) == 0)
    return 0;
  if (func39(data[19], data[15]) == 0)
    return 0;
  if (func40(data[0], data[1]) == 0)
    return 0;
  if (func41(data[18], data[1]) == 0)
    return 0;
  if (func42(data[16], data[5], data[1]) == 0)
    return 0;
  if (func43(data[14]) == 0)
    return 0;
  if (func44(data[0]) == 0)
    return 0;
  if (func45(data[19]) == 0)
    return 0;
  if (func46(data[4], data[19]) == 0)
    return 0;
  if (func47(data[8], data[7]) == 0)
    return 0;
  if (func48(data[6], data[7]) == 0)
    return 0;
  if (func49(data[18], data[1]) == 0)
    return 0;
  if (func50(data[2], data[8]) == 0)
    return 0;
  if (func51(data[2], data[13]) == 0)
    return 0;
  if (func52(data[3], data[8]) == 0)
    return 0;
  if (func53(data[16], data[18]) == 0)
    return 0;
  if (func54(data[10], data[9]) == 0)
    return 0;
  if (func55(data[18], data[1]) == 0)
    return 0;
  if (func56(data[0], data[1]) == 0)
    return 0;
  if (func57(data[16], data[1]) == 0)
    return 0;
  if (func58(data[18], data[1]) == 0)
    return 0;
  if (func59(data[18], data[3]) == 0)
    return 0;
  if (func60(data[9], data[1]) == 0)
    return 0;
  if (func61(data[0], data[1]) == 0)
    return 0;
  if (func62(data[13], data[1]) == 0)
    return 0;
  if (func63(data[18], data[1]) == 0)
    return 0;
  if (func64(data[0], data[1]) == 0)
    return 0;
  if (func65(data[0], data[1]) == 0)
    return 0;
  if (func66(data[11], data[14]) == 0)
    return 0;
  if (func67(data[5], data[11]) == 0)
    return 0;
  if (func68(data[18], data[1]) == 0)
    return 0;
  if (func69(data[0], data[6]) == 0)
    return 0;
  if (func70(data[2]) == 0)
    return 0;
  if (func71(data[0], data[1]) == 0)
    return 0;
  if (func72(data[9], data[10]) == 0)
    return 0;
  if (func73(data[10], data[8]) == 0)
    return 0;
  if (func74(data[19], data[17]) == 0)
    return 0;
  if (func75(data[0], data[17], data[8]) == 0)
    return 0;
  if (func76(data[17], data[18]) == 0)
    return 0;
  if (func77(data[18], data[9]) == 0)
    return 0;
  if (func78(data[3], data[6]) == 0)
    return 0;
  if (func79(data[16]) == 0)
    return 0;
  if (func80(data[7], data[3], data[17]) == 0)
    return 0;
  if (func81(data[0], data[1]) == 0)
    return 0;
  if (func82(data[10], data[18]) == 0)
    return 0;
  if (func83(data[6], data[7]) == 0)
    return 0;
  if (func84(data[0], data[6]) == 0)
    return 0;
  if (func85(data[12]) == 0)
    return 0;
  if (func86(data[0], data[1]) == 0)
    return 0;
  if (func87(data[6], data[1]) == 0)
    return 0;
  if (func88(data[18], data[1]) == 0)
    return 0;
  if (func89(data[0], data[6]) == 0)
    return 0;
  if (func90(data[0], data[1]) == 0)
    return 0;
  if (func91(data[18], data[1]) == 0)
    return 0;
  if (func92(data[0], data[6]) == 0)
    return 0;
  if (func93(data[13], data[10]) == 0)
    return 0;
  if (func94(data[2]) == 0)
    return 0;
  if (func95(data[0], data[1]) == 0)
    return 0;
  if (func96(data[0], data[11]) == 0)
    return 0;
  if (func97(data[18], data[1]) == 0)
    return 0;
  if (func98(data[0], data[6]) == 0)
    return 0;
  if (func99(data[0], data[19]) == 0)
    return 0;
  if (func100(data[14], data[18], data[3]) == 0)
    return 0;
  if (func101(data[14]) == 0)
    return 0;
  if (func102(data[6], data[1]) == 0)
    return 0;
  if (func103(data[5], data[1]) == 0)
    return 0;
  if (func104(data[14], data[3], data[10]) == 0)
    return 0;
  if (func105(data[18], data[1]) == 0)
    return 0;
  if (func106(data[0], data[6]) == 0)
    return 0;
  if (func107(data[6]) == 0)
    return 0;
  if (func108(data[9], data[10]) == 0)
    return 0;
  if (func109(data[7]) == 0)
    return 0;
  if (func110(data[9], data[17]) == 0)
    return 0;
  if (func111(data[16], data[15], data[18]) == 0)
    return 0;
  if (func112(data[0], data[16]) == 0)
    return 0;
  if (func113(data[18], data[3]) == 0)
    return 0;
  if (func114(data[3]) == 0)
    return 0;
  if (func115(data[0], data[1]) == 0)
    return 0;
  if (func116(data[14]) == 0)
    return 0;
  if (func117(data[0]) == 0)
    return 0;
  if (func118(data[19]) == 0)
    return 0;
  if (func119(data[0], data[6]) == 0)
    return 0;
  if (func120(data[9], data[5], data[0]) == 0)
    return 0;
  if (func121(data[0], data[1]) == 0)
    return 0;
  if (func122(data[12], data[4]) == 0)
    return 0;
  if (func123(data[14]) == 0)
    return 0;
  if (func124(data[0]) == 0)
    return 0;
  if (func125(data[19]) == 0)
    return 0;
  if (func126(data[14]) == 0)
    return 0;
  if (func127(data[1], data[9]) == 0)
    return 0;
  if (func128(data[19]) == 0)
    return 0;
  if (func129(data[11], data[14]) == 0)
    return 0;
  if (func130(data[11], data[9]) == 0)
    return 0;
  if (func131(data[12], data[15]) == 0)
    return 0;
  if (func132(data[0], data[2]) == 0)
    return 0;
  if (func133(data[13], data[1]) == 0)
    return 0;
  if (func134(data[7]) == 0)
    return 0;
  if (func135(data[13], data[5]) == 0)
    return 0;
  if (func136(data[12], data[14]) == 0)
    return 0;
  if (func137(data[9], data[4]) == 0)
    return 0;
  if (func138(data[6]) == 0)
    return 0;
  if (func139(data[14]) == 0)
    return 0;
  if (func140(data[0]) == 0)
    return 0;
  if (func141(data[19]) == 0)
    return 0;
  if (func142(data[11], data[9]) == 0)
    return 0;
  if (func143(data[12], data[15]) == 0)
    return 0;
  if (func144(data[7]) == 0)
    return 0;
  if (func145(data[19], data[7]) == 0)
    return 0;
  if (func146(data[4], data[6]) == 0)
    return 0;
  if (func147(data[6]) == 0)
    return 0;
  if (func148(data[0]) == 0)
    return 0;
  if (func149(data[19]) == 0)
    return 0;
  if (func150(data[11], data[9]) == 0)
    return 0;
  if (func151(data[12], data[15]) == 0)
    return 0;
  if (func152(data[13], data[1]) == 0)
    return 0;
  if (func153(data[7]) == 0)
    return 0;
  if (func154(data[2], data[8]) == 0)
    return 0;
  if (func155(data[13], data[5]) == 0)
    return 0;
  if (func156(data[6]) == 0)
    return 0;
  if (func157(data[6]) == 0)
    return 0;
  if (func158(data[14]) == 0)
    return 0;
  if (func159(data[0]) == 0)
    return 0;
  if (func160(data[19]) == 0)
    return 0;
  if (func161(data[12], data[15]) == 0)
    return 0;
  if (func162(data[5]) == 0)
    return 0;
  if (func163(data[6], data[7], data[3]) == 0)
    return 0;
  if (func164(data[13], data[5]) == 0)
    return 0;
  if (func165(data[6]) == 0)
    return 0;
  if (func166(data[14]) == 0)
    return 0;
  if (func167(data[0]) == 0)
    return 0;
  if (func168(data[19]) == 0)
    return 0;
  if (func169(data[11], data[9]) == 0)
    return 0;
  if (func170(data[9], data[11]) == 0)
    return 0;
  if (func171(data[6]) == 0)
    return 0;
  if (func172(data[14]) == 0)
    return 0;
  if (func173(data[19]) == 0)
    return 0;
  if (func174(data[11], data[9]) == 0)
    return 0;
  if (func175(data[12], data[15]) == 0)
    return 0;
  if (func176(data[7]) == 0)
    return 0;
  if (func177(data[13], data[5]) == 0)
    return 0;
  if (func178(data[0]) == 0)
    return 0;
  if (func179(data[19]) == 0)
    return 0;
  if (func180(data[11], data[9]) == 0)
    return 0;
  if (func181(data[12], data[15]) == 0)
    return 0;
  if (func182(data[13], data[1]) == 0)
    return 0;
  if (func183(data[7], data[17]) == 0)
    return 0;
  if (func184(data[7]) == 0)
    return 0;
  if (func185(data[6]) == 0)
    return 0;
  if (func186(data[4], data[12]) == 0)
    return 0;
  if (func187(data[2], data[8]) == 0)
    return 0;
  if (func188(data[16]) == 0)
    return 0;
  if (func189(data[6]) == 0)
    return 0;
  if (func190(data[4], data[12]) == 0)
    return 0;
  if (func191(data[16]) == 0)
    return 0;
  if (func192(data[19]) == 0)
    return 0;
  if (func193(data[19], data[4], data[2]) == 0)
    return 0;
  if (func194(data[11], data[9]) == 0)
    return 0;
  if (func195(data[12], data[15]) == 0)
    return 0;
  if (func196(data[13], data[15]) == 0)
    return 0;
  if (func197(data[13], data[1]) == 0)
    return 0;
  if (func198(data[19], data[5]) == 0)
    return 0;
  if (func199(data[13], data[5]) == 0)
    return 0;
  if (func200(data[6]) == 0)
    return 0;
  if (func201(data[4], data[12]) == 0)
    return 0;
  if (func202(data[5], data[17]) == 0)
    return 0;
  if (func203(data[16]) == 0)
    return 0;
  if (func204(data[6]) == 0)
    return 0;
  if (func205(data[7], data[3], data[17]) == 0)
    return 0;
  if (func206(data[4], data[12]) == 0)
    return 0;
  if (func207(data[16]) == 0)
    return 0;
  if (func208(data[0]) == 0)
    return 0;
  if (func209(data[19]) == 0)
    return 0;
  if (func210(data[11], data[9]) == 0)
    return 0;
  if (func211(data[13], data[1]) == 0)
    return 0;
  if (func212(data[7]) == 0)
    return 0;
  if (func213(data[13], data[5]) == 0)
    return 0;
  if (func214(data[6]) == 0)
    return 0;
  if (func215(data[4], data[12]) == 0)
    return 0;
  if (func216(data[6]) == 0)
    return 0;
  if (func217(data[4], data[12]) == 0)
    return 0;
  if (func218(data[16]) == 0)
    return 0;
  if (func219(data[16]) == 0)
    return 0;
  if (func220(data[0]) == 0)
    return 0;
  if (func221(data[19]) == 0)
    return 0;
  if (func222(data[0], data[1]) == 0)
    return 0;
  if (func223(data[0], data[1]) == 0)
    return 0;
  if (func224(data[14]) == 0)
    return 0;
  if (func225(data[0]) == 0)
    return 0;
  if (func226(data[19]) == 0)
    return 0;
  if (func227(data[0], data[1]) == 0)
    return 0;
  if (func228(data[0], data[1]) == 0)
    return 0;
  if (func229(data[18], data[1]) == 0)
    return 0;
  if (func230(data[0], data[1]) == 0)
    return 0;
  if (func231(data[17], data[3]) == 0)
    return 0;
  if (func232(data[0], data[1]) == 0)
    return 0;
  if (func233(data[18], data[1]) == 0)
    return 0;
  if (func234(data[0], data[1]) == 0)
    return 0;
  if (func235(data[18], data[4]) == 0)
    return 0;
  if (func236(data[18], data[1]) == 0)
    return 0;
  if (func237(data[0], data[1]) == 0)
    return 0;
  if (func238(data[2], data[8]) == 0)
    return 0;
  if (func239(data[13], data[0]) == 0)
    return 0;
  if (func240(data[0], data[1]) == 0)
    return 0;
  if (func241(data[0], data[1]) == 0)
    return 0;
  if (func242(data[18], data[1]) == 0)
    return 0;
  if (func243(data[0], data[6]) == 0)
    return 0;
  if (func244(data[2]) == 0)
    return 0;
  if (func245(data[0], data[1]) == 0)
    return 0;
  if (func246(data[1], data[11]) == 0)
    return 0;
  if (func247(data[18], data[1]) == 0)
    return 0;
  if (func248(data[0], data[6]) == 0)
    return 0;
  if (func249(data[2]) == 0)
    return 0;
  if (func250(data[0], data[1]) == 0)
    return 0;
  if (func251(data[4], data[6]) == 0)
    return 0;
  if (func252(data[0], data[1]) == 0)
    return 0;
  if (func253(data[18], data[1]) == 0)
    return 0;
  if (func254(data[16], data[15], data[18]) == 0)
    return 0;
  if (func255(data[0], data[6]) == 0)
    return 0;
  if (func256(data[2]) == 0)
    return 0;
  if (func257(data[16], data[18]) == 0)
    return 0;
  if (func258(data[0], data[1]) == 0)
    return 0;
  if (func259(data[0], data[6]) == 0)
    return 0;
  if (func260(data[9], data[13]) == 0)
    return 0;
  if (func261(data[2]) == 0)
    return 0;
  if (func262(data[2]) == 0)
    return 0;
  if (func263(data[0], data[1]) == 0)
    return 0;
  if (func264(data[0], data[1]) == 0)
    return 0;
  if (func265(data[0], data[6]) == 0)
    return 0;
  if (func266(data[7], data[4]) == 0)
    return 0;
  if (func267(data[16], data[7]) == 0)
    return 0;
  if (func268(data[0], data[1]) == 0)
    return 0;
  if (func269(data[0], data[1]) == 0)
    return 0;
  if (func270(data[18], data[1]) == 0)
    return 0;
  if (func271(data[13], data[3]) == 0)
    return 0;
  if (func272(data[2]) == 0)
    return 0;
  if (func273(data[0], data[1]) == 0)
    return 0;
  if (func274(data[2]) == 0)
    return 0;
  if (func275(data[0], data[1]) == 0)
    return 0;
  if (func276(data[14]) == 0)
    return 0;
  if (func277(data[0]) == 0)
    return 0;
  if (func278(data[19]) == 0)
    return 0;
  if (func279(data[0], data[6]) == 0)
    return 0;
  if (func280(data[2]) == 0)
    return 0;
  if (func281(data[0], data[1]) == 0)
    return 0;
  if (func282(data[8], data[0]) == 0)
    return 0;
  if (func283(data[14]) == 0)
    return 0;
  if (func284(data[19]) == 0)
    return 0;
  if (func285(data[14]) == 0)
    return 0;
  if (func286(data[19]) == 0)
    return 0;
  if (func287(data[11], data[9]) == 0)
    return 0;
  if (func288(data[12], data[15]) == 0)
    return 0;
  if (func289(data[13], data[1]) == 0)
    return 0;
  if (func290(data[7]) == 0)
    return 0;
  if (func291(data[13], data[5]) == 0)
    return 0;
  if (func292(data[6]) == 0)
    return 0;
  if (func293(data[14]) == 0)
    return 0;
  if (func294(data[19]) == 0)
    return 0;
  if (func295(data[11], data[9]) == 0)
    return 0;
  if (func296(data[12], data[15]) == 0)
    return 0;
  if (func297(data[12], data[4]) == 0)
    return 0;
  if (func298(data[7]) == 0)
    return 0;
  if (func299(data[13], data[5]) == 0)
    return 0;
  if (func300(data[12], data[14]) == 0)
    return 0;
  if (func301(data[6]) == 0)
    return 0;
  if (func302(data[14]) == 0)
    return 0;
  if (func303(data[19]) == 0)
    return 0;
  if (func304(data[11], data[9]) == 0)
    return 0;
  if (func305(data[12], data[15]) == 0)
    return 0;
  if (func306(data[13], data[1]) == 0)
    return 0;
  if (func307(data[6]) == 0)
    return 0;
  if (func308(data[19]) == 0)
    return 0;
  if (func309(data[12], data[15]) == 0)
    return 0;
  if (func310(data[13], data[1]) == 0)
    return 0;
  if (func311(data[13], data[5]) == 0)
    return 0;
  if (func312(data[6]) == 0)
    return 0;
  if (func313(data[14]) == 0)
    return 0;
  if (func314(data[0]) == 0)
    return 0;
  if (func315(data[9]) == 0)
    return 0;
  if (func316(data[11], data[9]) == 0)
    return 0;
  if (func317(data[13], data[1]) == 0)
    return 0;
  if (func318(data[7]) == 0)
    return 0;
  if (func319(data[13], data[4]) == 0)
    return 0;
  if (func320(data[16]) == 0)
    return 0;
  if (func321(data[4]) == 0)
    return 0;
  if (func322(data[0]) == 0)
    return 0;
  if (func323(data[19]) == 0)
    return 0;
  if (func324(data[11], data[9]) == 0)
    return 0;
  if (func325(data[12], data[15]) == 0)
    return 0;
  if (func326(data[7]) == 0)
    return 0;
  if (func327(data[13], data[5]) == 0)
    return 0;
  if (func328(data[0]) == 0)
    return 0;
  if (func329(data[19]) == 0)
    return 0;
  if (func330(data[11], data[9]) == 0)
    return 0;
  if (func331(data[12], data[15]) == 0)
    return 0;
  if (func332(data[13], data[1]) == 0)
    return 0;
  if (func333(data[4]) == 0)
    return 0;
  if (func334(data[13], data[5]) == 0)
    return 0;
  if (func335(data[6]) == 0)
    return 0;
  if (func336(data[4], data[12]) == 0)
    return 0;
  if (func337(data[16]) == 0)
    return 0;
  if (func338(data[6]) == 0)
    return 0;
  if (func339(data[16]) == 0)
    return 0;
  if (func340(data[0]) == 0)
    return 0;
  if (func341(data[19]) == 0)
    return 0;
  if (func342(data[11], data[9]) == 0)
    return 0;
  if (func343(data[12], data[15]) == 0)
    return 0;
  if (func344(data[13], data[1]) == 0)
    return 0;
  if (func345(data[7]) == 0)
    return 0;
  if (func346(data[13], data[5]) == 0)
    return 0;
  if (func347(data[6]) == 0)
    return 0;
  if (func348(data[4], data[12]) == 0)
    return 0;
  if (func349(data[16]) == 0)
    return 0;
  if (func350(data[6]) == 0)
    return 0;
  if (func351(data[4], data[12]) == 0)
    return 0;
  if (func352(data[16]) == 0)
    return 0;
  if (func353(data[0]) == 0)
    return 0;
  if (func354(data[19]) == 0)
    return 0;
  if (func355(data[11], data[9]) == 0)
    return 0;
  if (func356(data[8], data[15]) == 0)
    return 0;
  if (func357(data[7], data[1]) == 0)
    return 0;
  if (func358(data[17]) == 0)
    return 0;
  if (func359(data[3], data[5]) == 0)
    return 0;
  if (func360(data[6]) == 0)
    return 0;
  if (func361(data[4], data[12]) == 0)
    return 0;
  if (func362(data[16]) == 0)
    return 0;
  if (func363(data[6]) == 0)
    return 0;
  if (func364(data[4], data[12]) == 0)
    return 0;
  if (func365(data[16]) == 0)
    return 0;
  if (func366(data[16]) == 0)
    return 0;
  if (func367(data[14]) == 0)
    return 0;
  if (func368(data[0]) == 0)
    return 0;
  if (func369(data[19]) == 0)
    return 0;
  if (func370(data[3], data[1]) == 0)
    return 0;
  if (func371(data[4], data[1]) == 0)
    return 0;
  if (func372(data[16]) == 0)
    return 0;
  if (func373(data[14]) == 0)
    return 0;
  if (func374(data[19]) == 0)
    return 0;
  if (func375(data[0], data[1]) == 0)
    return 0;
  if (func376(data[0], data[1]) == 0)
    return 0;
  if (func377(data[2]) == 0)
    return 0;
  if (func378(data[0], data[1]) == 0)
    return 0;
  if (func379(data[0], data[1]) == 0)
    return 0;
  if (func380(data[18], data[1]) == 0)
    return 0;
  if (func381(data[0], data[6]) == 0)
    return 0;
  if (func382(data[2]) == 0)
    return 0;
  if (func383(data[8], data[1]) == 0)
    return 0;
  if (func384(data[5], data[1]) == 0)
    return 0;
  if (func385(data[18], data[1]) == 0)
    return 0;
  if (func386(data[0], data[6]) == 0)
    return 0;
  if (func387(data[2]) == 0)
    return 0;
  if (func388(data[0], data[1]) == 0)
    return 0;
  if (func389(data[18], data[1]) == 0)
    return 0;
  if (func390(data[0], data[6]) == 0)
    return 0;
  if (func391(data[2]) == 0)
    return 0;
  if (func392(data[18], data[1]) == 0)
    return 0;
  if (func393(data[0], data[6]) == 0)
    return 0;
  if (func394(data[2]) == 0)
    return 0;
  if (func395(data[0], data[1]) == 0)
    return 0;
  if (func396(data[0], data[1]) == 0)
    return 0;
  if (func397(data[18], data[1]) == 0)
    return 0;
  if (func398(data[9]) == 0)
    return 0;
  if (func399(data[0], data[1]) == 0)
    return 0;
  if (func400(data[0], data[1]) == 0)
    return 0;
  if (func401(data[8], data[6]) == 0)
    return 0;
  if (func402(data[12]) == 0)
    return 0;
  if (func403(data[0], data[1]) == 0)
    return 0;
  if (func404(data[12]) == 0)
    return 0;
  if (func405(data[1]) == 0)
    return 0;
  if (func406(data[0], data[6]) == 0)
    return 0;
  if (func407(data[4]) == 0)
    return 0;
  if (func408(data[0], data[1]) == 0)
    return 0;
  if (func409(data[14]) == 0)
    return 0;
  if (func410(data[14]) == 0)
    return 0;

  fprintf(stderr, "BINGO\n");
  abort();
  return 1;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (api(Data, Size)) {
    // Should've crashed before getting here.
    return 0;
  }
  return 0;
}


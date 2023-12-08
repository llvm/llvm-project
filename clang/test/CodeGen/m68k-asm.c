// REQUIRES: m68k-registered-target
// RUN: %clang -target m68k -S %s -o - | FileCheck %s

// Test special escaped character in inline assembly
void escaped() {
  // '.' -> '.'
  // CHECK: move.l #66, %d1
  __asm__ ("move%.l #66, %%d1" ::);
  // '#' -> '#'
  // CHECK: move.l #66, %d1
  __asm__ ("move.l %#66, %%d1" ::);
  // '/' -> '%'
  // CHECK: move.l #66, %d1
  __asm__ ("move.l #66, %/d1" ::);
  // '$' -> 's'
  // CHECK: muls %d0, %d1
  __asm__ ("mul%$ %%d0, %%d1" ::);
  // '&' -> 'd'
  // CHECK: move.l %d0, %d1
  __asm__ ("move.l %%%&0, %%d1" ::);
}

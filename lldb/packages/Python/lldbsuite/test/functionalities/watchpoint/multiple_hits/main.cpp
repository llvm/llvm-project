//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>
alignas(16) uint8_t buf[32];
// This uses inline assembly to generate an instruction that writes to a large
// block of memory. If it fails on your compiler/architecture, please add
// appropriate code to generate a large write to "buf". If you cannot write at
// least 2*sizeof(void*) bytes with a single instruction, you will have to skip
// this test.

int main() {
#if defined(__i386__) || defined(__x86_64__)
  asm volatile ("movdqa %%xmm0, %0" : : "m"(buf));
#elif defined(__arm__)
  asm volatile ("stm %0, { r0, r1, r2, r3 }" : : "r"(buf));
#elif defined(__aarch64__)
  asm volatile ("stp x0, x1, %0" : : "m"(buf));
#elif defined(__mips__)
  asm volatile ("lw $2, %0" : : "m"(buf));
#endif
  return 0;
}

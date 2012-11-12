// REQUIRES: x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O1 -S -o - %s | FileCheck %s

typedef long long int64_t;
typedef unsigned char uint8_t;
typedef int64_t x86_reg;

void avg_pixels8_mmx2(uint8_t *block, const uint8_t *pixels, int line_size, int h)
{
	__asm__ volatile("# %0 %1 %2 %3"
     :"+g"(h), "+S"(pixels), "+D"(block)
     :"r" ((x86_reg)line_size)         
     :"%""rax", "memory");
// CHECK: # %ecx %rsi %rdi %rdx
 }

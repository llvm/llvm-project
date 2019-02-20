//===-- main.c ------------------------------------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

void func() {
  unsigned int ymmvalues[16];
  for (int i = 0 ; i < 16 ; i++)
  {
    unsigned char val = (0x80 | i);
    ymmvalues[i] = (val << 24) | (val << 16) | (val << 8) | val;
  }

  unsigned int ymmallones = 0xFFFFFFFF;
#if defined(__AVX__)
  __asm__("int3;"
          "vbroadcastss %1, %%ymm0;"
          "vbroadcastss %0, %%ymm0;"
          "vbroadcastss %2, %%ymm1;"
          "vbroadcastss %0, %%ymm1;"
          "vbroadcastss %3, %%ymm2;"
          "vbroadcastss %0, %%ymm2;"
          "vbroadcastss %4, %%ymm3;"
          "vbroadcastss %0, %%ymm3;"
          "vbroadcastss %5, %%ymm4;"
          "vbroadcastss %0, %%ymm4;"
          "vbroadcastss %6, %%ymm5;"
          "vbroadcastss %0, %%ymm5;"
          "vbroadcastss %7, %%ymm6;"
          "vbroadcastss %0, %%ymm6;"
          "vbroadcastss %8, %%ymm7;"
          "vbroadcastss %0, %%ymm7;"
#if defined(__x86_64__)
          "vbroadcastss %1, %%ymm8;"
          "vbroadcastss %0, %%ymm8;"
          "vbroadcastss %2, %%ymm9;"
          "vbroadcastss %0, %%ymm9;"
          "vbroadcastss %3, %%ymm10;"
          "vbroadcastss %0, %%ymm10;"
          "vbroadcastss %4, %%ymm11;"
          "vbroadcastss %0, %%ymm11;"
          "vbroadcastss %5, %%ymm12;"
          "vbroadcastss %0, %%ymm12;"
          "vbroadcastss %6, %%ymm13;"
          "vbroadcastss %0, %%ymm13;"
          "vbroadcastss %7, %%ymm14;"
          "vbroadcastss %0, %%ymm14;"
          "vbroadcastss %8, %%ymm15;"
          "vbroadcastss %0, %%ymm15;"
#endif
          ::"m"(ymmallones),
          "m"(ymmvalues[0]), "m"(ymmvalues[1]), "m"(ymmvalues[2]), "m"(ymmvalues[3]),
          "m"(ymmvalues[4]), "m"(ymmvalues[5]), "m"(ymmvalues[6]), "m"(ymmvalues[7])
              );
#endif

#if defined(__AVX512F__)
  unsigned int zmmvalues[32];
  for (int i = 0 ; i < 32 ; i++)
  {
    unsigned char val = (0x80 | i);
    zmmvalues[i] = (val << 24) | (val << 16) | (val << 8) | val;
  }

  __asm__("int3;"
          "vbroadcastss %1, %%zmm0;"
          "vbroadcastss %0, %%zmm0;"
          "vbroadcastss %2, %%zmm1;"
          "vbroadcastss %0, %%zmm1;"
          "vbroadcastss %3, %%zmm2;"
          "vbroadcastss %0, %%zmm2;"
          "vbroadcastss %4, %%zmm3;"
          "vbroadcastss %0, %%zmm3;"
          "vbroadcastss %5, %%zmm4;"
          "vbroadcastss %0, %%zmm4;"
          "vbroadcastss %6, %%zmm5;"
          "vbroadcastss %0, %%zmm5;"
          "vbroadcastss %7, %%zmm6;"
          "vbroadcastss %0, %%zmm6;"
          "vbroadcastss %8, %%zmm7;"
          "vbroadcastss %0, %%zmm7;"
#if defined(__x86_64__)
          "vbroadcastss %1, %%zmm8;"
          "vbroadcastss %0, %%zmm8;"
          "vbroadcastss %2, %%zmm9;"
          "vbroadcastss %0, %%zmm9;"
          "vbroadcastss %3, %%zmm10;"
          "vbroadcastss %0, %%zmm10;"
          "vbroadcastss %4, %%zmm11;"
          "vbroadcastss %0, %%zmm11;"
          "vbroadcastss %5, %%zmm12;"
          "vbroadcastss %0, %%zmm12;"
          "vbroadcastss %6, %%zmm13;"
          "vbroadcastss %0, %%zmm13;"
          "vbroadcastss %7, %%zmm14;"
          "vbroadcastss %0, %%zmm14;"
          "vbroadcastss %8, %%zmm15;"
          "vbroadcastss %0, %%zmm15;"
          "vbroadcastss %1, %%zmm16;"
          "vbroadcastss %0, %%zmm16;"
          "vbroadcastss %2, %%zmm17;"
          "vbroadcastss %0, %%zmm17;"
          "vbroadcastss %3, %%zmm18;"
          "vbroadcastss %0, %%zmm18;"
          "vbroadcastss %4, %%zmm19;"
          "vbroadcastss %0, %%zmm19;"
          "vbroadcastss %5, %%zmm20;"
          "vbroadcastss %0, %%zmm20;"
          "vbroadcastss %6, %%zmm21;"
          "vbroadcastss %0, %%zmm21;"
          "vbroadcastss %7, %%zmm22;"
          "vbroadcastss %0, %%zmm22;"
          "vbroadcastss %8, %%zmm23;"
          "vbroadcastss %0, %%zmm23;"
          "vbroadcastss %1, %%zmm24;"
          "vbroadcastss %0, %%zmm24;"
          "vbroadcastss %2, %%zmm25;"
          "vbroadcastss %0, %%zmm25;"
          "vbroadcastss %3, %%zmm26;"
          "vbroadcastss %0, %%zmm26;"
          "vbroadcastss %4, %%zmm27;"
          "vbroadcastss %0, %%zmm27;"
          "vbroadcastss %5, %%zmm28;"
          "vbroadcastss %0, %%zmm28;"
          "vbroadcastss %6, %%zmm29;"
          "vbroadcastss %0, %%zmm29;"
          "vbroadcastss %7, %%zmm30;"
          "vbroadcastss %0, %%zmm30;"
          "vbroadcastss %8, %%zmm31;"
          "vbroadcastss %0, %%zmm31;"
#endif
          ::"m"(ymmallones),
          "m"(zmmvalues[0]), "m"(zmmvalues[1]), "m"(zmmvalues[2]), "m"(zmmvalues[3]),
          "m"(zmmvalues[4]), "m"(zmmvalues[5]), "m"(zmmvalues[6]), "m"(zmmvalues[7])
  );
#endif
}

int main(int argc, char const *argv[]) { func(); }

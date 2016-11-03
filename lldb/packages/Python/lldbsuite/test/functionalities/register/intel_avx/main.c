//===-- main.c ------------------------------------------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

void func() {
  unsigned int ymmvalues[16];
  unsigned char val;
  unsigned char i;
  for (i = 0 ; i < 16 ; i++)
  {
    val = (0x80 | i);
    ymmvalues[i] = (val << 24) | (val << 16) | (val << 8) | val;
  }

  unsigned int ymmallones = 0xFFFFFFFF;
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
          "vbroadcastss %9, %%ymm8;"
          "vbroadcastss %0, %%ymm8;"
          "vbroadcastss %10, %%ymm9;"
          "vbroadcastss %0, %%ymm9;"
          "vbroadcastss %11, %%ymm10;"
          "vbroadcastss %0, %%ymm10;"
          "vbroadcastss %12, %%ymm11;"
          "vbroadcastss %0, %%ymm11;"
          "vbroadcastss %13, %%ymm12;"
          "vbroadcastss %0, %%ymm12;"
          "vbroadcastss %14, %%ymm13;"
          "vbroadcastss %0, %%ymm13;"
          "vbroadcastss %15, %%ymm14;"
          "vbroadcastss %0, %%ymm14;"
          "vbroadcastss %16, %%ymm15;"
          "vbroadcastss %0, %%ymm15;"
#endif
          ::"m"(ymmallones),
          "m"(ymmvalues[0]), "m"(ymmvalues[1]), "m"(ymmvalues[2]), "m"(ymmvalues[3]),
          "m"(ymmvalues[4]), "m"(ymmvalues[5]), "m"(ymmvalues[6]), "m"(ymmvalues[7])
#if defined(__x86_64__)
          ,
          "m"(ymmvalues[8]), "m"(ymmvalues[9]), "m"(ymmvalues[10]), "m"(ymmvalues[11]),
          "m"(ymmvalues[12]), "m"(ymmvalues[13]), "m"(ymmvalues[14]), "m"(ymmvalues[15])
#endif
              );
}

int main(int argc, char const *argv[]) { func(); }

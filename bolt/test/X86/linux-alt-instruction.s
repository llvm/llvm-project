# REQUIRES: system-linux

## Check that BOLT correctly parses the Linux kernel .altinstructions section
## and annotates alternative instructions.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-normalized --alt-inst-feature-size=2 -o %t.out \
# RUN:   | FileCheck %s

## Older kernels used to have padlen field in alt_instr. Check compatibility.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --defsym PADLEN=1 \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-normalized --alt-inst-has-padlen -o %t.out \
# RUN:   | FileCheck %s

## Check with a larger size of "feature" field in alt_instr.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   --defsym FEATURE_SIZE_4=1 %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-normalized --alt-inst-feature-size=4 -o %t.out \
# RUN:   | FileCheck %s

## Check that out-of-bounds read is handled properly.

# RUN: not llvm-bolt %t.exe --print-normalized --alt-inst-feature-size=2 -o %t.out

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 2 alternative instruction entries

  .text
  .globl _start
  .type _start, %function
_start:
# CHECK: Binary Function "_start"
.L0:
  rdtsc
# CHECK:      rdtsc
# CHECK-SAME: AltInst: 1
# CHECK-SAME: AltInst2: 2
  nop
# CHECK-NEXT: nop
# CHECK-SAME: AltInst: 1
# CHECK-SAME: AltInst2: 2
  nop
  nop
.L1:
  ret
  .size _start, .-_start

  .section .altinstr_replacement,"ax",@progbits
.A0:
  lfence
  rdtsc
.A1:
  rdtscp
.Ae:

## Alternative instruction info.
  .section .altinstructions,"a",@progbits

  .long .L0 - .   # org instruction
  .long .A0 - .   # alt instruction
.ifdef FEATURE_SIZE_4
  .long 0x72      # feature flags
.else
  .word 0x72      # feature flags
.endif
  .byte .L1 - .L0 # org size
  .byte .A1 - .A0 # alt size
.ifdef PADLEN
  .byte 0
.endif

  .long .L0 - .   # org instruction
  .long .A1 - .   # alt instruction
.ifdef FEATURE_SIZE_4
  .long 0x3b      # feature flags
.else
  .word 0x3b      # feature flags
.endif
  .byte .L1 - .L0 # org size
  .byte .Ae - .A1 # alt size
.ifdef PADLEN
  .byte 0
.endif

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits

# REQUIRES: system-linux

## Check that BOLT correctly parses the Linux kernel .altinstructions section
## and annotates alternative instructions.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.exe --print-cfg --alt-inst-feature-size=2 -o %t.out \
# RUN:   | FileCheck %s

## Older kernels used to have padlen field in alt_instr. Check compatibility.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --defsym PADLEN=1 \
# RUN:   %s -o %t.padlen.o
# RUN: %clang %cflags -nostdlib %t.padlen.o -o %t.padlen.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.padlen.exe --print-cfg --alt-inst-has-padlen -o %t.padlen.out \
# RUN:   | FileCheck %s

## Check with a larger size of "feature" field in alt_instr.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   --defsym FEATURE_SIZE_4=1 %s -o %t.fs4.o
# RUN: %clang %cflags -nostdlib %t.fs4.o -o %t.fs4.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr,--no-pie
# RUN: llvm-bolt %t.fs4.exe --print-cfg --alt-inst-feature-size=4 -o %t.fs4.out \
# RUN:   | FileCheck %s

## Check that out-of-bounds read is handled properly.

# RUN: not llvm-bolt %t.fs4.exe --alt-inst-feature-size=2 -o %t.fs4.out

## Check that BOLT automatically detects structure fields in .altinstructions.

# RUN: llvm-bolt %t.exe --print-cfg -o %t.out | FileCheck %s
# RUN: llvm-bolt %t.exe --print-cfg -o %t.padlen.out | FileCheck %s
# RUN: llvm-bolt %t.exe --print-cfg -o %t.fs4.out | FileCheck %s

# CHECK:      BOLT-INFO: Linux kernel binary detected
# CHECK:      BOLT-INFO: parsed 3 alternative instruction entries

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
# CHECK-SAME: AltInst3: 3
  nop
# CHECK-NEXT: nop
# CHECK-SAME: AltInst: 1
# CHECK-SAME: AltInst2: 2
# CHECK-SAME: AltInst3: 3
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
.A2:
  pushf
  pop %rax
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
  .byte .A2 - .A1 # alt size
.ifdef PADLEN
  .byte 0
.endif

  .long .L0 - .   # org instruction
  .long .A2 - .   # alt instruction
.ifdef FEATURE_SIZE_4
  .long 0x110     # feature flags
.else
  .word 0x110     # feature flags
.endif
  .byte .L1 - .L0 # org size
  .byte .Ae - .A2 # alt size
.ifdef PADLEN
  .byte 0
.endif

## ORC unwind for "pushf; pop %rax" alternative sequence.
  .section .orc_unwind,"a",@progbits
  .align 4
  .section .orc_unwind_ip,"a",@progbits
  .align 4

  .section .orc_unwind
  .2byte 8
  .2byte 0
  .2byte 0x205
  .section .orc_unwind_ip
  .long _start - .

  .section .orc_unwind
  .2byte 16
  .2byte 0
  .2byte 0x205
  .section .orc_unwind_ip
  .long .L0 + 1 - .

  .section .orc_unwind
  .2byte 8
  .2byte 0
  .2byte 0x205
  .section .orc_unwind_ip
  .long .L0 + 2 - .

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits

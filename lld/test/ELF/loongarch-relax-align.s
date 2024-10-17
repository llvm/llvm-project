# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.64.o
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.32.o -o %t.32
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.64.o -o %t.64
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.32.o --no-relax -o %t.32n
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.64.o --no-relax -o %t.64n
# RUN: llvm-objdump -td --no-show-raw-insn %t.32 | FileCheck %s --check-prefix=RELAX32
# RUN: llvm-objdump -td --no-show-raw-insn %t.64 | FileCheck %s --check-prefixes=RELAX64,SRELAX64
# RUN: llvm-objdump -td --no-show-raw-insn %t.32n | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-objdump -td --no-show-raw-insn %t.64n | FileCheck %s --check-prefix=NORELAX

## Test the R_LARCH_ALIGN without symbol index.
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.o64.o --defsym=old=1
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.o64.o -o %t.o64
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.text2=0x20000 -e 0 %t.o64.o --no-relax -o %t.o64n
# RUN: llvm-objdump -td --no-show-raw-insn %t.o64 | FileCheck %s --check-prefixes=RELAX64,ORELAX64
# RUN: llvm-objdump -td --no-show-raw-insn %t.o64n | FileCheck %s --check-prefix=ONORELAX

## -r keeps section contents unchanged.
# RUN: ld.lld -r %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr --no-show-raw-insn %t.64.r | FileCheck %s --check-prefix=CHECKR

# RELAX32-DAG: {{0*}}10000 l .text  {{0*}}00 .Lalign_symbol
# RELAX32-DAG: {{0*}}10000 l .text  {{0*}}44 .Ltext_start
# RELAX32-DAG: {{0*}}10038 l .text  {{0*}}0c .L1
# RELAX32-DAG: {{0*}}10040 l .text  {{0*}}04 .L2
# RELAX32-DAG: {{0*}}20000 l .text2 {{0*}}14 .Ltext2_start

# RELAX32:      <.Ltext_start>:
# RELAX32-NEXT:   break 1
# RELAX32-NEXT:   break 2
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   break 3
# RELAX32-NEXT:   break 4
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   pcalau12i     $a0, 0
# RELAX32-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# RELAX32-NEXT:   pcalau12i     $a0, 0
# RELAX32-NEXT:   addi.{{[dw]}} $a0, $a0, 56
# RELAX32-NEXT:   pcalau12i     $a0, 0
# RELAX32-NEXT:   addi.{{[dw]}} $a0, $a0, 64
# RELAX32-EMPTY:
# RELAX32-NEXT: <.L1>:
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   nop
# RELAX32-EMPTY:
# RELAX32-NEXT: <.L2>:
# RELAX32-NEXT:   break 5

# RELAX32:      <.Ltext2_start>:
# RELAX32-NEXT:   pcalau12i     $a0, 0
# RELAX32-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   nop
# RELAX32-NEXT:   break 6

# NORELAX-DAG: {{0*}}10000 l .text  {{0*}}00 .Lalign_symbol
# NORELAX-DAG: {{0*}}10000 l .text  {{0*}}44 .Ltext_start
# NORELAX-DAG: {{0*}}10038 l .text  {{0*}}0c .L1
# NORELAX-DAG: {{0*}}10040 l .text  {{0*}}04 .L2
# NORELAX-DAG: {{0*}}20000 l .text2 {{0*}}14 .Ltext2_start

# NORELAX:      <.Ltext_start>:
# NORELAX-NEXT:   break 1
# NORELAX-NEXT:   break 2
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   break 3
# NORELAX-NEXT:   break 4
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   pcalau12i     $a0, 0
# NORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# NORELAX-NEXT:   pcalau12i     $a0, 0
# NORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 56
# NORELAX-NEXT:   pcalau12i     $a0, 0
# NORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 64
# NORELAX-EMPTY:
# NORELAX-NEXT: <.L1>:
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   nop
# NORELAX-EMPTY:
# NORELAX-NEXT: <.L2>:
# NORELAX-NEXT:   break 5

# NORELAX:      <.Ltext2_start>:
# NORELAX-NEXT:   pcalau12i     $a0, 0
# NORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   nop
# NORELAX-NEXT:   break 6



# ORELAX64-DAG: {{0*}}00001 l *ABS*  {{0*}}00 old
# SRELAX64-DAG: {{0*}}10000 l .text  {{0*}}00 .Lalign_symbol
# RELAX64-DAG: {{0*}}10000 l .text  {{0*}}34 .Ltext_start
# RELAX64-DAG: {{0*}}1002c l .text  {{0*}}08 .L1
# RELAX64-DAG: {{0*}}10030 l .text  {{0*}}04 .L2
# RELAX64-DAG: {{0*}}20000 l .text2 {{0*}}14 .Ltext2_start

# RELAX64:      <.Ltext_start>:
# RELAX64-NEXT:   break 1
# RELAX64-NEXT:   break 2
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   break 3
# RELAX64-NEXT:   break 4
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   pcaddi  $a0, -8
# RELAX64-NEXT:   pcaddi  $a0, 2
# RELAX64-NEXT:   pcaddi  $a0, 2
# RELAX64-EMPTY:
# RELAX64-NEXT: <.L1>:
# RELAX64-NEXT:   nop
# RELAX64-EMPTY:
# RELAX64-NEXT: <.L2>:
# RELAX64-NEXT:   break 5

# RELAX64:      <.Ltext2_start>:
# RELAX64-NEXT:   pcaddi     $a0, 0
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   nop
# RELAX64-NEXT:   break 6


# ONORELAX-DAG: {{0*}}00001 l *ABS*  {{0*}}00 old
# ONORELAX-DAG: {{0*}}10000 l .text  {{0*}}44 .Ltext_start
# ONORELAX-DAG: {{0*}}10038 l .text  {{0*}}0c .L1
# ONORELAX-DAG: {{0*}}10040 l .text  {{0*}}04 .L2
# ONORELAX-DAG: {{0*}}20000 l .text2 {{0*}}14 .Ltext2_start

# ONORELAX:      <.Ltext_start>:
# ONORELAX-NEXT:   break 1
# ONORELAX-NEXT:   break 2
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   break 3
# ONORELAX-NEXT:   break 4
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   pcalau12i     $a0, 0
# ONORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# ONORELAX-NEXT:   pcalau12i     $a0, 0
# ONORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 56
# ONORELAX-NEXT:   pcalau12i     $a0, 0
# ONORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 64
# ONORELAX-EMPTY:
# ONORELAX-NEXT: <.L1>:
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   nop
# ONORELAX-EMPTY:
# ONORELAX-NEXT: <.L2>:
# ONORELAX-NEXT:   break 5

# ONORELAX:      <.Ltext2_start>:
# ONORELAX-NEXT:   pcalau12i     $a0, 0
# ONORELAX-NEXT:   addi.{{[dw]}} $a0, $a0, 0
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   nop
# ONORELAX-NEXT:   break 6






# CHECKR:      <.Ltext2_start>:
# CHECKR-NEXT:   pcalau12i $a0, 0
# CHECKR-NEXT:   {{0*}}00: R_LARCH_PCALA_HI20 .Ltext2_start
# CHECKR-NEXT:   {{0*}}00: R_LARCH_RELAX      *ABS*
# CHECKR-NEXT:   addi.d    $a0, $a0, 0
# CHECKR-NEXT:   {{0*}}04: R_LARCH_PCALA_LO12 .Ltext2_start
# CHECKR-NEXT:   {{0*}}04: R_LARCH_RELAX      *ABS*
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   {{0*}}08: R_LARCH_ALIGN      .Lalign_symbol+0x4
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   nop
# CHECKR-NEXT:   break 6

.macro .fake_p2align_4 max=0
  .ifdef old
    .if \max==0
      .reloc ., R_LARCH_ALIGN, 0xc
      nop; nop; nop
    .endif
  .else
    .reloc ., R_LARCH_ALIGN, .Lalign_symbol + 0x4 + (\max << 8)
    nop; nop; nop
  .endif
.endm

  .text
.Lalign_symbol:
.Ltext_start:
  break 1
  break 2
## +0x8: Emit 2 nops, delete 1 nop.
  .fake_p2align_4

  break 3
## +0x14: Emit 3 nops > 8 bytes, not emit.
  .fake_p2align_4 8

  break 4
  .fake_p2align_4 8
## +0x18: Emit 2 nops <= 8 bytes.

## Compensate
.ifdef old
  nop; nop
.endif

## +0x20: Test symbol value and symbol size can be handled.
  la.pcrel $a0, .Ltext_start
  la.pcrel $a0, .L1
  la.pcrel $a0, .L2

## +0x38: Emit 2 nops, delete 1 nop.
.L1:
  .fake_p2align_4
.L2:
  break 5
  .size .L1, . - .L1
  .size .L2, . - .L2
  .size .Ltext_start, . - .Ltext_start

## Test another text section.
  .section .text2,"ax",@progbits
.Ltext2_start:
  la.pcrel $a0, .Ltext2_start
  .fake_p2align_4
  break 6
  .size .Ltext2_start, . - .Ltext2_start

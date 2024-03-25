# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:   .text_caller1 0x10010000 : { *(.text_caller1) } \
# RUN:   .text_caller2 0x10020000 : { *(.text_caller2) } \
# RUN:   .text_caller3 0x10030000 : { *(.text_caller3) } \
# RUN:   }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le --defsym AUX=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so
# RUN: ld.lld -T %t.script %t1.o %t2.so -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=REL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 --defsym AUX=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so
# RUN: ld.lld -T %t.script %t1.o %t2.so -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=REL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64le --defsym AUX=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so
# RUN: ld.lld -T %t.script %t1.o %t2.so -o %t --no-power10-stubs
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL-NOP10
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=SEC-NOP10
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=REL-NOP10
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=CHECK-NOP10

# RUN: llvm-mc -filetype=obj -triple=powerpc64 --defsym AUX=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so
# RUN: ld.lld -T %t.script %t1.o %t2.so -o %t --no-power10-stubs
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL-NOP10
# RUN: llvm-readelf -S -d %t | FileCheck %s --check-prefix=SEC-NOP10
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=REL-NOP10
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=CHECK-NOP10

## The test is created to check that when a function without TOC access an
## external function, a r12 setup stub is inserted.

# SYMBOL: Symbol table '.dynsym' contains 4 entries:
# SYMBOL:      1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT                UND callee_global_stother0
# SYMBOL-NEXT: 2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x20>] UND callee_global_stother1
# SYMBOL-NEXT: 3: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x60>] UND callee_global_TOC

# SYMBOL: Symbol table '.symtab' contains 12 entries:
# SYMBOL:      2: 0000000010010000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   6 caller1
# SYMBOL-NEXT: 3: 0000000010020000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   7 caller2
# SYMBOL-NEXT: 4: 0000000010030000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   8 caller3
# SYMBOL:      6: 0000000010010010    32 FUNC    LOCAL  DEFAULT                  6 __plt_pcrel_callee_global_stother0
# SYMBOL-NEXT: 7: 0000000010020010    32 FUNC    LOCAL  DEFAULT                  7 __plt_pcrel_callee_global_stother1
# SYMBOL-NEXT: 8: 0000000010030010    32 FUNC    LOCAL  DEFAULT                  8 __plt_pcrel_callee_global_TOC
# SYMBOL-NEXT: 9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT                UND callee_global_stother0
# SYMBOL-NEXT: 10: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x20>] UND callee_global_stother1
# SYMBOL-NEXT: 11: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x60>] UND callee_global_TOC

# SYMBOL-NOP10: Symbol table '.symtab' contains 12 entries:
# SYMBOL-NOP10:      2: 0000000010010000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   6 caller1
# SYMBOL-NOP10-NEXT: 3: 0000000010020000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   7 caller2
# SYMBOL-NOP10-NEXT: 4: 0000000010030000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   8 caller3
# SYMBOL-NOP10:      6: 0000000010010010    32 FUNC    LOCAL  DEFAULT                  6 __plt_pcrel_callee_global_stother0
# SYMBOL-NOP10-NEXT: 7: 0000000010020010    32 FUNC    LOCAL  DEFAULT                  7 __plt_pcrel_callee_global_stother1
# SYMBOL-NOP10-NEXT: 8: 0000000010030010    32 FUNC    LOCAL  DEFAULT                  8 __plt_pcrel_callee_global_TOC
# SYMBOL-NOP10-NEXT: 9: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT                UND callee_global_stother0
# SYMBOL-NOP10-NEXT: 10: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x20>] UND callee_global_stother1
# SYMBOL-NOP10-NEXT: 11: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [<other: 0x60>] UND callee_global_TOC

## DT_PLTGOT points to .plt
# SEC: .plt              NOBITS          0000000010030168 040168 000028 00  WA  0   0  8
# SEC-OG: .plt              NOBITS          0000000010030158 040158 000028 00  WA  0   0  8
# SEC: 0x0000000000000003 (PLTGOT)      0x10030168
# SEC-OG: 0x0000000000000003 (PLTGOT)      0x10030168

## DT_PLTGOT points to .plt
# SEC-NOP10: .plt              NOBITS          0000000010030168 040168 000028 00  WA  0   0  8
# SEC-NOP10: 0x0000000000000003 (PLTGOT)      0x10030168

## The first 2 entries in the .plt are reserved for the dynamic linkers
## usage. The JMP_SLOT relocations are stored at .plt[2], .plt[3], .plt[4].
## Check that we emit 3 R_PPC64_JMP_SLOT in .rela.plt.
# REL:      .rela.plt {
# REL-NEXT:   0x10030178 R_PPC64_JMP_SLOT callee_global_stother0 0x0
# REL-NEXT-OG:   0x10030168 R_PPC64_JMP_SLOT callee_global_stother0 0x0
# REL-NEXT:   0x10030180 R_PPC64_JMP_SLOT callee_global_stother1 0x0
# REL-NEXT-OG:   0x10030170 R_PPC64_JMP_SLOT callee_global_stother1 0x0
# REL-NEXT:   0x10030188 R_PPC64_JMP_SLOT callee_global_TOC 0x0
# REL-NEXT-OG:   0x10030178 R_PPC64_JMP_SLOT callee_global_TOC 0x0
# REL-NEXT: }

# REL-NOP10:      .rela.plt {
# REL-NOP10-NEXT:   0x10030178 R_PPC64_JMP_SLOT callee_global_stother0 0x0
# REL-NOP10-NEXT:   0x10030180 R_PPC64_JMP_SLOT callee_global_stother1 0x0
# REL-NOP10-NEXT:   0x10030188 R_PPC64_JMP_SLOT callee_global_TOC 0x0
# REL-NOP10-NEXT: }

# CHECK-LABEL: <caller1>:
# CHECK:       10010000: bl 0x10010010
# CHECK-NEXT:  10010004: blr

# CHECK-NOP10-LABEL: <caller1>:
# CHECK-NOP10:       10010000: bl 0x10010010
# CHECK-NOP10-NEXT:  10010004: blr

## .plt[2] - 0x10010010 = 0x10030178 - 0x10010010 = 0x20168 = 131432
# CHECK-LABEL: <__plt_pcrel_callee_global_stother0>:
# CHECK:       10010010: pld 12, 131432(0), 1
# CHECK-NEXT:  10010018: mtctr 12
# CHECK-NEXT:  1001001c: bctr

## No P10; branch to next inst to get addr
# CHECK-NOP10-LABEL: <__plt_pcrel_callee_global_stother0>:
# CHECK-NOP10:       10010010: mflr 12
# CHECK-NOP10-NEXT:            bcl 20, 31, 0x10010018
# CHECK-NOP10-NEXT:  10010018: mflr 11
# CHECK-NOP10-NEXT:            mtlr 12
# CHECK-NOP10-NEXT:            addis 12, 11, 2
# CHECK-NOP10-NEXT:            ld 12, 352(12)
# CHECK-NOP10-NEXT:            mtctr 12
# CHECK-NOP10-NEXT:            bctr

# CHECK-LABEL: <caller2>:
# CHECK:       10020000: bl 0x10020010
# CHECK-NEXT:  10020004: blr

# CHECK-NOP10-LABEL: <caller2>:
# CHECK-NOP10:       10020000: bl 0x10020010
# CHECK-NOP10-NEXT:  10020004: blr

## .plt[3] - 0x10020010 = 0x10030180 - 0x10020010 = 0x10170 = 65904
# CHECK-LABEL: <__plt_pcrel_callee_global_stother1>:
# CHECK:       10020010: pld 12, 65904(0), 1
# CHECK-NEXT:  10020018: mtctr 12
# CHECK-NEXT:  1002001c: bctr

## no P10; branch to next inst to get addr
## .plt[3]-r11 = 0x10030170-0x10020018 = 65536*1+344
# CHECK-NOP10-LABEL: <__plt_pcrel_callee_global_stother1>:
# CHECK-NOP10:       10020010: mflr 12
# CHECK-NOP10-NEXT:            bcl 20, 31, 0x10020018
# CHECK-NOP10-NEXT:  10020018: mflr 11
# CHECK-NOP10-NEXT:            mtlr 12
# CHECK-NOP10-NEXT:            addis 12, 11, 1
# CHECK-NOP10-NEXT:            ld 12, 360(12)
# CHECK-NOP10-NEXT:            mtctr 12
# CHECK-NOP10-NEXT:            bctr

# CHECK-LABEL: <caller3>:
# CHECK:       10030000: bl 0x10030010
# CHECK-NEXT:  10030004: blr

# CHECK-NOP10-LABEL: <caller3>:
# CHECK-NOP10:       10030000: bl 0x10030010
# CHECK-NOP10-NEXT:  10030004: blr

## .plt[4] - 0x10030010 = 0x10030188 - 0x10030010 = 0x178 = 376
# CHECK-LABEL: <__plt_pcrel_callee_global_TOC>:
# CHECK:       10030010: pld 12, 376(0), 1
# CHECK-NEXT:  10030018: mtctr 12
# CHECK-NEXT:  1003001c: bctr

## no P10; branch to next inst to get addr
## .plt[4]-r11 = 0x10030178-0x10030018 = 65536*0+352
# CHECK-NOP10-LABEL: <__plt_pcrel_callee_global_TOC>:
# CHECK-NOP10-NEXT:  10030010: mflr 12
# CHECK-NOP10-NEXT:            bcl 20, 31, 0x10030018
# CHECK-NOP10-NEXT:  10030018: mflr 11
# CHECK-NOP10-NEXT:            mtlr 12
# CHECK-NOP10-NEXT:            addis 12, 11, 0
# CHECK-NOP10-NEXT:            ld 12, 368(12)
# CHECK-NOP10-NEXT:            mtctr 12
# CHECK-NOP10-NEXT:            bctr

.ifdef AUX
.section .text_caller1, "ax", %progbits
caller1:
  .localentry caller1, 1
  bl callee_global_stother0@notoc
  blr
.section .text_caller2, "ax", %progbits
caller2:
  .localentry caller2, 1
  bl callee_global_stother1@notoc
  blr

.section .text_caller3, "ax", %progbits
caller3:
  .localentry caller3, 1
  bl callee_global_TOC@notoc
  blr

.else
func_extern:
  blr
.globl callee_global_stother0
callee_global_stother0:
  blr
.globl callee_global_stother1
callee_global_stother1:
  .localentry callee_global_stother1, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl func_extern@notoc
  blr
.globl callee_global_TOC
callee_global_TOC:
.Lfunc_gep1:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
  .localentry callee_global_TOC, .Lfunc_lep1-.Lfunc_gep1
  addis 4, 2, global@toc@ha
  lwz 3, global@toc@l(4)
  blr
global:
  .long  0
  .size  global, 4
.endif

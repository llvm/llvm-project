# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %t/c.s -o %t/c.o
# RUN: ld.lld %t/a.o -o %t/a
# RUN: llvm-objdump -t --section-headers %t/a | FileCheck %s

## This checks that:
## 1) Address of _etext is the first location after the last read-only loadable segment.
## 2) Address of _edata points to the end of the last non SHT_NOBITS section.
##    That is how gold/bfd do. At the same time specs says: "If the address of _edata is
##    greater than the address of _etext, the address of _end is same as the address
##    of _edata." (https://docs.oracle.com/cd/E53394_01/html/E54766/u-etext-3c.html).
## 3) Address of _end is different from _edata because of 2.
## 4) Addresses of _edata == edata, _end == end and _etext == etext.
# CHECK:      Sections:
# CHECK-NEXT:  Idx Name          Size     VMA              Type
# CHECK-NEXT:    0               00000000 0000000000000000
# CHECK-NEXT:    1 .text         00000001 0000000000201158 TEXT
# CHECK-NEXT:    2 .data         00000002 0000000000202159 DATA
# CHECK-NEXT:    3 .bss          00000006 000000000020215c BSS
# CHECK:      SYMBOL TABLE:
# CHECK-NEXT:  000000000020215b g       .data 0000000000000000 _edata
# CHECK-NEXT:  0000000000202162 g       .bss  0000000000000000 _end
# CHECK-NEXT:  0000000000201159 g       .text 0000000000000000 _etext
# CHECK-NEXT:  0000000000201158 g       .text 0000000000000000 _start
# CHECK-NEXT:  000000000020215b g       .data 0000000000000000 edata
# CHECK-NEXT:  0000000000202162 g       .bss  0000000000000000 end
# CHECK-NEXT:  0000000000201159 g       .text 0000000000000000 etext

# RUN: ld.lld -r %t/a.o -o %t/a.ro
# RUN: llvm-objdump -t %t/a.ro | FileCheck %s --check-prefix=RELOCATABLE
# RELOCATABLE:       0000000000000000 *UND* 0000000000000000 _edata
# RELOCATABLE-NEXT:  0000000000000000 *UND* 0000000000000000 _end
# RELOCATABLE-NEXT:  0000000000000000 *UND* 0000000000000000 _etext

## If a relocatable object file defines non-reserved identifiers (by C and C++)
## edata/end/etext, don't redefine them. Note: GNU ld redefines the reserved
## _edata while we don't for simplicty.
# RUN: ld.lld %t/b.o -o %t/b
# RUN: llvm-objdump -t %t/b | FileCheck %s --check-prefix=CHECK2
# RUN: ld.lld %t/c.o -o %t/c
# RUN: llvm-objdump -t %t/c | FileCheck %s --check-prefix=CHECK2
## PROVIDE does not redefine defined symbols, even if COMMON.
# RUN: ld.lld %t/c.o %t/lds -o %t/c
# RUN: llvm-objdump -t %t/c | FileCheck %s --check-prefix=CHECK2

# CHECK2:       [[#%x,]] g     O .bss   0000000000000001 _edata
# CHECK2-NEXT:  [[#%x,]] g     O .bss   0000000000000001 edata
# CHECK2-NEXT:  [[#%x,]] g     O .bss   0000000000000001 end
# CHECK2-NEXT:  [[#%x,]] g     O .bss   0000000000000001 etext

#--- a.s
.global _edata,_end,_etext,_start,edata,end,etext
.text
_start:
  nop
.data
  .word 1
.bss
  .align 4
  .space 6

#--- b.s
.bss
.macro def x
  .globl \x
  .type \x, @object
  \x: .byte 0
  .size \x, 1
.endm
def _edata
def edata
def end
def etext

#--- c.s
.comm _edata,1,1
.comm edata,1,1
.comm end,1,1
.comm etext,1,1

#--- lds
SECTIONS {
  .text : { *(.text) }

  PROVIDE(etext = .);
  PROVIDE(edata = .);
  PROVIDE(end = .);
}

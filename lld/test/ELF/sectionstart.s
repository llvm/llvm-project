# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o --section-start .text=0x100000 \
# RUN:   --section-start=.data=0x110000 --section-start .bss=0x200000 -o %t
# RUN: llvm-objdump --section-headers %t | FileCheck %s

# CHECK:      Sections:
# CHECK-NEXT:  Idx Name          Size     VMA              Type
# CHECK-NEXT:    0               00000000 0000000000000000
# CHECK-NEXT:    1 .text         00000001 0000000000100000 TEXT
# CHECK-NEXT:    2 .data         00000004 0000000000110000 DATA
# CHECK-NEXT:    3 .bss          00000004 0000000000200000 BSS

## Check that PHDRS are allocated below .text if .text is below default
## base for non-pie case

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x3000 -o %t
# RUN: llvm-readelf -l %t | FileCheck --check-prefix=CHECK-TEXT %s

# CHECK-TEXT: Program Headers:
# CHECK-TEXT-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-TEXT-NEXT:  PHDR 0x000040 0x0000000000002040 0x0000000000002040 0x000118 0x000118 R 0x8
# CHECK-TEXT-NEXT:  LOAD 0x000000 0x0000000000002000 0x0000000000002000 0x000158 0x000158 R 0x1000
# CHECK-TEXT-NEXT:  LOAD 0x001000 0x0000000000003000 0x0000000000003000 0x000001 0x000001 R E 0x1000

## Check that PHDRS are deleted if they don't fit

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x0 -o %t
# RUN: llvm-readelf -l %t | FileCheck --check-prefix=CHECK-TEXT-ZERO %s

# CHECK-TEXT-ZERO: Program Headers:
# CHECK-TEXT-ZERO-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-TEXT-ZERO-NEXT:  LOAD 0x001000 0x0000000000000000 0x0000000000000000 0x000001 0x000001 R E 0x1000
# CHECK-TEXT-ZERO-NEXT:  LOAD 0x001001 0x0000000000001001 0x0000000000001001 0x000004 0x000008 RW 0x1000

## The same, but dropped "0x" prefix.
# RUN: ld.lld %t.o --section-start .text=100000 \
# RUN:   --section-start .data=110000 --section-start .bss=0x200000 -o %t1
# RUN: llvm-objdump --section-headers %t1 | FileCheck %s

## Use -Ttext, -Tdata, -Tbss as replacement for --section-start:
# RUN: ld.lld %t.o -Ttext=0x100000 -Tdata=0x110000 -Tbss=0x200000 -o %t4
# RUN: llvm-objdump --section-headers %t4 | FileCheck %s

## The same, but dropped "0x" prefix.
# RUN: ld.lld %t.o -Ttext=100000 -Tdata=110000 -Tbss=200000 -o %t5
# RUN: llvm-objdump --section-headers %t5 | FileCheck %s

## Check form without assignment:
# RUN: ld.lld %t.o -Ttext 0x100000 -Tdata 0x110000 -Tbss 0x200000 -o %t4
# RUN: llvm-objdump --section-headers %t4 | FileCheck %s

## Errors:
# RUN: not ld.lld %t.o --section-start .text100000 -o /dev/null 2>&1 \
# RUN:    | FileCheck -check-prefix=ERR1 %s
# ERR1: invalid argument: --section-start .text100000

# RUN: not ld.lld %t.o --section-start .text=1Q0000 -o /dev/null 2>&1 \
# RUN:    | FileCheck -check-prefix=ERR2 %s
# ERR2: invalid argument: --section-start .text=1Q0000

# RUN: not ld.lld %t.o -Ttext=1w0000 -o /dev/null 2>&1 \
# RUN:    | FileCheck -check-prefix=ERR3 %s
# ERR3: invalid argument: -Ttext=1w0000

# RUN: not ld.lld %t.o -Tbss=1w0000 -o /dev/null 2>&1 \
# RUN:    | FileCheck -check-prefix=ERR4 %s
# ERR4: invalid argument: -Tbss=1w0000

# RUN: not ld.lld %t.o -Tdata=1w0000 -o /dev/null 2>&1 \
# RUN:    | FileCheck -check-prefix=ERR5 %s
# ERR5: invalid argument: -Tdata=1w0000

.text
.globl _start
_start:
 nop

.data
.long 0

.bss
.zero 4

# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x2002000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -S -r %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-readelf -x .branch_lt %t | FileCheck --check-prefix=BRANCH-LE %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -S -r %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-readelf -x .branch_lt %t | FileCheck --check-prefix=BRANCH-BE %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# SEC: Name       Type     Address          Off     Size   ES Flg Lk Inf Al
# SEC: .got       PROGBITS 0000000002002028 2002028 000008 00  WA  0   0  8
# SEC: .branch_lt PROGBITS 0000000002002030 2002030 000018 00  WA  0   0  8

# SEC: There are no relocations in this file.

## high@localentry (high+8), .text_high+16 and .text_low+8
# BRANCH-LE:      0x02002030 08200002 00000000 10200002 00000000
# BRANCH-LE-NEXT: 0x02002040 08200000 00000000
# BRANCH-BE:      0x02002030 00000000 02002008 00000000 02002010
# BRANCH-BE-NEXT: 0x02002040 00000000 00002008

# CHECK:      _start:
# CHECK-NEXT:     2000:       bl .+24
# CHECK-NEXT:                 bl .+20
# CHECK-NEXT:                 bl .+16
# CHECK-NEXT:                 bl .+33554428

## &.branch_lt[0] - .TOC. = .branch_lt - (.got+0x8000) = -32760
# CHECK:      __long_branch_high:
# CHECK-NEXT:     2018:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32760(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

## &.branch_lt[1] - .TOC. = .branch_lt - (.got+0x8000) = -32752
# CHECK:      __long_branch_:
# CHECK-NEXT:     2028:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32752(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

.section .text_low, "ax", %progbits
.globl _start
_start:
bl high          # Need a thunk
bl high          # Need a thunk
bl high          # Need a thunk
bl high
bl .text_high+16 # Need a thunk
blr

# CHECK:      Disassembly of section .text_high:
# CHECK-EMPTY:
# CHECK-NEXT: high:
# CHECK-NEXT:  2002000:       addis 2, 12, 1
# CHECK-NEXT:                 addi 2, 2, -32728
# CHECK-NEXT:                 bl .-33554432
# CHECK-NEXT:                 bl .+12
# CHECK:      __long_branch_:
# CHECK-NEXT:  2002018:       addis 12, 2, 0
# CHECK-NEXT:                 ld 12, -32744(12)
# CHECK-NEXT:                 mtctr 12
# CHECK-NEXT:                 bctr

.section .text_high, "ax", %progbits
.globl high
high:
addis 2, 12, .TOC.-high@ha
addi 2, 2, .TOC.-high@l
.localentry high, 8
bl .text_low+8
bl .text_low+8 # Need a thunk
blr

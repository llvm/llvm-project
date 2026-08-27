# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0xa000 : { *(.text_high) } \
# RUN:       }' > %t.lds

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld -pie -T %t.lds %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=SEC-PI -DOFF0=a0b0 -DOFF1=a0b8 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## --pic-veneer selects PI long-branch thunks in a non-PIC link.
# RUN: ld.lld --pic-veneer -T %t.lds %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=SEC-PI -DOFF0=a010 -DOFF1=a018 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# SEC: There are no relocations in this file.

# SEC-PI:      Relocation section '.rela.dyn' {{.*}} contains 2 entries:
# SEC-PI:      {{0+}}[[OFF0]] {{[0-9a-f]+}} R_PPC64_RELATIVE a004
# SEC-PI-NEXT: {{0+}}[[OFF1]] {{[0-9a-f]+}} R_PPC64_RELATIVE a010

# CHECK-LABEL: <_start>:
# CHECK-NEXT:  2000: bt 2, 0x2040
# CHECK-NEXT:        bt+ 2, 0x2040
# CHECK-NEXT:        bf 2, 0xa004
# CHECK-NEXT:        bt 2, 0x2020
# CHECK-NEXT:        blr
# CHECK-NEXT:        trap
# CHECK-NEXT:        trap
# CHECK-NEXT:        trap
# CHECK-EMPTY:
# CHECK-NEXT: <__long_branch_>:
# CHECK-NEXT:  2020: addis 12, 2, 0
# CHECK-NEXT:        ld 12, {{.*}}(12)
# CHECK-NEXT:        mtctr 12
# CHECK-NEXT:        bctr
# CHECK-NEXT:        ...
# CHECK-EMPTY:

# CHECK-NEXT: <__long_branch_high>:
# CHECK-NEXT:  2040: addis 12, 2, 0
# CHECK-NEXT:        ld 12, {{.*}}(12)
# CHECK-NEXT:        mtctr 12
# CHECK-NEXT:        bctr

.section .text_low, "ax", @progbits
.globl _start
_start:
beq high           # Need a thunk
beq+ high          # Need a thunk
bne high
beq .text_high+16  # Need a thunk
blr

# CHECK-LABEL: <.text_high>:
# CHECK-NEXT:      a000: nop
# CHECK-EMPTY:
# CHECK-LABEL: <high>:
# CHECK-NEXT:      a004: bf 0, 0x2008
# CHECK-NEXT:            bt 1, 0x2008

.section .text_high, "ax", @progbits
nop
.globl high
high:
bge .text_low+8    # Need a thunk
bgt .text_low+8    # Need a thunk

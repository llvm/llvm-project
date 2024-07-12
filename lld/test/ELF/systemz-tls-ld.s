# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=LD-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=LD %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t.so | FileCheck --check-prefix=LD-DATA %s

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t | FileCheck --check-prefix=LE-DATA %s

# LD-REL: Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# LD-REL: 00000000000024f8 0000000000000036 R_390_TLS_DTPMOD 0

## _GLOBAL_OFFSET_TABLE is at 0x24e0
# LD:      larl    %r12, 0x24e0

## GOT offset of the LDM TLS module ID is at 0x23e0
# LD-NEXT: lgrl    %r2, 0x23e0
# LD-NEXT: brasl   %r14, 0x13c0
# LD-NEXT: la      %r2, 0(%r2,%r7)

## DTP offset for a is at 0x23e8
# LD-NEXT: lgrl    %r1, 0x23e8
# LD-NEXT: lgf     %r1, 0(%r1,%r2)

## DTP offset for b is at 0x23f0
# LD-NEXT: lgrl    %r1, 0x23f0
# LD-NEXT: lgf     %r1, 0(%r1,%r2)

## DTP offset for c is at 0x23f8
# LD-NEXT: lgrl    %r1, 0x23f8
# LD-NEXT: lgf     %r1, 0(%r1,%r2)

## Constant pool holding GOT offsets of TLS module ID and DTP offsets:
# TLS module ID: 0x24f8 / 0x18
# a: 8
# b: 12
# c: 16
# LD-DATA: 23e0 00000000 00000018 00000000 00000008
# LD-DATA: 23f0 00000000 0000000c 00000000 00000010

# NOREL: no relocations

## _GLOBAL_OFFSET_TABLE is at 0x1002230
# LE:      larl    %r12, 0x1002230

## GOT offset of the LDM TLS module ID is at 0x1002210
# LE-NEXT: lgrl    %r2, 0x1002210
# LE-NEXT: brcl    0,
# LE-NEXT: la      %r2, 0(%r2,%r7)

## TP offset for a is at 0x1002218
# LE-NEXT: lgrl    %r1, 0x1002218
# LE-NEXT: lgf     %r1, 0(%r1,%r2)

## TP offset for b is at 0x1002220
# LE-NEXT: lgrl    %r1, 0x1002220
# LE-NEXT: lgf     %r1, 0(%r1,%r2)

## TP offset for c is at 0x1002228
# LE-NEXT: lgrl    %r1, 0x1002228
# LE-NEXT: lgf     %r1, 0(%r1,%r2)

## zeroed LDM / TP offsets:
# LDM TLS: 0
# a: -8
# b: -4
# c: 0
# LE-DATA: 1002210 00000000 00000000 ffffffff fffffff8
# LE-DATA: 1002220 ffffffff fffffffc 00000000 00000000


ear     %r7,%a0
sllg    %r7,%r1,32
ear     %r7,%a1
larl    %r12,_GLOBAL_OFFSET_TABLE_

lgrl    %r2,.LC0
brasl   %r14,__tls_get_offset@PLT:tls_ldcall:a
la      %r2,0(%r2,%r7)

lgrl    %r1, .LC1
lgf     %r1,0(%r1,%r2)

lgrl    %r1, .LC2
lgf     %r1,0(%r1,%r2)

lgrl    %r1, .LC3
lgf     %r1,0(%r1,%r2)

        .section        .data.rel.ro,"aw"
        .align  8
.LC0:
        .quad   a@TLSLDM
.LC1:
        .quad   a@DTPOFF
.LC2:
        .quad   b@DTPOFF
.LC3:
        .quad   c@DTPOFF

	.section .tbss
	.globl a
	.globl b
	.globl c
	.zero 8
a:
	.zero 4
b:
	.zero 4
c:

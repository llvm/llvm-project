# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: echo '.tbss; .globl b, c; b: .zero 4; c:' | llvm-mc -filetype=obj -triple=s390x-unknown-linux - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=GD-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t.so | FileCheck --check-prefix=GD-DATA %s

# RUN: ld.lld %t.o %t1.o -o %t.le
# RUN: llvm-readelf -r %t.le | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.le | FileCheck --check-prefix=LE %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t.le | FileCheck --check-prefix=LE-DATA %s

# RUN: ld.lld %t.o %t1.so -o %t.ie
# RUN: llvm-readelf -r %t.ie | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.ie | FileCheck --check-prefix=IE %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t.ie | FileCheck --check-prefix=IE-DATA %s

# GD-REL: Relocation section '.rela.dyn' at offset {{.*}} contains 6 entries:
# GD-REL:      0000000000002570 0000000200000036 R_390_TLS_DTPMOD 0000000000000008 a + 0
# GD-REL-NEXT: 0000000000002578 0000000200000037 R_390_TLS_DTPOFF 0000000000000008 a + 0
# GD-REL-NEXT: 0000000000002580 0000000300000036 R_390_TLS_DTPMOD 000000000000000c b + 0
# GD-REL-NEXT: 0000000000002588 0000000300000037 R_390_TLS_DTPOFF 000000000000000c b + 0
# GD-REL-NEXT: 0000000000002590 0000000400000036 R_390_TLS_DTPMOD 0000000000000010 c + 0
# GD-REL-NEXT: 0000000000002598 0000000400000037 R_390_TLS_DTPOFF 0000000000000010 c + 0

## _GLOBAL_OFFSET_TABLE is at 0x2558
# GD:      larl    %r12, 0x2558

## GOT offset of the TLS module ID / offset pair for a is at 0x2460
# GD-NEXT: lgrl    %r2, 0x2460
# GD-NEXT: brasl   %r14, 0x1440
# GD-NEXT: lgf     %r2, 0(%r2,%r7)

## GOT offset of the TLS module ID / offset pair for b is at 0x2468
# GD-NEXT: lgrl    %r2, 0x2468
# GD-NEXT: brasl   %r14, 0x1440
# GD-NEXT: lgf     %r2, 0(%r2,%r7)

## GOT offset of the TLS module ID / offset pair for c is at 0x2470
# GD-NEXT: lgrl    %r2, 0x2470
# GD-NEXT: brasl   %r14, 0x1440
# GD-NEXT: lgf     %r2, 0(%r2,%r7)

## Constant pool holding GOT offsets of TLS module ID / offset pairs:
# a: 0x2570 / 0x18
# b: 0x2580 / 0x28
# c: 0x2590 / 0x38
# GD-DATA:      2460 00000000 00000018 00000000 00000028
# GD-DATA-NEXT: 2470 00000000 00000038

# NOREL: no relocations

## _GLOBAL_OFFSET_TABLE is at 0x1002230
# LE:      larl    %r12, 0x1002230

## TP offset of a is at 0x1002218
# LE-NEXT: lgrl    %r2, 0x1002218
# LE-NEXT: jgnop
# LE-NEXT: lgf     %r2, 0(%r2,%r7)

## TP offset of b is at 0x1002220
# LE-NEXT: lgrl    %r2, 0x1002220
# LE-NEXT: jgnop
# LE-NEXT: lgf     %r2, 0(%r2,%r7)

## TP offset of c is at 0x1002228
# LE-NEXT: lgrl    %r2, 0x1002228
# LE-NEXT: jgnop
# LE-NEXT: lgf     %r2, 0(%r2,%r7)

## TP offsets
# a: -8
# b: -4
# c: 0
# LE-DATA:      1002218 ffffffff fffffff8 ffffffff fffffffc
# LE-DATA-NEXT: 1002228 00000000 00000000


# IE-REL: Relocation section '.rela.dyn' at offset {{.*}} contains 2 entries:
# IE-REL:      0000000001002430 0000000200000038 R_390_TLS_TPOFF 0000000000000000 b + 0
# IE-REL-NEXT: 0000000001002438 0000000300000038 R_390_TLS_TPOFF 0000000000000000 c + 0

## _GLOBAL_OFFSET_TABLE is at 0x1002418
# IE:      larl    %r12, 0x1002418

## TP offset of a is at 0x1002340
# IE-NEXT: lgrl    %r2, 0x1002340
# IE-NEXT: jgnop
# IE-NEXT: lgf     %r2, 0(%r2,%r7)

## GOT offset of the TP offset for b is at 0x1002348
# IE-NEXT: lgrl    %r2, 0x1002348
# IE-NEXT: lg      %r2, 0(%r2,%r12)
# IE-NEXT: lgf     %r2, 0(%r2,%r7)

## GOT offset of the TP offset for c is at 0x1002350
# IE-NEXT: lgrl    %r2, 0x1002350
# IE-NEXT: lg      %r2, 0(%r2,%r12)
# IE-NEXT: lgf     %r2, 0(%r2,%r7)

## TP offsets (a) / GOT offset of TP offsets (b, c)
# a: -4
# b: 0x1002430 / 0x18
# c: 0x1002438 / 0x20
# IE-DATA:      1002340 ffffffff fffffffc 00000000 00000018
# IE-DATA-NEXT: 1002350 00000000 00000020


ear     %r7,%a0
sllg    %r7,%r1,32
ear     %r7,%a1
larl    %r12,_GLOBAL_OFFSET_TABLE_

lgrl    %r2,.LC0
brasl   %r14,__tls_get_offset@PLT:tls_gdcall:a
lgf     %r2,0(%r2,%r7)

lgrl    %r2,.LC1
brasl   %r14,__tls_get_offset@PLT:tls_gdcall:b
lgf     %r2,0(%r2,%r7)

lgrl    %r2,.LC2
brasl   %r14,__tls_get_offset@PLT:tls_gdcall:c
lgf     %r2,0(%r2,%r7)

        .section        .data.rel.ro,"aw"
        .align  8
.LC0:
        .quad   a@TLSGD
.LC1:
        .quad   b@TLSGD
.LC2:
        .quad   c@TLSGD

	.section .tbss
	.globl a
	.zero 8
a:
	.zero 4

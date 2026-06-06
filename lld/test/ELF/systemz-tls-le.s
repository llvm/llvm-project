# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s
# RUN: llvm-objdump --section .data.rel.ro --full-contents %t | FileCheck --check-prefix=LE-DATA %s

# NOREL: no relocations

## TP offset for a is at 0x1002200
# LE:      lgrl    %r1, 0x1002200
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for b is at 0x1002208
# LE-NEXT: lgrl    %r1, 0x1002208
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for c is at 0x1002210
# LE-NEXT: lgrl    %r1, 0x1002210
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offsets:
# a: -8
# b: -4
# c: 0
# LE-DATA: 1002200 ffffffff fffffff8 ffffffff fffffffc
# LE-DATA: 1002210 00000000 00000000

ear     %r7,%a0
sllg    %r7,%r1,32
ear     %r7,%a1

lgrl    %r1, .LC0
lgf     %r1,0(%r1,%r7)

lgrl    %r1, .LC1
lgf     %r1,0(%r1,%r7)

lgrl    %r1, .LC2
lgf     %r1,0(%r1,%r7)

        .section        .data.rel.ro,"aw"
        .align  8
.LC0:
        .quad   a@ntpoff
.LC1:
        .quad   b@ntpoff
.LC2:
        .quad   c@ntpoff

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

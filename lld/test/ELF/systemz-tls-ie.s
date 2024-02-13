# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o

# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=IE %s
# RUN: llvm-objdump --section .data --full-contents %t.so | FileCheck --check-prefix=IE-DATA %s

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s
# RUN: llvm-objdump --section .data --full-contents %t | FileCheck --check-prefix=LE-DATA %s
# RUN: llvm-objdump --section .got --full-contents %t | FileCheck --check-prefix=LE-GOT %s

# IE-REL: Relocation section '.rela.dyn' at offset {{.*}} contains 4 entries:
# IE-REL: 0000000000003478 000000000000000c R_390_RELATIVE 2460
# IE-REL: 0000000000002460 0000000100000038 R_390_TLS_TPOFF 0000000000000008 a + 0
# IE-REL: 0000000000002468 0000000200000038 R_390_TLS_TPOFF 000000000000000c b + 0
# IE-REL: 0000000000002470 0000000300000038 R_390_TLS_TPOFF 0000000000000010 c + 0

## TP offset for a is at 0x2460
# IE:      lgrl    %r1, 0x2460
# IE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for b is at 0x2468
# IE-NEXT: lgrl    %r1, 0x2468
# IE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for c is at 0x2470
# IE-NEXT: lgrl    %r1, 0x2470
# IE-NEXT: lgf     %r1, 0(%r1,%r7)

## Data element: TP offset for a is at 0x2460 (relocated via R_390_RELATIVE above)
# IE-DATA: 3478 00000000 00000000

# NOREL: no relocations

## TP offset for a is at 0x1002250
# LE:      lgrl    %r1, 0x1002250
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for b is at 0x1002258
# LE-NEXT: lgrl    %r1, 0x1002258
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## TP offset for c is at 0x1002260
# LE-NEXT: lgrl    %r1, 0x1002260
# LE-NEXT: lgf     %r1, 0(%r1,%r7)

## Data element: TP offset for a is at 0x1002250
# LE-DATA: 00000000 01002250

## TP offsets in GOT:
# a: -8
# b: -4
# c: 0
# LE-GOT: 1002238 00000000 00000000 00000000 00000000
# LE-GOT: 1002248 00000000 00000000 ffffffff fffffff8
# LE-GOT: 1002258 ffffffff fffffffc 00000000 00000000

ear     %r7,%a0
sllg    %r7,%r1,32
ear     %r7,%a1

lgrl    %r1, a@indntpoff
lgf     %r1,0(%r1,%r7)

lgrl    %r1, b@indntpoff
lgf     %r1,0(%r1,%r7)

lgrl    %r1, c@indntpoff
lgf     %r1,0(%r1,%r7)

	.data
	.reloc .,R_390_TLS_IE64,a
	.space 8

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

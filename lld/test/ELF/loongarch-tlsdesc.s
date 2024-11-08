# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so
# RUN: llvm-mc -filetype=obj -triple=loongarch32 --defsym ELF32=1 a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=loongarch32 --defsym ELF32=1 c.s -o c.32.o
# RUN: ld.lld -shared -soname=c.32.so c.32.o -o c.32.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -h -d a.64.so | FileCheck %s --check-prefix=GD64

# RUN: ld.lld -shared -z now a.64.o c.64.o -o rel.64.so -z rel
# RUN: llvm-readobj -r -x .got rel.64.so | FileCheck --check-prefix=GD64-REL %s

## FIXME: The transition frome TLSDESC to IE/LE has not yet been implemented.
## Keep the dynamic relocations and hand them over to dynamic linker.

# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readobj -r -x .got a.64.le | FileCheck --check-prefix=LE64-RELA %s

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s

## 32-bit code is mostly the same. We only test a few variants.

# RUN: ld.lld -shared -z now a.32.o c.32.o -o rel.32.so -z rel
# RUN: llvm-readobj -r -x .got rel.32.so | FileCheck --check-prefix=GD32-REL %s

# GD64-RELA:      .rela.dyn {
# GD64-RELA-NEXT:   0x20400 R_LARCH_TLS_DESC64 - 0x7FF
# GD64-RELA-NEXT:   0x203E0 R_LARCH_TLS_DESC64 a 0x0
# GD64-RELA-NEXT:   0x203F0 R_LARCH_TLS_DESC64 c 0x0
# GD64-RELA-NEXT: }
# GD64-RELA:      Hex dump of section '.got':
# GD64-RELA-NEXT: 0x000203e0 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x000203f0 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT: 0x00020400 00000000 00000000 00000000 00000000 .

# GD64-REL:      .rel.dyn {
# GD64-REL-NEXT:   0x203E8 R_LARCH_TLS_DESC64 -
# GD64-REL-NEXT:   0x203C8 R_LARCH_TLS_DESC64 a
# GD64-REL-NEXT:   0x203D8 R_LARCH_TLS_DESC64 c
# GD64-REL-NEXT: }
# GD64-REL:      Hex dump of section '.got':
# GD64-REL-NEXT: 0x000203c8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000203d8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000203e8 00000000 00000000 ff070000 00000000 .

# GD64:      .got     00000030 00000000000203e0

## &.got[a]-. = 0x203e0 - 0x102e0: 0x10 pages, page offset 0x3e0
# GD64:        102e0: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 992
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a1, $a0, $tp

## &.got[b]-. = 0x203e0+32 - 0x102f4: 0x10 pages, page offset 0x400
# GD64:        102f4: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1024
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a2, $a0, $tp

## &.got[c]-. = 0x23e0+16 - 0x10308: 0x10 pages, page offset 0x3f0
# GD64:        10308: pcalau12i $a0, 16
# GD64-NEXT:          addi.d $a0, $a0, 1008
# GD64-NEXT:          ld.d $ra, $a0, 0
# GD64-NEXT:          jirl $ra, $ra, 0
# GD64-NEXT:          add.d $a3, $a0, $tp

# LE64-RELA:      .rela.dyn {
# LE64-RELA-NEXT:   0x30250 R_LARCH_TLS_DESC64 - 0x8
# LE64-RELA-NEXT:   0x30260 R_LARCH_TLS_DESC64 - 0x800
# LE64-RELA-NEXT:   0x30270 R_LARCH_TLS_DESC64 - 0x7FF
# LE64-RELA-NEXT: }
# LE64-RELA:      Hex dump of section '.got':
# LE64-RELA-NEXT: 0x00030250 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030260 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030270 00000000 00000000 00000000 00000000 .

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x303D8 R_LARCH_TLS_DESC64 - 0x8
# IE64-RELA-NEXT:   0x303F8 R_LARCH_TLS_DESC64 - 0x7FF
# IE64-RELA-NEXT:   0x303E8 R_LARCH_TLS_DESC64 c 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x000303d8 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x000303e8 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x000303f8 00000000 00000000 00000000 00000000 .

# GD32-REL:      .rel.dyn {
# GD32-REL-NEXT:    0x20270 R_LARCH_TLS_DESC32 -
# GD32-REL-NEXT:    0x20260 R_LARCH_TLS_DESC32 a
# GD32-REL-NEXT:    0x20268 R_LARCH_TLS_DESC32 c
# GD32-REL-NEXT: }
# GD32-REL:      Hex dump of section '.got':
# GD32-REL-NEXT: 0x00020260 00000000 00000000 00000000 00000000 .
# GD32-REL-NEXT: 0x00020270 00000000 ff070000                   .

#--- a.s
.macro add dst, src1, src2
.ifdef ELF32
add.w \dst, \src1, \src2
.else
add.d \dst, \src1, \src2
.endif
.endm

la.tls.desc $a0, a
add $a1, $a0, $tp

la.tls.desc $a0, b
add $a2, $a0, $tp

la.tls.desc $a0, c
add $a3, $a0, $tp

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 2039  ## Place b at 0x7ff
b:
.zero 1

#--- c.s
.section .tbss,"awT",@nobits
.globl c
c: .zero 4

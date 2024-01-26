# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so
# RUN: llvm-mc -filetype=obj -triple=riscv32 --defsym ELF32=1 a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv32 --defsym ELF32=1 c.s -o c.32.o
# RUN: ld.lld -shared -soname=c.32.so c.32.o -o c.32.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.so | FileCheck %s --check-prefix=GD64

# RUN: ld.lld -shared -z now a.64.o c.64.o -o rel.64.so -z rel
# RUN: llvm-readobj -r -x .got rel.64.so | FileCheck --check-prefix=GD64-REL %s

# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readelf -r a.64.le | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.le | FileCheck %s --check-prefix=LE64

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r a.64.ie | FileCheck --check-prefix=IE64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.64.ie | FileCheck %s --check-prefix=IE64

## 32-bit code is mostly the same. We only test a few variants. The IE optimization uses the LW instruction.

# RUN: ld.lld -shared -z now a.32.o c.32.o -o rel.32.so -z rel
# RUN: llvm-readobj -r -x .got rel.32.so | FileCheck --check-prefix=GD32-REL %s
# RUN: ld.lld -e 0 -z now a.32.o c.32.so -o a.32.ie
# RUN: llvm-objdump --no-show-raw-insn -M no-aliases -h -d a.32.ie | FileCheck %s --check-prefix=IE32

# GD64-RELA:      .rela.dyn {
# GD64-RELA-NEXT:   0x2408 R_RISCV_TLSDESC - 0x7FF
# GD64-RELA-NEXT:   0x23E8 R_RISCV_TLSDESC a 0x0
# GD64-RELA-NEXT:   0x23F8 R_RISCV_TLSDESC c 0x0
# GD64-RELA-NEXT: }
# GD64-RELA:      Hex dump of section '.got':
# GD64-RELA-NEXT: 0x000023e0 20230000 00000000 00000000 00000000 #
# GD64-RELA-NEXT: 0x000023f0 00000000 00000000 00000000 00000000 .

# GD64-REL:      .rel.dyn {
# GD64-REL-NEXT:   0x23F0 R_RISCV_TLSDESC -
# GD64-REL-NEXT:   0x23D0 R_RISCV_TLSDESC a
# GD64-REL-NEXT:   0x23E0 R_RISCV_TLSDESC c
# GD64-REL-NEXT: }
# GD64-REL:      Hex dump of section '.got':
# GD64-REL-NEXT: 0x000023c8 08230000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000023d8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000023e8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000023f8 ff070000 00000000                   .

# GD32-REL:      .rel.dyn {
# GD32-REL-NEXT:   0x2274 R_RISCV_TLSDESC -
# GD32-REL-NEXT:   0x2264 R_RISCV_TLSDESC a
# GD32-REL-NEXT:   0x226C R_RISCV_TLSDESC c
# GD32-REL-NEXT: }
# GD32-REL:      Hex dump of section '.got':
# GD32-REL-NEXT: 0x00002260 00220000 00000000 00000000 00000000 .
# GD32-REL-NEXT: 0x00002270 00000000 00000000 ff070000          .

# GD64:      .got     00000038 00000000000023e0

## &.got[a]-. = 0x23e0+8 - 0x12e0 = 0x1108
# GD64:        12e0: auipc   a0, 0x1
# GD64-NEXT:         ld      a1, 0x108(a0)
# GD64-NEXT:         addi    a0, a0, 0x108
# GD64-NEXT:         jalr    t0, 0x0(a1)
# GD64-NEXT:         add     a0, a0, tp

## &.got[b]-. = 0x23e0+40 - 0x12f4 = 0x1114
# GD64-NEXT:   12f4: auipc   a2, 0x1
# GD64-NEXT:         ld      a3, 0x114(a2)
# GD64-NEXT:         addi    a0, a2, 0x114
# GD64-NEXT:         jalr    t0, 0x0(a3)
# GD64-NEXT:         add     a0, a0, tp

## &.got[c]-. = 0x23e0+24 - 0x1308 = 0x10f0
# GD64-NEXT:   1308: auipc   a4, 0x1
# GD64-NEXT:         ld      a5, 0xf0(a4)
# GD64-NEXT:         addi    a0, a4, 0xf0
# GD64-NEXT:         jalr    t0, 0x0(a5)
# GD64-NEXT:         add     a0, a0, tp

# NOREL: no relocations

# LE64-LABEL: <.text>:
## st_value(a) = 8
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    a0, zero, 0x8
# LE64-NEXT:         add     a0, a0, tp
## st_value(b) = 2047
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    a0, zero, 0x7ff
# LE64-NEXT:         add     a0, a0, tp
## st_value(c) = 2048
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         addi    zero, zero, 0x0
# LE64-NEXT:         lui     a0, 0x1
# LE64-NEXT:         addi    a0, a0, -0x800
# LE64-NEXT:         add     a0, a0, tp

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x123B0 R_RISCV_TLS_TPREL64 c 0x0
# IE64-RELA-NEXT: }

# IE64:       .got     00000010 00000000000123a8

## a and b are optimized to use LE. c is optimized to IE.
# IE64-LABEL: <.text>:
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    a0, zero, 0x8
# IE64-NEXT:         add     a0, a0, tp
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    a0, zero, 0x7ff
# IE64-NEXT:         add     a0, a0, tp
## &.got[c]-. = 0x123a8+8 - 0x112b8 = 0x10f8
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:         addi    zero, zero, 0x0
# IE64-NEXT:  112b8: auipc   a0, 0x1
# IE64-NEXT:         ld      a0, 0xf8(a0)
# IE64-NEXT:         add     a0, a0, tp

# IE32:       .got     00000008 00012248

# IE32-LABEL: <.text>:
## st_value(a) = 8
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    a0, zero, 0x8
# IE32-NEXT:         add     a0, a0, tp
## st_value(b) = 2047
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    a0, zero, 0x7ff
# IE32-NEXT:         add     a0, a0, tp
## &.got[c]-. = 0x12248+4 - 0x111cc = 0x1080
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:         addi    zero, zero, 0x0
# IE32-NEXT:  111cc: auipc   a0, 0x1
# IE32-NEXT:         lw      a0, 0x80(a0)
# IE32-NEXT:         add     a0, a0, tp

#--- a.s
.macro load dst, src
.ifdef ELF32
lw \dst, \src
.else
ld \dst, \src
.endif
.endm

.Ltlsdesc_hi0:
  auipc a0, %tlsdesc_hi(a)
  load  a1, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a0)
  addi  a0, a0, %tlsdesc_add_lo(.Ltlsdesc_hi0)
  jalr  t0, 0(a1), %tlsdesc_call(.Ltlsdesc_hi0)
  add   a0, a0, tp

.Ltlsdesc_hi1:
  auipc a2, %tlsdesc_hi(b)
  load  a3, %tlsdesc_load_lo(.Ltlsdesc_hi1)(a2)
  addi  a0, a2, %tlsdesc_add_lo(.Ltlsdesc_hi1)
  jalr  t0, 0(a3), %tlsdesc_call(.Ltlsdesc_hi1)
  add   a0, a0, tp

.Ltlsdesc_hi2:
  auipc a4, %tlsdesc_hi(c)
  load  a5, %tlsdesc_load_lo(.Ltlsdesc_hi2)(a4)
  addi  a0, a4, %tlsdesc_add_lo(.Ltlsdesc_hi2)
  jalr  t0, 0(a5), %tlsdesc_call(.Ltlsdesc_hi2)
  add   a0, a0, tp

.section .tbss
.globl a
.zero 8
a:
.zero 2039  ## Place b at 0x7ff
b:
.zero 1

#--- c.s
.tbss
.globl c
c: .zero 4

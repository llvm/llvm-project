# REQUIRES: loongarch
## Test for `TLS DESC` with extreme code model.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 c.s -o c.64.o
# RUN: ld.lld -shared -soname=c.64.so c.64.o -o c.64.so

# RUN: ld.lld -shared -z now a.64.o c.64.o -o a.64.so
# RUN: llvm-readobj -r -x .got a.64.so | FileCheck --check-prefix=GD64-RELA %s
# RUN: llvm-objdump --no-show-raw-insn -h -d a.64.so | FileCheck %s --check-prefix=GD64

# RUN: ld.lld -shared -z now a.64.o c.64.o -o rel.64.so -z rel
# RUN: llvm-readobj -r -x .got rel.64.so | FileCheck --check-prefix=GD64-REL %s

## Cannot convert TLSDESC to IE/LE with the extreme code model.
## Only check the GOT slot.
# RUN: ld.lld -e 0 -z now a.64.o c.64.o -o a.64.le
# RUN: llvm-readobj -r -x .got a.64.le 2>&1 | FileCheck --check-prefix=LE64-RELA %s

# RUN: ld.lld -e 0 -z now a.64.o c.64.so -o a.64.ie
# RUN: llvm-readobj -r -x .got a.64.ie | FileCheck --check-prefix=IE64-RELA %s

# GD64-RELA:       .rela.dyn {
# GD64-RELA-NEXT:    0x20420 R_LARCH_TLS_DESC64 - 0x7FF
# GD64-RELA-NEXT:    0x20400 R_LARCH_TLS_DESC64 a 0x0
# GD64-RELA-NEXT:    0x20410 R_LARCH_TLS_DESC64 c 0x0
# GD64-RELA-NEXT:  }

# GD64-RELA:       Hex dump of section '.got':
# GD64-RELA-NEXT:  0x00020400 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT:  0x00020410 00000000 00000000 00000000 00000000 .
# GD64-RELA-NEXT:  0x00020420 00000000 00000000 00000000 00000000 .

# GD64-REL:      .rel.dyn {
# GD64-REL-NEXT:   0x20408 R_LARCH_TLS_DESC64 -
# GD64-REL-NEXT:   0x203E8 R_LARCH_TLS_DESC64 a
# GD64-REL-NEXT:   0x203F8 R_LARCH_TLS_DESC64 c
# GD64-REL-NEXT: }
# GD64-REL:      Hex dump of section '.got':
# GD64-REL-NEXT: 0x000203e8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x000203f8 00000000 00000000 00000000 00000000 .
# GD64-REL-NEXT: 0x00020408 00000000 00000000 ff070000 00000000 .

# GD64:      .got           00000030 0000000000020400
## &.got[a]-. = 0x20400 - 0x102e0: 0x10 pages, page offset 0x400
# GD64:        102e0: pcalau12i	$a0, 16
# GD64-NEXT:          addi.d	$a1, $zero, 1024
# GD64-NEXT:          lu32i.d	$a1, 0
# GD64-NEXT:          lu52i.d	$a1, $a1, 0
# GD64-NEXT:          add.d	$a0, $a0, $a1
# GD64-NEXT:          ld.d	$ra, $a0, 0
# GD64-NEXT:          jirl	$ra, $ra, 0
# GD64-NEXT:          add.d	$a1, $a0, $tp
## &.got[b]-. = 0x20400+32 - 0x10300: 0x10 pages, page offset 0x420
# GD64-NEXT:   10300: pcalau12i	$a0, 16
# GD64-NEXT:          addi.d	$a2, $zero, 1056
# GD64-NEXT:          lu32i.d	$a2, 0
# GD64-NEXT:          lu52i.d	$a2, $a2, 0
# GD64-NEXT:          add.d	$a0, $a0, $a2
# GD64-NEXT:          ld.d	$ra, $a0, 0
# GD64-NEXT:          jirl	$ra, $ra, 0
# GD64-NEXT:          add.d	$a2, $a1, $tp
## &.got[b]-. = 0x20400+16 - 0x10320: 0x10 pages, page offset 0x410
# GD64-NEXT:   10320: pcalau12i	$a0, 16
# GD64-NEXT:          addi.d	$a3, $zero, 1040
# GD64-NEXT:          lu32i.d	$a3, 0
# GD64-NEXT:          lu52i.d	$a3, $a3, 0
# GD64-NEXT:          add.d	$a0, $a0, $a3
# GD64-NEXT:          ld.d	$ra, $a0, 0
# GD64-NEXT:          jirl	$ra, $ra, 0
# GD64-NEXT:          add.d	$a3, $a2, $tp

# LE64-RELA:      .rela.dyn {
# LE64-RELA-NEXT:   0x30270 R_LARCH_TLS_DESC64 - 0x8
# LE64-RELA-NEXT:   0x30280 R_LARCH_TLS_DESC64 - 0x800
# LE64-RELA-NEXT:   0x30290 R_LARCH_TLS_DESC64 - 0x7FF
# LE64-RELA-NEXT: }
# LE64-RELA:      Hex dump of section '.got':
# LE64-RELA-NEXT: 0x00030270 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030280 00000000 00000000 00000000 00000000 .
# LE64-RELA-NEXT: 0x00030290 00000000 00000000 00000000 00000000 .

# IE64-RELA:      .rela.dyn {
# IE64-RELA-NEXT:   0x30410 R_LARCH_TLS_DESC64 - 0x8
# IE64-RELA-NEXT:   0x30438 R_LARCH_TLS_DESC64 - 0x7FF
# IE64-RELA-NEXT:   0x30420 R_LARCH_TLS_DESC64 c 0x0
# IE64-RELA-NEXT:   0x30430 R_LARCH_TLS_TPREL64 c 0x0
# IE64-RELA-NEXT: }
# IE64-RELA:      Hex dump of section '.got':
# IE64-RELA-NEXT: 0x00030410 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030420 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030430 00000000 00000000 00000000 00000000 .
# IE64-RELA-NEXT: 0x00030440 00000000 00000000                  .

#--- a.s
## The extreme TLSDESC form: pcalau12i/addi.d/lu32i.d/lu52i.d build the address,
## then ld.d + jirl call the resolvers, with R_LARCH_TLS_DESC_PC_HI20/LO12,
## R_LARCH_TLS_DESC64_PC_LO20/HI12, R_LARCH_TLS_DESC_LD and CALL.
la.tls.desc $a0, $a1, a
add.d $a1, $a0, $tp

la.tls.desc $a0, $a2, b
add.d $a2, $a1, $tp

la.tls.desc $a0, $a3, c
add.d $a3, $a2, $tp

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

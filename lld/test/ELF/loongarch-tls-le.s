# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --defsym ELF32=1 %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.64.o

# RUN: ld.lld %t.32.o -o %t.32
# RUN: llvm-nm -p %t.32 | FileCheck --check-prefixes=NM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=LE,LE32 %s

# RUN: ld.lld %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=LE,LE64 %s

# RUN: not ld.lld -shared %t.32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_LARCH_TLS_LE_HI20 against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12 against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_HI20 against a cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12 against a cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_HI20_R against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12_R against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_HI20_R against a cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12_R against a cannot be used with -shared

# NM: {{0*}}00000008 b .LANCHOR0
# NM: {{0*}}00000800 B a

## .LANCHOR0@tprel = 8
## a@tprel = 0x800
# LE:      lu12i.w $a0, 0
# LE-NEXT: ori $a0, $a0, 8
# LE-NEXT: lu12i.w $a1, 0
# LE-NEXT: ori $a1, $a1, 2048

# LE32:      add.w   $a0, $a0, $tp
# LE32-NEXT: addi.w  $a0, $a0, 8
# LE32-NEXT: lu12i.w $a0, 1
# LE32-NEXT: add.w   $a0, $a0, $tp
# LE32-NEXT: addi.w  $a0, $a0, -2048

# LE64:      add.d   $a0, $a0, $tp
# LE64-NEXT: addi.d  $a0, $a0, 8
# LE64-NEXT: lu12i.w $a0, 1
# LE64-NEXT: add.d   $a0, $a0, $tp
# LE64-NEXT: addi.d  $a0, $a0, -2048

# LE-EMPTY:

.macro add dst, src1, src2, src3
.ifdef ELF32
add.w \dst, \src1, \src2, \src3
.else
add.d \dst, \src1, \src2, \src3
.endif
.endm
.macro addi dst, src1, src2
.ifdef ELF32
addi.w \dst, \src1, \src2
.else
addi.d \dst, \src1, \src2
.endif
.endm

.text

_start:
la.tls.le $a0, .LANCHOR0
la.tls.le $a1, a

lu12i.w $a0, %le_hi20_r(.LANCHOR0)
add $a0, $a0, $tp, %le_add_r(.LANCHOR0)
addi $a0, $a0, %le_lo12_r(.LANCHOR0)

lu12i.w $a0, %le_hi20_r(a)
add $a0, $a0, $tp, %le_add_r(a)
addi $a0, $a0, %le_lo12_r(a)

.section .tbss,"awT",@nobits
.space 8
.LANCHOR0:
.space 0x800-8
.globl a
a:
.zero 4

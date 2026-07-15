# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+relax --defsym ELF32=1 %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t.64.o

# RUN: ld.lld %t.32.o -o %t.32
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=RELAX32 %s

# RUN: ld.lld %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=RELAX64 %s

# RELAX32-LABEL: <_start>:
## .LANCHOR0@tprel = 8
# RELAX32-NEXT:    addi.w  $a0, $tp, 8 
# RELAX32-NEXT:    ld.w    $a1, $a0, 0
# RELAX32-NEXT:    ld.w    $a2, $tp, 8
## .a@tprel - 4 = 0x7fc
# RELAX32-NEXT:    addi.w  $a1, $zero, 1
# RELAX32-NEXT:    addi.w $a1, $a1, 2
# RELAX32-NEXT:    st.w   $a1, $tp, 2044
## .a@tprel = 0x800
# RELAX32-NEXT:    lu12i.w $a0, 1
# RELAX32-NEXT:    add.w   $a0, $a0, $tp
# RELAX32-NEXT:    addi.w  $a0, $a0, -2048

# RELAX64-LABEL: <_start>:
## .LANCHOR0@tprel = 8
# RELAX64-NEXT:    addi.d  $a0, $tp, 8 
# RELAX64-NEXT:    ld.d    $a1, $a0, 0
# RELAX64-NEXT:    ld.d    $a2, $tp, 8
## .a@tprel - 4 = 0x7fc
# RELAX64-NEXT:    addi.d  $a1, $zero, 1
# RELAX64-NEXT:    addi.d $a1, $a1, 2
# RELAX64-NEXT:    st.d   $a1, $tp, 2044
## .a@tprel = 0x800
# RELAX64-NEXT:    lu12i.w $a0, 1
# RELAX64-NEXT:    add.d   $a0, $a0, $tp
# RELAX64-NEXT:    addi.d  $a0, $a0, -2048

.macro add dst, src1, src2, src3
.ifdef ELF32
  add.w \dst, \src1, \src2, \src3
.else
  add.d \dst, \src1, \src2, \src3
.endif
.endm
.macro inst op dst, src1, src2
.ifdef ELF32
  .ifc      \op, addi
    addi.w  \dst, \src1, \src2
  .else;    .ifc   \op, ld
    ld.w    \dst, \src1, \src2
  .else;    .ifc   \op, st
    st.w    \dst, \src1, \src2
  .else;    .ifc   \op, ldptr
    ldptr.w \dst, \src1, \src2
  .else
    .error "Unknown op in ELF32 mode"
  .endif; .endif; .endif; .endif
.else
  .ifc      \op, addi
    addi.d  \dst, \src1, \src2
  .else;    .ifc   \op, ld
    ld.d    \dst, \src1, \src2
  .else;    .ifc   \op, st
    st.d    \dst, \src1, \src2
  .else;    .ifc   \op, ldptr
    ldptr.d \dst, \src1, \src2
  .else
    .error "Unknown op in ELF64 mode"
  .endif; .endif; .endif; .endif
.endif
.endm

.macro addi dst, src1, src2
  inst addi \dst, \src1, \src2
.endm
.macro ld dst, src1, src2
  inst ld \dst, \src1, \src2
.endm
.macro st dst, src1, src2
  inst st \dst, \src1, \src2
.endm
.macro ldptr dst, src1, src2
  inst ldptr \dst, \src1, \src2
.endm

_start:
  ## Test instructions not in pairs.
  lu12i.w $a0, %le_hi20_r(.LANCHOR0)
  add $a0, $a0, $tp, %le_add_r(.LANCHOR0)
  addi $a0, $a0, %le_lo12_r(.LANCHOR0)
  ld $a1, $a0, 0
  ld $a2, $a0, %le_lo12_r(.LANCHOR0)

  ## hi20(a-4) = hi20(0x7fc) = 0. relaxable
  ## Test non-adjacent instructions.
  lu12i.w $a0, %le_hi20_r(a-4)
  addi $a1, $zero, 0x1
  add $a0, $a0, $tp, %le_add_r(a-4)
  addi $a1, $a1, 0x2
  st $a1, $a0, %le_lo12_r(a-4)

  ## hi20(a) = hi20(0x800) = 1. not relaxable
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

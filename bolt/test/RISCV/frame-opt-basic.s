# REQUIRES: system-linux

# RUN: %clang %cflags64 -march=rv64gc %s -o %t.exe
# RUN: link_fdata --no-lbr --nmtool llvm-nm %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:   -frame-opt=all -simplify-conditional-tail-calls=false \
# RUN:   -eliminate-unreachable=false | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.out | \
# RUN:   FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: llvm-readelf --unwind %t.out | FileCheck --check-prefix=CHECK-CFI %s

  .text
  .reloc 0, R_RISCV_NONE
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
# FDATA: 1 _start #_start# 100
  addi sp, sp, -16
  .cfi_def_cfa sp, 16
  sd s1, 8(sp)
  .cfi_offset s1, -8
.Lbranch:
  beqz a0, .Lcold
.Lhot:
  addi s1, s1, 1
  addi a0, s1, 0
.Lexit:
  ld s1, 8(sp)
  .cfi_restore s1
  addi sp, sp, 16
  .cfi_def_cfa sp, 0
  ret
.Lcold:
  addi a0, a0, 2
  j .Lexit
  .cfi_endproc
  .size _start, .-_start

# CHECK: BOLT-INFO: FOP optimized
# CHECK: BOLT-INFO: FRAME ANALYSIS: 0 function(s) {{.*}} could not have its frame indices restored.
# CHECK: BOLT-INFO: Shrink wrapping moved 1 spills inserting load/stores
# CHECK-OBJDUMP: <_start>:
# CHECK-OBJDUMP: beqz
# CHECK-OBJDUMP-NEXT: sd s1, 0x8(sp)
# CHECK-CFI: DW_CFA_def_cfa: reg2 +0
# CHECK-CFI: DW_CFA_offset: reg9 -8

## Keep cold stubs after the hot path, relaxing compressed conditional branches
## as needed, and preserve the source's unwind state. Beyond the 4 KiB branch
## range, keep the stub next to the source.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --emit-relocs -e small %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:   --split-functions --split-all-cold --pad-funcs-before=small:2097152,large:2097152
# RUN: llvm-objdump -d --disassemble-symbols=small,large %t.bolt > %t.check
# RUN: llvm-dwarfdump --debug-frame %t.bolt >> %t.check
# RUN: FileCheck %s < %t.check

# CHECK-LABEL: <small>:
# CHECK: beqz a0, 0x[[#%x,STUB:]]
# CHECK-NEXT: li t1, 0x1
# CHECK: ret
# CHECK-NEXT: [[#STUB]]: {{.*}}auipc t1,
# CHECK-LABEL: <large>:
# CHECK-NEXT: {{.*}}bnez a0,
# CHECK-NEXT: {{.*}}auipc t1,
# CHECK: 0x[[#STUB]]: CFA=X2+16

  .text
  .globl small
  .type small, @function
small:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 small 0 0 100
  addi sp, sp, -16
  .cfi_def_cfa_offset 16
small_branch:
  bnez a0, small_hot
# FDATA: 1 small #small_branch# 1 small #small_cold# 0 0
# FDATA: 1 small #small_branch# 1 small #small_hot# 0 100
small_cold:
  li t1, 0
  addi sp, sp, 16
  .cfi_def_cfa_offset 0
  ret
small_hot:
  .cfi_def_cfa_offset 16
  li t1, 1
  .rept 128
  addi a1, a1, 1
  .endr
  addi sp, sp, 16
  .cfi_def_cfa_offset 0
  ret
  .cfi_endproc
  .size small, .-small

  .globl large
  .type large, @function
large:
# FDATA: 0 [unknown] 0 1 large 0 0 100
  bnez a0, large_hot
# FDATA: 1 large 0 1 large #large_cold# 0 0
# FDATA: 1 large 0 1 large #large_hot# 0 100
large_cold:
  li t1, 0
  ret
large_hot:
  li t1, 1
  .rept 2200
  addi a1, a1, 1
  .endr
  ret
  .size large, .-large
  .reloc 0, R_RISCV_NONE

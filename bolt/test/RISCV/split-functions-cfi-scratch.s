## A register can be dead at the branch target but still be needed to unwind
## from the trampoline. Keep the CFA register intact while crossing fragments.

# RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions \
# RUN:   --split-strategy=random2 --bolt-seed=1 --pad-funcs-before=_start:2097152
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK-NOT: auipc t0,
# CHECK: auipc t1,
# CHECK-NOT: auipc t0,

  .text
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  mv t0, sp
  .cfi_def_cfa t0, 0
  beqz a0, .Lcold
  .cfi_def_cfa sp, 0
  li t0, 0
  li t1, 0
  ret
.Lcold:
  .cfi_def_cfa sp, 0
  li t0, 0
  li t1, 0
  ret
  .cfi_endproc
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE

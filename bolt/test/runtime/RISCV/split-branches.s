## Reduced from clang's hot/cold JAL overflow. Both directions must preserve
## live t0; a function with no dead temporary must remain contiguous.
# REQUIRES: system-linux
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:   --reorder-functions=cdsort --split-functions --split-all-cold \
# RUN:   --pad-funcs-before=foo:2097152 | FileCheck %s
# RUN: %t.bolt
# RUN: llvm-objdump -d --disassemble-symbols=foo,foo.cold.0 %t.bolt | FileCheck %s --check-prefix=CODE
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=SYMS
# RUN: llvm-bolt %t.exe -o %t.all.bolt --split-functions --split-strategy=all
# RUN: %t.all.bolt
# RUN: llvm-bolt %t.exe -o %t.random.bolt --split-functions \
# RUN:   --split-strategy=randomN --bolt-seed=1
# RUN: %t.random.bolt
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj %s -o %t.c.o
# RUN: link_fdata %s %t.c.o %t.c.fdata
# RUN: llvm-strip --strip-unneeded %t.c.o
# RUN: ld.lld --emit-relocs %t.c.o -o %t.c.exe
# RUN: llvm-bolt %t.c.exe -o %t.c.bolt --data=%t.c.fdata --reorder-blocks=ext-tsp \
# RUN:   --reorder-functions=cdsort --split-functions --split-all-cold \
# RUN:   --pad-funcs-before=foo:2097152 | FileCheck %s
# RUN: %t.c.bolt
# RUN: llvm-mc -triple=riscv64 -filetype=obj --defsym ARG_SCRATCH=1 %s -o %t.arg.o
# RUN: llvm-strip --strip-unneeded %t.arg.o
# RUN: ld.lld --emit-relocs %t.arg.o -o %t.arg.exe
# RUN: llvm-bolt %t.arg.exe -o %t.arg.bolt --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:   --reorder-functions=cdsort --split-functions --split-all-cold \
# RUN:   --pad-funcs-before=foo:2097152 | FileCheck %s
# RUN: %t.arg.bolt
# RUN: llvm-objdump -d --disassemble-symbols=foo,foo.cold.0 %t.arg.bolt | FileCheck %s --check-prefix=ARG

# CHECK: RISC-V relaxed branches in
# CODE: <foo>:
# CODE: auipc t1,
# CODE: jr
# CODE: <foo.cold.0>:
# CODE: auipc t1,
# CODE: jr
# SYMS: foo.cold.0
# SYMS-NOT: unsplittable.cold
# ARG: <foo>:
# ARG: auipc a0,
# ARG: <foo.cold.0>:
# ARG: auipc a0,

  .text
  .globl _start
  .type _start,@function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  addi sp, sp, -16
  li t0, 42
  li a0, 0
  call foo
  bnez a0, fail
  li a0, 1
  call foo
  bnez a0, fail
  li a0, 0
  call unsplittable
  bnez a0, fail
  li a0, 1
  call unsplittable
fail:
  li a7, 93
  ecall
  .size _start, .-_start

  .globl foo
  .type foo,@function
foo:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 foo 0 0 100
branch:
  beqz a0, cold
# FDATA: 1 foo #branch# 1 foo #cold# 0 0
# FDATA: 1 foo #branch# 1 foo #hot# 0 100
hot:
  .ifdef ARG_SCRATCH
  li a1, 9
  .else
  li t1, 9
  .endif
  addi a0, t0, -42
  ret
cold:
  .ifdef ARG_SCRATCH
  li a1, 7
  .else
  li t1, 7
  .endif
  j hot
  .cfi_endproc
  .size foo, .-foo

  .globl unsplittable
  .type unsplittable,@function
unsplittable:
# FDATA: 0 [unknown] 0 1 unsplittable 0 0 100
keep_branch:
  beqz a0, keep_cold
# FDATA: 1 unsplittable #keep_branch# 1 unsplittable #keep_cold# 0 0
# FDATA: 1 unsplittable #keep_branch# 1 unsplittable #keep_hot# 0 100
keep_hot:
  .irp reg, t0, t1, t2, t3, t4, t5, t6, a0, a1, a2, a3, a4, a5, a6, a7
  sw \reg, 0(sp)
  .endr
  addi a0, t0, -42
  ret
keep_cold:
  .irp reg, t0, t1, t2, t3, t4, t5, t6, a0, a1, a2, a3, a4, a5, a6, a7
  sw \reg, 0(sp)
  .endr
  addi a0, t0, -42
  ret
  .size unsplittable, .-unsplittable

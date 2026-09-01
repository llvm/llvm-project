# REQUIRES: system-linux, x86-registered-target

## Verify that x86 ICP materializes the promoted target address with a
## PC-relative LEA instruction, automatically for PIE inputs.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/main.s -o %t.o
# RUN: link_fdata %t/main.s %t.o %t/fdata

## PIE inputs enable x86 ICP PIC mode automatically.
# RUN: ld.lld -pie --entry=reg_call --emit-relocs %t.o -o %t.pie
# RUN: llvm-bolt %t.pie -o %t.pie.out --relocs --data=%t/fdata --icp=calls \
# RUN:   --icp-calls-topn=1 --lite=0 --use-gnu-stack \
# RUN:   --custom-allocation-vma=0x80000000
# RUN: llvm-objdump -d --no-show-raw-insn %t.pie.out | \
# RUN:   FileCheck %s --check-prefix=PIE-LEA

# PIE-LEA-LABEL: <reg_call_site>:
# PIE-LEA:       leaq {{.*}}(%rip), %r11
# PIE-LEA-SAME:  <target>

#--- main.s
.text
.globl reg_call
.type reg_call,@function
reg_call:
reg_call_site:
  callq *%rax
# FDATA: 1 reg_call #reg_call_site# 1 target 0 0 100
  retq
.size reg_call, .-reg_call

.globl dummy
.type dummy,@function
dummy:
  callq target
  retq
.size dummy, .-dummy

.globl target
.type target,@function
target:
  retq
.size target, .-target

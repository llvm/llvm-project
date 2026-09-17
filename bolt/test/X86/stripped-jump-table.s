## A stripped input has no symbol evidence for fragment relationships. Accept
## entries that resolve to valid instructions in discovered functions and force
## movable jump tables.
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -e _start %t.o -o %t.exe
# RUN: llvm-strip --strip-all %t.exe -o %t.stripped
# RUN: llvm-bolt %t.stripped -o %t.bolt --allow-stripped \
# RUN:   --recover-relocations --print-normalized 2>&1 | FileCheck %s

# CHECK: BOLT-INFO: forcing --jump-tables=move for relocation recovery
# CHECK: jmpq {{.*}} # JUMPTABLE

.text
.globl _start
.type _start, @function
_start:
.cfi_startproc
  callq dispatch
  retq
.cfi_endproc
.size _start, .-_start

.globl dispatch
.type dispatch, @function
dispatch:
.cfi_startproc
  andl $1, %edi
  leaq .LJTI0(%rip), %rax
  movslq (%rax,%rdi,4), %rcx
  addq %rax, %rcx
  jmpq *%rcx
.Llocal:
  retq
.cfi_endproc
.size dispatch, .-dispatch

.globl other
.type other, @function
other:
.cfi_startproc
  retq
.cfi_endproc
.size other, .-other

.section .rodata,"a",@progbits
.p2align 2
.LJTI0:
  .long .Llocal-.LJTI0
  .long other-.LJTI0

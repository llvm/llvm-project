## Check that BOLT in lite mode processes fragments as expected.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/main.s -o %t.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/baz.s -o %t.baz.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/baz2.s -o %t.baz2.o
# RUN: link_fdata %s %t.o %t.main.fdata
# RUN: link_fdata %s %t.baz.o %t.baz.fdata
# RUN: link_fdata %s %t.baz2.o %t.baz2.fdata
# RUN: merge-fdata %t.main.fdata %t.baz.fdata %t.baz2.fdata > %t.fdata
# RUN: %clang %cflags %t.o %t.baz.o %t.baz2.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --lite=1 --data %t.fdata -v=1 -print-cfg \
# RUN:   2>&1 | FileCheck %s

# CHECK: BOLT-INFO: processing main.cold.1 as a sibling of non-ignored function
# CHECK: BOLT-INFO: processing foo.cold.1/1(*2) as a sibling of non-ignored function
# CHECK: BOLT-INFO: processing bar.cold.1/1(*2) as a sibling of non-ignored function
# CHECK: BOLT-INFO: processing baz.cold.1 as a sibling of non-ignored function
# CHECK: BOLT-INFO: processing baz.cold.1/1(*2) as a sibling of non-ignored function
# CHECK: BOLT-INFO: processing baz.cold.1/2(*2) as a sibling of non-ignored function

# CHECK: Binary Function "main.cold.1" after building cfg
# CHECK: Parent : main

# CHECK: Binary Function "foo.cold.1/1(*2)" after building cfg
# CHECK: Parent : foo

# CHECK: Binary Function "bar.cold.1/1(*2)" after building cfg
# CHECK: Parent : bar/1(*2)

# CHECK: Binary Function "baz.cold.1" after building cfg
# CHECK: Parent : baz{{$}}

# CHECK: Binary Function "baz.cold.1/1(*2)" after building cfg
# CHECK: Parent : baz/1(*2)

# CHECK: Binary Function "baz.cold.1/2(*2)" after building cfg
# CHECK: Parent : baz/2(*2)

#--- main.s
.file "main.s"
  .globl main
  .type main, %function
main:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 main 0 1 0
  cmpl	$0x0, %eax
  je	main.cold.1
  retq
  .cfi_endproc
.size main, .-main

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 foo 0 1 0
  cmpl	$0x0, %eax
  je	foo.cold.1
  retq
  .cfi_endproc
.size foo, .-foo

  .local bar
  .type bar, %function
bar:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 bar/1 0 1 0
  cmpl	$0x0, %eax
  je	bar.cold.1
  retq
  .cfi_endproc
.size bar, .-bar

  .globl baz
  .type baz, %function
baz:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 baz 0 1 0
  cmpl	$0x0, %eax
  je	baz.cold.1
  retq
  .cfi_endproc
.size baz, .-baz

  .section .text.cold
  .globl main.cold.1
  .type main.cold.1, %function
main.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size main.cold.1, .-main.cold.1

  .local foo.cold.1
  .type foo.cold.1, %function
foo.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size foo.cold.1, .-foo.cold.1

  .local bar.cold.1
  .type bar.cold.1, %function
bar.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size bar.cold.1, .-bar.cold.1

  .globl baz.cold.1
  .type baz.cold.1, %function
baz.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size baz.cold.1, .-baz.cold.1

#--- baz.s
.file "baz.s"
  .local baz
  .type baz, %function
baz:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 baz/1 0 1 0
  cmpl	$0x0, %eax
  je	baz.cold.1
  retq
  .cfi_endproc
.size baz, .-baz

  .section .text.cold
  .local baz.cold.1
  .type baz.cold.1, %function
baz.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size baz.cold.1, .-baz.cold.1

#--- baz2.s
.file "baz2.s"
  .local baz
  .type baz, %function
baz:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 baz/2 0 1 0
  cmpl	$0x0, %eax
  je	baz.cold.1
  retq
  .cfi_endproc
.size baz, .-baz

  .section .text.cold
  .local baz.cold.1
  .type baz.cold.1, %function
baz.cold.1:
  .cfi_startproc
  pushq	%rbp
  movq	%rsp, %rbp
  movl	$0x0, %eax
  popq	%rbp
  retq
  .cfi_endproc
.size baz.cold.1, .-baz.cold.1

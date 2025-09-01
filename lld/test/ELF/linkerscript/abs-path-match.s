# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o

# RUN: llvm-mc -filetype=obj -triple=x86_64 foo.s -o foo.o
# RUN: llvm-objcopy --rename-section .text=.text_foo  foo.o foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64 bar.s -o bar.o
# RUN: llvm-objcopy --rename-section .text=.text_bar  bar.o bar.o

# RUN: ld.lld -r main.o %t/foo.o %t/bar.o -T script.ld -o main_abs.o

# RUN: llvm-objdump -S main_abs.o > main_abs
# RUN: llvm-objdump -S main_abs.o | FileCheck %s
# CHECK: Disassembly of section .goo:


#--- foo.s
    .text
    .globl	foo
    .p2align	4
    .type	foo,@function
foo:
    nop


#--- bar.s
    .text
    .globl	bar
    .p2align	4
    .type	bar,@function
bar:      
    nop


#--- main.s
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:
	callq	foo@PLT
	callq	bar@PLT
	retq


#--- script.ld
SECTIONS {
  .text : { *(.text) }
  .goo : {
    bar.o(.text_bar);
    foo.o(.text_foo);
  }
}
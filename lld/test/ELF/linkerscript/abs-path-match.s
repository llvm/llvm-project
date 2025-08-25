# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o

# RUN: llvm-mc -filetype=obj -triple=x86_64 foo.s -o foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bar.s -o bar.o

# RUN: ld.lld main.o %t/foo.o %t/bar.o -T script.ld -o main_abs.o -Map=main_abs.map

# RUN: FileCheck %s < main_abs.map
# CHECK: .goo
# CHECK: bar.o:(.text_bar)
# CHECK: bar
# CHECK: foo.o:(.text_foo) 
# CHECK: foo
# CHECK-NOT: .text_bar
# CHECK-NOT: .text_foo

#--- foo.s
    .section .text_foo, "ax", %progbits
    .globl	foo
    .p2align	4
    .type	foo,@function
foo:
    nop


#--- bar.s
    .section .text_bar, "ax", %progbits
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
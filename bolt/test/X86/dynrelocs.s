# This reproduces a bug when rewriting dynamic relocations in X86 as
# BOLT incorrectly attributes R_X86_64_64 dynamic relocations
# to the wrong section when the -jump-tables=move flag is used. We
#	expect the relocations to belong to the .bolt.org.rodata section but
#	it is attributed to a new .rodata section that only contains jump
#	table entries, created by BOLT. BOLT will only create this new .rodata
# section if both -jump-tables=move is used and a hot function with
# jt is present in the input binary, triggering a scenario where the
# dynamic relocs rewriting gets confused on where to put .rodata relocs.

# It is uncommon to end up with dynamic relocations against .rodata,
# but it can happen. In these cases we cannot corrupt the
# output binary by writing out dynamic relocs incorrectly. The linker
# avoids emitting relocs against read-only sections but we override
# this behvior with the -z notext flag. During runtime, these pages
# are mapped with write permission and then changed to read-only after
# the dynamic linker finishes processing the dynamic relocs.

# In this test, we create a reference to a dynamic object that will
# imply in R_X86_64_64 being used for .rodata. Now BOLT, when creating
# a new .rodata to hold jump table entries, needs to remember to emit
# these dynamic relocs against the original .rodata, and not the new
# one it just created.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   %p/Inputs/define_bar.s -o %t.2.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.2.o -o %t.so -shared
# RUN: ld.lld -z notext %t.o -o %t.exe -q  %t.so
# RUN: llvm-bolt -data %t.fdata %t.exe -relocs -o %t.out -lite=0 \
# RUN:   -jump-tables=move
# RUN: llvm-readobj -rs %t.out | FileCheck --check-prefix=READOBJ %s

# Verify that BOLT outputs the dynamic reloc at the correct address,
# which is the start of the .bolt.org.rodata section.
# READOBJ:        Relocations [
# READOBJ:          Section ([[#]]) .rela.dyn {
# READOBJ-NEXT:        0x[[#%X,ADDR:]] R_X86_64_64 bar 0x10
# READOBJ:        Symbols [
# READOBJ:           Name: .bolt.org.rodata
# READOBJ-NEXT:      Value: 0x[[#ADDR]]

  # Create a hot function with jump table
  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 6
	movq	.LJUMPTABLE(,%rdi,8), %rax
b: jmpq *%rax
# FDATA: 1 _start #b# 1 _start #c# 0 3
c:
  mov $1, %rax
d:
  xorq %rax, %rax
  ret
  .cfi_endproc
  .size _start, .-_start

  # This is the section that needs to be tested.
  .section .rodata
	.align 4
  # We will have a R_X86_64_64 here or R_X86_64_COPY if this section
  # is non-writable. We use -z notext to force the linker to accept dynamic
  # relocations in read-only sections and make it a R_X86_64_64.
  .quad bar + 16  # Reference a dynamic object (such as a vtable ref)
  # Add other contents to this section: a hot jump table that will be
	# copied by BOLT into a new section.
.LJUMPTABLE:
	.quad	c
	.quad	c
	.quad	d
	.quad	d

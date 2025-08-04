// Check that we are able to rewrite binaries when we fail to identify a
// suitable location to put new code and user supplies a custom one via
// --custom-allocation-vma. This happens more obviously if the binary has
// segments mapped to very high addresses.

// In this example, my.reserved.section is mapped to a segment to be loaded
// at address 0x10000000000, while regular text should be at 0x200000. We
// pick a vma in the middle at 0x700000 to carve space for BOLT to put data,
// since BOLT's usual route of allocating after the last segment will put
// code far away and that will blow up relocations from main.

// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/main.s -o %t.o
// RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-T %t/main.ls
// RUN: llvm-bolt %t.exe -o %t.bolt --custom-allocation-vma=0x700000

//--- main.s
  .type            reserved_space,@object
  .section        .my.reserved.section,"awx",@nobits
  .globl           reserved_space
  .p2align         4, 0x0
reserved_space:
  .zero  0x80000000
  .size   reserved_space, 0x80000000

	.text
  .globl main
  .globl _start
  .type main, %function
_start:
main:
	.cfi_startproc
  nop
  nop
  nop
  retq
	.cfi_endproc
.size main, .-main

//--- main.ls
SECTIONS
{
    .my.reserved.section 1<<40 : {
      *(.my.reserved.section);
    }
} INSERT BEFORE .comment;

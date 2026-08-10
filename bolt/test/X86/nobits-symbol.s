## Check that llvm-bolt doesn't choke on symbols defined in nobits sections.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t
#

  .type            symbol_in_nobits,@object
  .section        .my.nobits.section,"awx",@nobits
  .globl           symbol_in_nobits
  .p2align         4, 0x0
symbol_in_nobits:
  .zero  0x100000
  .size   symbol_in_nobits, 0x100000

	.text
  .globl main
  .type main, %function
main:
	.cfi_startproc
.LBB06:
  retq
	.cfi_endproc
.size main, .-main

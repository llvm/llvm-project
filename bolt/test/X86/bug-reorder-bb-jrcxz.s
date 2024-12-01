## Check that BOLT handles code with jrcxz instruction that has a one-byte
## signed offset restriction. If we try to separate jrcxz instruction from its
## destination, e.g. by placing it in a different code fragment, then the link
## step will fail.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q

## Disable relocation mode to leave main fragment in its original location.

# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --reorder-blocks=ext-tsp \
# RUN:   --split-functions --relocs=0

	.text
	.globl main
	.type	main,@function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 1
# FDATA: 1 main 0 1 main #.hot# 0 1
.cfi_startproc
  jrcxz .Lcold
.hot:
  ret

.Lcold:
  xorl %eax,%eax
  ret
.cfi_endproc
.size  main,.-main

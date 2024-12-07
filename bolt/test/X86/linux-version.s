# REQUIRES: system-linux

## Check that BOLT correctly detects the Linux kernel version

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -nostdlib %t.o -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr

# RUN: llvm-bolt %t.exe -o %t.out | FileCheck %s

# CHECK: BOLT-INFO: Linux kernel version is 6.6.61

	.text
	.globl	f
	.type	f, @function
f:
	ret
	.size	f, .-f

	.globl	linux_banner
	.section	.rodata
	.align 16
	.type	linux_banner, @object
	.size	linux_banner, 22
linux_banner:
	.string	"Linux version 6.6.61\n"

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits

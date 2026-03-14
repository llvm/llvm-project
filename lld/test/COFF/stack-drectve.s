# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t.obj
# RUN: lld-link /out:%t.exe /entry:main %t.obj
# RUN: llvm-readobj --file-headers %t.exe | FileCheck %s

# CHECK: SizeOfStackReserve: 20480
# CHECK: SizeOfStackCommit: 12288

	.text
	.globl main
main:
	mov $42, %eax
	ret

	.section	.drectve,"yn"
	.ascii	" -stack:0x5000,0x3000"

## Check that llvm-bolt will unnecessarily relax ADR instruction.
## ADR below references containing function that is split. But ADR is always
## in the main fragment, thus there is no need to relax it.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=randomN \
# RUN:   2>&1 | FileCheck %s
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s

# CHECK-NOT: adrp

	.text
  .globl _start
  .type _start, %function
_start:
	.cfi_startproc
  adr x1, _start
	cmp	x1, x11
	b.hi	.L1

	mov	x0, #0x0

.L1:
	ret	x30

	.cfi_endproc
.size _start, .-_start

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

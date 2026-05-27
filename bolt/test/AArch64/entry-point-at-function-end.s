## Promote unmarked code after a CFI-bounded predecessor into __BOLT_UNMARKED_TAIL.
## Covers:
##   1) FDE-only region (no symtab entry; BOLT names it __BOLT_FDE_FUNC* or .text/N)
##   2) Named function (FDE range == symbol size) -> named_sub + tail
## Caller is in a separate section; -Wl,-q keeps PC-relative relocs on the bl.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -pie -Wl,-q -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm -v=1 2>&1 | FileCheck %s

# CHECK: Binary Function "{{(__BOLT_FDE_FUNC.*|\.text/[0-9]+)}}" after disassembly
# CHECK: Size        : 0x14
# CHECK: MaxSize     : 0x14
# CHECK: Binary Function "{{.*}}__BOLT_UNMARKED_TAIL{{.*}}" after disassembly
# CHECK: Size        : 0x8
# CHECK: MaxSize     : 0x8

# CHECK: Binary Function "named_sub" after disassembly
# CHECK: Size        : 0x14
# CHECK: MaxSize     : 0x14
# CHECK: Binary Function "{{.*}}__BOLT_UNMARKED_TAIL{{.*}}" after disassembly
# CHECK: Size        : 0x10
# CHECK: MaxSize     : 0x10

# CHECK-COUNT-2: bl	__BOLT_UNMARKED_TAILat{{.*}}

	.section	.text.fdeonly,"ax",@progbits

## Scenario 1: FDE-only predecessor (no symbol table entry).
.Lfde_sub:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	w0, #42
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc

.Ltail_after_fde:
	nop
	ret

	.section	.text.named,"ax",@progbits

## Scenario 2: named predecessor (FDE address range matches .size).
	.globl	named_sub
	.type	named_sub, %function
named_sub:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	w0, #7
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc
	.size	named_sub, .-named_sub

.Ltail_after_named:
	mov	x0, x0
	mov	w1, #7
	add	w0, w0, w1
	ret

	.section	.text.entry,"ax",@progbits

	.globl	_start
	.type	_start, %function
_start:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	bl	.Ltail_after_fde
	bl	.Ltail_after_named
	ldp	x29, x30, [sp], #16
	mov	w0, #0
	ret
	.cfi_endproc
	.size	_start, .-_start

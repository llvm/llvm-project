## Test that BOLT correctly handles entry points at exact FDE boundaries.
##
## This test simulates a real-world scenario where:
## 1. A large function (big_func) contains an FDE with no symbol
## 2. The FDE covers a sub-function within big_func [fde_start, fde_end)
## 3. Inlined code follows immediately after the FDE (no symbol, no FDE)
## 4. .init branches to exactly the FDE end address via section relocation
##
## Without the fix, getOrCreateLocalLabel() would return getFunctionEndLabel()
## when offset == FDE size. The fix ensures a proper entry point label is created.
##

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -pie -Wl,-q -Wl,--init=_init -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm -v=1 2>&1 | FileCheck %s

## The FDE-only function should get an entry point for the inlined code
# CHECK: BOLT-WARNING: FDE {{.*}} has no corresponding symbol table entry
# CHECK: Binary Function "{{.*}}__BOLT_FDE_FUNC{{.*}}" after disassembly
# CHECK: Size        : 0x14
# CHECK: MaxSize     : 0x1c
# CHECK: IsMultiEntry: 1

## _init function in .init section
	.section .init,"ax",@progbits
	.globl	_init
	.type	_init, %function
_init:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	## Branch to inlined code that's right after FDE end
	## Linker converts this to: R_AARCH64_CALL26 .text + offset
	bl	.Linlined_code
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc
	.size	_init, .-_init

## Main code in .text section
	.text

## Large function that contains the FDE and inlined code
	.globl	big_func
	.type	big_func, %function
big_func:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	w0, #1
	add	w0, w0, #1
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc
	.size	big_func, .-big_func

## FDE-covered code WITHOUT a symbol - simulates sub-function within big_func
## This has its own FDE but no symbol in the symbol table
.Lfde_start:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	w0, #42
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc
## FDE ends here - inlined code follows immediately

## Inlined code at exactly FDE end (offset == FDE size)
## _init branches here - this is what triggers the bug without the fix
.Linlined_code:
	nop
	ret

	.globl	_start
	.type	_start, %function
_start:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	bl	_init
	bl	big_func
	ldp	x29, x30, [sp], #16
	mov	w0, #0
	ret
	.cfi_endproc
	.size	_start, .-_start

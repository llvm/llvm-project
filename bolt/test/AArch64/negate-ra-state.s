# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags  %t.o -o %t.exe -Wl,-q

# RUN: llvm-objdump %t.exe -d > %t.exe.dump
# RUN: llvm-objdump --dwarf=frames %t.exe > %t.exe.dump-dwarf
# RUN: match-dwarf %t.exe.dump %t.exe.dump-dwarf foo > %t.match-dwarf.txt

# RUN: llvm-bolt %t.exe -o %t.exe.bolt

# RUN: llvm-objdump %t.exe.bolt -d > %t.exe.bolt.dump
# RUN: llvm-objdump --dwarf=frames %t.exe.bolt  > %t.exe.bolt.dump-dwarf
# RUN: match-dwarf %t.exe.bolt.dump %t.exe.bolt.dump-dwarf foo > %t.bolt.match-dwarf.txt

# RUN: diff %t.match-dwarf.txt %t.bolt.match-dwarf.txt

	.text
	.globl	foo
	.p2align	2
	.type	foo,@function
foo:
	.cfi_startproc
	hint	#25
	.cfi_negate_ra_state
	sub	sp, sp, #16
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	str	w0, [sp, #12]
	ldr	w8, [sp, #12]
	add	w0, w8, #1
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #16
	hint	#29
	.cfi_negate_ra_state
	ret
.Lfunc_end1:
	.size	foo, .Lfunc_end1-foo
	.cfi_endproc

	.global _start
	.type _start, %function
_start:
	b foo

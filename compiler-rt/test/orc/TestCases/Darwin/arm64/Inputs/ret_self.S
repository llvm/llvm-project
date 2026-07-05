	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0	sdk_version 14, 4
	.globl	_ret_self
	.p2align	2
_ret_self:
	adrp	x0, _ret_self@PAGE
	add	x0, x0, _ret_self@PAGEOFF
	ret

.subsections_via_symbols


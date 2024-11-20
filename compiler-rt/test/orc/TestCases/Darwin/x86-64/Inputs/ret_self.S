// A function that returns its own address. Handy for testing whether JIT'd code
// and JIT symbol tables agree on addresses.

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	_ret_self
	.p2align	4, 0x90
_ret_self:
	leaq	_ret_self(%rip), %rax
	retq

.subsections_via_symbols

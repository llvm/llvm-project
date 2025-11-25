	.text
	.file	"reloc-asm.c"
	.globl	foo
	.p2align	4, 0x90
	.type	foo,@function
foo:
	s_load_dwordx2 s[0:1], s[4:5], 0x0                         // 000000000000: C0060002 00000000
	v_mov_b32_e32 v2, 42                                       // 000000000008: 7E0402AA
	s_waitcnt lgkmcnt(0)                                       // 00000000000C: BF8C007F
	v_mov_b32_e32 v0, s0                                       // 000000000010: 7E000200
	v_mov_b32_e32 v1, s1                                       // 000000000014: 7E020201
	flat_store_dword v[0:1], v2                                // 000000000018: DC700000 00000200
	s_endpgm
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo

	.ident	"clang"
	.section	".note.GNU-stack","",@progbits
	.addrsig

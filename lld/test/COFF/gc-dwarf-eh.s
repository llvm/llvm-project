# REQUIRES: x86

# RUN: llvm-mc -triple=i686-windows-gnu %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmingw -lldmap:%t.map -out:%t.exe -opt:ref -entry:main %t.obj -verbose 2>&1 | FileCheck %s
# RUN: FileCheck %s --check-prefix=MAP --input-file=%t.map

# CHECK: Discarded _unused

# MAP: In Symbol
# MAP: gc-dwarf-eh.s.tmp.obj:(.text)
# MAP: {{ ___gxx_personality_v0$}}

	.def	_main; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,_main
	.globl	_main
	.cfi_startproc
	.cfi_personality 0, ___gxx_personality_v0
_main:
	xorl	%eax, %eax
	ret
	.cfi_endproc

	.def	___gxx_personality_v0; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,___gxx_personality_v0
	.globl	___gxx_personality_v0
___gxx_personality_v0:
	ret

	.def	_unused; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,_unused
	.globl	_unused
	.cfi_startproc
	.cfi_personality 0, ___gxx_personality_v0
_unused:
	ret
	.cfi_endproc

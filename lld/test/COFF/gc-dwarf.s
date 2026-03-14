# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -lldmap:%t.map -out:%t.exe -opt:ref -entry:main %t.obj -verbose 2>&1 | FileCheck %s
# RUN: FileCheck %s --check-prefix=MAP --input-file=%t.map

# CHECK:      Discarded unused1
# CHECK-NEXT: Discarded unused2
# CHECK-NOT: Discarded

# MAP: In Symbol
# MAP: gc-dwarf.s.tmp.obj:(.text)
# MAP: {{ main$}}
# MAP: gc-dwarf.s.tmp.obj:(.text)
# MAP: {{ used$}}

	.def	 @feat.00; .scl	3; .type	0; .endef
	.globl	@feat.00
.set @feat.00, 0

	.def	 main; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,main
	.globl	main
main:
	callq used
	xorl	%eax, %eax
	retq

	.def	 used; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,used
	.globl	used
used:
	retq


	.def	 unused1; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,unused1
	.globl	unused1
unused1:
	retq

	.def	 unused2; .scl	2; .type	32; .endef
	.section	.text,"xr",one_only,unused2
	.globl	unused2
unused2:
	retq

# This isn't valid DWARF, but LLD doesn't care. Make up some data that
# references the functions above.
.section .debug_info,"r"
.long main@IMGREL
.long unused1@IMGREL
.long unused2@IMGREL

# Similarly, .eh_frame unwind info should not keep functions alive. Again, this
# is not valid unwind info, but it doesn't matter for testing purposes.
.section .eh_frame,"r"
.long main@IMGREL
.long unused1@IMGREL
.long unused2@IMGREL

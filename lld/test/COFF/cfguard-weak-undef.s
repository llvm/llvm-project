# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj /out:%t.exe /entry:entry /subsystem:console /guard:cf

	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 2048

	.globl	entry
entry:
	retq

	.data
	.globl	funcs
funcs:
	.quad	weakfunc

	.section	.gfids$y,"dr"
	.symidx	weakfunc
	.section	.giats$y,"dr"
	.section	.gljmp$y,"dr"
	.weak	weakfunc
	.addrsig
	.addrsig_sym weakfunc

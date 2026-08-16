# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t.obj
# RUN: lld-link %t.obj /dll /noentry /out:%t.dll /opt:noref,icf
# RUN: llvm-readobj --coff-exports %t.dll | FileCheck %s

## This test models the object produced by Inputs/icf-eh-funclet.cpp, built with:
##   cl.exe /c /EHsc /O2 /GS- /d2FH4- icf-eh-funclet.cpp
## which is a reduced reproducer for llvm/llvm-project#41566.
##
## A C++ EH catch/cleanup funclet is emitted as an associative, executable comdat
## (".text$x"). A funclet's own unwind info (.pdata/.xdata) is emitted in comdats
## that are associative to the funclet's *parent* function (not to the funclet)
## and that chain to the parent's exception-handling data (FuncInfo). The funclet
## body carries no reference to that unwind info, so two funclets with identical
## code look mergeable to ICF even when their parents differ. Folding them would
## leave the single surviving funclet described by two .pdata entries covering
## the same address range but with conflicting unwind info, so a thrown exception
## unwinds with the wrong parent's FuncInfo and is dispatched to the wrong catch
## handler (or terminate()).
##
## f1c and f2c are identical funclets whose parents (f1, f2) differ, and whose
## unwind info (f1c_xdata, f2c_xdata) chains to those different parents. They must
## NOT be folded and must keep distinct addresses.
# CHECK:      Name: f1c
# CHECK-NEXT: RVA: 0x1012
# CHECK:      Name: f2c
# CHECK-NEXT: RVA: 0x1015

## g1c and g2c are identical funclets whose parents (g1, g2) are themselves
## identical and fold, so folding the funclets is safe and must still happen.
# CHECK:      Name: g1c
# CHECK-NEXT: RVA: [[G:0x[0-9A-Fa-f]+]]
# CHECK:      Name: g2c
# CHECK-NEXT: RVA: [[G]]

## --- Parents f1/f2: different bodies, so they do not fold. ---
	.section	.text,"xr",one_only,f1
	.globl	f1
f1:
	movl	$1, %eax
	retq

	.section	.text,"xr",one_only,f2
	.globl	f2
f2:
	movl	$2, %eax
	retq

## Catch funclets for f1/f2: identical bodies in associative .text$x comdats.
	.section	.text$x,"xr",associative,f1
	.globl	f1c
f1c:
	nop
	nop
	retq
.Lf1c_end:

	.section	.text$x,"xr",associative,f2
	.globl	f2c
f2c:
	nop
	nop
	retq
.Lf2c_end:

## Funclet unwind info (.xdata), associative to the *parent*, chaining to it.
## Because they reference different parents, f1c_xdata and f2c_xdata differ.
	.section	.xdata,"dr",associative,f1
f1c_xdata:
	.long	1
	.long	f1@IMGREL
	.section	.xdata,"dr",associative,f2
f2c_xdata:
	.long	1
	.long	f2@IMGREL

## Funclet .pdata (RUNTIME_FUNCTION), associative to the *parent*.
	.section	.pdata,"dr",associative,f1
	.long	f1c@IMGREL
	.long	.Lf1c_end@IMGREL
	.long	f1c_xdata@IMGREL
	.section	.pdata,"dr",associative,f2
	.long	f2c@IMGREL
	.long	.Lf2c_end@IMGREL
	.long	f2c_xdata@IMGREL

## --- Parents g1/g2: identical bodies, so they fold. ---
	.section	.text,"xr",one_only,g1
	.globl	g1
g1:
	movl	$7, %eax
	retq

	.section	.text,"xr",one_only,g2
	.globl	g2
g2:
	movl	$7, %eax
	retq

	.section	.text$x,"xr",associative,g1
	.globl	g1c
g1c:
	int3
	int3
	retq
.Lg1c_end:

	.section	.text$x,"xr",associative,g2
	.globl	g2c
g2c:
	int3
	int3
	retq
.Lg2c_end:

	.section	.xdata,"dr",associative,g1
g1c_xdata:
	.long	1
	.long	g1@IMGREL
	.section	.xdata,"dr",associative,g2
g2c_xdata:
	.long	1
	.long	g2@IMGREL

	.section	.pdata,"dr",associative,g1
	.long	g1c@IMGREL
	.long	.Lg1c_end@IMGREL
	.long	g1c_xdata@IMGREL
	.section	.pdata,"dr",associative,g2
	.long	g2c@IMGREL
	.long	.Lg2c_end@IMGREL
	.long	g2c_xdata@IMGREL

	.section	.drectve,"yn"
	.ascii	" -export:f1c -export:f2c -export:g1c -export:g2c"

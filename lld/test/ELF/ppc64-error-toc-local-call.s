# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

## This test checks that the linker produces errors when it is missing the nop
## after a local call to a callee with st_other=1.

# CHECK: (.text+0xC): call to save_callee lacks nop, can't restore toc
# CHECK: (.text+0x1C): call to save_callee lacks nop, can't restore toc

callee:
	.localentry	callee, 1
	blr                                                # 0x0

caller:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	.localentry	caller, .Lfunc_lep1-.Lfunc_gep1
	bl callee                                          # 0xC
	blr

caller_tail:
.Lfunc_gep2:
	addis 2, 12, .TOC.-.Lfunc_gep2@ha
	addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
	.localentry	caller_tail, .Lfunc_lep2-.Lfunc_gep2
	b callee                                           # 0x1C

# REQUIRES: x86
## Regression test for https://github.com/llvm/llvm-project/issues/63039

## Use an old version to ensure we do *not* have any compact-unwind.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos9 %s -o %t.o

## Pre-cond: smoke-check that there is really no compact-unwind entries - only dwarfs.
# RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s --check-prefix=PRE
# PRE-NOT: Contents of __compact_unwind section:
# PRE-NOT: Entry at offset
# PRE: .eh_frame contents:
# PRE: {{[0-9a-f]+}} {{.*}} CIE
# PRE:   Format:                DWARF32
# PRE:   Version:               1

## Link should succeed (ie., not crashed due to bug in icf code).
# RUN: %lld -lSystem -lc++ --icf=all -arch x86_64 -arch x86_64 -platform_version macos 11.0 11.0 %t.o -o %t.out

## Post-cond: verify that the final binary has expected eh-frame contents.
# RUN: llvm-objdump --macho --syms --dwarf=frames %t.out | FileCheck %s --check-prefix=POST
# POST-LABEL: SYMBOL TABLE:
# POST: [[#%x,EXCEPT_ADDR:]] l   O __TEXT,__gcc_except_tab GCC_except_table0
# POST: [[#%x,EXCEPT_ADDR]]  l   O __TEXT,__gcc_except_tab GCC_except_table1
# POST: [[#%.16x,F0_ADDR:]]  g   F __TEXT,__text _f0
# POST: [[#%.16x,F1_ADDR:]]  g   F __TEXT,__text _f1
# POST: [[#%.16x,G_ADDR:]]   g   F __TEXT,__text _g

# POST-LABEL: .eh_frame contents:
# POST: {{.*}} FDE cie={{.+}} pc=[[#%x,G_ADDR]]...{{.+}}

# POST: {{.*}} FDE cie={{.+}} pc=[[#%x,F0_ADDR]]...{{.+}}
# POST: Format:       DWARF32 
# POST: LSDA Address: [[#%.16x,EXCEPT_ADDR]]

# POST: {{.*}} FDE cie={{.+}} pc=[[#%x,F1_ADDR]]...{{.+}}
# POST Format:       DWARF32 
# POST LSDA Address: [[#%.16x,EXCEPT_ADDR]]

	.section        __TEXT,__text,regular,pure_instructions
	.globl	_f0
_f0:
	.cfi_startproc
	.cfi_lsda 16, Lexception0
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	callq	_g
	retq
	.cfi_endproc
	
	.section	__TEXT,__gcc_except_tab
GCC_except_table0:
Lexception0:
	.byte	255
                                
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_f1
_f1:
	.cfi_startproc
	.cfi_lsda 16, Lexception1
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	callq	_g
	retq
	.cfi_endproc
	
	.section	__TEXT,__gcc_except_tab
GCC_except_table1:
Lexception1:
	.byte	255

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_g
_g:               
	.cfi_startproc
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	.cfi_def_cfa_register %rbp	
	retq
	.cfi_endproc
	
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main             
_main:                            
	retq

.subsections_via_symbols

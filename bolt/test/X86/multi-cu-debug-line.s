## Test that BOLT correctly handles debug line information for functions
## that belong to multiple compilation units (e.g., inline functions in
## common header files). This is the assembly version of the multi-cu-debug-line.test.
## The test covers two scenarios:
## 1. Normal processing: .debug_line section shows lines for the function
##    in all CUs where it was compiled, with no duplicate rows within CUs
## 2. Functions not processed: When BOLT doesn't process functions (using
##    --funcs with nonexistent function), original debug info is preserved

# REQUIRES: system-linux

## Split the input into separate object files
# RUN: split-file %s %t

## Assemble the separate object files
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/multi-cu-file1.s -o %t/multi-cu-file1.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/multi-cu-file2.s -o %t/multi-cu-file2.o

## Link them together with debug info
# RUN: %clang %cflags %t/multi-cu-file1.o %t/multi-cu-file2.o -o %t.exe -Wl,-q

## Test 1: Normal BOLT processing (functions are processed/optimized)
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-line %t.bolt > %t.debug-line.txt
# RUN: FileCheck %s --check-prefix=BASIC --input-file %t.debug-line.txt

## Check that debug line information is present for both compilation units
# BASIC: debug_line[{{.*}}]
# BASIC: file_names[{{.*}}]:
# BASIC: name: "{{.*}}multi-cu-file1.c"
# BASIC: debug_line[{{.*}}]
# BASIC: file_names[{{.*}}]:
# BASIC: name: "{{.*}}multi-cu-file2.c"

## Use our helper script to create a normalized table without addresses
# RUN: process-debug-line %t.debug-line.txt > %t.normalized-debug-line.txt
# RUN: FileCheck %s --check-prefix=NORMALIZED --input-file %t.normalized-debug-line.txt

## Check that we have line entries for the inline function (lines 5, 6, 7) from multi-cu-common.h
## in both compilation units
# NORMALIZED: multi-cu-file1.c 5 {{[0-9]+}} multi-cu-common.h
# NORMALIZED: multi-cu-file1.c 6 {{[0-9]+}} multi-cu-common.h
# NORMALIZED: multi-cu-file1.c 7 {{[0-9]+}} multi-cu-common.h
# NORMALIZED: multi-cu-file2.c 5 {{[0-9]+}} multi-cu-common.h
# NORMALIZED: multi-cu-file2.c 6 {{[0-9]+}} multi-cu-common.h
# NORMALIZED: multi-cu-file2.c 7 {{[0-9]+}} multi-cu-common.h

## Verify that we have line entries for the inline function in multiple CUs
## by checking that the header file appears multiple times in different contexts
# RUN: grep -c "multi-cu-common.h" %t.debug-line.txt > %t.header-count.txt
# RUN: FileCheck %s --check-prefix=MULTI-CU --input-file %t.header-count.txt

## The header should appear in debug line info for multiple CUs
# MULTI-CU: {{[2-9]|[1-9][0-9]+}}

## Check that there are no duplicate line table rows within the same CU
## This verifies the fix for the bug where duplicate entries were created
# RUN: sort %t.normalized-debug-line.txt | uniq -c | \
# RUN:   awk '$1 > 1 {print "DUPLICATE_ROW: " $0}' > %t.duplicates.txt
# RUN: FileCheck %s --check-prefix=NO-DUPLICATES --input-file %t.duplicates.txt --allow-empty

## Should have no duplicate normalized rows (file should be empty)
## Note: Cross-CU duplicates are expected and valid (same function in different CUs)
## but within-CU duplicates would indicate a bug
# NO-DUPLICATES-NOT: DUPLICATE_ROW

## Test 2: Functions not processed by BOLT (using --funcs with nonexistent function)
## This tests the code path where BOLT preserves original debug info
# RUN: llvm-bolt %t.exe -o %t.not-emitted.bolt --update-debug-sections --funcs=nonexistent_function
# RUN: llvm-dwarfdump --debug-line %t.not-emitted.bolt > %t.not-emitted.debug-line.txt
# RUN: FileCheck %s --check-prefix=PRESERVED-BASIC --input-file %t.not-emitted.debug-line.txt

## Check that debug line information is still present for both compilation units when functions aren't processed
# PRESERVED-BASIC: debug_line[{{.*}}]
# PRESERVED-BASIC: file_names[{{.*}}]:
# PRESERVED-BASIC: name: "{{.*}}multi-cu-file1.c"
# PRESERVED-BASIC: debug_line[{{.*}}]
# PRESERVED-BASIC: file_names[{{.*}}]:
# PRESERVED-BASIC: name: "{{.*}}multi-cu-file2.c"

## Create normalized output for the not-emitted case
# RUN: process-debug-line %t.not-emitted.debug-line.txt > %t.not-emitted.normalized.txt
# RUN: FileCheck %s --check-prefix=PRESERVED-NORMALIZED --input-file %t.not-emitted.normalized.txt

## Check that we have line entries for the inline function (lines 5, 6, 7) from multi-cu-common.h
## in both compilation units (preserved from original)
# PRESERVED-NORMALIZED: multi-cu-file1.c 5 {{[0-9]+}} multi-cu-common.h
# PRESERVED-NORMALIZED: multi-cu-file1.c 6 {{[0-9]+}} multi-cu-common.h
# PRESERVED-NORMALIZED: multi-cu-file1.c 7 {{[0-9]+}} multi-cu-common.h
# PRESERVED-NORMALIZED: multi-cu-file2.c 5 {{[0-9]+}} multi-cu-common.h
# PRESERVED-NORMALIZED: multi-cu-file2.c 6 {{[0-9]+}} multi-cu-common.h
# PRESERVED-NORMALIZED: multi-cu-file2.c 7 {{[0-9]+}} multi-cu-common.h

## Verify that we have line entries for the inline function in multiple CUs (preserved)
## by checking that the header file appears multiple times in different contexts
# RUN: grep -c "multi-cu-common.h" %t.not-emitted.debug-line.txt > %t.preserved-header-count.txt
# RUN: FileCheck %s --check-prefix=PRESERVED-MULTI-CU --input-file %t.preserved-header-count.txt

## The header should appear in debug line info for multiple CUs (preserved from original)
# PRESERVED-MULTI-CU: {{[2-9]|[1-9][0-9]+}}

## Check that original debug info is preserved for main functions
# RUN: grep "multi-cu-file1.c.*multi-cu-file1.c" %t.not-emitted.normalized.txt > %t.preserved-main.txt
# RUN: FileCheck %s --check-prefix=PRESERVED-MAIN --input-file %t.preserved-main.txt

# PRESERVED-MAIN: multi-cu-file1.c {{[0-9]+}} {{[0-9]+}} multi-cu-file1.c

## Check that original debug info is preserved for file2 functions
# RUN: grep "multi-cu-file2.c.*multi-cu-file2.c" %t.not-emitted.normalized.txt > %t.preserved-file2.txt
# RUN: FileCheck %s --check-prefix=PRESERVED-FILE2 --input-file %t.preserved-file2.txt

# PRESERVED-FILE2: multi-cu-file2.c {{[0-9]+}} {{[0-9]+}} multi-cu-file2.c

## Note: We do not check for duplicates in Test 2 since we are preserving original debug info as-is
## and the original may contain patterns that would be flagged as duplicates by our normalization

;--- multi-cu-file1.s
	.file	"multi-cu-file1.c"
	.file	1 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-file1.c"
	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.loc	1 4 0                           # bolt/test/Inputs/multi-cu-file1.c:4:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
.Ltmp0:
	.loc	1 5 7 prologue_end              # bolt/test/Inputs/multi-cu-file1.c:5:7
	movl	$5, -8(%rbp)
	.loc	1 6 39                          # bolt/test/Inputs/multi-cu-file1.c:6:39
	movl	-8(%rbp), %edi
	.loc	1 6 16 is_stmt 0                # bolt/test/Inputs/multi-cu-file1.c:6:16
	callq	common_inline_function
	.loc	1 6 7                           # bolt/test/Inputs/multi-cu-file1.c:6:7
	movl	%eax, -12(%rbp)
	.loc	1 7 35 is_stmt 1                # bolt/test/Inputs/multi-cu-file1.c:7:35
	movl	-12(%rbp), %esi
	.loc	1 7 3 is_stmt 0                 # bolt/test/Inputs/multi-cu-file1.c:7:3
	leaq	.L.str(%rip), %rdi
	movb	$0, %al
	callq	printf
	.loc	1 8 3 is_stmt 1                 # bolt/test/Inputs/multi-cu-file1.c:8:3
	xorl	%eax, %eax
	.loc	1 8 3 epilogue_begin is_stmt 0  # bolt/test/Inputs/multi-cu-file1.c:8:3
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function common_inline_function
	.type	common_inline_function,@function
common_inline_function:                 # @common_inline_function
.Lfunc_begin1:
	.file	2 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-common.h"
	.loc	2 4 0 is_stmt 1                 # bolt/test/Inputs/multi-cu-common.h:4:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp2:
	.loc	2 5 16 prologue_end             # bolt/test/Inputs/multi-cu-common.h:5:16
	movl	-4(%rbp), %eax
	.loc	2 5 18 is_stmt 0                # bolt/test/Inputs/multi-cu-common.h:5:18
	shll	%eax
	.loc	2 5 7                           # bolt/test/Inputs/multi-cu-common.h:5:7
	movl	%eax, -8(%rbp)
	.loc	2 6 10 is_stmt 1                # bolt/test/Inputs/multi-cu-common.h:6:10
	movl	-8(%rbp), %eax
	addl	$10, %eax
	movl	%eax, -8(%rbp)
	.loc	2 7 10                          # bolt/test/Inputs/multi-cu-common.h:7:10
	movl	-8(%rbp), %eax
	.loc	2 7 3 epilogue_begin is_stmt 0  # bolt/test/Inputs/multi-cu-common.h:7:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	common_inline_function, .Lfunc_end1-common_inline_function
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"File1: Result is %d\n"
	.size	.L.str, 21

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xbe DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x11 DW_TAG_variable
	.long	59                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str
	.byte	3                               # Abbrev [3] 0x3b:0xc DW_TAG_array_type
	.long	71                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x40:0x6 DW_TAG_subrange_type
	.long	78                              # DW_AT_type
	.byte	21                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x47:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x4e:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	7                               # Abbrev [7] 0x55:0x36 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	193                             # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x6e:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	193                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x7c:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	193                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x8b:0x36 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string7                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	193                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xa4:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string10                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	193                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xb2:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string9                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	193                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xc1:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0" # string offset=0
.Linfo_string1:
	.asciz	"/repo/llvm-project/bolt/test/Inputs/multi-cu-file1.c" # string offset=43
.Linfo_string2:
	.asciz	"/repo/llvm-project" # string offset=125
.Linfo_string3:
	.asciz	"char"                          # string offset=173
.Linfo_string4:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=178
.Linfo_string5:
	.asciz	"main"                          # string offset=198
.Linfo_string6:
	.asciz	"int"                           # string offset=203
.Linfo_string7:
	.asciz	"common_inline_function"        # string offset=207
.Linfo_string8:
	.asciz	"value"                         # string offset=230
.Linfo_string9:
	.asciz	"result"                        # string offset=236
.Linfo_string10:
	.asciz	"x"                             # string offset=243
	.ident	"clang version 18.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym common_inline_function
	.addrsig_sym printf
	.section	.debug_line,"",@progbits
.Lline_table_start0:

;--- multi-cu-file2.s
	.file	"multi-cu-file2.c"
	.file	1 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-file2.c"
	.text
	.globl	helper_function                 # -- Begin function helper_function
	.p2align	4
	.type	helper_function,@function
helper_function:                        # @helper_function
.Lfunc_begin0:
	.loc	1 4 0                           # bolt/test/Inputs/multi-cu-file2.c:4:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
.Ltmp0:
	.loc	1 5 7 prologue_end              # bolt/test/Inputs/multi-cu-file2.c:5:7
	movl	$10, -4(%rbp)
	.loc	1 6 39                          # bolt/test/Inputs/multi-cu-file2.c:6:39
	movl	-4(%rbp), %edi
	.loc	1 6 16 is_stmt 0                # bolt/test/Inputs/multi-cu-file2.c:6:16
	callq	common_inline_function
	.loc	1 6 7                           # bolt/test/Inputs/multi-cu-file2.c:6:7
	movl	%eax, -8(%rbp)
	.loc	1 7 42 is_stmt 1                # bolt/test/Inputs/multi-cu-file2.c:7:42
	movl	-8(%rbp), %esi
	.loc	1 7 3 is_stmt 0                 # bolt/test/Inputs/multi-cu-file2.c:7:3
	leaq	.L.str(%rip), %rdi
	movb	$0, %al
	callq	printf
	.loc	1 8 1 epilogue_begin is_stmt 1  # bolt/test/Inputs/multi-cu-file2.c:8:1
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	helper_function, .Lfunc_end0-helper_function
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function common_inline_function
	.type	common_inline_function,@function
common_inline_function:                 # @common_inline_function
.Lfunc_begin1:
	.file	2 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-common.h"
	.loc	2 4 0                           # bolt/test/Inputs/multi-cu-common.h:4:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp2:
	.loc	2 5 16 prologue_end             # bolt/test/Inputs/multi-cu-common.h:5:16
	movl	-4(%rbp), %eax
	.loc	2 5 18 is_stmt 0                # bolt/test/Inputs/multi-cu-common.h:5:18
	shll	%eax
	.loc	2 5 7                           # bolt/test/Inputs/multi-cu-common.h:5:7
	movl	%eax, -8(%rbp)
	.loc	2 6 10 is_stmt 1                # bolt/test/Inputs/multi-cu-common.h:6:10
	movl	-8(%rbp), %eax
	addl	$10, %eax
	movl	%eax, -8(%rbp)
	.loc	2 7 10                          # bolt/test/Inputs/multi-cu-common.h:7:10
	movl	-8(%rbp), %eax
	.loc	2 7 3 epilogue_begin is_stmt 0  # bolt/test/Inputs/multi-cu-common.h:7:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	common_inline_function, .Lfunc_end1-common_inline_function
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"File2: Helper result is %d\n"
	.size	.L.str, 28

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xba DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x11 DW_TAG_variable
	.long	59                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str
	.byte	3                               # Abbrev [3] 0x3b:0xc DW_TAG_array_type
	.long	71                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x40:0x6 DW_TAG_subrange_type
	.long	78                              # DW_AT_type
	.byte	28                              # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x47:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x4e:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	7                               # Abbrev [7] 0x55:0x32 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x6a:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	189                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x78:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	189                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x87:0x36 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string6                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	189                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xa0:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string10                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	189                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xae:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string9                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	189                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xbd:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0" # string offset=0
.Linfo_string1:
	.asciz	"/repo/llvm-project/bolt/test/Inputs/multi-cu-file2.c" # string offset=43
.Linfo_string2:
	.asciz	"/repo/llvm-project" # string offset=125
.Linfo_string3:
	.asciz	"char"                          # string offset=173
.Linfo_string4:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=178
.Linfo_string5:
	.asciz	"helper_function"               # string offset=198
.Linfo_string6:
	.asciz	"common_inline_function"        # string offset=214
.Linfo_string7:
	.asciz	"int"                           # string offset=237
.Linfo_string8:
	.asciz	"value"                         # string offset=241
.Linfo_string9:
	.asciz	"result"                        # string offset=247
.Linfo_string10:
	.asciz	"x"                             # string offset=254
	.ident	"clang version 18.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym common_inline_function
	.addrsig_sym printf
	.section	.debug_line,"",@progbits
.Lline_table_start0:

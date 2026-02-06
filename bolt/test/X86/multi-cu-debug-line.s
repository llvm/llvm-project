## Test that BOLT correctly handles debug line information for functions
## that belong to multiple compilation units (e.g., inline functions in
## common header files). This is the assembly version of the multi-cu-debug-line.test.
## The test covers two scenarios:
## 1. Normal processing: .debug_line section shows lines for the function
##    in all CUs where it was compiled, with no duplicate rows within CUs
## 2. Functions not processed: When BOLT doesn't process functions (using
##    --funcs with nonexistent function), original debug info is preserved

# REQUIRES: system-linux

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/multi-cu-file1.s -o %t/multi-cu-file1.o
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %t/multi-cu-file2.s -o %t/multi-cu-file2.o
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

;--- multi-cu-file1.s
	.text
	.file	1 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-file1.c"
	.file	2 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-common.h"

	.globl	main
	.type	main,@function
main:
.Lfunc_begin0:
	.loc	1 4 0
	callq	common_inline_function
	.loc	1 8 0
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.type	common_inline_function,@function
common_inline_function:
.Lfunc_begin1:
	.loc	2 5 0
	movl	$42, %eax
	.loc	2 6 0
	addl	$10, %eax
	.loc	2 7 0
	retq
.Lfunc_end1:
	.size	common_inline_function, .Lfunc_end1-common_inline_function

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
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x30 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x10 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	2                               # Abbrev [2] 0x3a:0x10 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.long	.Linfo_string4                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0"
.Linfo_string1:
	.asciz	"/repo/llvm-project/bolt/test/Inputs/multi-cu-file1.c"
.Linfo_string2:
	.asciz	"/repo/llvm-project"
.Linfo_string3:
	.asciz	"main"
.Linfo_string4:
	.asciz	"common_inline_function"

	.section	.debug_line,"",@progbits
.Lline_table_start0:

;--- multi-cu-file2.s
	.text
	.file	1 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-file2.c"
	.file	2 "/repo/llvm-project" "bolt/test/Inputs/multi-cu-common.h"

	.globl	helper_function
	.type	helper_function,@function
helper_function:
.Lfunc_begin0:
	.loc	1 4 0
	callq	common_inline_function
	.loc	1 8 0
	retq
.Lfunc_end0:
	.size	helper_function, .Lfunc_end0-helper_function

	.type	common_inline_function,@function
common_inline_function:
.Lfunc_begin1:
	.loc	2 5 0
	movl	$42, %eax
	.loc	2 6 0
	addl	$10, %eax
	.loc	2 7 0
	retq
.Lfunc_end1:
	.size	common_inline_function, .Lfunc_end1-common_inline_function

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
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x30 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x10 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	2                               # Abbrev [2] 0x3a:0x10 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.long	.Linfo_string4                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0"
.Linfo_string1:
	.asciz	"/repo/llvm-project/bolt/test/Inputs/multi-cu-file2.c"
.Linfo_string2:
	.asciz	"/repo/llvm-project"
.Linfo_string3:
	.asciz	"helper_function"
.Linfo_string4:
	.asciz	"common_inline_function"

	.section	.debug_line,"",@progbits
.Lline_table_start0:

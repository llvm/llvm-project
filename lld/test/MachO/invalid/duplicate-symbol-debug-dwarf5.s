# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t-dup.o
# RUN: not %lld -dylib -o /dev/null %t.o %t-dup.o 2>&1 | FileCheck %s -DFILE_1=%t.o -DFILE_2=%t-dup.o

## Check that ld64.lld prints source locations of both functions and variables
## when using dwarf-5.

# CHECK:      error: duplicate symbol: _f
# CHECK-NEXT: >>> defined in v.c:3 (.{{[/\\]}}v.c:3)
# CHECK-NEXT: >>>            [[FILE_1]]
# CHECK-NEXT: >>> defined in v.c:3 (.{{[/\\]}}v.c:3)
# CHECK-NEXT: >>>            [[FILE_2]]

# CHECK:      error: duplicate symbol: _x
# CHECK-NEXT: >>> defined in v.c:1 (.{{[/\\]}}v.c:1)
# CHECK-NEXT: >>>            [[FILE_1]]

## Everything below is the unmodified output of
##   clang --target=x86_64-apple-macos11 -gdwarf-5 -O0 -S v.c -fno-ident \
##     -fdebug-prefix-map=$PWD=.
## for this v.c:
##   int x = 1;
##
##   int f(void) {
##     return x;
##   }

	.build_version macos, 11, 0
	.section	__TEXT,__text,regular,pure_instructions
	.att_syntax
	.file	0 "." "v.c" md5 0x0b77d92fd01660d83743038538e2318e
	.globl	_f                              ## -- Begin function f
	.p2align	4
_f:                                     ## @f
Lfunc_begin0:
	.loc	0 3 0                           ## v.c:3:0
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
Ltmp0:
	.loc	0 4 10 prologue_end             ## v.c:4:10
	movl	_x(%rip), %eax
	.loc	0 4 3 epilogue_begin is_stmt 0  ## v.c:4:3
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DATA,__data
	.globl	_x                              ## @x
	.p2align	2, 0x0
_x:
	.long	1                               ## 0x1

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ## Abbreviation Code
	.byte	17                              ## DW_TAG_compile_unit
	.byte	1                               ## DW_CHILDREN_yes
	.byte	37                              ## DW_AT_producer
	.byte	37                              ## DW_FORM_strx1
	.byte	19                              ## DW_AT_language
	.byte	5                               ## DW_FORM_data2
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.ascii	"\202|"                         ## DW_AT_LLVM_sysroot
	.byte	37                              ## DW_FORM_strx1
	.byte	114                             ## DW_AT_str_offsets_base
	.byte	23                              ## DW_FORM_sec_offset
	.byte	16                              ## DW_AT_stmt_list
	.byte	23                              ## DW_FORM_sec_offset
	.byte	27                              ## DW_AT_comp_dir
	.byte	37                              ## DW_FORM_strx1
	.byte	17                              ## DW_AT_low_pc
	.byte	27                              ## DW_FORM_addrx
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	115                             ## DW_AT_addr_base
	.byte	23                              ## DW_FORM_sec_offset
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	2                               ## Abbreviation Code
	.byte	46                              ## DW_TAG_subprogram
	.byte	0                               ## DW_CHILDREN_no
	.byte	17                              ## DW_AT_low_pc
	.byte	27                              ## DW_FORM_addrx
	.byte	18                              ## DW_AT_high_pc
	.byte	6                               ## DW_FORM_data4
	.byte	64                              ## DW_AT_frame_base
	.byte	24                              ## DW_FORM_exprloc
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	39                              ## DW_AT_prototyped
	.byte	25                              ## DW_FORM_flag_present
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	3                               ## Abbreviation Code
	.byte	52                              ## DW_TAG_variable
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	73                              ## DW_AT_type
	.byte	19                              ## DW_FORM_ref4
	.byte	63                              ## DW_AT_external
	.byte	25                              ## DW_FORM_flag_present
	.byte	58                              ## DW_AT_decl_file
	.byte	11                              ## DW_FORM_data1
	.byte	59                              ## DW_AT_decl_line
	.byte	11                              ## DW_FORM_data1
	.byte	2                               ## DW_AT_location
	.byte	24                              ## DW_FORM_exprloc
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	4                               ## Abbreviation Code
	.byte	36                              ## DW_TAG_base_type
	.byte	0                               ## DW_CHILDREN_no
	.byte	3                               ## DW_AT_name
	.byte	37                              ## DW_FORM_strx1
	.byte	62                              ## DW_AT_encoding
	.byte	11                              ## DW_FORM_data1
	.byte	11                              ## DW_AT_byte_size
	.byte	11                              ## DW_FORM_data1
	.byte	0                               ## EOM(1)
	.byte	0                               ## EOM(2)
	.byte	0                               ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	5                               ## DWARF version number
	.byte	1                               ## DWARF Unit Type
	.byte	8                               ## Address Size (in bytes)
Lset1 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	1                               ## Abbrev [1] 0xc:0x37 DW_TAG_compile_unit
	.byte	0                               ## DW_AT_producer
	.short	29                              ## DW_AT_language
	.byte	1                               ## DW_AT_name
	.byte	2                               ## DW_AT_LLVM_sysroot
Lset2 = Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
	.long	Lset2
Lset3 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset3
	.byte	3                               ## DW_AT_comp_dir
	.byte	0                               ## DW_AT_low_pc
Lset4 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset4
Lset5 = Laddr_table_base0-Lsection_info0 ## DW_AT_addr_base
	.long	Lset5
	.byte	2                               ## Abbrev [2] 0x24:0xf DW_TAG_subprogram
	.byte	0                               ## DW_AT_low_pc
Lset6 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset6
	.byte	1                               ## DW_AT_frame_base
	.byte	86
	.byte	6                               ## DW_AT_name
	.byte	0                               ## DW_AT_decl_file
	.byte	3                               ## DW_AT_decl_line
                                        ## DW_AT_prototyped
	.long	62                              ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                               ## Abbrev [3] 0x33:0xb DW_TAG_variable
	.byte	4                               ## DW_AT_name
	.long	62                              ## DW_AT_type
                                        ## DW_AT_external
	.byte	0                               ## DW_AT_decl_file
	.byte	1                               ## DW_AT_decl_line
	.byte	2                               ## DW_AT_location
	.byte	161
	.byte	1
	.byte	4                               ## Abbrev [4] 0x3e:0x4 DW_TAG_base_type
	.byte	5                               ## DW_AT_name
	.byte	5                               ## DW_AT_encoding
	.byte	4                               ## DW_AT_byte_size
	.byte	0                               ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	32                              ## Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.byte	0                               ## string offset=0 ; 
	.asciz	"v.c"                           ## string offset=1 ; v.c
	.asciz	"/"                             ## string offset=5 ; /
	.asciz	"."                             ## string offset=7 ; .
	.asciz	"f"                             ## string offset=9 ; f
	.asciz	"x"                             ## string offset=11 ; x
	.asciz	"int"                           ## string offset=13 ; int
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	1
	.long	5
	.long	7
	.long	11
	.long	13
	.long	9
	.section	__DWARF,__debug_addr,regular,debug
Lsection_info0:
Lset7 = Ldebug_addr_end0-Ldebug_addr_start0 ## Length of contribution
	.long	Lset7
Ldebug_addr_start0:
	.short	5                               ## DWARF version number
	.byte	8                               ## Address size
	.byte	0                               ## Segment selector size
Laddr_table_base0:
	.quad	Lfunc_begin0
	.quad	_x
Ldebug_addr_end0:
	.section	__DWARF,__debug_names,regular,debug
Ldebug_names_begin:
Lset8 = Lnames_end0-Lnames_start0       ## Header: unit length
	.long	Lset8
Lnames_start0:
	.short	5                               ## Header: version
	.short	0                               ## Header: padding
	.long	1                               ## Header: compilation unit count
	.long	0                               ## Header: local type unit count
	.long	0                               ## Header: foreign type unit count
	.long	3                               ## Header: bucket count
	.long	3                               ## Header: name count
Lset9 = Lnames_abbrev_end0-Lnames_abbrev_start0 ## Header: abbreviation table size
	.long	Lset9
	.long	8                               ## Header: augmentation string size
	.ascii	"LLVM0700"                      ## Header: augmentation string
Lset10 = Lcu_begin0-Lsection_info       ## Compilation unit 0
	.long	Lset10
	.long	1                               ## Bucket 0
	.long	0                               ## Bucket 1
	.long	3                               ## Bucket 2
	.long	177675                          ## Hash in Bucket 0
	.long	177693                          ## Hash in Bucket 0
	.long	193495088                       ## Hash in Bucket 2
	.long	9                               ## String in Bucket 0: f
	.long	11                              ## String in Bucket 0: x
	.long	13                              ## String in Bucket 2: int
Lset11 = Lnames0-Lnames_entries0        ## Offset in Bucket 0
	.long	Lset11
Lset12 = Lnames1-Lnames_entries0        ## Offset in Bucket 0
	.long	Lset12
Lset13 = Lnames2-Lnames_entries0        ## Offset in Bucket 2
	.long	Lset13
Lnames_abbrev_start0:
	.byte	1                               ## Abbrev code
	.byte	46                              ## DW_TAG_subprogram
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	2                               ## Abbrev code
	.byte	52                              ## DW_TAG_variable
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	3                               ## Abbrev code
	.byte	36                              ## DW_TAG_base_type
	.byte	3                               ## DW_IDX_die_offset
	.byte	19                              ## DW_FORM_ref4
	.byte	4                               ## DW_IDX_parent
	.byte	25                              ## DW_FORM_flag_present
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev
	.byte	0                               ## End of abbrev list
Lnames_abbrev_end0:
Lnames_entries0:
Lnames0:
	.byte	1                               ## Abbreviation code
	.long	36                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: f
Lnames1:
	.byte	2                               ## Abbreviation code
	.long	51                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: x
Lnames2:
	.byte	3                               ## Abbreviation code
	.long	62                              ## DW_IDX_die_offset
	.byte	0                               ## DW_IDX_parent
                                        ## End of list: int
	.p2align	2, 0x0
Lnames_end0:
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:

	.text
	.file	"debug-fission-simple.cpp"
	.file	1 "" "debug-fission-simple.cpp"
	.section	.text._Z7doStuffi,"ax",@progbits
	.globl	_Z7doStuffi                     # -- Begin function _Z7doStuffi
	.p2align	4, 0x90
	.type	_Z7doStuffi,@function
_Z7doStuffi:                            # @_Z7doStuffi
.Lfunc_begin0:
	.loc	1 3 0                           # debug-fission-simple.cpp:3:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	1 4 11 prologue_end             # debug-fission-simple.cpp:4:11
	cmpl	$5, -4(%rbp)
.Ltmp1:
	.loc	1 4 7 is_stmt 0                 # debug-fission-simple.cpp:4:7
	jne	.LBB0_2
# %bb.1:                                # %if.then
.Ltmp2:
	.loc	1 5 16 is_stmt 1                # debug-fission-simple.cpp:5:16
	movl	_ZL3foo, %eax
	.loc	1 5 14 is_stmt 0                # debug-fission-simple.cpp:5:14
	addl	$1, %eax
	.loc	1 5 9                           # debug-fission-simple.cpp:5:9
	addl	-4(%rbp), %eax
	movl	%eax, -4(%rbp)
	.loc	1 5 5                           # debug-fission-simple.cpp:5:5
	jmp	.LBB0_3
.LBB0_2:                                # %if.else
	.loc	1 7 9 is_stmt 1                 # debug-fission-simple.cpp:7:9
	movl	-4(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -4(%rbp)
.Ltmp3:
.LBB0_3:                                # %if.end
	.loc	1 8 10                          # debug-fission-simple.cpp:8:10
	movl	-4(%rbp), %eax
	.loc	1 8 3 is_stmt 0                 # debug-fission-simple.cpp:8:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp4:
.Lfunc_end0:
	.size	_Z7doStuffi, .Lfunc_end0-_Z7doStuffi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z8doStuff2i,"ax",@progbits
	.globl	_Z8doStuff2i                    # -- Begin function _Z8doStuff2i
	.p2align	4, 0x90
	.type	_Z8doStuff2i,@function
_Z8doStuff2i:                           # @_Z8doStuff2i
.Lfunc_begin1:
	.loc	1 11 0 is_stmt 1                # debug-fission-simple.cpp:11:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp5:
	.loc	1 12 14 prologue_end            # debug-fission-simple.cpp:12:14
	movl	-4(%rbp), %eax
	addl	$3, %eax
	movl	%eax, -4(%rbp)
	.loc	1 12 3 is_stmt 0                # debug-fission-simple.cpp:12:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp6:
.Lfunc_end1:
	.size	_Z8doStuff2i, .Lfunc_end1-_Z8doStuff2i
	.cfi_endproc
                                        # -- End function
	.section	.text._Z6_startv,"ax",@progbits
	.globl	_Z6_startv                      # -- Begin function _Z6_startv
	.p2align	4, 0x90
	.type	_Z6_startv,@function
_Z6_startv:                             # @_Z6_startv
.Lfunc_begin2:
	.loc	1 15 0 is_stmt 1                # debug-fission-simple.cpp:15:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
.Ltmp7:
	.loc	1 16 7 prologue_end             # debug-fission-simple.cpp:16:7
	movl	$4, -4(%rbp)
	.loc	1 17 18                         # debug-fission-simple.cpp:17:18
	movl	-4(%rbp), %edi
	.loc	1 17 10 is_stmt 0               # debug-fission-simple.cpp:17:10
	callq	_Z7doStuffi
	.loc	1 17 3                          # debug-fission-simple.cpp:17:3
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp8:
.Lfunc_end2:
	.size	_Z6_startv, .Lfunc_end2-_Z6_startv
	.cfi_endproc
                                        # -- End function
	.type	_ZL3foo,@object                 # @_ZL3foo
	.data
	.p2align	2
_ZL3foo:
	.long	2                               # 0x2
	.size	_ZL3foo, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.ascii	"\263B"                         # DW_AT_GNU_addr_base
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	1                               # Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	436953012669069206              # DW_AT_GNU_dwo_id
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"" # string offset=0
.Lskel_string1:
	.asciz	"debug-fission-simple-convert.dwo"      # string offset=47
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"foo"                           # string offset=0
.Linfo_string1:
	.asciz	"int"                           # string offset=4
.Linfo_string2:
	.asciz	"_ZL3foo"                       # string offset=8
.Linfo_string3:
	.asciz	"_Z7doStuffi"                   # string offset=16
.Linfo_string4:
	.asciz	"doStuff"                       # string offset=28
.Linfo_string5:
	.asciz	"_Z8doStuff2i"                  # string offset=36
.Linfo_string6:
	.asciz	"doStuff2"                      # string offset=49
.Linfo_string7:
	.asciz	"_Z6_startv"                    # string offset=58
.Linfo_string8:
	.asciz	"_start"                        # string offset=69
.Linfo_string9:
	.asciz	"val"                           # string offset=76
.Linfo_string10:
	.asciz	"clang version 13.0.0" # string offset=80
.Linfo_string11:
	.asciz	"debug-fission-simple.cpp"      # string offset=214
.Linfo_string12:
	.asciz	"debug-fission-simple-convert.dwo"      # string offset=239
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	4
	.long	8
	.long	16
	.long	28
	.long	36
	.long	49
	.long	58
	.long	69
	.long	76
	.long	80
	.long	214
	.long	239
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x73 DW_TAG_compile_unit
	.byte	10                              # DW_AT_producer
	.short	4                               # DW_AT_language
	.byte	11                              # DW_AT_name
	.byte	12                              # DW_AT_GNU_dwo_name
	.quad	436953012669069206              # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0xc DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	37                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	0
	.byte	2                               # DW_AT_linkage_name
	.byte	3                               # Abbrev [3] 0x25:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x29:0x1c DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	37                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x39:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	37                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x45:0x1c DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	5                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	37                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x55:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	37                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x61:0x1c DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	7                               # DW_AT_linkage_name
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	37                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x71:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.long	37                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
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
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	_ZL3foo
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_begin2
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	25                              # DIE offset
	.byte	160                             # Attributes: VARIABLE, STATIC
	.asciz	"foo"                           # External Name
	.long	41                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuff"                       # External Name
	.long	69                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuff2"                      # External Name
	.long	97                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"_start"                        # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	37                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 13"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z7doStuffi
	.addrsig_sym _ZL3foo
	.section	.debug_line,"",@progbits
.Lline_table_start0:

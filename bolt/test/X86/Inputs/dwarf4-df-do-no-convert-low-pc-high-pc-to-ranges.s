	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "." "main.cpp"
	.loc	1 5 0                           # main.cpp:5:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movl	%edi, %ebx
.Ltmp0:
	.loc	1 6 7 prologue_end              # main.cpp:6:7
	cmpq	$0, _Z13may_not_existv@GOTPCREL(%rip)
	je	.LBB0_2
.Ltmp1:
# %bb.1:                                # %if.then
	#DEBUG_VALUE: main:argc <- $ebx
	#DEBUG_VALUE: main:argv <- $rsi
	.loc	1 7 5                           # main.cpp:7:5
	callq	_Z13may_not_existv@PLT
.Ltmp2:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
.LBB0_2:                                # %if.end
	#DEBUG_VALUE: main:argc <- $ebx
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	#DEBUG_VALUE: main:j <- 0
	#DEBUG_VALUE: helper:i <- $ebx
	.loc	1 2 10                          # main.cpp:2:10
	incl	%ebx
.Ltmp3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: helper:i <- $ebx
	#DEBUG_VALUE: main:j <- $ebx
	.loc	1 10 3                          # main.cpp:10:3
	movl	%ebx, %eax
	.loc	1 10 3 epilogue_begin is_stmt 0 # main.cpp:10:3
	popq	%rbx
.Ltmp4:
	#DEBUG_VALUE: helper:i <- $eax
	#DEBUG_VALUE: main:j <- $eax
	.cfi_def_cfa_offset 8
	retq
.Ltmp5:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_loc.dwo,"e",@progbits
.Ldebug_loc0:
	.byte	3
	.byte	0
	.long	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	3
	.byte	2
	.long	.Ltmp3-.Ltmp1
	.short	1                               # Loc expr size
	.byte	83                              # super-register DW_OP_reg3
	.byte	3
	.byte	3
	.long	.Lfunc_end0-.Ltmp3
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	0
.Ldebug_loc1:
	.byte	3
	.byte	0
	.long	.Ltmp2-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	3
	.byte	1
	.long	.Lfunc_end0-.Ltmp2
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0
.Ldebug_loc2:
	.byte	3
	.byte	1
	.long	.Ltmp3-.Ltmp2
	.short	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	0                               # 0
	.byte	159                             # DW_OP_stack_value
	.byte	3
	.byte	3
	.long	.Ltmp4-.Ltmp3
	.short	1                               # Loc expr size
	.byte	83                              # super-register DW_OP_reg3
	.byte	3
	.byte	4
	.long	.Lfunc_end0-.Ltmp4
	.short	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.byte	0
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
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
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
	.quad	-5076722043903325778            # DW_AT_GNU_dwo_id
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=38
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_ZL6helperi"                   # string offset=0
.Linfo_string1:
	.asciz	"helper"                        # string offset=12
.Linfo_string2:
	.asciz	"int"                           # string offset=19
.Linfo_string3:
	.asciz	"i"                             # string offset=23
.Linfo_string4:
	.asciz	"_Z13may_not_existv"            # string offset=25
.Linfo_string5:
	.asciz	"may_not_exist"                 # string offset=44
.Linfo_string6:
	.asciz	"main"                          # string offset=58
.Linfo_string7:
	.asciz	"argc"                          # string offset=63
.Linfo_string8:
	.asciz	"argv"                          # string offset=68
.Linfo_string9:
	.asciz	"char"                          # string offset=73
.Linfo_string10:
	.asciz	"j"                             # string offset=78
.Linfo_string11:
	.asciz	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 640e07c49037cca41a1bfbeb916b569d8c950aea)" # string offset=80
.Linfo_string12:
	.asciz	"main.cpp"                      # string offset=185
.Linfo_string13:
	.asciz	"main.dwo"                      # string offset=194
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	12
	.long	19
	.long	23
	.long	25
	.long	44
	.long	58
	.long	63
	.long	68
	.long	73
	.long	78
	.long	80
	.long	185
	.long	194
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x88 DW_TAG_compile_unit
	.byte	11                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	12                              # DW_AT_name
	.byte	13                              # DW_AT_GNU_dwo_name
	.quad	-5076722043903325778            # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0x13 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	1                               # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x23:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x2c:0x4 DW_TAG_base_type
	.byte	2                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x30:0x4f DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x3f:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc0-.debug_loc.dwo     # DW_AT_location
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x4b:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc1-.debug_loc.dwo     # DW_AT_location
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	132                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x57:0xc DW_TAG_variable
	.long	.Ldebug_loc2-.debug_loc.dwo     # DW_AT_location
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x63:0x15 DW_TAG_inlined_subroutine
	.long	25                              # DW_AT_abstract_origin
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	9                               # DW_AT_call_line
	.byte	32                              # DW_AT_call_column
	.byte	9                               # Abbrev [9] 0x70:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	83
	.long	35                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x78:0x6 DW_TAG_GNU_call_site
	.long	127                             # DW_AT_abstract_origin
	.byte	1                               # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x7f:0x5 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	12                              # Abbrev [12] 0x84:0x5 DW_TAG_pointer_type
	.long	137                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x89:0x5 DW_TAG_pointer_type
	.long	142                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x8e:0x4 DW_TAG_base_type
	.byte	9                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
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
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
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
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
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
	.byte	6                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	7                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	8                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	110                             # DW_AT_linkage_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp2
	.quad	.Ltmp1
	.quad	.Ltmp3
	.quad	.Ltmp4
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	48                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	25                              # DIE offset
	.byte	176                             # Attributes: FUNCTION, STATIC
	.asciz	"helper"                        # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	44                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	142                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.weak	_Z13may_not_existv
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 640e07c49037cca41a1bfbeb916b569d8c950aea)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z13may_not_existv
	.section	.debug_line,"",@progbits
.Lline_table_start0:

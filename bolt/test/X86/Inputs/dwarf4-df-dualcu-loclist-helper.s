# clang++ -g -gdwarf-4 -emit-llvm -S helper.cpp
# llc -O2 -mtriple=x86_64-unknown-linux-gnu helper.ll
# int z = 0;
# int d = 0;
#
# int helper(int z_, int d_) {
#  z += z_;
#  d += d_;
#  return z * d;
# }

	.text
	.file	"helper.cpp"
	.file	1 "." "helper.cpp"
	.globl	_Z6helperii                     # -- Begin function _Z6helperii
	.p2align	4, 0x90
	.type	_Z6helperii,@function
_Z6helperii:                            # @_Z6helperii
.Lfunc_begin0:
	.loc	1 4 0                           # helper.cpp:4:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: helper:z_ <- $edi
	#DEBUG_VALUE: helper:d_ <- $esi
	movl	%esi, %eax
.Ltmp0:
	.loc	1 5 4 prologue_end              # helper.cpp:5:4
	addl	z(%rip), %edi
.Ltmp1:
	#DEBUG_VALUE: helper:z_ <- [DW_OP_LLVM_entry_value 1] $edi
	.loc	1 6 4                           # helper.cpp:6:4
	addl	d(%rip), %eax
	.loc	1 5 4                           # helper.cpp:5:4
	movl	%edi, z(%rip)
	.loc	1 6 4                           # helper.cpp:6:4
	movl	%eax, d(%rip)
	.loc	1 7 11                          # helper.cpp:7:11
	imull	%edi, %eax
	.loc	1 7 2 is_stmt 0                 # helper.cpp:7:2
	retq
.Ltmp2:
.Lfunc_end0:
	.size	_Z6helperii, .Lfunc_end0-_Z6helperii
	.cfi_endproc
                                        # -- End function
	.type	z,@object                       # @z
	.bss
	.globl	z
	.p2align	2
z:
	.long	0                               # 0x0
	.size	z, 4

	.type	d,@object                       # @d
	.globl	d
	.p2align	2
d:
	.long	0                               # 0x0
	.size	d, 4

	.section	.debug_loc.dwo,"e",@progbits
.Ldebug_loc0:
	.byte	3
	.byte	2
	.long	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	3
	.byte	3
	.long	.Lfunc_end0-.Ltmp1
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
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
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	-5732284259998169791            # DW_AT_GNU_dwo_id
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"."                             # string offset=0
.Lskel_string1:
	.asciz	"helper.dwo"                    # string offset=2
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"z"                             # string offset=0
.Linfo_string1:
	.asciz	"int"                           # string offset=2
.Linfo_string2:
	.asciz	"d"                             # string offset=6
.Linfo_string3:
	.asciz	"_Z6helperii"                   # string offset=8
.Linfo_string4:
	.asciz	"helper"                        # string offset=20
.Linfo_string5:
	.asciz	"z_"                            # string offset=27
.Linfo_string6:
	.asciz	"d_"                            # string offset=30
.Linfo_string7:
	.asciz	"clang version 15.0.0"          # string offset=33
.Linfo_string8:
	.asciz	"helper.cpp"                    # string offset=54
.Linfo_string9:
	.asciz	"helper.dwo"                    # string offset=65
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	2
	.long	6
	.long	8
	.long	20
	.long	27
	.long	30
	.long	33
	.long	54
	.long	65
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
	.byte	7                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	8                               # DW_AT_name
	.byte	9                               # DW_AT_GNU_dwo_name
	.quad	-5732284259998169791            # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0xb DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	0
	.byte	3                               # Abbrev [3] 0x24:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x28:0xb DW_TAG_variable
	.byte	2                               # DW_AT_name
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	1
	.byte	4                               # Abbrev [4] 0x33:0x27 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x43:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc0-.debug_loc.dwo     # DW_AT_location
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x4f:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
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
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
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
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
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
	.byte	6                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	z
	.quad	d
	.quad	.Lfunc_begin0
	.quad	.Ltmp1
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:

# clang++ helper2.cpp -O2 -g2 -gdwarf-4 -S -o helper2.s
# int z1 = 0;
# int d1 = 0;
#
# int helper31(int z_, int d_) {
#  z1 += z_;
#  d1 += d_;
#  return z1 * d1;
# }
#
#
# int z = 0;
# int d = 0;
#
# int helper3(int z_, int d_) {
#  z += z_;
#  d += d_;
#  return z * d;
# }

	.text
	.file	"helper2.cpp"
	.file	1 "/dwarf4CrossCULocList" "helper2.cpp"
	.globl	_Z8helper31ii                   # -- Begin function _Z8helper31ii
	.p2align	4, 0x90
	.type	_Z8helper31ii,@function
_Z8helper31ii:                          # @_Z8helper31ii
.Lfunc_begin0:
	.loc	1 4 0                           # helper2.cpp:4:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: helper31:z_ <- $edi
	#DEBUG_VALUE: helper31:d_ <- $esi
	movl	%esi, %eax
.Ltmp0:
	.loc	1 5 5 prologue_end              # helper2.cpp:5:5
	addl	z1(%rip), %edi
.Ltmp1:
	#DEBUG_VALUE: helper31:z_ <- [DW_OP_LLVM_entry_value 1] $edi
	.loc	1 6 5                           # helper2.cpp:6:5
	addl	d1(%rip), %eax
	.loc	1 5 5                           # helper2.cpp:5:5
	movl	%edi, z1(%rip)
	.loc	1 6 5                           # helper2.cpp:6:5
	movl	%eax, d1(%rip)
	.loc	1 7 12                          # helper2.cpp:7:12
	imull	%edi, %eax
	.loc	1 7 2 is_stmt 0                 # helper2.cpp:7:2
	retq
.Ltmp2:
.Lfunc_end0:
	.size	_Z8helper31ii, .Lfunc_end0-_Z8helper31ii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper3ii                    # -- Begin function _Z7helper3ii
	.p2align	4, 0x90
	.type	_Z7helper3ii,@function
_Z7helper3ii:                           # @_Z7helper3ii
.Lfunc_begin1:
	.loc	1 14 0 is_stmt 1                # helper2.cpp:14:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: helper3:z_ <- $edi
	#DEBUG_VALUE: helper3:d_ <- $esi
	movl	%esi, %eax
.Ltmp3:
	.loc	1 15 4 prologue_end             # helper2.cpp:15:4
	addl	z(%rip), %edi
.Ltmp4:
	#DEBUG_VALUE: helper3:z_ <- [DW_OP_LLVM_entry_value 1] $edi
	.loc	1 16 4                          # helper2.cpp:16:4
	addl	d(%rip), %eax
	.loc	1 15 4                          # helper2.cpp:15:4
	movl	%edi, z(%rip)
	.loc	1 16 4                          # helper2.cpp:16:4
	movl	%eax, d(%rip)
	.loc	1 17 11                         # helper2.cpp:17:11
	imull	%edi, %eax
	.loc	1 17 2 is_stmt 0                # helper2.cpp:17:2
	retq
.Ltmp5:
.Lfunc_end1:
	.size	_Z7helper3ii, .Lfunc_end1-_Z7helper3ii
	.cfi_endproc
                                        # -- End function
	.type	z1,@object                      # @z1
	.bss
	.globl	z1
	.p2align	2, 0x0
z1:
	.long	0                               # 0x0
	.size	z1, 4

	.type	d1,@object                      # @d1
	.globl	d1
	.p2align	2, 0x0
d1:
	.long	0                               # 0x0
	.size	d1, 4

	.type	z,@object                       # @z
	.globl	z
	.p2align	2, 0x0
z:
	.long	0                               # 0x0
	.size	z, 4

	.type	d,@object                       # @d
	.globl	d
	.p2align	2, 0x0
d:
	.long	0                               # 0x0
	.size	d, 4

	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Lfunc_begin1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
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
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	14                              # DW_FORM_strp
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
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
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
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	6                               # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xb:0xef DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	z1
	.byte	3                               # Abbrev [3] 0x3f:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x46:0x15 DW_TAG_variable
	.long	.Linfo_string5                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	d1
	.byte	2                               # Abbrev [2] 0x5b:0x15 DW_TAG_variable
	.long	.Linfo_string6                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	z
	.byte	2                               # Abbrev [2] 0x70:0x15 DW_TAG_variable
	.long	.Linfo_string7                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	d
	.byte	4                               # Abbrev [4] 0x85:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0xa2:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0xb1:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0xbf:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string10                 # DW_AT_linkage_name
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0xdc:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0xeb:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	63                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 37d6c1cc7d4dd3a8a47ba62254bc88521bd50d66)" # string offset=0
.Linfo_string1:
	.asciz	"helper2.cpp"                   # string offset=101
.Linfo_string2:
	.asciz	"/dwarf4CrossCULocList" # string offset=113
.Linfo_string3:
	.asciz	"z1"                            # string offset=172
.Linfo_string4:
	.asciz	"int"                           # string offset=175
.Linfo_string5:
	.asciz	"d1"                            # string offset=179
.Linfo_string6:
	.asciz	"z"                             # string offset=182
.Linfo_string7:
	.asciz	"d"                             # string offset=184
.Linfo_string8:
	.asciz	"_Z8helper31ii"                 # string offset=186
.Linfo_string9:
	.asciz	"helper31"                      # string offset=200
.Linfo_string10:
	.asciz	"_Z7helper3ii"                  # string offset=209
.Linfo_string11:
	.asciz	"helper3"                       # string offset=222
.Linfo_string12:
	.asciz	"z_"                            # string offset=230
.Linfo_string13:
	.asciz	"d_"                            # string offset=233
	.ident	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 37d6c1cc7d4dd3a8a47ba62254bc88521bd50d66)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

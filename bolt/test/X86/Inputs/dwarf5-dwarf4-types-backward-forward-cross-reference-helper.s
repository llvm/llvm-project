# helper1.cpp
# struct Foo3a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# struct Foo4 {
#   char *c1;
#   char *c2;
# };
#
# int foo2() {
#   Foo3a f;
#   Foo4 f2;
#   return 0;
# }

# helper2.cpp
# struct Foo4a {
#  char *c1;
#  char *c2;
#  char *c3;
# };
# struct Foo5 {
#   char *c1;
#   char *c2;
# };
#
# int foo3() {
#   Foo4a f;
#   Foo5 f2;
#   return 0;
# }



	.text
	.file	"llvm-link"
	.globl	_Z4foo2v                        # -- Begin function _Z4foo2v
	.p2align	4, 0x90
	.type	_Z4foo2v,@function
_Z4foo2v:                               # @_Z4foo2v
.Lfunc_begin0:
	.file	1 "/dwarf5-dwarf4-types-forward-cross-reference-test" "helper1.cpp"
	.loc	1 11 0                          # helper1.cpp:11:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	1 14 3 prologue_end             # helper1.cpp:14:3
	xorl	%eax, %eax
	.loc	1 14 3 epilogue_begin is_stmt 0 # helper1.cpp:14:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z4foo2v, .Lfunc_end0-_Z4foo2v
	.cfi_endproc
                                        # -- End function
	.globl	_Z4foo3v                        # -- Begin function _Z4foo3v
	.p2align	4, 0x90
	.type	_Z4foo3v,@function
_Z4foo3v:                               # @_Z4foo3v
.Lfunc_begin1:
	.file	2 "/dwarf5-dwarf4-types-forward-cross-reference-test" "helper2.cpp"
	.loc	2 11 0 is_stmt 1                # helper2.cpp:11:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp2:
	.loc	2 14 3 prologue_end             # helper2.cpp:14:3
	xorl	%eax, %eax
	.loc	2 14 3 epilogue_begin is_stmt 0 # helper2.cpp:14:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	_Z4foo3v, .Lfunc_end1-_Z4foo3v
	.cfi_endproc
                                        # -- End function
	.section	.debug_types,"G",@progbits,10693860647081617285,comdat
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-7752883426627934331            # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string14                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string12                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string13                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_types,"G",@progbits,17604755499357858397,comdat
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-841988574351693219             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string16                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string12                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_types,"G",@progbits,10955924554604642151,comdat
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-7490819519104909465            # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string17                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string12                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string13                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_types,"G",@progbits,5738727807022258601,comdat
	.long	.Ldebug_info_end3-.Ldebug_info_start3 # Length of Unit
.Ldebug_info_start3:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	5738727807022258601             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string18                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string12                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end3:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
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
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                               # Abbreviation Code
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
	.byte	16                              # DW_FORM_ref_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end4-.Ldebug_info_start4 # Length of Unit
.Ldebug_info_start4:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x73 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x2a:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string4                  # DW_AT_linkage_name
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	100                             # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x47:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	107                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x55:0xe DW_TAG_variable <-- Manually modified s/8/10
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string15                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	.Lmanual_label                  # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x64:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
.Lmanual_label_forward:
	.byte	9                               # Abbrev [9] 0x6b:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-7752883426627934331            # DW_AT_signature
	.byte	9                               # Abbrev [9] 0x74:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-841988574351693219             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end4:
.Lcu_begin1:
	.long	.Ldebug_info_end5-.Ldebug_info_start5 # Length of Unit
.Ldebug_info_start5:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x73 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string3                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x2a:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string7                  # DW_AT_linkage_name
	.long	.Linfo_string8                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	100                             # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x47:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string9                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	107                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x55:0xe DW_TAG_variable <-- Manually modified s/8/10
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string15                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	.Lmanual_label_forward                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x64:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
.Lmanual_label:
	.byte	9                               # Abbrev [9] 0x6b:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-7490819519104909465            # DW_AT_signature
	.byte	9                               # Abbrev [9] 0x74:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	5738727807022258601             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end5:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)" # string offset=0
.Linfo_string1:
	.asciz	"helper1.cpp"                   # string offset=105
.Linfo_string2:
	.asciz	"/dwarf5-dwarf4-types-forward-cross-reference-test" # string offset=117
.Linfo_string3:
	.asciz	"helper2.cpp"                   # string offset=204
.Linfo_string4:
	.asciz	"_Z4foo2v"                      # string offset=216
.Linfo_string5:
	.asciz	"foo2"                          # string offset=225
.Linfo_string6:
	.asciz	"int"                           # string offset=230
.Linfo_string7:
	.asciz	"_Z4foo3v"                      # string offset=234
.Linfo_string8:
	.asciz	"foo3"                          # string offset=243
.Linfo_string9:
	.asciz	"f"                             # string offset=248
.Linfo_string10:
	.asciz	"c1"                            # string offset=250
.Linfo_string11:
	.asciz	"char"                          # string offset=253
.Linfo_string12:
	.asciz	"c2"                            # string offset=258
.Linfo_string13:
	.asciz	"c3"                            # string offset=261
.Linfo_string14:
	.asciz	"Foo3a"                         # string offset=264
.Linfo_string15:
	.asciz	"f2"                            # string offset=270
.Linfo_string16:
	.asciz	"Foo4"                          # string offset=273
.Linfo_string17:
	.asciz	"Foo4a"                         # string offset=278
.Linfo_string18:
	.asciz	"Foo5"                          # string offset=284
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)"
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

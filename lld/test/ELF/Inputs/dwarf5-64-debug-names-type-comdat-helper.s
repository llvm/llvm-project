	.text
	.file	"helper.cpp"
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4, 0x90
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.file	0 "/home/ayermolo/local/tasks/T138552329/typeDedupSmall" "helper.cpp" md5 0x305ec66c221c583021f8375b300e2591
	.loc	0 2 0                           # helper.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 4 3 prologue_end              # helper.cpp:4:3
	xorl	%eax, %eax
	.loc	0 4 3 epilogue_begin is_stmt 0  # helper.cpp:4:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
                                        # -- End function
	.file	1 "." "header.h" md5 0x53699580704254cb1dd2a83230f8a7ea
	.section	.debug_info,"G",@progbits,1175092228111723119,comdat
.Ltu_begin0:
	.long	4294967295                      # DWARF64 Mark
	.quad	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.quad	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	1175092228111723119             # Type Signature
	.quad	59                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x28:0x2d DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.quad	.Lline_table_start0             # DW_AT_stmt_list
	.quad	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x3b:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	9                               # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x41:0x9 DW_TAG_member
	.byte	7                               # DW_AT_name
	.long	75                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4b:0x5 DW_TAG_pointer_type
	.long	80                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x50:0x4 DW_TAG_base_type
	.byte	8                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	37                              # DW_FORM_strx1
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
	.byte	37                              # DW_FORM_strx1
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
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	37                              # DW_FORM_strx1
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	4294967295                      # DWARF64 Mark
	.quad	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.quad	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	6                               # Abbrev [6] 0x18:0x4d DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.quad	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.quad	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.quad	.Laddr_table_base0              # DW_AT_addr_base
	.byte	7                               # Abbrev [7] 0x3b:0x1c DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	87                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x4b:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	91                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x57:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	9                               # Abbrev [9] 0x5b:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	1175092228111723119             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	4294967295                      # DWARF64 Mark
	.quad	84                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0 (git@github.com:ayermolo/llvm-project.git 2a059ae838c2e444f47dc1dcdfefb6fc876a53c1)" # string offset=0
.Linfo_string1:
	.asciz	"helper.cpp"                    # string offset=105
.Linfo_string2:
	.asciz	"/home/ayermolo/local/tasks/T138552329/typeDedupSmall" # string offset=116
.Linfo_string3:
	.asciz	"foo"                           # string offset=169
.Linfo_string4:
	.asciz	"_Z3foov"                       # string offset=173
.Linfo_string5:
	.asciz	"int"                           # string offset=181
.Linfo_string6:
	.asciz	"f"                             # string offset=185
.Linfo_string7:
	.asciz	"Foo2a"                         # string offset=187
.Linfo_string8:
	.asciz	"c1"                            # string offset=193
.Linfo_string9:
	.asciz	"char"                          # string offset=196
	.section	.debug_str_offsets,"",@progbits
	.quad	.Linfo_string0
	.quad	.Linfo_string1
	.quad	.Linfo_string2
	.quad	.Linfo_string4
	.quad	.Linfo_string3
	.quad	.Linfo_string5
	.quad	.Linfo_string6
	.quad	.Linfo_string8
	.quad	.Linfo_string9
	.quad	.Linfo_string7
	.section	.debug_addr,"",@progbits
	.long	4294967295                      # DWARF64 Mark
	.quad	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	4294967295                      # DWARF64 Mark
	.quad	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	1                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	5                               # Header: bucket count
	.long	5                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.quad	.Lcu_begin0                     # Compilation unit 0
	.quad	.Ltu_begin0                     # Type unit 0
	.long	0                               # Bucket 0
	.long	0                               # Bucket 1
	.long	0                               # Bucket 2
	.long	1                               # Bucket 3
	.long	2                               # Bucket 4
	.long	193495088                       # Hash in Bucket 3
	.long	193491849                       # Hash in Bucket 4
	.long	259227804                       # Hash in Bucket 4
	.long	2090147939                      # Hash in Bucket 4
	.long	-1257882357                     # Hash in Bucket 4
	.quad	.Linfo_string5                  # String in Bucket 3: int
	.quad	.Linfo_string3                  # String in Bucket 4: foo
	.quad	.Linfo_string7                  # String in Bucket 4: Foo2a
	.quad	.Linfo_string9                  # String in Bucket 4: char
	.quad	.Linfo_string4                  # String in Bucket 4: _Z3foov
	.quad	.Lnames2-.Lnames_entries0       # Offset in Bucket 3
	.quad	.Lnames0-.Lnames_entries0       # Offset in Bucket 4
	.quad	.Lnames3-.Lnames_entries0       # Offset in Bucket 4
	.quad	.Lnames4-.Lnames_entries0       # Offset in Bucket 4
	.quad	.Lnames1-.Lnames_entries0       # Offset in Bucket 4
.Lnames_abbrev_start0:
	.ascii	"\350\004"                      # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\354\004"                      # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\210\t"                        # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\310\013"                      # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\214\t"                        # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames2:
	.ascii	"\210\t"                        # Abbreviation code
	.long	87                              # DW_IDX_die_offset
	.byte	0                               # End of list: int
.Lnames0:
	.ascii	"\310\013"                      # Abbreviation code
	.long	59                              # DW_IDX_die_offset
	.byte	0                               # End of list: foo
.Lnames3:
	.ascii	"\354\004"                      # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	59                              # DW_IDX_die_offset
	.ascii	"\350\004"                      # Abbreviation code
	.long	91                              # DW_IDX_die_offset
	.byte	0                               # End of list: Foo2a
.Lnames4:
	.ascii	"\214\t"                        # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	80                              # DW_IDX_die_offset
	.byte	0                               # End of list: char
.Lnames1:
	.ascii	"\310\013"                      # Abbreviation code
	.long	59                              # DW_IDX_die_offset
	.byte	0                               # End of list: _Z3foov
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 18.0.0 (git@github.com:ayermolo/llvm-project.git 2a059ae838c2e444f47dc1dcdfefb6fc876a53c1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

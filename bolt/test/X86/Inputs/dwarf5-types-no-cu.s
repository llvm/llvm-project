# clang++ helper.cpp -g2 -gdwarf-5 -gno-pubnames -fdebug-types-section -S -o helperTypes.s
# struct Foo1 {
#   char a1;
#   char a2;
#   char a3;
# };
#
# struct Foo2 {
#   int b1;
#   int b2;
# };
#
# Foo1 f1;
# Foo2 f2;

# Manually removed Compile Unit .debug_info section.

	.text
	.file	"helper.cpp"
	.file	0 "/home" "helper.cpp" md5 0xd58ef77d520bf2e6491a2e387a3501f1
	.section	.debug_info,"G",@progbits,5391472263833448044,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	5391472263833448044             # Type Signature
	.long	35                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x32 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	8                               # DW_AT_name
	.byte	3                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x29:0x9 DW_TAG_member
	.byte	4                               # DW_AT_name
	.long	69                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x32:0x9 DW_TAG_member
	.byte	6                               # DW_AT_name
	.long	69                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	1                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3b:0x9 DW_TAG_member
	.byte	7                               # DW_AT_name
	.long	69                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	2                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x45:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_info,"G",@progbits,5322170643381124694,comdat
.Ltu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	5322170643381124694             # Type Signature
	.long	35                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x29 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x19 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	13                              # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x29:0x9 DW_TAG_member
	.byte	10                              # DW_AT_name
	.long	60                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x32:0x9 DW_TAG_member
	.byte	12                              # DW_AT_name
	.long	60                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.byte	4                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x3c:0x4 DW_TAG_base_type
	.byte	11                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.type	f1,@object                      # @f1
	.bss
	.globl	f1
f1:
	.zero	3
	.size	f1, 3

	.type	f2,@object                      # @f2
	.globl	f2
	.p2align	2, 0x0
f2:
	.zero	8
	.size	f2, 8

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
	.byte	5                               # Abbreviation Code
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
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	7                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_str_offsets,"",@progbits
	.long	60                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0git (git@github.com:llvm/llvm-project.git 44dc1e0baae7c4b8a02ba06dcf396d3d452aa873)" # string offset=0
.Linfo_string1:
	.asciz	"helper.cpp"                    # string offset=104
.Linfo_string2:
	.asciz	"/home" # string offset=115
.Linfo_string3:
	.asciz	"f1"                            # string offset=153
.Linfo_string4:
	.asciz	"a1"                            # string offset=156
.Linfo_string5:
	.asciz	"char"                          # string offset=159
.Linfo_string6:
	.asciz	"a2"                            # string offset=164
.Linfo_string7:
	.asciz	"a3"                            # string offset=167
.Linfo_string8:
	.asciz	"Foo1"                          # string offset=170
.Linfo_string9:
	.asciz	"f2"                            # string offset=175
.Linfo_string10:
	.asciz	"b1"                            # string offset=178
.Linfo_string11:
	.asciz	"int"                           # string offset=181
.Linfo_string12:
	.asciz	"b2"                            # string offset=185
.Linfo_string13:
	.asciz	"Foo2"                          # string offset=188
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	f1
	.quad	f2
.Ldebug_addr_end0:
	.ident	"clang version 18.0.0git (git@github.com:llvm/llvm-project.git 44dc1e0baae7c4b8a02ba06dcf396d3d452aa873)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

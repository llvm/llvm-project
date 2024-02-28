# clang++ -g2 -gdwarf-5 -gpubnames -fdebug-types-section -S
# header.h
# struct Foo2a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# helper.cpp
# #include "header.h"
# int fooint;
# struct Foo2Int {
#    int *c1;
#    int *c2;
# };
#
# int foo() {
#   Foo2Int fint;
#   Foo2a f;
#   return 0;
# }

	.text
	.file	"helper.cpp"
	.file	0 "/typeDedup" "helper.cpp" md5 0xc33186b2db66a78883b1546aace9855d
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4, 0x90
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.loc	0 8 0                           # helper.cpp:8:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 11 3 prologue_end             # helper.cpp:11:3
	xorl	%eax, %eax
	.loc	0 11 3 epilogue_begin is_stmt 0 # helper.cpp:11:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
                                        # -- End function
	.type	fooint,@object                  # @fooint
	.bss
	.globl	fooint
	.p2align	2, 0x0
fooint:
	.long	0                               # 0x0
	.size	fooint, 4

	.file	1 "." "header.h" md5 0xfea7bb1f22c47f129e15695f7137a1e7
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
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
	.byte	2                               # Abbreviation Code
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
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x97 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0xb DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	46                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x2e:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x32:0x27 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	5                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	46                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x42:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	89                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x4d:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	119                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x59:0x19 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	10                              # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	7                               # Abbrev [7] 0x5f:0x9 DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	114                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	7                               # Abbrev [7] 0x68:0x9 DW_TAG_member
	.byte	9                               # DW_AT_name
	.long	114                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x72:0x5 DW_TAG_pointer_type
	.long	46                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x77:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	14                              # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	7                               # Abbrev [7] 0x7d:0x9 DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	153                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	7                               # Abbrev [7] 0x86:0x9 DW_TAG_member
	.byte	9                               # DW_AT_name
	.long	153                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	7                               # Abbrev [7] 0x8f:0x9 DW_TAG_member
	.byte	13                              # DW_AT_name
	.long	153                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x99:0x5 DW_TAG_pointer_type
	.long	158                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x9e:0x4 DW_TAG_base_type
	.byte	12                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	64                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"helper.cpp"                    # string offset=24
.Linfo_string2:
	.asciz	"/home/ayermolo/local/tasks/T138552329/typeDedup" # string offset=35
.Linfo_string3:
	.asciz	"fooint"                        # string offset=83
.Linfo_string4:
	.asciz	"int"                           # string offset=90
.Linfo_string5:
	.asciz	"foo"                           # string offset=94
.Linfo_string6:
	.asciz	"_Z3foov"                       # string offset=98
.Linfo_string7:
	.asciz	"fint"                          # string offset=106
.Linfo_string8:
	.asciz	"Foo2Int"                       # string offset=111
.Linfo_string9:
	.asciz	"c1"                            # string offset=119
.Linfo_string10:
	.asciz	"c2"                            # string offset=122
.Linfo_string11:
	.asciz	"f"                             # string offset=125
.Linfo_string12:
	.asciz	"Foo2a"                         # string offset=127
.Linfo_string13:
	.asciz	"char"                          # string offset=133
.Linfo_string14:
	.asciz	"c3"                            # string offset=138
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string6
	.long	.Linfo_string5
	.long	.Linfo_string7
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string8
	.long	.Linfo_string11
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string12
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	fooint
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	7                               # Header: bucket count
	.long	7                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	2                               # Bucket 2
	.long	3                               # Bucket 3
	.long	0                               # Bucket 4
	.long	5                               # Bucket 5
	.long	7                               # Bucket 6
	.long	-1257882357                     # Hash in Bucket 0
	.long	-1168750522                     # Hash in Bucket 2
	.long	193495088                       # Hash in Bucket 3
	.long	259227804                       # Hash in Bucket 3
	.long	193491849                       # Hash in Bucket 5
	.long	2090147939                      # Hash in Bucket 5
	.long	-35356620                       # Hash in Bucket 6
	.long	.Linfo_string6                  # String in Bucket 0: _Z3foov
	.long	.Linfo_string8                  # String in Bucket 2: Foo2Int
	.long	.Linfo_string4                  # String in Bucket 3: int
	.long	.Linfo_string12                 # String in Bucket 3: Foo2a
	.long	.Linfo_string5                  # String in Bucket 5: foo
	.long	.Linfo_string13                 # String in Bucket 5: char
	.long	.Linfo_string3                  # String in Bucket 6: fooint
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames6-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 6
.Lnames_abbrev_start0:
	.ascii	"\230."                         # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\230\023"                      # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\230$"                         # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\2304"                         # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames3:
.L0:
	.ascii	"\230."                         # Abbreviation code
	.long	50                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z3foov
.Lnames4:
.L5:
	.ascii	"\230\023"                      # Abbreviation code
	.long	89                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: Foo2Int
.Lnames0:
.L2:
	.ascii	"\230$"                         # Abbreviation code
	.long	46                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames5:
.L3:
	.ascii	"\230\023"                      # Abbreviation code
	.long	119                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: Foo2a
.Lnames2:
	.ascii	"\230."                         # Abbreviation code
	.long	50                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
.Lnames6:
.L1:
	.ascii	"\230$"                         # Abbreviation code
	.long	158                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
.Lnames1:
.L4:
	.ascii	"\2304"                         # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: fooint
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 18.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

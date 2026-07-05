# struct Foo {
#  char *c1;
#  char *c2;
#  char *c3;
# };
# struct Foo2 {
#  char *c1;
#  char *c2;
# };
# int main(int argc, char *argv[]) {
#  Foo f;
#  f.c1 = argv[argc];
#  f.c2 = argv[argc + 1];
#  f.c3 = argv[argc + 2];
#  Foo2 f2;
#  f.c1 = argv[argc + 3];
#  f.c2 = argv[argc + 4];
#  return 0;
# }

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "dwarf4-types-dwarf5-test" "main.cpp"
	.loc	1 10 0                          # main.cpp:10:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp0:
	.loc	1 12 9 prologue_end             # main.cpp:12:9
	movq	-16(%rbp), %rax
	movslq	-8(%rbp), %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 12 7 is_stmt 0                # main.cpp:12:7
	movq	%rax, -40(%rbp)
	.loc	1 13 9 is_stmt 1                # main.cpp:13:9
	movq	-16(%rbp), %rax
	.loc	1 13 14 is_stmt 0               # main.cpp:13:14
	movl	-8(%rbp), %ecx
	.loc	1 13 19                         # main.cpp:13:19
	addl	$1, %ecx
	.loc	1 13 9                          # main.cpp:13:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 13 7                          # main.cpp:13:7
	movq	%rax, -32(%rbp)
	.loc	1 14 9 is_stmt 1                # main.cpp:14:9
	movq	-16(%rbp), %rax
	.loc	1 14 14 is_stmt 0               # main.cpp:14:14
	movl	-8(%rbp), %ecx
	.loc	1 14 19                         # main.cpp:14:19
	addl	$2, %ecx
	.loc	1 14 9                          # main.cpp:14:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 14 7                          # main.cpp:14:7
	movq	%rax, -24(%rbp)
	.loc	1 16 9 is_stmt 1                # main.cpp:16:9
	movq	-16(%rbp), %rax
	.loc	1 16 14 is_stmt 0               # main.cpp:16:14
	movl	-8(%rbp), %ecx
	.loc	1 16 19                         # main.cpp:16:19
	addl	$3, %ecx
	.loc	1 16 9                          # main.cpp:16:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 16 7                          # main.cpp:16:7
	movq	%rax, -40(%rbp)
	.loc	1 17 9 is_stmt 1                # main.cpp:17:9
	movq	-16(%rbp), %rax
	.loc	1 17 14 is_stmt 0               # main.cpp:17:14
	movl	-8(%rbp), %ecx
	.loc	1 17 19                         # main.cpp:17:19
	addl	$4, %ecx
	.loc	1 17 9                          # main.cpp:17:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 17 7                          # main.cpp:17:7
	movq	%rax, -32(%rbp)
	.loc	1 18 2 is_stmt 1                # main.cpp:18:2
	xorl	%eax, %eax
	.loc	1 18 2 epilogue_begin is_stmt 0 # main.cpp:18:2
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_types,"G",@progbits,7448148824980338162,comdat
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	7448148824980338162             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string12                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string9                  # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string11                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_types,"G",@progbits,5322170643381124694,comdat
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	5322170643381124694             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string14                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string9                  # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string10                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
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
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
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
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x9c DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x2a:0x52 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	124                             # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x43:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	124                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x51:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	131                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x5f:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	148                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x6d:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	157                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x7c:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x83:0x5 DW_TAG_pointer_type
	.long	136                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x88:0x5 DW_TAG_pointer_type
	.long	141                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x8d:0x7 DW_TAG_base_type
	.long	.Linfo_string7                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0x94:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	7448148824980338162             # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x9d:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	5322170643381124694             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 15.0.0" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=146
.Linfo_string2:
	.asciz	"dwarf4-types-dwarf5-test" # string offset=155
.Linfo_string3:
	.asciz	"main"                          # string offset=218
.Linfo_string4:
	.asciz	"int"                           # string offset=223
.Linfo_string5:
	.asciz	"argc"                          # string offset=227
.Linfo_string6:
	.asciz	"argv"                          # string offset=232
.Linfo_string7:
	.asciz	"char"                          # string offset=237
.Linfo_string8:
	.asciz	"f"                             # string offset=242
.Linfo_string9:
	.asciz	"c1"                            # string offset=244
.Linfo_string10:
	.asciz	"c2"                            # string offset=247
.Linfo_string11:
	.asciz	"c3"                            # string offset=250
.Linfo_string12:
	.asciz	"Foo"                           # string offset=253
.Linfo_string13:
	.asciz	"f2"                            # string offset=257
.Linfo_string14:
	.asciz	"Foo2"                          # string offset=260
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

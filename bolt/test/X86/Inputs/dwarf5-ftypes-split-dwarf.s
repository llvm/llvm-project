# clang++ main.cpp -g2 -gsplit-dwarf=split -gdwarf-5 -fdebug-types-section -S
# struct Foo {
#   char *c1;
#   char *c2;
#   char *c3;
# };
#
# struct Foo2 {
#   char *c1;
#   char *c2;
# };
#
# int main(int argc, char *argv[]) {
#   Foo f;
#   f.c1 = argv[argc];
#   f.c2 = argv[argc + 1];
#   f.c3 = argv[argc + 2];
#   Foo2 f2;
#   f.c1 = argv[argc + 3];
#   f.c2 = argv[argc + 4];
#   return 0;
# }

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "." "main.cpp" md5 0xa832f464c853be0f9c52da29cd913807
	.loc	0 12 0                          # main.cpp:12:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp0:
	.loc	0 14 10 prologue_end            # main.cpp:14:10
	movq	-16(%rbp), %rax
	movslq	-8(%rbp), %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	0 14 8 is_stmt 0                # main.cpp:14:8
	movq	%rax, -40(%rbp)
	.loc	0 15 10 is_stmt 1               # main.cpp:15:10
	movq	-16(%rbp), %rax
	.loc	0 15 15 is_stmt 0               # main.cpp:15:15
	movl	-8(%rbp), %ecx
	.loc	0 15 20                         # main.cpp:15:20
	addl	$1, %ecx
	.loc	0 15 10                         # main.cpp:15:10
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	0 15 8                          # main.cpp:15:8
	movq	%rax, -32(%rbp)
	.loc	0 16 10 is_stmt 1               # main.cpp:16:10
	movq	-16(%rbp), %rax
	.loc	0 16 15 is_stmt 0               # main.cpp:16:15
	movl	-8(%rbp), %ecx
	.loc	0 16 20                         # main.cpp:16:20
	addl	$2, %ecx
	.loc	0 16 10                         # main.cpp:16:10
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	0 16 8                          # main.cpp:16:8
	movq	%rax, -24(%rbp)
	.loc	0 18 10 is_stmt 1               # main.cpp:18:10
	movq	-16(%rbp), %rax
	.loc	0 18 15 is_stmt 0               # main.cpp:18:15
	movl	-8(%rbp), %ecx
	.loc	0 18 20                         # main.cpp:18:20
	addl	$3, %ecx
	.loc	0 18 10                         # main.cpp:18:10
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	0 18 8                          # main.cpp:18:8
	movq	%rax, -40(%rbp)
	.loc	0 19 10 is_stmt 1               # main.cpp:19:10
	movq	-16(%rbp), %rax
	.loc	0 19 15 is_stmt 0               # main.cpp:19:15
	movl	-8(%rbp), %ecx
	.loc	0 19 20                         # main.cpp:19:20
	addl	$4, %ecx
	.loc	0 19 10                         # main.cpp:19:10
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	0 19 8                          # main.cpp:19:8
	movq	%rax, -32(%rbp)
	.loc	0 20 3 is_stmt 1                # main.cpp:20:3
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	7448148824980338162             # Type Signature
	.long	31                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x33 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1f:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	9                               # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x25:0x9 DW_TAG_member
	.byte	6                               # DW_AT_name
	.long	65                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x2e:0x9 DW_TAG_member
	.byte	7                               # DW_AT_name
	.long	65                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x37:0x9 DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	65                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x41:0x5 DW_TAG_pointer_type
	.long	70                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x46:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	5322170643381124694             # Type Signature
	.long	31                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x2a DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1f:0x19 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	11                              # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x25:0x9 DW_TAG_member
	.byte	6                               # DW_AT_name
	.long	56                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x2e:0x9 DW_TAG_member
	.byte	7                               # DW_AT_name
	.long	56                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x38:0x5 DW_TAG_pointer_type
	.long	61                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x3d:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	4780348136649610820
	.byte	1                               # Abbrev [1] 0x14:0x14 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	1                               # DW_AT_dwo_name
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	12                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=38
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string1
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	64                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"main"                          # string offset=0
.Linfo_string1:
	.asciz	"int"                           # string offset=5
.Linfo_string2:
	.asciz	"argc"                          # string offset=9
.Linfo_string3:
	.asciz	"argv"                          # string offset=14
.Linfo_string4:
	.asciz	"char"                          # string offset=19
.Linfo_string5:
	.asciz	"f"                             # string offset=24
.Linfo_string6:
	.asciz	"c1"                            # string offset=26
.Linfo_string7:
	.asciz	"c2"                            # string offset=29
.Linfo_string8:
	.asciz	"c3"                            # string offset=32
.Linfo_string9:
	.asciz	"Foo"                           # string offset=35
.Linfo_string10:
	.asciz	"f2"                            # string offset=39
.Linfo_string11:
	.asciz	"Foo2"                          # string offset=42
.Linfo_string12:
	.asciz	"clang version 15.0.0" 			# string offset=47
.Linfo_string13:
	.asciz	"main.cpp"                      # string offset=68
.Linfo_string14:
	.asciz	"main.dwo"                      # string offset=77
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	5
	.long	9
	.long	14
	.long	19
	.long	24
	.long	26
	.long	29
	.long	32
	.long	35
	.long	39
	.long	42
	.long	47
	.long	68
	.long	77
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	4780348136649610820
	.byte	6                               # Abbrev [6] 0x14:0x67 DW_TAG_compile_unit
	.byte	12                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	13                              # DW_AT_name
	.byte	14                              # DW_AT_dwo_name
	.byte	7                               # Abbrev [7] 0x1a:0x3c DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	0                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	86                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x29:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x34:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	90                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x3f:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	104                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x4a:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	17                              # DW_AT_decl_line
	.long	113                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x56:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x5a:0x5 DW_TAG_pointer_type
	.long	95                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x5f:0x5 DW_TAG_pointer_type
	.long	100                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x64:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	10                              # Abbrev [10] 0x68:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	7448148824980338162             # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x71:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	5322170643381124694             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end2:
	.section	.debug_abbrev.dwo,"e",@progbits
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
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	5                               # DW_TAG_formal_parameter
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
	.section	.debug_line.dwo,"e",@progbits
.Ltmp2:
	.long	.Ldebug_line_end0-.Ldebug_line_start0 # unit length
.Ldebug_line_start0:
	.short	5
	.byte	8
	.byte	0
	.long	.Lprologue_end0-.Lprologue_start0
.Lprologue_start0:
	.byte	1
	.byte	1
	.byte	1
	.byte	-5
	.byte	14
	.byte	1
	.byte	1
	.byte	1
	.byte	8
	.byte	1
	.ascii	"/home/ayermolo/local/tasks/T104766233"
	.byte	0
	.byte	3
	.byte	1
	.byte	8
	.byte	2
	.byte	15
	.byte	5
	.byte	30
	.byte	1
	.ascii	"main.cpp"
	.byte	0
	.byte	0
	.byte	0xa8, 0x32, 0xf4, 0x64
	.byte	0xc8, 0x53, 0xbe, 0x0f
	.byte	0x9c, 0x52, 0xda, 0x29
	.byte	0xcd, 0x91, 0x38, 0x07
.Lprologue_end0:
.Ldebug_line_end0:
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	40                              # Compilation Unit Length
	.long	26                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	40                              # Compilation Unit Length
	.long	104                             # DIE offset
	.byte	16                              # Attributes: TYPE, EXTERNAL
	.asciz	"Foo"                           # External Name
	.long	113                             # DIE offset
	.byte	16                              # Attributes: TYPE, EXTERNAL
	.asciz	"Foo2"                          # External Name
	.long	86                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	100                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

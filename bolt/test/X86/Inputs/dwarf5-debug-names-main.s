# clang++ -g2 -gdwarf-5 -gpubnames -fdebug-types-section
# header.h
# struct Foo2a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# main.cpp
# #include "header.h"
# extern int fooint;
# namespace {
# struct t1 {
# int i;
# };
# }
# template <int *> struct t2 {
#   t1 v1;
# };
# struct t3 {
#   t2<&fooint> v1;
# };
# t3 v1;
#
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
#  Foo2 f2;
#  Foo2a f3;
#  return 0;
# }
	.text
	.file	"main.cpp"
	.file	0 "/typeDedup" "main.cpp" md5 0x04e636082b2b8a95a6ca39dde52372ae
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.loc	0 25 0                          # main.cpp:25:0
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
	.loc	0 29 2 prologue_end             # main.cpp:29:2
	xorl	%eax, %eax
	.loc	0 29 2 epilogue_begin is_stmt 0 # main.cpp:29:2
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	v1,@object                      # @v1
	.bss
	.globl	v1
	.p2align	2, 0x0
v1:
	.zero	4
	.size	v1, 4

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
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
	.byte	48                              # DW_TAG_template_value_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
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
	.byte	11                              # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x11a DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0xb DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	46                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	1
	.byte	3                               # Abbrev [3] 0x2e:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	8                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x34:0x9 DW_TAG_member
	.byte	3                               # DW_AT_name
	.long	62                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x3e:0x19 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	7                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	5                               # Abbrev [5] 0x44:0x9 DW_TAG_template_value_parameter
	.long	87                              # DW_AT_type
	.byte	3                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	159
	.byte	4                               # Abbrev [4] 0x4d:0x9 DW_TAG_member
	.byte	3                               # DW_AT_name
	.long	97                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x57:0x5 DW_TAG_pointer_type
	.long	92                              # DW_AT_type
	.byte	7                               # Abbrev [7] 0x5c:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x60:0x12 DW_TAG_namespace
	.byte	3                               # Abbrev [3] 0x61:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	6                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x67:0x9 DW_TAG_member
	.byte	5                               # DW_AT_name
	.long	92                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x72:0x48 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.long	92                              # DW_AT_type
                                        # DW_AT_external
	.byte	10                              # Abbrev [10] 0x81:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.long	92                              # DW_AT_type
	.byte	10                              # Abbrev [10] 0x8c:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	25                              # DW_AT_decl_line
	.long	186                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x97:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.byte	13                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	26                              # DW_AT_decl_line
	.long	200                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xa2:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.byte	18                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	27                              # DW_AT_decl_line
	.long	234                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xad:0xc DW_TAG_variable
	.byte	3                               # DW_AT_location
	.byte	145
	.ascii	"\260\177"
	.byte	20                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	28                              # DW_AT_decl_line
	.long	259                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0xba:0x5 DW_TAG_pointer_type
	.long	191                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xbf:0x5 DW_TAG_pointer_type
	.long	196                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0xc4:0x4 DW_TAG_base_type
	.byte	12                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0xc8:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	17                              # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0xce:0x9 DW_TAG_member
	.byte	14                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	17                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0xd7:0x9 DW_TAG_member
	.byte	15                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	18                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0xe0:0x9 DW_TAG_member
	.byte	16                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	19                              # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0xea:0x19 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	19                              # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	21                              # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0xf0:0x9 DW_TAG_member
	.byte	14                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	22                              # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0xf9:0x9 DW_TAG_member
	.byte	15                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	23                              # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	3                               # Abbrev [3] 0x103:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	21                              # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x109:0x9 DW_TAG_member
	.byte	14                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0x112:0x9 DW_TAG_member
	.byte	15                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0x11b:0x9 DW_TAG_member
	.byte	16                              # DW_AT_name
	.long	191                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	92                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/home/ayermolo/local/tasks/T138552329/typeDedup" # string offset=33
.Linfo_string3:
	.asciz	"v1"                            # string offset=81
.Linfo_string4:
	.asciz	"t3"                            # string offset=84
.Linfo_string5:
	.asciz	"t2<&fooint>"                   # string offset=87
.Linfo_string6:
	.asciz	"int"                           # string offset=99
.Linfo_string7:
	.asciz	"(anonymous namespace)"         # string offset=103
.Linfo_string8:
	.asciz	"t1"                            # string offset=125
.Linfo_string9:
	.asciz	"i"                             # string offset=128
.Linfo_string10:
	.asciz	"main"                          # string offset=130
.Linfo_string11:
	.asciz	"argc"                          # string offset=135
.Linfo_string12:
	.asciz	"argv"                          # string offset=140
.Linfo_string13:
	.asciz	"char"                          # string offset=145
.Linfo_string14:
	.asciz	"f"                             # string offset=150
.Linfo_string15:
	.asciz	"Foo"                           # string offset=152
.Linfo_string16:
	.asciz	"c1"                            # string offset=156
.Linfo_string17:
	.asciz	"c2"                            # string offset=159
.Linfo_string18:
	.asciz	"c3"                            # string offset=162
.Linfo_string19:
	.asciz	"f2"                            # string offset=165
.Linfo_string20:
	.asciz	"Foo2"                          # string offset=168
.Linfo_string21:
	.asciz	"f3"                            # string offset=173
.Linfo_string22:
	.asciz	"Foo2a"                         # string offset=176
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string6
	.long	.Linfo_string9
	.long	.Linfo_string8
	.long	.Linfo_string5
	.long	.Linfo_string4
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string16
	.long	.Linfo_string17
	.long	.Linfo_string18
	.long	.Linfo_string15
	.long	.Linfo_string19
	.long	.Linfo_string20
	.long	.Linfo_string21
	.long	.Linfo_string22
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	fooint
	.quad	v1
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
	.long	11                              # Header: bucket count
	.long	11                              # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	3                               # Bucket 1
	.long	5                               # Bucket 2
	.long	0                               # Bucket 3
	.long	0                               # Bucket 4
	.long	6                               # Bucket 5
	.long	8                               # Bucket 6
	.long	9                               # Bucket 7
	.long	11                              # Bucket 8
	.long	0                               # Bucket 9
	.long	0                               # Bucket 10
	.long	259227804                       # Hash in Bucket 0
	.long	2090147939                      # Hash in Bucket 0
	.long	193491849                       # Hash in Bucket 1
	.long	958480634                       # Hash in Bucket 1
	.long	2090263771                      # Hash in Bucket 2
	.long	5863786                         # Hash in Bucket 5
	.long	5863852                         # Hash in Bucket 5
	.long	193495088                       # Hash in Bucket 6
	.long	5863788                         # Hash in Bucket 7
	.long	2090499946                      # Hash in Bucket 7
	.long	-1929613044                     # Hash in Bucket 8
	.long	.Linfo_string22                 # String in Bucket 0: Foo2a
	.long	.Linfo_string13                 # String in Bucket 0: char
	.long	.Linfo_string15                 # String in Bucket 1: Foo
	.long	.Linfo_string5                  # String in Bucket 1: t2<&fooint>
	.long	.Linfo_string20                 # String in Bucket 2: Foo2
	.long	.Linfo_string8                  # String in Bucket 5: t1
	.long	.Linfo_string3                  # String in Bucket 5: v1
	.long	.Linfo_string6                  # String in Bucket 6: int
	.long	.Linfo_string4                  # String in Bucket 7: t3
	.long	.Linfo_string10                 # String in Bucket 7: main
	.long	.Linfo_string7                  # String in Bucket 8: (anonymous namespace)
	.long	.Lnames10-.Lnames_entries0      # Offset in Bucket 0
	.long	.Lnames7-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames8-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames9-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 6
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 7
	.long	.Lnames6-.Lnames_entries0       # Offset in Bucket 7
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 8
.Lnames_abbrev_start0:
	.ascii	"\2309"                         # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.ascii	"\270\023"                      # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
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
	.ascii	"\230."                         # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames10:
.L1:
	.ascii	"\230\023"                      # Abbreviation code
	.long	259                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: Foo2a
.Lnames7:
.L8:
	.ascii	"\230$"                         # Abbreviation code
	.long	196                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
.Lnames8:
.L0:
	.ascii	"\230\023"                      # Abbreviation code
	.long	200                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: Foo
.Lnames1:
.L2:
	.ascii	"\230\023"                      # Abbreviation code
	.long	62                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t2<&fooint>
.Lnames9:
.L9:
	.ascii	"\230\023"                      # Abbreviation code
	.long	234                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: Foo2
.Lnames4:
.L5:
	.ascii	"\270\023"                      # Abbreviation code
	.long	97                              # DW_IDX_die_offset
	.long	.L3-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: t1
.Lnames5:
.L7:
	.ascii	"\2304"                         # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: v1
.Lnames2:
.L10:
	.ascii	"\230$"                         # Abbreviation code
	.long	92                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames0:
.L6:
	.ascii	"\230\023"                      # Abbreviation code
	.long	46                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t3
.Lnames6:
.L4:
	.ascii	"\230."                         # Abbreviation code
	.long	114                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames3:
.L3:
	.ascii	"\2309"                         # Abbreviation code
	.long	96                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: (anonymous namespace)
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 18.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

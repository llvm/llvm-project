# clang++ -gsplit-dwarf -g2 -gdwarf-5 -gpubnames -fdebug-types-section -fdebug-compilation-dir='.' -S
# header.h
# struct Foo2a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# include "header.h"
# struct Foo2 {
#  char *c1;
# };
# int main(int argc, char *argv[]) {
#  Foo2 f2;
#  Foo2a f3;
#  return 0;
# }

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "." "main.cpp" md5 0x9c5cea5bb78d3fc265cd175110bfe903
	.loc	0 5 0                           # main.cpp:5:0
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
	.loc	0 8 2 prologue_end              # main.cpp:8:2
	xorl	%eax, %eax
	.loc	0 8 2 epilogue_begin is_stmt 0  # main.cpp:8:2
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
	.quad	5322170643381124694             # Type Signature
	.long	33                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x23 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.byte	6                               # DW_AT_comp_dir
	.byte	7                               # DW_AT_dwo_name
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x21:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	9                               # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0x9 DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	49                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x31:0x5 DW_TAG_pointer_type
	.long	54                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x36:0x4 DW_TAG_base_type
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
	.quad	1175092228111723119             # Type Signature
	.long	33                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x35 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.byte	6                               # DW_AT_comp_dir
	.byte	7                               # DW_AT_dwo_name
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x21:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	13                              # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0x9 DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	67                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x30:0x9 DW_TAG_member
	.byte	11                              # DW_AT_name
	.long	67                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x39:0x9 DW_TAG_member
	.byte	12                              # DW_AT_name
	.long	67                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x43:0x5 DW_TAG_pointer_type
	.long	72                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x48:0x4 DW_TAG_base_type
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
	.quad	5962099678818150071
	.byte	1                               # Abbrev [1] 0x14:0x14 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
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
	.asciz	"main"                          # string offset=53
.Lskel_string2:
	.asciz	"int"                           # string offset=58
.Lskel_string3:
	.asciz	"char"                          # string offset=62
.Lskel_string4:
	.asciz	"Foo2"                          # string offset=67
.Lskel_string5:
	.asciz	"Foo2a"                         # string offset=72
.Lskel_string6:
	.asciz	"main.dwo"                      # string offset=78
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string6
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	68                              # Length of String Offsets Set
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
	.asciz	"f2"                            # string offset=24
.Linfo_string6:
	.asciz	"/home/ayermolo/local/tasks/T138552329/typeDedupSplit" # string offset=27
.Linfo_string7:
	.asciz	"main.dwo"                      # string offset=80
.Linfo_string8:
	.asciz	"c1"                            # string offset=89
.Linfo_string9:
	.asciz	"Foo2"                          # string offset=92
.Linfo_string10:
	.asciz	"f3"                            # string offset=97
.Linfo_string11:
	.asciz	"c2"                            # string offset=100
.Linfo_string12:
	.asciz	"c3"                            # string offset=103
.Linfo_string13:
	.asciz	"Foo2a"                         # string offset=106
.Linfo_string14:
	.asciz	"clang version 18.0.0git (git@github.com:ayermolo/llvm-project.git db35fa8fc524127079662802c4735dbf397f86d0)" # string offset=112
.Linfo_string15:
	.asciz	"main.cpp"                      # string offset=220
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	5
	.long	9
	.long	14
	.long	19
	.long	24
	.long	27
	.long	80
	.long	89
	.long	92
	.long	97
	.long	100
	.long	103
	.long	106
	.long	112
	.long	220
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	5962099678818150071
	.byte	6                               # Abbrev [6] 0x14:0x67 DW_TAG_compile_unit
	.byte	14                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	15                              # DW_AT_name
	.byte	7                               # DW_AT_dwo_name
	.byte	7                               # Abbrev [7] 0x1a:0x3c DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	0                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x29:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x34:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	90                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x3f:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	104                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x4a:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	80
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
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
	.quad	5322170643381124694             # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x71:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	1175092228111723119             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end2:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	2
	.ascii	"/home/ayermolo/local/tasks/T138552329/typeDedupSplit"
	.byte	0
	.byte	46
	.byte	0
	.byte	3
	.byte	1
	.byte	8
	.byte	2
	.byte	15
	.byte	5
	.byte	30
	.byte	2
	.ascii	"main.cpp"
	.byte	0
	.byte	0
	.byte	0x9c, 0x5c, 0xea, 0x5b
	.byte	0xb7, 0x8d, 0x3f, 0xc2
	.byte	0x65, 0xcd, 0x17, 0x51
	.byte	0x10, 0xbf, 0xe9, 0x03
	.ascii	"header.h"
	.byte	0
	.byte	1
	.byte	0xfe, 0xa7, 0xbb, 0x1f
	.byte	0x22, 0xc4, 0x7f, 0x12
	.byte	0x9e, 0x15, 0x69, 0x5f
	.byte	0x71, 0x37, 0xa1, 0xe7
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
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	2                               # Header: foreign type unit count
	.long	5                               # Header: bucket count
	.long	5                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.quad	5322170643381124694             # Type unit 0
	.quad	1175092228111723119             # Type unit 1
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	0                               # Bucket 2
	.long	3                               # Bucket 3
	.long	4                               # Bucket 4
	.long	2090263771                      # Hash in Bucket 1
	.long	2090499946                      # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 3
	.long	259227804                       # Hash in Bucket 4
	.long	2090147939                      # Hash in Bucket 4
	.long	.Lskel_string4                  # String in Bucket 1: Foo2
	.long	.Lskel_string1                  # String in Bucket 1: main
	.long	.Lskel_string2                  # String in Bucket 3: int
	.long	.Lskel_string5                  # String in Bucket 4: Foo2a
	.long	.Lskel_string3                  # String in Bucket 4: char
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 4
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
	.ascii	"\310\013"                      # Abbrev code
	.byte	46                              # DW_TAG_subprogram
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
.Lnames3:
	.ascii	"\354\004"                      # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	33                              # DW_IDX_die_offset
	.ascii	"\350\004"                      # Abbreviation code
	.long	104                             # DW_IDX_die_offset
	.byte	0                               # End of list: Foo2
.Lnames0:
	.ascii	"\310\013"                      # Abbreviation code
	.long	26                              # DW_IDX_die_offset
	.byte	0                               # End of list: main
.Lnames1:
	.ascii	"\210\t"                        # Abbreviation code
	.long	86                              # DW_IDX_die_offset
	.byte	0                               # End of list: int
.Lnames4:
	.ascii	"\354\004"                      # Abbreviation code
	.byte	1                               # DW_IDX_type_unit
	.long	33                              # DW_IDX_die_offset
	.ascii	"\350\004"                      # Abbreviation code
	.long	113                             # DW_IDX_die_offset
	.byte	0                               # End of list: Foo2a
.Lnames2:
	.ascii	"\214\t"                        # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	54                              # DW_IDX_die_offset
	.ascii	"\214\t"                        # Abbreviation code
	.byte	1                               # DW_IDX_type_unit
	.long	72                              # DW_IDX_die_offset
	.ascii	"\210\t"                        # Abbreviation code
	.long	100                             # DW_IDX_die_offset
	.byte	0                               # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 18.0.0git (git@github.com:ayermolo/llvm-project.git db35fa8fc524127079662802c4735dbf397f86d0)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

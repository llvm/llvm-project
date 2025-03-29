# clang++ -g2 -gdwarf-5 -gpubnames -fdebug-types-section
# header.h
# struct Foo2a {
#   char *c1;
# };
# main.cpp
# #include "header.h"
# int main() {
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
	.file	0 "/typeDedupSmall" "main.cpp" md5 0x6e787b94dac2f817bb35cfde8006dc82
	.loc	0 2 0                           # main.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
.Ltmp0:
	.loc	0 4 2 prologue_end              # main.cpp:4:2
	xorl	%eax, %eax
	.loc	0 4 2 epilogue_begin is_stmt 0  # main.cpp:4:2
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.file	1 "." "header.h" md5 0x53699580704254cb1dd2a83230f8a7ea
	.section	.debug_info,"G",@progbits,1175092228111723119,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	1175092228111723119             # Type Signature
	.long	35                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x25 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x10 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	8                               # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x29:0x9 DW_TAG_member
	.byte	6                               # DW_AT_name
	.long	51                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x33:0x5 DW_TAG_pointer_type
	.long	56                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x38:0x4 DW_TAG_base_type
	.byte	7                               # DW_AT_name
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
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	6                               # Abbrev [6] 0xc:0x40 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	7                               # Abbrev [7] 0x23:0x1b DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	62                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x32:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	66                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x3e:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	9                               # Abbrev [9] 0x42:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	1175092228111723119             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	40                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0git (git@github.com:ayermolo/llvm-project.git db35fa8fc524127079662802c4735dbf397f86d0)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=108
.Linfo_string2:
	.asciz	"/home/typeDedupSmall" # string offset=117
.Linfo_string3:
	.asciz	"main"                          # string offset=170
.Linfo_string4:
	.asciz	"int"                           # string offset=175
.Linfo_string5:
	.asciz	"f3"                            # string offset=179
.Linfo_string6:
	.asciz	"Foo2a"                         # string offset=182
.Linfo_string7:
	.asciz	"c1"                            # string offset=188
.Linfo_string8:
	.asciz	"char"                          # string offset=191
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string6
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
	.long	1                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	4                               # Header: bucket count
	.long	4                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	3                               # Bucket 2
	.long	4                               # Bucket 3
	.long	193495088                       # Hash in Bucket 0
	.long	259227804                       # Hash in Bucket 0
	.long	2090499946                      # Hash in Bucket 2
	.long	2090147939                      # Hash in Bucket 3
	.long	.Linfo_string4                  # String in Bucket 0: int
	.long	.Linfo_string6                  # String in Bucket 0: Foo2a
	.long	.Linfo_string3                  # String in Bucket 2: main
	.long	.Linfo_string8                  # String in Bucket 3: char
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 3
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
.Lnames1:
	.ascii	"\210\t"                        # Abbreviation code
	.long	62                              # DW_IDX_die_offset
	.byte	0                               # End of list: int
.Lnames2:
	.ascii	"\354\004"                      # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
	.ascii	"\350\004"                      # Abbreviation code
	.long	66                              # DW_IDX_die_offset
	.byte	0                               # End of list: Foo2a
.Lnames0:
	.ascii	"\310\013"                      # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # End of list: main
.Lnames3:
	.ascii	"\214\t"                        # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	56                              # DW_IDX_die_offset
	.byte	0                               # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 18.0.0git (git@github.com:ayermolo/llvm-project.git db35fa8fc524127079662802c4735dbf397f86d0)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

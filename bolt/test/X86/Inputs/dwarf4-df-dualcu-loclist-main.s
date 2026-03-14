# clang++ -g -gdwarf-4 -emit-llvm -S main.cpp
# llc -O2 -mtriple=x86_64-unknown-linux-gnu main.ll
# void use(int * x, int * y) {
# *x += 4;
# *y -= 2;
# }
#
# int helper(int z_, int d_);
# int x = 0;
# int y = 1;
# int  main(int argc, char *argv[]) {
#    x = argc;
#    y = argc + 3;
#    use(&x, &y);
#    return helper(x, y);
# }

	.text
	.file	"main.cpp"
	.file	1 "." "main.cpp"
	.globl	_Z3usePiS_                      # -- Begin function _Z3usePiS_
	.p2align	4, 0x90
	.type	_Z3usePiS_,@function
_Z3usePiS_:                             # @_Z3usePiS_
.Lfunc_begin0:
	.loc	1 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: use:x <- $rdi
	#DEBUG_VALUE: use:y <- $rsi
	.loc	1 2 4 prologue_end              # main.cpp:2:4
	addl	$4, (%rdi)
	.loc	1 3 4                           # main.cpp:3:4
	addl	$-2, (%rsi)
	.loc	1 4 1                           # main.cpp:4:1
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_Z3usePiS_, .Lfunc_end0-_Z3usePiS_
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	1 9 0                           # main.cpp:9:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
                                        # kill: def $edi killed $edi def $rdi
	.loc	1 2 4 prologue_end              # main.cpp:2:4
	leal	4(%rdi), %eax
	movl	%eax, x(%rip)
.Ltmp1:
	#DEBUG_VALUE: use:x <- undef
	.loc	1 3 4                           # main.cpp:3:4
	leal	1(%rdi), %esi
.Ltmp2:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	movl	%esi, y(%rip)
.Ltmp3:
	#DEBUG_VALUE: use:y <- undef
	.loc	1 13 11                         # main.cpp:13:11
	movl	%eax, %edi
.Ltmp4:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	jmp	_Z6helperii                     # TAILCALL
.Ltmp5:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.type	x,@object                       # @x
	.bss
	.globl	x
	.p2align	2
x:
	.long	0                               # 0x0
	.size	x, 4

	.type	y,@object                       # @y
	.data
	.globl	y
	.p2align	2
y:
	.long	1                               # 0x1
	.size	y, 4

	.section	.debug_loc.dwo,"e",@progbits
.Ldebug_loc0:
	.byte	3
	.byte	3
	.long	.Ltmp4-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	3
	.byte	5
	.long	.Lfunc_end1-.Ltmp4
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	0
.Ldebug_loc1:
	.byte	3
	.byte	3
	.long	.Ltmp2-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	3
	.byte	6
	.long	.Lfunc_end1-.Ltmp2
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.ascii	"\263B"                         # DW_AT_GNU_addr_base
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	1                               # Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	-4922924650784735988            # DW_AT_GNU_dwo_id
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"."                             # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=2
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"x"                             # string offset=0
.Linfo_string1:
	.asciz	"int"                           # string offset=2
.Linfo_string2:
	.asciz	"y"                             # string offset=6
.Linfo_string3:
	.asciz	"_Z3usePiS_"                    # string offset=8
.Linfo_string4:
	.asciz	"use"                           # string offset=19
.Linfo_string5:
	.asciz	"_Z6helperii"                   # string offset=23
.Linfo_string6:
	.asciz	"helper"                        # string offset=35
.Linfo_string7:
	.asciz	"main"                          # string offset=42
.Linfo_string8:
	.asciz	"argc"                          # string offset=47
.Linfo_string9:
	.asciz	"argv"                          # string offset=52
.Linfo_string10:
	.asciz	"char"                          # string offset=57
.Linfo_string11:
	.asciz	"clang version 15.0.0"          # string offset=62
.Linfo_string12:
	.asciz	"main.cpp"                      # string offset=83
.Linfo_string13:
	.asciz	"main.dwo"                      # string offset=92
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	2
	.long	6
	.long	8
	.long	19
	.long	23
	.long	35
	.long	42
	.long	47
	.long	52
	.long	57
	.long	62
	.long	83
	.long	92
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xbd DW_TAG_compile_unit
	.byte	11                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	12                              # DW_AT_name
	.byte	13                              # DW_AT_GNU_dwo_name
	.quad	-4922924650784735988            # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0xb DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	0
	.byte	3                               # Abbrev [3] 0x24:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x28:0xb DW_TAG_variable
	.byte	2                               # DW_AT_name
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	1
	.byte	4                               # Abbrev [4] 0x33:0x1b DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	78                              # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x3f:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	84                              # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x46:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	92                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4e:0x17 DW_TAG_subprogram
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	7                               # Abbrev [7] 0x54:0x8 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	101                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x5c:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	101                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x65:0x5 DW_TAG_pointer_type
	.long	36                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x6a:0x3b DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	10                              # Abbrev [10] 0x79:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc0-.debug_loc.dwo     # DW_AT_location
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
	.byte	10                              # Abbrev [10] 0x85:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc1-.debug_loc.dwo     # DW_AT_location
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	185                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0x91:0xd DW_TAG_inlined_subroutine
	.long	78                              # DW_AT_abstract_origin
	.byte	3                               # DW_AT_low_pc
	.long	.Ltmp3-.Lfunc_begin1            # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	12                              # DW_AT_call_line
	.byte	4                               # DW_AT_call_column
	.byte	12                              # Abbrev [12] 0x9e:0x6 DW_TAG_GNU_call_site
	.long	165                             # DW_AT_abstract_origin
                                        # DW_AT_GNU_tail_call
	.byte	4                               # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xa5:0x14 DW_TAG_subprogram
	.byte	5                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	36                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xae:0x5 DW_TAG_formal_parameter
	.long	36                              # DW_AT_type
	.byte	14                              # Abbrev [14] 0xb3:0x5 DW_TAG_formal_parameter
	.long	36                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0xb9:0x5 DW_TAG_pointer_type
	.long	190                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xbe:0x5 DW_TAG_pointer_type
	.long	195                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0xc3:0x4 DW_TAG_base_type
	.byte	10                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
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
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
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
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
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
	.byte	23                              # DW_FORM_sec_offset
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.ascii	"\225B"                         # DW_AT_GNU_tail_call
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	x
	.quad	y
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp5
	.quad	.Ltmp4
	.quad	.Ltmp2
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:

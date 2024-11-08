# clang++ -g2 -fdebug-types-section -gdwarf-5 -ffunction-sections
# int foo(int i) {
#   if (i == 1)
#     return 2;
#   return 1;
# }
# int main(int argc, char* argv[]) {
#   int j = argc;
#   if (j ==3)
#     j+= foo(argc);
#   return j;
# }

	.text
	.file	"main.cpp"
	.section	.text._Z3fooi,"ax",@progbits
	.globl	_Z3fooi                         # -- Begin function _Z3fooi
	.p2align	4, 0x90
	.type	_Z3fooi,@function
_Z3fooi:                                # @_Z3fooi
.Lfunc_begin0:
	.file	0 "/test" "main.cpp" md5 0x485e4f4a4784830c686905cfd190e444
	.loc	0 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -8(%rbp)
.Ltmp0:
	.loc	0 2 9 prologue_end              # main.cpp:2:9
	cmpl	$1, -8(%rbp)
.Ltmp1:
	.loc	0 2 7 is_stmt 0                 # main.cpp:2:7
	jne	.LBB0_2
# %bb.1:                                # %if.then
.Ltmp2:
	.loc	0 3 5 is_stmt 1                 # main.cpp:3:5
	movl	$2, -4(%rbp)
	jmp	.LBB0_3
.Ltmp3:
.LBB0_2:                                # %if.end
	.loc	0 4 3                           # main.cpp:4:3
	movl	$1, -4(%rbp)
.LBB0_3:                                # %return
	.loc	0 5 1                           # main.cpp:5:1
	movl	-4(%rbp), %eax
	.loc	0 5 1 epilogue_begin is_stmt 0  # main.cpp:5:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp4:
.Lfunc_end0:
	.size	_Z3fooi, .Lfunc_end0-_Z3fooi
	.cfi_endproc
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	0 6 0 is_stmt 1                 # main.cpp:6:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp5:
	.loc	0 7 11 prologue_end             # main.cpp:7:11
	movl	-8(%rbp), %eax
	.loc	0 7 7 is_stmt 0                 # main.cpp:7:7
	movl	%eax, -20(%rbp)
.Ltmp6:
	.loc	0 8 9 is_stmt 1                 # main.cpp:8:9
	cmpl	$3, -20(%rbp)
.Ltmp7:
	.loc	0 8 7 is_stmt 0                 # main.cpp:8:7
	jne	.LBB1_2
# %bb.1:                                # %if.then
.Ltmp8:
	.loc	0 9 13 is_stmt 1                # main.cpp:9:13
	movl	-8(%rbp), %edi
	.loc	0 9 9 is_stmt 0                 # main.cpp:9:9
	callq	_Z3fooi
	.loc	0 9 6                           # main.cpp:9:6
	addl	-20(%rbp), %eax
	movl	%eax, -20(%rbp)
.Ltmp9:
.LBB1_2:                                # %if.end
	.loc	0 10 10 is_stmt 1               # main.cpp:10:10
	movl	-20(%rbp), %eax
	.loc	0 10 3 epilogue_begin is_stmt 0 # main.cpp:10:3
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp10:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
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
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
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
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xc:0x7f DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2b:0x1c DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x3b:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x47:0x31 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x56:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x61:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	124                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x6c:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	108
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	120                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x78:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x7c:0x5 DW_TAG_pointer_type
	.long	129                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x81:0x5 DW_TAG_pointer_type
	.long	134                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x86:0x4 DW_TAG_base_type
	.byte	10                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	52                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/test" # string offset=33
.Linfo_string3:
	.asciz	"_Z3fooi"                       # string offset=71
.Linfo_string4:
	.asciz	"foo"                           # string offset=79
.Linfo_string5:
	.asciz	"int"                           # string offset=83
.Linfo_string6:
	.asciz	"main"                          # string offset=87
.Linfo_string7:
	.asciz	"i"                             # string offset=92
.Linfo_string8:
	.asciz	"argc"                          # string offset=94
.Linfo_string9:
	.asciz	"argv"                          # string offset=99
.Linfo_string10:
	.asciz	"char"                          # string offset=104
.Linfo_string11:
	.asciz	"j"                             # string offset=109
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.ident	"clang version 18.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z3fooi
	.section	.debug_line,"",@progbits
.Lline_table_start0:

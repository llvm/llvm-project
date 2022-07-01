	.text
	.file	"main.cpp"
	.section	.text._Z3usePiS_,"ax",@progbits
	.globl	_Z3usePiS_                      # -- Begin function _Z3usePiS_
	.p2align	4, 0x90
	.type	_Z3usePiS_,@function
_Z3usePiS_:                             # @_Z3usePiS_
.Lfunc_begin0:
	.file	0 "." "main.cpp" md5 0xe3a18fae8565a087d09d6076b542cdab
	.loc	0 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: use:x <- $rdi
	#DEBUG_VALUE: use:y <- $rsi
	.loc	0 2 6 prologue_end              # main.cpp:2:6
	addl	$4, (%rdi)
	.loc	0 3 6                           # main.cpp:3:6
	addl	$-2, (%rsi)
	.loc	0 4 1                           # main.cpp:4:1
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_Z3usePiS_, .Lfunc_end0-_Z3usePiS_
	.cfi_endproc
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	0 12 0                          # main.cpp:12:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
.Ltmp1:
	#DEBUG_VALUE: main:x <- $edi
	movl	%edi, %ebx
.Ltmp2:
	#DEBUG_VALUE: main:y <- [DW_OP_plus_uconst 3, DW_OP_stack_value] undef
	#DEBUG_VALUE: use:x <- undef
	#DEBUG_VALUE: use:y <- undef
	.loc	0 2 6 prologue_end              # main.cpp:2:6
	leal	4(%rbx), %r14d
.Ltmp3:
	#DEBUG_VALUE: main:x <- $r14d
	.loc	0 14 20                         # main.cpp:14:20
	addl	fooVar0(%rip), %ebx
	.loc	0 14 30 is_stmt 0               # main.cpp:14:30
	addl	fooVar1(%rip), %ebx
.Ltmp4:
	.loc	0 3 6 is_stmt 1                 # main.cpp:3:6
	addl	fooVar2(%rip), %ebx
.Ltmp5:
	#DEBUG_VALUE: main:y <- undef
	.loc	0 16 19                         # main.cpp:16:19
	movl	%r14d, %edi
.Ltmp6:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	callq	_Z4foo0i
.Ltmp7:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	movl	%eax, %ebp
	#DEBUG_VALUE: main:x <- $r14d
	.loc	0 16 29 is_stmt 0               # main.cpp:16:29
	movl	%r14d, %edi
	callq	_Z4foo1i
.Ltmp8:
	movl	%eax, %r15d
	#DEBUG_VALUE: main:x <- $r14d
	.loc	0 16 39                         # main.cpp:16:39
	movl	%r14d, %edi
	callq	_Z4foo2i
.Ltmp9:
                                        # kill: def $eax killed $eax def $rax
	.loc	0 16 13                         # main.cpp:16:13
	addl	%ebx, %ebp
	.loc	0 16 17                         # main.cpp:16:17
	addl	%r15d, %ebp
	.loc	0 16 37                         # main.cpp:16:37
	addl	%ebp, %eax
	addl	$5, %eax
	.loc	0 16 4                          # main.cpp:16:4
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
.Ltmp10:
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp11:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_loclists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	3                               # Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
	.long	.Ldebug_loc1-.Lloclists_table_base0
	.long	.Ldebug_loc2-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	1                               # DW_LLE_base_addressx
	.byte	1                               #   base address index
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    #   starting offset
	.uleb128 .Ltmp6-.Lfunc_begin1           #   ending offset
	.byte	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin1           #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   ending offset
	.byte	4                               # Loc expr size
	.byte	163                             # DW_OP_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc1:
	.byte	1                               # DW_LLE_base_addressx
	.byte	1                               #   base address index
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    #   starting offset
	.uleb128 .Ltmp7-.Lfunc_begin1           #   ending offset
	.byte	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp7-.Lfunc_begin1           #   starting offset
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   ending offset
	.byte	4                               # Loc expr size
	.byte	163                             # DW_OP_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_loc2:
	.byte	1                               # DW_LLE_base_addressx
	.byte	1                               #   base address index
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin1           #   starting offset
	.uleb128 .Ltmp3-.Lfunc_begin1           #   ending offset
	.byte	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 .Ltmp3-.Lfunc_begin1           #   starting offset
	.uleb128 .Ltmp10-.Lfunc_begin1          #   ending offset
	.byte	1                               # Loc expr size
	.byte	94                              # super-register DW_OP_reg14
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
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
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
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
	.quad	8833000344697388042
	.byte	1                               # Abbrev [1] 0x14:0x1c DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	1                               # DW_AT_dwo_name
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 # Length
.Ldebug_list_header_start1:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges1:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	12                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=60
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string1
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	76                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z3usePiS_"                    # string offset=0
.Linfo_string1:
	.asciz	"use"                           # string offset=11
.Linfo_string2:
	.asciz	"x"                             # string offset=15
.Linfo_string3:
	.asciz	"int"                           # string offset=17
.Linfo_string4:
	.asciz	"y"                             # string offset=21
.Linfo_string5:
	.asciz	"_Z4foo0i"                      # string offset=23
.Linfo_string6:
	.asciz	"foo0"                          # string offset=32
.Linfo_string7:
	.asciz	"_Z4foo1i"                      # string offset=37
.Linfo_string8:
	.asciz	"foo1"                          # string offset=46
.Linfo_string9:
	.asciz	"_Z4foo2i"                      # string offset=51
.Linfo_string10:
	.asciz	"foo2"                          # string offset=60
.Linfo_string11:
	.asciz	"main"                          # string offset=65
.Linfo_string12:
	.asciz	"argc"                          # string offset=70
.Linfo_string13:
	.asciz	"argv"                          # string offset=75
.Linfo_string14:
	.asciz	"char"                          # string offset=80
.Linfo_string15:
	.asciz	"clang version 15.0.0" # string offset=85
.Linfo_string16:
	.asciz	"main.cpp"                      # string offset=219
.Linfo_string17:
	.asciz	"main.dwo"                      # string offset=228
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	11
	.long	15
	.long	17
	.long	21
	.long	23
	.long	32
	.long	37
	.long	46
	.long	51
	.long	60
	.long	65
	.long	70
	.long	75
	.long	80
	.long	85
	.long	219
	.long	228
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	8833000344697388042
	.byte	1                               # Abbrev [1] 0x14:0xdf DW_TAG_compile_unit
	.byte	15                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	16                              # DW_AT_name
	.byte	17                              # DW_AT_dwo_name
	.byte	2                               # Abbrev [2] 0x1a:0x1b DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.long	53                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x26:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	58                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x2d:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	66                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x35:0x16 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	5                               # Abbrev [5] 0x3a:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	75                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x42:0x8 DW_TAG_formal_parameter
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	75                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4b:0x5 DW_TAG_pointer_type
	.long	80                              # DW_AT_type
	.byte	7                               # Abbrev [7] 0x50:0x4 DW_TAG_base_type
	.byte	3                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	8                               # Abbrev [8] 0x54:0x63 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0x63:0x9 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_location
	.byte	12                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x6c:0x9 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	13                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	228                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x75:0x9 DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
	.byte	11                              # Abbrev [11] 0x7e:0x8 DW_TAG_variable
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
	.byte	12                              # Abbrev [12] 0x86:0x9 DW_TAG_inlined_subroutine
	.long	53                              # DW_AT_abstract_origin
	.byte	0                               # DW_AT_ranges
	.byte	0                               # DW_AT_call_file
	.byte	15                              # DW_AT_call_line
	.byte	4                               # DW_AT_call_column
	.byte	13                              # Abbrev [13] 0x8f:0xd DW_TAG_call_site
	.long	183                             # DW_AT_call_origin
	.byte	2                               # DW_AT_call_return_pc
	.byte	14                              # Abbrev [14] 0x95:0x6 DW_TAG_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_call_value
	.byte	126
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x9c:0xd DW_TAG_call_site
	.long	198                             # DW_AT_call_origin
	.byte	3                               # DW_AT_call_return_pc
	.byte	14                              # Abbrev [14] 0xa2:0x6 DW_TAG_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_call_value
	.byte	126
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xa9:0xd DW_TAG_call_site
	.long	213                             # DW_AT_call_origin
	.byte	4                               # DW_AT_call_return_pc
	.byte	14                              # Abbrev [14] 0xaf:0x6 DW_TAG_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	2                               # DW_AT_call_value
	.byte	126
	.byte	0
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0xb7:0xf DW_TAG_subprogram
	.byte	5                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	80                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xc0:0x5 DW_TAG_formal_parameter
	.long	80                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0xc6:0xf DW_TAG_subprogram
	.byte	7                               # DW_AT_linkage_name
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xcf:0x5 DW_TAG_formal_parameter
	.long	80                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	15                              # Abbrev [15] 0xd5:0xf DW_TAG_subprogram
	.byte	9                               # DW_AT_linkage_name
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	80                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	16                              # Abbrev [16] 0xde:0x5 DW_TAG_formal_parameter
	.long	80                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0xe4:0x5 DW_TAG_pointer_type
	.long	233                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xe9:0x5 DW_TAG_pointer_type
	.long	238                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0xee:0x4 DW_TAG_base_type
	.byte	14                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
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
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
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
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
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
	.byte	9                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
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
	.byte	12                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	72                              # DW_TAG_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	127                             # DW_AT_call_origin
	.byte	19                              # DW_FORM_ref4
	.byte	125                             # DW_AT_call_return_pc
	.byte	27                              # DW_FORM_addrx
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	73                              # DW_TAG_call_site_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	126                             # DW_AT_call_value
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_rnglists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end2-.Ldebug_list_header_start2 # Length
.Ldebug_list_header_start2:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_dwo_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
	.byte	1                               # DW_RLE_base_addressx
	.byte	1                               #   base address index
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin1           #   starting offset
	.uleb128 .Ltmp3-.Lfunc_begin1           #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp4-.Lfunc_begin1           #   starting offset
	.uleb128 .Ltmp5-.Lfunc_begin1           #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end2:
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp7
	.quad	.Ltmp8
	.quad	.Ltmp9
.Ldebug_addr_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	84                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	53                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"use"                           # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	80                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	238                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

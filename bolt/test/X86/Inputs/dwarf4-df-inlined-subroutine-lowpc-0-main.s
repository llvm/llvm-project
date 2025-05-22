# __attribute__((always_inline))
# int doStuff(int val) {
#   if (val)
#     ++val;
#   return val;
# }
# __attribute__((always_inline))
# int doStuffSame(int val) {
#       if (val)
#             ++val;
#         return val;
# }
#
#
# int main(int argc, const char** argv) {
#     return  doStuff(argc) + doStuffSame(argc);
# }
# Assembly manually modified to set DW_AT_low_pc to 0 for DW_TAG_inlined_subroutine(s)
	.text
	.file	"main.cpp"
	.section	.text._Z7doStuffi,"ax",@progbits
	.globl	_Z7doStuffi                     # -- Begin function _Z7doStuffi
	.p2align	4, 0x90
	.type	_Z7doStuffi,@function
_Z7doStuffi:                            # @_Z7doStuffi
.Lfunc_begin0:
	.file	1 "." "main.cpp"
	.loc	1 2 0                           # main.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	1 3 7 prologue_end              # main.cpp:3:7
	cmpl	$0, -4(%rbp)
.Ltmp1:
	.loc	1 3 7 is_stmt 0                 # main.cpp:3:7
	je	.LBB0_2
# %bb.1:                                # %if.then
.Ltmp2:
	.loc	1 4 5 is_stmt 1                 # main.cpp:4:5
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
.Ltmp3:
.LBB0_2:                                # %if.end
	.loc	1 5 10                          # main.cpp:5:10
	movl	-4(%rbp), %eax
	.loc	1 5 3 epilogue_begin is_stmt 0  # main.cpp:5:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp4:
.Lfunc_end0:
	.size	_Z7doStuffi, .Lfunc_end0-_Z7doStuffi
	.cfi_endproc
                                        # -- End function
	.section	.text._Z11doStuffSamei,"ax",@progbits
	.globl	_Z11doStuffSamei                # -- Begin function _Z11doStuffSamei
	.p2align	4, 0x90
	.type	_Z11doStuffSamei,@function
_Z11doStuffSamei:                       # @_Z11doStuffSamei
.Lfunc_begin1:
	.loc	1 8 0 is_stmt 1                 # main.cpp:8:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp5:
	.loc	1 9 11 prologue_end             # main.cpp:9:11
	cmpl	$0, -4(%rbp)
.Ltmp6:
	.loc	1 9 11 is_stmt 0                # main.cpp:9:11
	je	.LBB1_2
# %bb.1:                                # %if.then
.Ltmp7:
	.loc	1 10 13 is_stmt 1               # main.cpp:10:13
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
.Ltmp8:
.LBB1_2:                                # %if.end
	.loc	1 11 16                         # main.cpp:11:16
	movl	-4(%rbp), %eax
	.loc	1 11 9 epilogue_begin is_stmt 0 # main.cpp:11:9
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp9:
.Lfunc_end1:
	.size	_Z11doStuffSamei, .Lfunc_end1-_Z11doStuffSamei
	.cfi_endproc
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin2:
	.loc	1 15 0 is_stmt 1                # main.cpp:15:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -12(%rbp)
	movl	%edi, -16(%rbp)
	movq	%rsi, -24(%rbp)
.Ltmp10:
	.loc	1 16 21 prologue_end            # main.cpp:16:21
	movl	-16(%rbp), %eax
	movl	%eax, -8(%rbp)
.Ltmp11:
	.loc	1 3 7                           # main.cpp:3:7
	cmpl	$0, -8(%rbp)
.Ltmp12:
	.loc	1 3 7 is_stmt 0                 # main.cpp:3:7
	je	.LBB2_2
# %bb.1:                                # %if.then.i
.Ltmp13:
	.loc	1 4 5 is_stmt 1                 # main.cpp:4:5
	movl	-8(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -8(%rbp)
.Ltmp14:
.LBB2_2:                                # %_Z7doStuffi.exit
	.loc	1 5 10                          # main.cpp:5:10
	movl	-8(%rbp), %eax
	movl	%eax, -28(%rbp)                 # 4-byte Spill
.Ltmp15:
	.loc	1 16 41                         # main.cpp:16:41
	movl	-16(%rbp), %eax
	movl	%eax, -4(%rbp)
.Ltmp16:
	.loc	1 9 11                          # main.cpp:9:11
	cmpl	$0, -4(%rbp)
.Ltmp17:
	.loc	1 9 11 is_stmt 0                # main.cpp:9:11
	je	.LBB2_4
# %bb.3:                                # %if.then.i4
.Ltmp18:
	.loc	1 10 13 is_stmt 1               # main.cpp:10:13
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
.Ltmp19:
.LBB2_4:                                # %_Z11doStuffSamei.exit
	.loc	1 0 13 is_stmt 0                # main.cpp:0:13
	movl	-28(%rbp), %eax                 # 4-byte Reload
	.loc	1 16 27 is_stmt 1               # main.cpp:16:27
	addl	-4(%rbp), %eax
	.loc	1 16 5 epilogue_begin is_stmt 0 # main.cpp:16:5
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp20:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
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
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	1388808602086005033             # DW_AT_GNU_dwo_id
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	0								# manually modified
	.quad	0								# manually modified
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=38
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z7doStuffi"                   # string offset=0
.Linfo_string1:
	.asciz	"doStuff"                       # string offset=12
.Linfo_string2:
	.asciz	"int"                           # string offset=20
.Linfo_string3:
	.asciz	"val"                           # string offset=24
.Linfo_string4:
	.asciz	"_Z11doStuffSamei"              # string offset=28
.Linfo_string5:
	.asciz	"doStuffSame"                   # string offset=45
.Linfo_string6:
	.asciz	"main"                          # string offset=57
.Linfo_string7:
	.asciz	"argc"                          # string offset=62
.Linfo_string8:
	.asciz	"argv"                          # string offset=67
.Linfo_string9:
	.asciz	"char"                          # string offset=72
.Linfo_string10:
	.asciz	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 63dbe7e808d07bdf25bad85301980bc323b0cd64)" # string offset=77
.Linfo_string11:
	.asciz	"main.cpp"                      # string offset=178
.Linfo_string12:
	.asciz	"main.dwo"                      # string offset=187
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	12
	.long	20
	.long	24
	.long	28
	.long	45
	.long	57
	.long	62
	.long	67
	.long	72
	.long	77
	.long	178
	.long	187
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xc8 DW_TAG_compile_unit
	.byte	10                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	11                              # DW_AT_name
	.byte	12                              # DW_AT_GNU_dwo_name
	.quad	1388808602086005033             # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0x15 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	67                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x25:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	77                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x2e:0x15 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	90                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x3a:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	100                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x43:0x13 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	5                               # Abbrev [5] 0x4d:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x56:0x4 DW_TAG_base_type
	.byte	2                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x5a:0x13 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	5                               # Abbrev [5] 0x64:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x6d:0x52 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	86                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x7c:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x87:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	191                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x92:0x16 DW_TAG_inlined_subroutine
	.long	67                              # DW_AT_abstract_origin
	.byte	3                               # DW_AT_low_pc
	.long	.Ltmp15-.Ltmp11                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	16                              # DW_AT_call_line
	.byte	13                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0x9f:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	77                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0xa8:0x16 DW_TAG_inlined_subroutine
	.long	90                              # DW_AT_abstract_origin
	.byte	4                               # DW_AT_low_pc
	.long	.Ltmp19-.Ltmp16                 # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	16                              # DW_AT_call_line
	.byte	29                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0xb5:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	100                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xbf:0x5 DW_TAG_pointer_type
	.long	196                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xc4:0x5 DW_TAG_pointer_type
	.long	201                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xc9:0x5 DW_TAG_const_type
	.long	206                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xce:0x4 DW_TAG_base_type
	.byte	9                               # DW_AT_name
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
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
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
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.byte	8                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
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
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	10                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_begin2
	.quad	0
	.quad	0
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	67                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuff"                       # External Name
	.long	90                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuffSame"                   # External Name
	.long	109                             # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	86                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	206                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 63dbe7e808d07bdf25bad85301980bc323b0cd64)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

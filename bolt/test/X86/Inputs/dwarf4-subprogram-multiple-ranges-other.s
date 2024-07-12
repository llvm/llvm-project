## clang++ -fbasic-block-sections=all -ffunction-sections -g2 -gdwarf-4
## int doStuffOther(int val) {
##   if (val)
##     ++val;
##   return val;
## }
##
## int mainOther(int argc, const char** argv) {
##     return  doStuffOther(argc);
## }
		.text
	.file	"mainOther.cpp"
	.section	.text._Z12doStuffOtheri,"ax",@progbits
	.globl	_Z12doStuffOtheri               # -- Begin function _Z12doStuffOtheri
	.p2align	4, 0x90
	.type	_Z12doStuffOtheri,@function
_Z12doStuffOtheri:                      # @_Z12doStuffOtheri
.Lfunc_begin0:
	.file	1 "." "mainOther.cpp"
	.loc	1 1 0                           # mainOther.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	1 2 8 prologue_end              # mainOther.cpp:2:8
	cmpl	$0, -4(%rbp)
.Ltmp1:
	.loc	1 2 8 is_stmt 0                 # mainOther.cpp:2:8
	je	_Z12doStuffOtheri.__part.2
	jmp	_Z12doStuffOtheri.__part.1
.LBB_END0_0:
	.cfi_endproc
	.section	.text._Z12doStuffOtheri,"ax",@progbits,unique,1
_Z12doStuffOtheri.__part.1:             # %if.then
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 3 6 is_stmt 1                 # mainOther.cpp:3:6
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
	jmp	_Z12doStuffOtheri.__part.2
.LBB_END0_1:
	.size	_Z12doStuffOtheri.__part.1, .LBB_END0_1-_Z12doStuffOtheri.__part.1
	.cfi_endproc
	.section	.text._Z12doStuffOtheri,"ax",@progbits,unique,2
_Z12doStuffOtheri.__part.2:             # %if.end
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 4 11                          # mainOther.cpp:4:11
	movl	-4(%rbp), %eax
	.loc	1 4 4 epilogue_begin is_stmt 0  # mainOther.cpp:4:4
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END0_2:
	.size	_Z12doStuffOtheri.__part.2, .LBB_END0_2-_Z12doStuffOtheri.__part.2
	.cfi_endproc
	.section	.text._Z12doStuffOtheri,"ax",@progbits
.Lfunc_end0:
	.size	_Z12doStuffOtheri, .Lfunc_end0-_Z12doStuffOtheri
                                        # -- End function
	.section	.text._Z9mainOtheriPPKc,"ax",@progbits
	.globl	_Z9mainOtheriPPKc               # -- Begin function _Z9mainOtheriPPKc
	.p2align	4, 0x90
	.type	_Z9mainOtheriPPKc,@function
_Z9mainOtheriPPKc:                      # @_Z9mainOtheriPPKc
.Lfunc_begin1:
	.loc	1 7 0 is_stmt 1                 # mainOther.cpp:7:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp2:
	.loc	1 8 27 prologue_end             # mainOther.cpp:8:27
	movl	-4(%rbp), %edi
	.loc	1 8 14 is_stmt 0                # mainOther.cpp:8:14
	callq	_Z12doStuffOtheri
	.loc	1 8 6 epilogue_begin            # mainOther.cpp:8:6
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END1_0:
	.cfi_endproc
.Lfunc_end1:
	.size	_Z9mainOtheriPPKc, .Lfunc_end1-_Z9mainOtheriPPKc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
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
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
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
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
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
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
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
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x9b DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges1                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x24 DW_TAG_subprogram
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	136                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x3f:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	136                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4e:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string6                  # DW_AT_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	136                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x6b:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	136                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x79:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	143                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x88:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x8f:0x5 DW_TAG_pointer_type
	.long	148                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x94:0x5 DW_TAG_pointer_type
	.long	153                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x99:0x5 DW_TAG_const_type
	.long	158                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x9e:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	_Z12doStuffOtheri.__part.1
	.quad	.LBB_END0_1
	.quad	_Z12doStuffOtheri.__part.2
	.quad	.LBB_END0_2
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	_Z12doStuffOtheri.__part.1
	.quad	.LBB_END0_1
	.quad	_Z12doStuffOtheri.__part.2
	.quad	.LBB_END0_2
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git df542e1ed82bd4e5a9e345d3a3ae63a76893a0cf)" # string offset=0
.Linfo_string1:
	.asciz	"mainOther.cpp"                 # string offset=104
.Linfo_string2:
	.asciz	"."                             # string offset=118
.Linfo_string3:
	.asciz	"_Z12doStuffOtheri"             # string offset=120
.Linfo_string4:
	.asciz	"doStuffOther"                  # string offset=138
.Linfo_string5:
	.asciz	"int"                           # string offset=151
.Linfo_string6:
	.asciz	"_Z9mainOtheriPPKc"             # string offset=155
.Linfo_string7:
	.asciz	"mainOther"                     # string offset=173
.Linfo_string8:
	.asciz	"val"                           # string offset=183
.Linfo_string9:
	.asciz	"argc"                          # string offset=187
.Linfo_string10:
	.asciz	"argv"                          # string offset=192
.Linfo_string11:
	.asciz	"char"                          # string offset=197
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git df542e1ed82bd4e5a9e345d3a3ae63a76893a0cf)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z12doStuffOtheri
	.section	.debug_line,"",@progbits
.Lline_table_start0:

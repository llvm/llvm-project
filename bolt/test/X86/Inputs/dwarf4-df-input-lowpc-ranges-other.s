## clang++ -fbasic-block-sections=all -ffunction-sections -g2 -gdwarf-4 -gsplit-dwarf -fdebug-compilation-dir='.'
## __attribute__((always_inline))
## int doStuffOther(int val) {
##   if (val)
##     ++val;
##   return val;
## }
## __attribute__((always_inline))
## int doStuffOther2(int val) {
##   int foo = 3;
##   return val + foo;
## }
##
##
## int mainOther(int argc, const char** argv) {
##     return  doStuffOther(argc) + doStuffOther2(argc);;
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
	.loc	1 2 0                           # mainOther.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	1 3 8 prologue_end              # mainOther.cpp:3:8
	cmpl	$0, -4(%rbp)
.Ltmp1:
	.loc	1 3 8 is_stmt 0                 # mainOther.cpp:3:8
	je	_Z12doStuffOtheri.__part.2
	jmp	_Z12doStuffOtheri.__part.1
.LBB_END0_0:
	.cfi_endproc
	.section	.text._Z12doStuffOtheri,"ax",@progbits,unique,1
_Z12doStuffOtheri.__part.1:             # %if.then
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 4 6 is_stmt 1                 # mainOther.cpp:4:6
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
	.loc	1 5 11                          # mainOther.cpp:5:11
	movl	-4(%rbp), %eax
	.loc	1 5 4 epilogue_begin is_stmt 0  # mainOther.cpp:5:4
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
	.section	.text._Z13doStuffOther2i,"ax",@progbits
	.globl	_Z13doStuffOther2i              # -- Begin function _Z13doStuffOther2i
	.p2align	4, 0x90
	.type	_Z13doStuffOther2i,@function
_Z13doStuffOther2i:                     # @_Z13doStuffOther2i
.Lfunc_begin1:
	.loc	1 8 0 is_stmt 1                 # mainOther.cpp:8:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp2:
	.loc	1 9 8 prologue_end              # mainOther.cpp:9:8
	movl	$3, -8(%rbp)
	.loc	1 10 11                         # mainOther.cpp:10:11
	movl	-4(%rbp), %eax
	.loc	1 10 15 is_stmt 0               # mainOther.cpp:10:15
	addl	-8(%rbp), %eax
	.loc	1 10 4 epilogue_begin           # mainOther.cpp:10:4
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END1_0:
	.cfi_endproc
.Lfunc_end1:
	.size	_Z13doStuffOther2i, .Lfunc_end1-_Z13doStuffOther2i
                                        # -- End function
	.section	.text._Z9mainOtheriPPKc,"ax",@progbits
	.globl	_Z9mainOtheriPPKc               # -- Begin function _Z9mainOtheriPPKc
	.p2align	4, 0x90
	.type	_Z9mainOtheriPPKc,@function
_Z9mainOtheriPPKc:                      # @_Z9mainOtheriPPKc
.Lfunc_begin2:
	.loc	1 14 0 is_stmt 1                # mainOther.cpp:14:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -16(%rbp)
	movq	%rsi, -24(%rbp)
.Ltmp3:
	.loc	1 15 27 prologue_end            # mainOther.cpp:15:27
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
.Ltmp4:
	.loc	1 3 8                           # mainOther.cpp:3:8
	cmpl	$0, -12(%rbp)
.Ltmp5:
	.loc	1 3 8 is_stmt 0                 # mainOther.cpp:3:8
	je	_Z9mainOtheriPPKc.__part.2
	jmp	_Z9mainOtheriPPKc.__part.1
.LBB_END2_0:
	.cfi_endproc
	.section	.text._Z9mainOtheriPPKc,"ax",@progbits,unique,3
_Z9mainOtheriPPKc.__part.1:             # %if.then.i
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 4 6 is_stmt 1                 # mainOther.cpp:4:6
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -12(%rbp)
	jmp	_Z9mainOtheriPPKc.__part.2
.LBB_END2_1:
	.size	_Z9mainOtheriPPKc.__part.1, .LBB_END2_1-_Z9mainOtheriPPKc.__part.1
	.cfi_endproc
	.section	.text._Z9mainOtheriPPKc,"ax",@progbits,unique,4
_Z9mainOtheriPPKc.__part.2:             # %_Z12doStuffOtheri.exit
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 5 11                          # mainOther.cpp:5:11
	movl	-12(%rbp), %eax
.Ltmp6:
	.loc	1 15 49                         # mainOther.cpp:15:49
	movl	-16(%rbp), %ecx
	movl	%ecx, -4(%rbp)
.Ltmp7:
	.loc	1 9 8                           # mainOther.cpp:9:8
	movl	$3, -8(%rbp)
	.loc	1 10 11                         # mainOther.cpp:10:11
	movl	-4(%rbp), %ecx
	.loc	1 10 15 is_stmt 0               # mainOther.cpp:10:15
	addl	-8(%rbp), %ecx
.Ltmp8:
	.loc	1 15 33 is_stmt 1               # mainOther.cpp:15:33
	addl	%ecx, %eax
	.loc	1 15 6 epilogue_begin is_stmt 0 # mainOther.cpp:15:6
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END2_2:
	.size	_Z9mainOtheriPPKc.__part.2, .LBB_END2_2-_Z9mainOtheriPPKc.__part.2
	.cfi_endproc
	.section	.text._Z9mainOtheriPPKc,"ax",@progbits
.Lfunc_end2:
	.size	_Z9mainOtheriPPKc, .Lfunc_end2-_Z9mainOtheriPPKc
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
	.ascii	"\262B"                         # DW_AT_GNU_ranges_base
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	1                               # Abbrev [1] 0xb:0x29 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	-1082921489565291703            # DW_AT_GNU_dwo_id
	.long	.debug_ranges                   # DW_AT_GNU_ranges_base
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges3                 # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
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
	.quad	_Z9mainOtheriPPKc.__part.1
	.quad	.LBB_END2_1
	.quad	_Z9mainOtheriPPKc.__part.2
	.quad	.LBB_END2_2
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp4
	.quad	.Lfunc_end2
	.quad	_Z9mainOtheriPPKc.__part.1
	.quad	.LBB_END2_1
	.quad	_Z9mainOtheriPPKc.__part.2
	.quad	.Ltmp6
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	_Z12doStuffOtheri.__part.1
	.quad	.LBB_END0_1
	.quad	_Z12doStuffOtheri.__part.2
	.quad	.LBB_END0_2
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	_Z9mainOtheriPPKc.__part.1
	.quad	.LBB_END2_1
	.quad	_Z9mainOtheriPPKc.__part.2
	.quad	.LBB_END2_2
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"."                             # string offset=0
.Lskel_string1:
	.asciz	"mainOther.dwo"                 # string offset=2
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z12doStuffOtheri"             # string offset=0
.Linfo_string1:
	.asciz	"doStuffOther"                  # string offset=18
.Linfo_string2:
	.asciz	"int"                           # string offset=31
.Linfo_string3:
	.asciz	"val"                           # string offset=35
.Linfo_string4:
	.asciz	"_Z13doStuffOther2i"            # string offset=39
.Linfo_string5:
	.asciz	"doStuffOther2"                 # string offset=58
.Linfo_string6:
	.asciz	"foo"                           # string offset=72
.Linfo_string7:
	.asciz	"_Z9mainOtheriPPKc"             # string offset=76
.Linfo_string8:
	.asciz	"mainOther"                     # string offset=94
.Linfo_string9:
	.asciz	"argc"                          # string offset=104
.Linfo_string10:
	.asciz	"argv"                          # string offset=109
.Linfo_string11:
	.asciz	"char"                          # string offset=114
.Linfo_string12:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git df542e1ed82bd4e5a9e345d3a3ae63a76893a0cf)" # string offset=119
.Linfo_string13:
	.asciz	"mainOther.cpp"                 # string offset=223
.Linfo_string14:
	.asciz	"mainOther.dwo"                 # string offset=237
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	18
	.long	31
	.long	35
	.long	39
	.long	58
	.long	72
	.long	76
	.long	94
	.long	104
	.long	109
	.long	114
	.long	119
	.long	223
	.long	237
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xde DW_TAG_compile_unit
	.byte	12                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	13                              # DW_AT_name
	.byte	14                              # DW_AT_GNU_dwo_name
	.quad	-1082921489565291703            # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0x14 DW_TAG_subprogram
	.long	.Ldebug_ranges0-.debug_ranges   # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	74                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x24:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	84                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x2d:0x1d DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	97                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x39:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	107                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x41:0x8 DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	115                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4a:0x13 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	93                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	7                               # Abbrev [7] 0x54:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	93                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x5d:0x4 DW_TAG_base_type
	.byte	2                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x61:0x1b DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	93                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	7                               # Abbrev [7] 0x6b:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	93                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x73:0x8 DW_TAG_variable
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	93                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x7c:0x59 DW_TAG_subprogram
	.long	.Ldebug_ranges1-.debug_ranges   # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	7                               # DW_AT_linkage_name
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	93                              # DW_AT_type
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x8b:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	93                              # DW_AT_type
	.byte	11                              # Abbrev [11] 0x96:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	14                              # DW_AT_decl_line
	.long	213                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xa1:0x15 DW_TAG_inlined_subroutine
	.long	74                              # DW_AT_abstract_origin
	.long	.Ldebug_ranges2-.debug_ranges   # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	15                              # DW_AT_call_line
	.byte	14                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0xad:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.long	84                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xb6:0x1e DW_TAG_inlined_subroutine
	.long	97                              # DW_AT_abstract_origin
	.byte	7                               # DW_AT_low_pc
	.long	.Ltmp8-.Ltmp7                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	15                              # DW_AT_call_line
	.byte	35                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0xc3:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	107                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0xcb:0x8 DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	115                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0xd5:0x5 DW_TAG_pointer_type
	.long	218                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0xda:0x5 DW_TAG_pointer_type
	.long	223                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0xdf:0x5 DW_TAG_const_type
	.long	228                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xe4:0x4 DW_TAG_base_type
	.byte	11                              # DW_AT_name
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
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
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
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
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
	.byte	9                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
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
	.byte	10                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
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
	.byte	12                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
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
	.byte	14                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	_Z12doStuffOtheri.__part.1
	.quad	_Z12doStuffOtheri.__part.2
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	_Z9mainOtheriPPKc.__part.1
	.quad	_Z9mainOtheriPPKc.__part.2
	.quad	.Lfunc_begin2
	.quad	.Ltmp7
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	52                              # Compilation Unit Length
	.long	74                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuffOther"                  # External Name
	.long	97                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"doStuffOther2"                 # External Name
	.long	124                             # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"mainOther"                     # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	52                              # Compilation Unit Length
	.long	93                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	228                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git df542e1ed82bd4e5a9e345d3a3ae63a76893a0cf)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

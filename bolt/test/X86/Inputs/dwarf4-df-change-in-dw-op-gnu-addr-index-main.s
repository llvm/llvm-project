# clang++ -O2 -g -gdwarf-4 -gsplit-dwarf main.cpp
# void use(int * x, int * y) {
#  static int Foo = *y + *x;
#  *x += 4;
#  *y -= Foo;
# }
#
# int x = 0;
# int y = 1;
# int  main(int argc, char *argv[]) {
#    x = argc;
#    y = argc + 3;
#    use(&x, &y);
#    return 0;
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
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
.Ltmp0:
	.loc	1 2 2 prologue_end              # main.cpp:2:2
	movzbl	_ZGVZ3usePiS_E3Foo(%rip), %eax
	testb	%al, %al
	je	.LBB0_1
.Ltmp1:
.LBB0_3:                                # %init.end
	#DEBUG_VALUE: use:x <- $r14
	#DEBUG_VALUE: use:y <- $rbx
	.loc	1 3 5                           # main.cpp:3:5
	addl	$4, (%r14)
	.loc	1 4 8                           # main.cpp:4:8
	movl	_ZZ3usePiS_E3Foo(%rip), %eax
	.loc	1 4 5 is_stmt 0                 # main.cpp:4:5
	subl	%eax, (%rbx)
	.loc	1 5 1 epilogue_begin is_stmt 1  # main.cpp:5:1
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
.Ltmp2:
	#DEBUG_VALUE: use:y <- [DW_OP_LLVM_entry_value 1] $rsi
	.cfi_def_cfa_offset 16
	popq	%r14
.Ltmp3:
	#DEBUG_VALUE: use:x <- [DW_OP_LLVM_entry_value 1] $rdi
	.cfi_def_cfa_offset 8
	retq
.Ltmp4:
.LBB0_1:                                # %init.check
	.cfi_def_cfa_offset 32
	#DEBUG_VALUE: use:x <- $r14
	#DEBUG_VALUE: use:y <- $rbx
	.loc	1 2 2                           # main.cpp:2:2
	leaq	_ZGVZ3usePiS_E3Foo(%rip), %rdi
	callq	__cxa_guard_acquire@PLT
.Ltmp5:
	testl	%eax, %eax
	je	.LBB0_3
.Ltmp6:
# %bb.2:                                # %init
	#DEBUG_VALUE: use:x <- $r14
	#DEBUG_VALUE: use:y <- $rbx
	.loc	1 2 24 is_stmt 0                # main.cpp:2:24
	movl	(%r14), %eax
	.loc	1 2 22                          # main.cpp:2:22
	addl	(%rbx), %eax
	.loc	1 2 2                           # main.cpp:2:2
	movl	%eax, _ZZ3usePiS_E3Foo(%rip)
	leaq	_ZGVZ3usePiS_E3Foo(%rip), %rdi
	callq	__cxa_guard_release@PLT
.Ltmp7:
	.loc	1 0 2                           # main.cpp:0:2
	jmp	.LBB0_3
.Ltmp8:
.Lfunc_end0:
	.size	_Z3usePiS_, .Lfunc_end0-_Z3usePiS_
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	.loc	1 10 6 prologue_end is_stmt 1   # main.cpp:10:6
	movl	%edi, x(%rip)
	.loc	1 11 13                         # main.cpp:11:13
	addl	$3, %edi
.Ltmp9:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	.loc	1 11 6 is_stmt 0                # main.cpp:11:6
	movl	%edi, y(%rip)
.Ltmp10:
	#DEBUG_VALUE: use:y <- undef
	#DEBUG_VALUE: use:x <- undef
	.loc	1 2 2 is_stmt 1                 # main.cpp:2:2
	movzbl	_ZGVZ3usePiS_E3Foo(%rip), %eax
	testb	%al, %al
	je	.LBB1_1
.Ltmp11:
.LBB1_4:                                # %_Z3usePiS_.exit
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	.loc	1 3 5                           # main.cpp:3:5
	addl	$4, x(%rip)
	.loc	1 4 8                           # main.cpp:4:8
	movl	_ZZ3usePiS_E3Foo(%rip), %eax
	.loc	1 4 5 is_stmt 0                 # main.cpp:4:5
	subl	%eax, y(%rip)
.Ltmp12:
	.loc	1 13 4 is_stmt 1                # main.cpp:13:4
	xorl	%eax, %eax
	retq
.Ltmp13:
.LBB1_1:                                # %init.check.i
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- $rsi
	pushq	%rax
	.cfi_def_cfa_offset 16
.Ltmp14:
	.loc	1 2 2                           # main.cpp:2:2
	leaq	_ZGVZ3usePiS_E3Foo(%rip), %rdi
	callq	__cxa_guard_acquire@PLT
.Ltmp15:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	testl	%eax, %eax
	je	.LBB1_3
.Ltmp16:
# %bb.2:                                # %init.i
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	.loc	1 2 24 is_stmt 0                # main.cpp:2:24
	movl	x(%rip), %eax
	.loc	1 2 22                          # main.cpp:2:22
	addl	y(%rip), %eax
	.loc	1 2 2                           # main.cpp:2:2
	movl	%eax, _ZZ3usePiS_E3Foo(%rip)
	leaq	_ZGVZ3usePiS_E3Foo(%rip), %rdi
	callq	__cxa_guard_release@PLT
.Ltmp17:
.LBB1_3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	.loc	1 0 2                           # main.cpp:0:2
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	jmp	.LBB1_4
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.type	_ZZ3usePiS_E3Foo,@object        # @_ZZ3usePiS_E3Foo
	.local	_ZZ3usePiS_E3Foo
	.comm	_ZZ3usePiS_E3Foo,4,4
	.type	_ZGVZ3usePiS_E3Foo,@object      # @_ZGVZ3usePiS_E3Foo
	.local	_ZGVZ3usePiS_E3Foo
	.comm	_ZGVZ3usePiS_E3Foo,8,8
	.type	x,@object                       # @x
	.bss
	.globl	x
	.p2align	2, 0x0
x:
	.long	0                               # 0x0
	.size	x, 4

	.type	y,@object                       # @y
	.data
	.globl	y
	.p2align	2, 0x0
y:
	.long	1                               # 0x1
	.size	y, 4

	.section	.debug_loc.dwo,"e",@progbits
.Ldebug_loc0:
	.byte	3
	.byte	3
	.long	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # DW_OP_reg5
	.byte	3
	.byte	5
	.long	.Ltmp3-.Ltmp1
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.byte	3
	.byte	6
	.long	.Ltmp4-.Ltmp3
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	3
	.byte	7
	.long	.Lfunc_end0-.Ltmp4
	.short	1                               # Loc expr size
	.byte	94                              # DW_OP_reg14
	.byte	0
.Ldebug_loc1:
	.byte	3
	.byte	3
	.long	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	3
	.byte	5
	.long	.Ltmp2-.Ltmp1
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.byte	3
	.byte	8
	.long	.Ltmp4-.Ltmp2
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	3
	.byte	7
	.long	.Lfunc_end0-.Ltmp4
	.short	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.byte	0
.Ldebug_loc2:
	.byte	3
	.byte	4
	.long	.Ltmp9-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.byte	3
	.byte	9
	.long	.Lfunc_end1-.Ltmp9
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.byte	0
.Ldebug_loc3:
	.byte	3
	.byte	4
	.long	.Ltmp11-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	3
	.byte	10
	.long	.Ltmp13-.Ltmp11
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.byte	3
	.byte	11
	.long	.Ltmp15-.Ltmp13
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.byte	3
	.byte	12
	.long	.Lfunc_end1-.Ltmp15
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
	.byte	1                               # Abbrev [1] 0xb:0x29 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	-3506999759942021708            # DW_AT_GNU_dwo_id
	.long	.debug_ranges                   # DW_AT_GNU_ranges_base
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"main.dwo"                      # string offset=38
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"Foo"                           # string offset=0
.Linfo_string1:
	.asciz	"int"                           # string offset=4
.Linfo_string2:
	.asciz	"x"                             # string offset=8
.Linfo_string3:
	.asciz	"y"                             # string offset=10
.Linfo_string4:
	.asciz	"_Z3usePiS_"                    # string offset=12
.Linfo_string5:
	.asciz	"use"                           # string offset=23
.Linfo_string6:
	.asciz	"main"                          # string offset=27
.Linfo_string7:
	.asciz	"argc"                          # string offset=32
.Linfo_string8:
	.asciz	"argv"                          # string offset=37
.Linfo_string9:
	.asciz	"char"                          # string offset=42
.Linfo_string10:
	.asciz	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 3a8db0f4bfb57348f49d9603119fa157114bbf8e)" # string offset=47
.Linfo_string11:
	.asciz	"main.cpp"                      # string offset=148
.Linfo_string12:
	.asciz	"main.dwo"                      # string offset=157
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	4
	.long	8
	.long	10
	.long	12
	.long	23
	.long	27
	.long	32
	.long	37
	.long	42
	.long	47
	.long	148
	.long	157
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xb1 DW_TAG_compile_unit
	.byte	10                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	11                              # DW_AT_name
	.byte	12                              # DW_AT_GNU_dwo_name
	.quad	-3506999759942021708            # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0x2a DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	93                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x25:0xb DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	67                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	0
	.byte	4                               # Abbrev [4] 0x30:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc0-.debug_loc.dwo     # DW_AT_location
	.long	99                              # DW_AT_abstract_origin
	.byte	4                               # Abbrev [4] 0x39:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc1-.debug_loc.dwo     # DW_AT_location
	.long	107                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x43:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x47:0xb DW_TAG_variable
	.byte	2                               # DW_AT_name
	.long	67                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	1
	.byte	6                               # Abbrev [6] 0x52:0xb DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	67                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	2
	.byte	7                               # Abbrev [7] 0x5d:0x17 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	8                               # Abbrev [8] 0x63:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	116                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x6b:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	116                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x74:0x5 DW_TAG_pointer_type
	.long	67                              # DW_AT_type
	.byte	10                              # Abbrev [10] 0x79:0x34 DW_TAG_subprogram
	.byte	4                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	67                              # DW_AT_type
                                        # DW_AT_external
	.byte	11                              # Abbrev [11] 0x88:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc2-.debug_loc.dwo     # DW_AT_location
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	67                              # DW_AT_type
	.byte	11                              # Abbrev [11] 0x94:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc3-.debug_loc.dwo     # DW_AT_location
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	173                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0xa0:0xc DW_TAG_inlined_subroutine
	.long	93                              # DW_AT_abstract_origin
	.long	.Ldebug_ranges0-.debug_ranges   # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	12                              # DW_AT_call_line
	.byte	4                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0xad:0x5 DW_TAG_pointer_type
	.long	178                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xb2:0x5 DW_TAG_pointer_type
	.long	183                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xb7:0x4 DW_TAG_base_type
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
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
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
	.byte	11                              # Abbreviation Code
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
	.byte	12                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
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
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	_ZZ3usePiS_E3Foo
	.quad	x
	.quad	y
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp1
	.quad	.Ltmp3
	.quad	.Ltmp4
	.quad	.Ltmp2
	.quad	.Ltmp9
	.quad	.Ltmp11
	.quad	.Ltmp13
	.quad	.Ltmp15
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	52                              # Compilation Unit Length
	.long	37                              # DIE offset
	.byte	160                             # Attributes: VARIABLE, STATIC
	.asciz	"use::Foo"                      # External Name
	.long	71                              # DIE offset
	.byte	32                              # Attributes: VARIABLE, EXTERNAL
	.asciz	"x"                             # External Name
	.long	82                              # DIE offset
	.byte	32                              # Attributes: VARIABLE, EXTERNAL
	.asciz	"y"                             # External Name
	.long	93                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"use"                           # External Name
	.long	121                             # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	52                              # Compilation Unit Length
	.long	67                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	183                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 3a8db0f4bfb57348f49d9603119fa157114bbf8e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _ZGVZ3usePiS_E3Foo
	.section	.debug_line,"",@progbits
.Lline_table_start0:

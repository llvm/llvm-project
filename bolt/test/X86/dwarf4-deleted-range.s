# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %tmain.o
# RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-children --name=main --debug-info %t.bolt \
# RUN:   | FileCheck %s

# CHECK: 		DW_TAG_inlined_subroutine
# CHECK:      DW_AT_low_pc
# CHECK-SAME: 0x0000000000000000

# CHECK:    DW_TAG_GNU_call_site
# CHECK:      DW_AT_low_pc
# CHECK-SAME: 0x0000000000000000

## Test that llvm-bolt correctly updates DIEs corresponding to deleted code.

# Test case built from the following source using:
#
#   clang -O2 -g ...
#
# Assembly modified with "je" -> "jmp" to introduce unreachable block.
#
# extern void puts(const char *);
#
# static void foo() {
#   puts("hi");
# }
#
# int main(int argc, char **argv) {
#   if (argc)
#     foo();
#   return 0;
# }

	.text
	.file	"unreachable.c"
	.file	1 "." "unreachable.c"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.loc	1 7 0                           # unreachable.c:7:0
	.cfi_startproc
# %bb.0:
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	.loc	1 8 7 prologue_end              # unreachable.c:8:7
	testl	%edi, %edi
.Ltmp0:
	.loc	1 8 7 is_stmt 0                 # unreachable.c:8:7
	jmp	.LBB0_2
.Ltmp1:
# %bb.1:
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	pushq	%rax
	.cfi_def_cfa_offset 16
.Ltmp2:
	.loc	1 4 3 is_stmt 1                 # unreachable.c:4:3
	movl	$.L.str, %edi
.Ltmp3:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	callq	puts
.Ltmp4:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	.loc	1 0 3 is_stmt 0                 # unreachable.c:0:3
	addq	$8, %rsp
.Ltmp5:
	.cfi_def_cfa_offset 8
.LBB0_2:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rsi
	.loc	1 10 3 is_stmt 1                # unreachable.c:10:3
	xorl	%eax, %eax
	retq
.Ltmp6:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"hi"
	.size	.L.str, 3

	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	84                              # DW_OP_reg4
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	84                              # DW_OP_reg4
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
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
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
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
	.byte	3                               # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
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
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
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
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	10                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
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
	.byte	11                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
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
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	12                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x11 DW_TAG_variable
	.long	59                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	.L.str
	.byte	3                               # Abbrev [3] 0x3b:0xc DW_TAG_array_type
	.long	71                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x40:0x6 DW_TAG_subrange_type
	.long	78                              # DW_AT_type
	.byte	3                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x47:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x4e:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	7                               # Abbrev [7] 0x55:0x8 DW_TAG_subprogram
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	1                               # DW_AT_inline
	.byte	8                               # Abbrev [8] 0x5d:0x59 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	205                             # DW_AT_type
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0x76:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	205                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x85:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	212                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x94:0x14 DW_TAG_inlined_subroutine
	.long	85                              # DW_AT_abstract_origin
	.quad	.Ltmp2                          # DW_AT_low_pc
	.long	.Ltmp5-.Ltmp2                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	9                               # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	11                              # Abbrev [11] 0xa8:0xd DW_TAG_GNU_call_site
	.long	182                             # DW_AT_abstract_origin
	.quad	.Ltmp4                          # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xb6:0xd DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	13                              # Abbrev [13] 0xbd:0x5 DW_TAG_formal_parameter
	.long	195                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	14                              # Abbrev [14] 0xc3:0x5 DW_TAG_pointer_type
	.long	200                             # DW_AT_type
	.byte	15                              # Abbrev [15] 0xc8:0x5 DW_TAG_const_type
	.long	71                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0xcd:0x7 DW_TAG_base_type
	.long	.Linfo_string8                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	14                              # Abbrev [14] 0xd4:0x5 DW_TAG_pointer_type
	.long	217                             # DW_AT_type
	.byte	14                              # Abbrev [14] 0xd9:0x5 DW_TAG_pointer_type
	.long	71                              # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 15.0.7                                                " # string offset=0
.Linfo_string1:
	.asciz	"unreachable.c"                 # string offset=69
.Linfo_string2:
	.asciz	"."                             # string offset=83
.Linfo_string3:
	.asciz	"char"                          # string offset=85
.Linfo_string4:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=90
.Linfo_string5:
	.asciz	"foo"                           # string offset=110
.Linfo_string6:
	.asciz	"puts"                          # string offset=114
.Linfo_string7:
	.asciz	"main"                          # string offset=119
.Linfo_string8:
	.asciz	"int"                           # string offset=124
.Linfo_string9:
	.asciz	"argc"                          # string offset=128
.Linfo_string10:
	.asciz	"argv"                          # string offset=133
	.section	.debug_line,"",@progbits
.Lline_table_start0:

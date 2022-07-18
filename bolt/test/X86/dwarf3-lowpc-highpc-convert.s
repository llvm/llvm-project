# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=3 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-3 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-ranges %t.bolt > %t.txt
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt >> %t.txt
# RUN: cat %t.txt | FileCheck --check-prefix=POSTCHECK %s

# This tests checks that DW_AT_high_pc[DW_FORM_ADDR] can be converted to DW_AT_ranges correctly in Dwarf3

# PRECHECK: version = 0x0003
# PRECHECK: DW_AT_low_pc
# PRECHECK-SAME: DW_FORM_addr
# PRECHECK-SAME: 0x[[#%.16x,ADDR_LOW:]]
# PRECHECK-NEXT: DW_AT_high_pc [DW_FORM_addr]	(0x[[#ADDR_LOW + 41]])

# POSTCHECK: [[#%.8x,OFFSET:]] [[#%.16x, ADDR_1_BEGIN:]]
# POSTCHECK-SAME: [[#%.16x, ADDR_1_END:]]
# POSTCHECK-NEXT: [[#OFFSET]]
# POSTCHECK-SAME: [[#%.16x, ADDR_2_BEGIN:]]
# POSTCHECK-SAME: [[#%.16x, ADDR_2_END:]]
# POSTCHECK: version = 0x0003
# POSTCHECK: DW_AT_low_pc
# POSTCHECK-SAME: DW_FORM_addr
# POSTCHECK-SAME: (0x0000000000000000)
# POSTCHECK-NEXT: DW_AT_ranges
# POSTCHECK-SAME: DW_FORM_sec_offset
# POSTCHECK-SAME: (0x[[#OFFSET]]
# POSTCHECK-NEXT: [0x[[#ADDR_1_BEGIN]]
# POSTCHECK-SAME: 0x[[#ADDR_1_END]]
# POSTCHECK-NEXT: [0x[[#ADDR_2_BEGIN]]
# POSTCHECK-SAME: 0x[[#ADDR_2_END]]


# clang++ -g -gdwarf-3 -emit-llvm -S -O2 main.cpp
#
# void use(int * x, int * y) {
# *x += 4;
# *y -= 2;
# }

# int x = 0;
# int y = 1;
# int  main(int argc, char *argv[]) {
#     x = argc;
#     y = argc + 3;
#     use(&x, &y);
#     return x+y;
# }

	.text
	.file	"main.cpp"
	.file	1 "/home/test" "main.cpp"
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
	.loc	1 8 0                           # main.cpp:8:0
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
	leal	1(%rdi), %eax
	movl	%eax, y(%rip)
.Ltmp2:
	#DEBUG_VALUE: use:y <- undef
	.loc	1 12 13                         # main.cpp:12:13
	leal	(%rdi,%rdi), %eax
	addl	$5, %eax
	.loc	1 12 5 is_stmt 0                # main.cpp:12:5
	retq
.Ltmp3:
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
	.byte	6                               # DW_FORM_data4
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	12                              # DW_FORM_flag
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	10                              # DW_FORM_block1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	1                               # DW_FORM_addr
	.byte	64                              # DW_AT_frame_base
	.byte	10                              # DW_FORM_block1
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	10                              # DW_FORM_block1
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.ascii	"\207@"                         # DW_AT_MIPS_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	12                              # DW_FORM_flag
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
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
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	1                               # DW_FORM_addr
	.byte	64                              # DW_AT_frame_base
	.byte	10                              # DW_FORM_block1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	12                              # DW_FORM_flag
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	10                              # DW_FORM_block1
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
	.byte	11                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	1                               # DW_FORM_addr
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	3                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x108 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.quad	.Lfunc_end1                     # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2e:0x16 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	68                              # DW_AT_type
	.byte	1                               # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	x
	.byte	3                               # Abbrev [3] 0x44:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x4b:0x16 DW_TAG_variable
	.long	.Linfo_string5                  # DW_AT_name
	.long	68                              # DW_AT_type
	.byte	1                               # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	y
	.byte	4                               # Abbrev [4] 0x61:0x26 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.quad	.Lfunc_end0                     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	135                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x78:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	148                             # DW_AT_abstract_origin
	.byte	5                               # Abbrev [5] 0x7f:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	159                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x87:0x24 DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_MIPS_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	1                               # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	7                               # Abbrev [7] 0x94:0xb DW_TAG_formal_parameter
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	171                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x9f:0xb DW_TAG_formal_parameter
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	171                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0xab:0x5 DW_TAG_pointer_type
	.long	68                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0xb0:0x51 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.quad	.Lfunc_end1                     # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	1                               # DW_AT_external
	.byte	10                              # Abbrev [10] 0xce:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	10                              # Abbrev [10] 0xdb:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	257                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xe8:0x18 DW_TAG_inlined_subroutine
	.long	135                             # DW_AT_abstract_origin
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.quad	.Ltmp2                          # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	11                              # DW_AT_call_line
	.byte	5                               # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x101:0x5 DW_TAG_pointer_type
	.long	262                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x106:0x5 DW_TAG_pointer_type
	.long	267                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x10b:0x7 DW_TAG_base_type
	.long	.Linfo_string11                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 14.0.0" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=134
.Linfo_string2:
	.asciz	"/home/test"               # string offset=143
.Linfo_string3:
	.asciz	"x"                             # string offset=159
.Linfo_string4:
	.asciz	"int"                           # string offset=161
.Linfo_string5:
	.asciz	"y"                             # string offset=165
.Linfo_string6:
	.asciz	"_Z3usePiS_"                    # string offset=167
.Linfo_string7:
	.asciz	"use"                           # string offset=178
.Linfo_string8:
	.asciz	"main"                          # string offset=182
.Linfo_string9:
	.asciz	"argc"                          # string offset=187
.Linfo_string10:
	.asciz	"argv"                          # string offset=192
.Linfo_string11:
	.asciz	"char"                          # string offset=197
	.ident	"clang version 14.0.0"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:

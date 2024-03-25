# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q -Wl,-gc-sections -fuse-ld=lld -Wl,--entry=main
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-dwarfdump --debug-info %t.bolt | FileCheck --check-prefix=POSTCHECK %s

# PRECHECK:       DW_TAG_subprogram
# PRECHECK-NEXT:  DW_AT_ranges
# PRECHECK-NEXT:    [0x0000000000000000
# PRECHECK-NEXT:    [0x0000000000000000
# PRECHECK-NEXT:    [0x0000000000000000
# PRECHECK-NEXT:    [0x0000000000000000
# PRECHECK-NEXT:  DW_AT_frame_base
# PRECHECK-NEXT:  DW_AT_linkage_name  ("_Z6helperi")
# PRECHECK-NEXT:  DW_AT_name  ("helper")

# POSTCHECK:      DW_TAG_subprogram
# POSTCHECK-NEXT: DW_AT_frame_base
# POSTCHECK-NEXT: DW_AT_linkage_name  ("_Z6helperi")
# POSTCHECK-NEXT: DW_AT_name  ("helper")
# POSTCHECK-NEXT: DW_AT_decl_file
# POSTCHECK-NEXT: DW_AT_decl_line
# POSTCHECK-NEXT: DW_AT_type
# POSTCHECK-NEXT: DW_AT_external
# POSTCHECK-NEXT: DW_AT_low_pc  (0x0000000000000000)
# POSTCHECK-NEXT: DW_AT_high_pc (0x0000000000000001)

## Tests BOLT path that handles DW_AT_ranges with no output function ranges.

## clang++ main.cpp -O0 -fno-inline-functions -fbasic-block-sections=all -g2 -S
## int helper(int argc) {
##   int x = argc;
##   if (x == 3)
##     x++;
##   else
##     x--;
##   return x;
## }
## int  main(int argc, char *argv[]) {
##   int x = argc;
##   if (x == 3)
##     x++;
##   else
##     x--;
##   return x;
## }

	.text
	.file	"main.cpp"
	.section	.text._Z6helperi,"ax",@progbits
	.globl	_Z6helperi                      # -- Begin function _Z6helperi
	.p2align	4, 0x90
	.type	_Z6helperi,@function
_Z6helperi:                             # @_Z6helperi
.Lfunc_begin0:
	.file	0 "/repro2" "main.cpp" md5 0x888a2704226ec400f256aa9c2207456c
	.loc	0 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	0 2 11 prologue_end             # main.cpp:2:11
	movl	-4(%rbp), %eax
	.loc	0 2 7 is_stmt 0                 # main.cpp:2:7
	movl	%eax, -8(%rbp)
.Ltmp1:
	.loc	0 3 9 is_stmt 1                 # main.cpp:3:9
	cmpl	$3, -8(%rbp)
.Ltmp2:
	.loc	0 3 7 is_stmt 0                 # main.cpp:3:7
	jne	_Z6helperi.__part.2
	jmp	_Z6helperi.__part.1
.LBB_END0_0:
	.cfi_endproc
	.section	.text._Z6helperi,"ax",@progbits,unique,1
_Z6helperi.__part.1:                    # %if.then
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 4 6 is_stmt 1                 # main.cpp:4:6
	movl	-8(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -8(%rbp)
	.loc	0 4 5 is_stmt 0                 # main.cpp:4:5
	jmp	_Z6helperi.__part.3
.LBB_END0_1:
	.size	_Z6helperi.__part.1, .LBB_END0_1-_Z6helperi.__part.1
	.cfi_endproc
	.section	.text._Z6helperi,"ax",@progbits,unique,2
_Z6helperi.__part.2:                    # %if.else
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 6 6 is_stmt 1                 # main.cpp:6:6
	movl	-8(%rbp), %eax
	addl	$-1, %eax
	movl	%eax, -8(%rbp)
	jmp	_Z6helperi.__part.3
.LBB_END0_2:
	.size	_Z6helperi.__part.2, .LBB_END0_2-_Z6helperi.__part.2
	.cfi_endproc
	.section	.text._Z6helperi,"ax",@progbits,unique,3
_Z6helperi.__part.3:                    # %if.end
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 7 10                          # main.cpp:7:10
	movl	-8(%rbp), %eax
	.loc	0 7 3 epilogue_begin is_stmt 0  # main.cpp:7:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END0_3:
	.size	_Z6helperi.__part.3, .LBB_END0_3-_Z6helperi.__part.3
	.cfi_endproc
	.section	.text._Z6helperi,"ax",@progbits
.Lfunc_end0:
	.size	_Z6helperi, .Lfunc_end0-_Z6helperi
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	0 9 0 is_stmt 1                 # main.cpp:9:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp3:
	.loc	0 10 11 prologue_end            # main.cpp:10:11
	movl	-8(%rbp), %eax
	.loc	0 10 7 is_stmt 0                # main.cpp:10:7
	movl	%eax, -20(%rbp)
.Ltmp4:
	.loc	0 11 9 is_stmt 1                # main.cpp:11:9
	cmpl	$3, -20(%rbp)
.Ltmp5:
	.loc	0 11 7 is_stmt 0                # main.cpp:11:7
	jne	main.__part.2
	jmp	main.__part.1
.LBB_END1_0:
	.cfi_endproc
	.section	.text.main,"ax",@progbits,unique,4
main.__part.1:                          # %if.then
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 12 6 is_stmt 1                # main.cpp:12:6
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	.loc	0 12 5 is_stmt 0                # main.cpp:12:5
	jmp	main.__part.3
.LBB_END1_1:
	.size	main.__part.1, .LBB_END1_1-main.__part.1
	.cfi_endproc
	.section	.text.main,"ax",@progbits,unique,5
main.__part.2:                          # %if.else
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 14 6 is_stmt 1                # main.cpp:14:6
	movl	-20(%rbp), %eax
	addl	$-1, %eax
	movl	%eax, -20(%rbp)
	jmp	main.__part.3
.LBB_END1_2:
	.size	main.__part.2, .LBB_END1_2-main.__part.2
	.cfi_endproc
	.section	.text.main,"ax",@progbits,unique,6
main.__part.3:                          # %if.end
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	0 15 10                         # main.cpp:15:10
	movl	-20(%rbp), %eax
	.loc	0 15 3 epilogue_begin is_stmt 0 # main.cpp:15:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB_END1_3:
	.size	main.__part.3, .LBB_END1_3-main.__part.3
	.cfi_endproc
	.section	.text.main,"ax",@progbits
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
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
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
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
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
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
	.byte	1                               # Abbrev [1] 0xc:0x82 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.byte	2                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2b:0x23 DW_TAG_subprogram
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	123                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x37:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	123                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x42:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	123                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x4e:0x2d DW_TAG_subprogram
	.byte	1                               # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	123                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x59:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	123                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0x64:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	127                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x6f:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	108
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	123                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x7b:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x7f:0x5 DW_TAG_pointer_type
	.long	132                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x84:0x5 DW_TAG_pointer_type
	.long	137                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0x89:0x4 DW_TAG_base_type
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
	.long	3                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
	.long	.Ldebug_ranges2-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .LBB_END0_1-_Z6helperi.__part.1 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .LBB_END0_2-_Z6helperi.__part.2 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	2                               #   start index
	.uleb128 .LBB_END0_3-_Z6helperi.__part.3 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	3                               #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	3                               # DW_RLE_startx_length
	.byte	4                               #   start index
	.uleb128 .LBB_END1_1-main.__part.1      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	5                               #   start index
	.uleb128 .LBB_END1_2-main.__part.2      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	6                               #   start index
	.uleb128 .LBB_END1_3-main.__part.3      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	7                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_ranges2:
	.byte	3                               # DW_RLE_startx_length
	.byte	0                               #   start index
	.uleb128 .LBB_END0_1-_Z6helperi.__part.1 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .LBB_END0_2-_Z6helperi.__part.2 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	2                               #   start index
	.uleb128 .LBB_END0_3-_Z6helperi.__part.3 #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	3                               #   start index
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	4                               #   start index
	.uleb128 .LBB_END1_1-main.__part.1      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	5                               #   start index
	.uleb128 .LBB_END1_2-main.__part.2      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	6                               #   start index
	.uleb128 .LBB_END1_3-main.__part.3      #   length
	.byte	3                               # DW_RLE_startx_length
	.byte	7                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	48                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:ayermolo/llvm-project.git a1d8664d409cac2a923176a8e9a731385bde279e)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=108
.Linfo_string2:
	.asciz	"/repro2" # string offset=117
.Linfo_string3:
	.asciz	"_Z6helperi"                    # string offset=162
.Linfo_string4:
	.asciz	"helper"                        # string offset=173
.Linfo_string5:
	.asciz	"int"                           # string offset=180
.Linfo_string6:
	.asciz	"main"                          # string offset=184
.Linfo_string7:
	.asciz	"argc"                          # string offset=189
.Linfo_string8:
	.asciz	"x"                             # string offset=194
.Linfo_string9:
	.asciz	"argv"                          # string offset=196
.Linfo_string10:
	.asciz	"char"                          # string offset=201
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	_Z6helperi.__part.1
	.quad	_Z6helperi.__part.2
	.quad	_Z6helperi.__part.3
	.quad	.Lfunc_begin0
	.quad	main.__part.1
	.quad	main.__part.2
	.quad	main.__part.3
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.ident	"clang version 19.0.0git (git@github.com:ayermolo/llvm-project.git a1d8664d409cac2a923176a8e9a731385bde279e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

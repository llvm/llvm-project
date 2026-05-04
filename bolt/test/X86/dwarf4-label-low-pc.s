
# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %tmain.o
# RUN: %clang %cflags -dwarf-4 %tmain.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt > %t.txt
# RUN: llvm-objdump -d %t.bolt >> %t.txt
# RUN: cat %t.txt | FileCheck --check-prefix=POSTCHECK %s

## This test checks that we correctly handle DW_AT_low_pc [DW_FORM_addr] that is part of DW_TAG_label.

# PRECHECK: version = 0x0004
# PRECHECK: DW_TAG_label
# PRECHECK-NEXT: DW_AT_name
# PRECHECK-NEXT: DW_AT_decl_file
# PRECHECK-NEXT: DW_AT_decl_line
# PRECHECK-NEXT:DW_AT_low_pc [DW_FORM_addr]
# PRECHECK: DW_TAG_label
# PRECHECK-NEXT: DW_AT_name
# PRECHECK-NEXT: DW_AT_decl_file
# PRECHECK-NEXT: DW_AT_decl_line
# PRECHECK-NEXT:DW_AT_low_pc [DW_FORM_addr]

# POSTCHECK: version = 0x0004
# POSTCHECK: DW_TAG_label
# POSTCHECK-NEXT: DW_AT_name
# POSTCHECK-NEXT: DW_AT_decl_file
# POSTCHECK-NEXT: DW_AT_decl_line
# POSTCHECK-NEXT:DW_AT_low_pc [DW_FORM_addr] (0x[[ADDR:[1-9a-f]*]]
# POSTCHECK: DW_TAG_label
# POSTCHECK-NEXT: DW_AT_name
# POSTCHECK-NEXT: DW_AT_decl_file
# POSTCHECK-NEXT: DW_AT_decl_line
# POSTCHECK-NEXT:DW_AT_low_pc [DW_FORM_addr] (0x[[ADDR2:[1-9a-f]*]]

# POSTCHECK: [[ADDR]]: 8b 45 f8
# POSTCHECK: [[ADDR2]]: 8b 45 f8

## clang++ main.cpp -g2 -gdwarf-4 -S
## int main() {
##   int a = 4;
##   if (a == 5)
##     goto LABEL1;
##   else
##     goto LABEL2;
##   LABEL1:a++;
##   LABEL2:a--;
##   return 0;
## }

		.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/home" "main.cpp"
	.loc	1 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
.Ltmp0:
	.loc	1 2 7 prologue_end              # main.cpp:2:7
	movl	$4, -8(%rbp)
.Ltmp1:
	.loc	1 3 9                           # main.cpp:3:9
	cmpl	$5, -8(%rbp)
.Ltmp2:
	.loc	1 3 7 is_stmt 0                 # main.cpp:3:7
	jne	.LBB0_2
# %bb.1:                                # %if.then
.Ltmp3:
	.loc	1 4 5 is_stmt 1                 # main.cpp:4:5
	jmp	.LBB0_3
.LBB0_2:                                # %if.else
	.loc	1 6 5                           # main.cpp:6:5
	jmp	.LBB0_4
.Ltmp4:
.LBB0_3:                                # %LABEL1
	#DEBUG_LABEL: main:LABEL1
	.loc	1 7 11                          # main.cpp:7:11
	movl	-8(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -8(%rbp)
.LBB0_4:                                # %LABEL2
.Ltmp5:
	#DEBUG_LABEL: main:LABEL2
	.loc	1 8 11                          # main.cpp:8:11
	movl	-8(%rbp), %eax
	addl	$-1, %eax
	movl	%eax, -8(%rbp)
	.loc	1 9 3                           # main.cpp:9:3
	xorl	%eax, %eax
	.loc	1 9 3 epilogue_begin is_stmt 0  # main.cpp:9:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp6:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
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
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
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
	.byte	52                              # DW_TAG_variable
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
	.byte	10                              # DW_TAG_label
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x6d DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	112                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x43:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	112                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x51:0xf DW_TAG_label
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.quad	.Ltmp4                          # DW_AT_low_pc
	.byte	4                               # Abbrev [4] 0x60:0xf DW_TAG_label
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.quad	.Ltmp5                          # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x70:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/home" # string offset=33
.Linfo_string3:
	.asciz	"main"                          # string offset=71
.Linfo_string4:
	.asciz	"int"                           # string offset=76
.Linfo_string5:
	.asciz	"a"                             # string offset=80
.Linfo_string6:
	.asciz	"LABEL1"                        # string offset=82
.Linfo_string7:
	.asciz	"LABEL2"                        # string offset=89
	.ident	"clang version 19.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

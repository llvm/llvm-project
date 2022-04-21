
# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %tmain.o
# RUN: %clang %cflags -dwarf-5 %tmain.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt -update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s

# RUN: llvm-dwarfdump --show-form --verbose --debug-addr %t.bolt > %t.txt
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt >> %t.txt

# This test checks that we correctly handle DW_AT_low_pc [DW_FORM_addrx] that is part of DW_TAG_label.

# PRECHECK: version = 0x0005
# PRECHECK: DW_TAG_label
# PRECHECK-NEXT: DW_AT_name
# PRECHECK-NEXT: DW_AT_decl_file
# PRECHECK-NEXT: DW_AT_decl_line
# PRECHECK-NEXT:DW_AT_low_pc [DW_FORM_addrx]  (indexed (00000001)
# PRECHECK: DW_TAG_label
# PRECHECK-NEXT: DW_AT_name
# PRECHECK-NEXT: DW_AT_decl_file
# PRECHECK-NEXT: DW_AT_decl_line
# PRECHECK-NEXT:DW_AT_low_pc [DW_FORM_addrx]  (indexed (00000002)


# POSTCHECK: Addrs: [
# POSTCHECK-NEXT: 0x
# POSTCHECK-NEXT: 0x
# POSTCHECK-NEXT: 0x[[#%.16x,ADDR:]]
# POSTCHECK-NEXT: 0x[[#%.16x,ADDR2:]]

# POSTCHECK: version = 0x0005
# POSTCHECK: DW_TAG_label
# POSTCHECK-NEXT: DW_AT_name
# POSTCHECK-NEXT: DW_AT_decl_file
# POSTCHECK-NEXT: DW_AT_decl_line
# POSTCHECK-NEXT:
# POSTCHECK-NEXT:DW_AT_low_pc [DW_FORM_addrx]  (indexed (00000002)
# POSTCHECK-SAME: [0x[[#ADDR]]
# POSTCHECK: DW_TAG_label
# POSTCHECK-NEXT: DW_AT_name
# POSTCHECK-NEXT: DW_AT_decl_file
# POSTCHECK-NEXT: DW_AT_decl_line
# POSTCHECK-NEXT:
# POSTCHECK-NEXT:DW_AT_low_pc [DW_FORM_addrx]  (indexed (00000003)
# POSTCHECK-SAME: [0x[[#ADDR2]]

# clang++ main.cpp -g -S
# int main() {
#   int a = 4;
#   if (a == 5)
#     goto LABEL1;
#   else
#     goto LABEL2;
#   LABEL1:a++;
#   LABEL2:a--;
#   return 0;
# }

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "/testLabel" "main.cpp" md5 0xa0bd66020d06f1303de7008e3c542050
	.loc	0 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
.Ltmp0:
	.loc	0 2 7 prologue_end              # main.cpp:2:7
	movl	$4, -8(%rbp)
.Ltmp1:
	.loc	0 3 9                           # main.cpp:3:9
	cmpl	$5, -8(%rbp)
.Ltmp2:
	.loc	0 3 7 is_stmt 0                 # main.cpp:3:7
	jne	.LBB0_2
# %bb.1:                                # %if.then
.Ltmp3:
	.loc	0 4 5 is_stmt 1                 # main.cpp:4:5
	jmp	.LBB0_3
.LBB0_2:                                # %if.else
	.loc	0 6 5                           # main.cpp:6:5
	jmp	.LBB0_4
.Ltmp4:
.LBB0_3:                                # %LABEL1
	#DEBUG_LABEL: main:LABEL1
	.loc	0 7 11                          # main.cpp:7:11
	movl	-8(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -8(%rbp)
.LBB0_4:                                # %LABEL2
.Ltmp5:
	#DEBUG_LABEL: main:LABEL2
	.loc	0 8 11                          # main.cpp:8:11
	movl	-8(%rbp), %eax
	addl	$-1, %eax
	movl	%eax, -8(%rbp)
	.loc	0 9 3                           # main.cpp:9:3
	xorl	%eax, %eax
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
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
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
	.byte	4                               # Abbreviation Code
	.byte	10                              # DW_TAG_label
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x41 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x25 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	72                              # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x32:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	72                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x3d:0x5 DW_TAG_label
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	1                               # DW_AT_low_pc
	.byte	4                               # Abbrev [4] 0x42:0x5 DW_TAG_label
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	2                               # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x48:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	36                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 15.0.0" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=134
.Linfo_string2:
	.asciz	"/testLabel" # string offset=143
.Linfo_string3:
	.asciz	"main"                          # string offset=190
.Linfo_string4:
	.asciz	"int"                           # string offset=195
.Linfo_string5:
	.asciz	"a"                             # string offset=199
.Linfo_string6:
	.asciz	"LABEL1"                        # string offset=201
.Linfo_string7:
	.asciz	"LABEL2"                        # string offset=208
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp4
	.quad	.Ltmp5
.Ldebug_addr_end0:
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

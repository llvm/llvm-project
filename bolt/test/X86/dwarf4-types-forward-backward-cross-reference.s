# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -gdwarf-4 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt | FileCheck --check-prefix=POSTCHECK %s

## This test checks that BOLT handles correctly backward and forward cross CU references
## for DWARF4 with -fdebug-types-section

# POSTCHECK: version = 0x0004
# POSTCHECK: DW_TAG_variable [10]
# POSTCHECK: DW_AT_type [DW_FORM_ref_addr] (0x0000000000000112 "Foo2a")

# POSTCHECK: version = 0x0004
# POSTCHECK: DW_TAG_variable [10]
# POSTCHECK: DW_AT_type [DW_FORM_ref_addr] (0x0000000000000094 "Foo")
# POSTCHECK: DW_TAG_variable [10]
# POSTCHECK: DW_AT_type [DW_FORM_ref_addr] (0x0000000000000190 "Foo3a")

# main.cpp
# struct Foo {
#  char *c1;
#  char *c2;
#  char *c3;
# };
# struct Foo2 {
#  char *c1;
#  char *c2;
# };
# int main(int argc, char *argv[]) {
#  Foo f;
#  f.c1 = argv[argc];
#  f.c2 = argv[argc + 1];
#  f.c3 = argv[argc + 2];
#  Foo2 f2;
#  f.c1 = argv[argc + 3];
#  f.c2 = argv[argc + 4];
#  return 0;
# }

# helper.cpp
# struct Foo2a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# struct Foo3 {
#   char *c1;
#   char *c2;
# };
#
# int foo() {
#   Foo2a f;
#   Foo3 f2;
#   return 0;
# }

# helper1.cpp
# struct Foo3a {
#   char *c1;
#   char *c2;
#   char *c3;
# };
# struct Foo4 {
#   char *c1;
#   char *c2;
# };
#
# int foo2() {
#   Foo3a f;
#   Foo4 f2;
#   return 0;
# }


	.text
	.file	"llvm-link"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/dwarf4-types-forward-cross-reference-test" "main.cpp"
	.loc	1 10 0                          # main.cpp:10:0
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
.Ltmp0:
	.loc	1 12 9 prologue_end             # main.cpp:12:9
	movq	-16(%rbp), %rax
	movslq	-8(%rbp), %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 12 7 is_stmt 0                # main.cpp:12:7
	movq	%rax, -40(%rbp)
	.loc	1 13 9 is_stmt 1                # main.cpp:13:9
	movq	-16(%rbp), %rax
	.loc	1 13 14 is_stmt 0               # main.cpp:13:14
	movl	-8(%rbp), %ecx
	.loc	1 13 19                         # main.cpp:13:19
	addl	$1, %ecx
	.loc	1 13 9                          # main.cpp:13:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 13 7                          # main.cpp:13:7
	movq	%rax, -32(%rbp)
	.loc	1 14 9 is_stmt 1                # main.cpp:14:9
	movq	-16(%rbp), %rax
	.loc	1 14 14 is_stmt 0               # main.cpp:14:14
	movl	-8(%rbp), %ecx
	.loc	1 14 19                         # main.cpp:14:19
	addl	$2, %ecx
	.loc	1 14 9                          # main.cpp:14:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 14 7                          # main.cpp:14:7
	movq	%rax, -24(%rbp)
	.loc	1 16 9 is_stmt 1                # main.cpp:16:9
	movq	-16(%rbp), %rax
	.loc	1 16 14 is_stmt 0               # main.cpp:16:14
	movl	-8(%rbp), %ecx
	.loc	1 16 19                         # main.cpp:16:19
	addl	$3, %ecx
	.loc	1 16 9                          # main.cpp:16:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 16 7                          # main.cpp:16:7
	movq	%rax, -40(%rbp)
	.loc	1 17 9 is_stmt 1                # main.cpp:17:9
	movq	-16(%rbp), %rax
	.loc	1 17 14 is_stmt 0               # main.cpp:17:14
	movl	-8(%rbp), %ecx
	.loc	1 17 19                         # main.cpp:17:19
	addl	$4, %ecx
	.loc	1 17 9                          # main.cpp:17:9
	movslq	%ecx, %rcx
	movq	(%rax,%rcx,8), %rax
	.loc	1 17 7                          # main.cpp:17:7
	movq	%rax, -32(%rbp)
	.loc	1 18 2 is_stmt 1                # main.cpp:18:2
	xorl	%eax, %eax
	.loc	1 18 2 epilogue_begin is_stmt 0 # main.cpp:18:2
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4, 0x90
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin1:
	.file	2 "/dwarf4-types-forward-cross-reference-test" "helper.cpp"
	.loc	2 11 0 is_stmt 1                # helper.cpp:11:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp2:
	.loc	2 14 3 prologue_end             # helper.cpp:14:3
	xorl	%eax, %eax
	.loc	2 14 3 epilogue_begin is_stmt 0 # helper.cpp:14:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	_Z3foov, .Lfunc_end1-_Z3foov
	.cfi_endproc
                                        # -- End function
	.globl	_Z4foo2v                        # -- Begin function _Z4foo2v
	.p2align	4, 0x90
	.type	_Z4foo2v,@function
_Z4foo2v:                               # @_Z4foo2v
.Lfunc_begin2:
	.file	3 "/dwarf4-types-forward-cross-reference-test" "helper1.cpp"
	.loc	3 11 0 is_stmt 1                # helper1.cpp:11:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp4:
	.loc	3 14 3 prologue_end             # helper1.cpp:14:3
	xorl	%eax, %eax
	.loc	3 14 3 epilogue_begin is_stmt 0 # helper1.cpp:14:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp5:
.Lfunc_end2:
	.size	_Z4foo2v, .Lfunc_end2-_Z4foo2v
	.cfi_endproc
                                        # -- End function
	.section	.debug_types,"G",@progbits,7448148824980338162,comdat
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	7448148824980338162             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string18                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string17                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_types,"G",@progbits,5322170643381124694,comdat
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	5322170643381124694             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string20                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_types,"G",@progbits,1175092228111723119,comdat
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	1175092228111723119             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string21                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string17                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_types,"G",@progbits,12995149649732825572,comdat
	.long	.Ldebug_info_end3-.Ldebug_info_start3 # Length of Unit
.Ldebug_info_start3:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-5451594423976726044            # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string22                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	2                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	2                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end3:
	.section	.debug_types,"G",@progbits,10693860647081617285,comdat
	.long	.Ldebug_info_end4-.Ldebug_info_start4 # Length of Unit
.Ldebug_info_start4:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-7752883426627934331            # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x42 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x2e DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string23                 # DW_AT_name
	.byte	24                              # DW_AT_byte_size
	.byte	3                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x3f:0xc DW_TAG_member
	.long	.Linfo_string17                 # DW_AT_name
	.long	76                              # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	16                              # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x4c:0x5 DW_TAG_pointer_type
	.long	81                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x51:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end4:
	.section	.debug_types,"G",@progbits,17604755499357858397,comdat
	.long	.Ldebug_info_end5-.Ldebug_info_start5 # Length of Unit
.Ldebug_info_start5:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-841988574351693219             # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0x36 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string24                 # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	3                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x27:0xc DW_TAG_member
	.long	.Linfo_string15                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	3                               # Abbrev [3] 0x33:0xc DW_TAG_member
	.long	.Linfo_string16                 # DW_AT_name
	.long	64                              # DW_AT_type
	.byte	3                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x40:0x5 DW_TAG_pointer_type
	.long	69                              # DW_AT_type
	.byte	5                               # Abbrev [5] 0x45:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end5:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
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
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
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
    .byte   12                              # Abbreviation Code <-- Manually added abbrev decl
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
    .byte   16                              # DW_FORM_ref_addr
    .byte   0                               # EOM(1)
    .byte   0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end6-.Ldebug_info_start6 # Length of Unit
.Ldebug_info_start6:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x9c DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x2a:0x52 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string5                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	124                             # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x43:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	124                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x51:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	131                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x5f:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string14                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	148                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x6d:0xe DW_TAG_variable <-- Manually modified s/9/12
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	72
	.long	.Linfo_string19                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	15                              # DW_AT_decl_line
	.long	.Lmanual_label                  # DW_AT_type <-- Manually modified
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x7c:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x83:0x5 DW_TAG_pointer_type
	.long	136                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x88:0x5 DW_TAG_pointer_type
	.long	141                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x8d:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
.Lmanual_label_forward:
	.byte	10                              # Abbrev [10] 0x94:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	7448148824980338162             # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x9d:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	5322170643381124694             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end6:
.Lcu_begin1:
	.long	.Ldebug_info_end7-.Ldebug_info_start7 # Length of Unit
.Ldebug_info_start7:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x73 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string3                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	11                              # Abbrev [11] 0x2a:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string7                  # DW_AT_linkage_name
	.long	.Linfo_string8                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	100                             # DW_AT_type
                                        # DW_AT_external
	.byte	12                              # Abbrev [12] 0x47:0xe DW_TAG_variable2 <-- Manually modified s/9/12
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string14                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	.Lmanual_label_forward          # DW_AT_type <-- Manually modified
	.byte	12                              # Abbrev [12] 0x55:0xe DW_TAG_variable <-- Manually modified s/9/12
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string19                 # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	.Lmanual_label2                 # DW_AT_type <-- Manually modified
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x64:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
.Lmanual_label:
	.byte	10                              # Abbrev [10] 0x6b:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	1175092228111723119             # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x74:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-5451594423976726044            # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end7:
.Lcu_begin2:
	.long	.Ldebug_info_end8-.Ldebug_info_start8 # Length of Unit
.Ldebug_info_start8:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	6                               # Abbrev [6] 0xb:0x73 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string4                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	11                              # Abbrev [11] 0x2a:0x3a DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string9                  # DW_AT_linkage_name
	.long	.Linfo_string10                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	100                             # DW_AT_type
                                        # DW_AT_external
	.byte	9                               # Abbrev [9] 0x47:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string14                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	12                              # DW_AT_decl_line
	.long	107                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x55:0xe DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	88
	.long	.Linfo_string19                 # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	116                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x64:0x7 DW_TAG_base_type
	.long	.Linfo_string6                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
.Lmanual_label2:
	.byte	10                              # Abbrev [10] 0x6b:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-7752883426627934331            # DW_AT_signature
	.byte	10                              # Abbrev [10] 0x74:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-841988574351693219             # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end8:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=105
.Linfo_string2:
	.asciz	"/dwarf4-types-forward-cross-reference-test" # string offset=114
.Linfo_string3:
	.asciz	"helper.cpp"                    # string offset=194
.Linfo_string4:
	.asciz	"helper1.cpp"                   # string offset=205
.Linfo_string5:
	.asciz	"main"                          # string offset=217
.Linfo_string6:
	.asciz	"int"                           # string offset=222
.Linfo_string7:
	.asciz	"_Z3foov"                       # string offset=226
.Linfo_string8:
	.asciz	"foo"                           # string offset=234
.Linfo_string9:
	.asciz	"_Z4foo2v"                      # string offset=238
.Linfo_string10:
	.asciz	"foo2"                          # string offset=247
.Linfo_string11:
	.asciz	"argc"                          # string offset=252
.Linfo_string12:
	.asciz	"argv"                          # string offset=257
.Linfo_string13:
	.asciz	"char"                          # string offset=262
.Linfo_string14:
	.asciz	"f"                             # string offset=267
.Linfo_string15:
	.asciz	"c1"                            # string offset=269
.Linfo_string16:
	.asciz	"c2"                            # string offset=272
.Linfo_string17:
	.asciz	"c3"                            # string offset=275
.Linfo_string18:
	.asciz	"Foo"                           # string offset=278
.Linfo_string19:
	.asciz	"f2"                            # string offset=282
.Linfo_string20:
	.asciz	"Foo2"                          # string offset=285
.Linfo_string21:
	.asciz	"Foo2a"                         # string offset=290
.Linfo_string22:
	.asciz	"Foo3"                          # string offset=296
.Linfo_string23:
	.asciz	"Foo3a"                         # string offset=301
.Linfo_string24:
	.asciz	"Foo4"                          # string offset=307
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)"
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)"
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 73027ae39b1492e5b6033358a13b86d7d1e781ae)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

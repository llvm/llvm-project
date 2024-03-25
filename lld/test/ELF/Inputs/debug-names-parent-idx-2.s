#-- input file: debug-names-parent-idx-2.cpp
## Generated with:
##
## - clang++ -g -O0 -gpubnames -fdebug-compilation-dir='parent-idx-test' \
##     -S debug-names-parent-idx-2.cpp -o debug-names-parent-idx-2.s
##
## foo.h contents:
##
## int foo();
##
## struct foo {
##   int x;
##   char y;
##   struct foo *foo_ptr;
## };
##
## namespace parent_test {
##   int foo();
## }
##
## debug-names-parent-index-2.cpp contents:
##
## #include "foo.h"
## int foo () {
##   struct foo struct2;
##   struct2.x = 1024;
##   struct2.y = 'r';
##   struct2.foo_ptr = nullptr;
##   return struct2.x * (int) struct2.y;
## }
##
## namespace parent_test {
## int foo () {
##   return 25;
## }
## }
##
	.text
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4, 0x90
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	movl	$1024, -16(%rbp)                # imm = 0x400
	movb	$114, -12(%rbp)
	movq	$0, -8(%rbp)
	movl	-16(%rbp), %eax
	movsbl	-12(%rbp), %ecx
	imull	%ecx, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
                                        # -- End function
	.globl	_ZN11parent_test3fooEv          # -- Begin function _ZN11parent_test3fooEv
	.p2align	4, 0x90
	.type	_ZN11parent_test3fooEv,@function
_ZN11parent_test3fooEv:                 # @_ZN11parent_test3fooEv
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp2:
	movl	$25, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	_ZN11parent_test3fooEv, .Lfunc_end1-_ZN11parent_test3fooEv
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
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
	.byte	1                               # Abbrev [1] 0xc:0x76 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x4 DW_TAG_base_type
	.byte	3                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x27:0x1c DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	5                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	35                              # DW_AT_type
                                        # DW_AT_external
	.byte	4                               # Abbrev [4] 0x37:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	86                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x43:0x13 DW_TAG_namespace
	.byte	4                               # DW_AT_name
	.byte	6                               # Abbrev [6] 0x45:0x10 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	7                               # DW_AT_linkage_name
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.long	35                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x56:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	6                               # DW_AT_name
	.byte	16                              # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	8                               # Abbrev [8] 0x5c:0x9 DW_TAG_member
	.byte	9                               # DW_AT_name
	.long	35                              # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	8                               # Abbrev [8] 0x65:0x9 DW_TAG_member
	.byte	10                              # DW_AT_name
	.long	120                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	4                               # DW_AT_data_member_location
	.byte	8                               # Abbrev [8] 0x6e:0x9 DW_TAG_member
	.byte	12                              # DW_AT_name
	.long	124                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.byte	8                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x78:0x4 DW_TAG_base_type
	.byte	11                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	9                               # Abbrev [9] 0x7c:0x5 DW_TAG_pointer_type
	.long	86                              # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	56                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 4df364bc93af49ae413ec1ae8328f34ac70730c4)" # string offset=0
.Linfo_string1:
	.asciz	"debug-names-parent-idx-2.cpp"  # string offset=104
.Linfo_string2:
	.asciz	"parent-idx-test"               # string offset=133
.Linfo_string3:
	.asciz	"int"                           # string offset=149
.Linfo_string4:
	.asciz	"foo"                           # string offset=153
.Linfo_string5:
	.asciz	"_Z3foov"                       # string offset=157
.Linfo_string6:
	.asciz	"parent_test"                   # string offset=165
.Linfo_string7:
	.asciz	"_ZN11parent_test3fooEv"        # string offset=177
.Linfo_string8:
	.asciz	"struct2"                       # string offset=200
.Linfo_string9:
	.asciz	"x"                             # string offset=208
.Linfo_string10:
	.asciz	"y"                             # string offset=210
.Linfo_string11:
	.asciz	"char"                          # string offset=212
.Linfo_string12:
	.asciz	"foo_ptr"                       # string offset=217
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	6                               # Header: bucket count
	.long	6                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	3                               # Bucket 2
	.long	5                               # Bucket 3
	.long	0                               # Bucket 4
	.long	6                               # Bucket 5
	.long	-1451972055                     # Hash in Bucket 1
	.long	-1257882357                     # Hash in Bucket 1
	.long	175265198                       # Hash in Bucket 2
	.long	193495088                       # Hash in Bucket 2
	.long	193491849                       # Hash in Bucket 3
	.long	2090147939                      # Hash in Bucket 5
	.long	.Linfo_string7                  # String in Bucket 1: _ZN11parent_test3fooEv
	.long	.Linfo_string5                  # String in Bucket 1: _Z3foov
	.long	.Linfo_string6                  # String in Bucket 2: parent_test
	.long	.Linfo_string3                  # String in Bucket 2: int
	.long	.Linfo_string4                  # String in Bucket 3: foo
	.long	.Linfo_string11                 # String in Bucket 5: char
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 5
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	5                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames4:
.L3:
	.byte	1                               # Abbreviation code
	.long	69                              # DW_IDX_die_offset
	.long	.L5-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _ZN11parent_test3fooEv
.Lnames2:
.L0:
	.byte	2                               # Abbreviation code
	.long	39                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z3foov
.Lnames3:
.L5:
	.byte	3                               # Abbreviation code
	.long	67                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: parent_test
.Lnames0:
.L2:
	.byte	4                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames1:
	.byte	2                               # Abbreviation code
	.long	39                              # DW_IDX_die_offset
	.byte	1                               # DW_IDX_parent
                                        # Abbreviation code
	.long	69                              # DW_IDX_die_offset
	.long	.L5-.Lnames_entries0            # DW_IDX_parent
.L4:
	.byte	5                               # Abbreviation code
	.long	86                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
.Lnames5:
.L1:
	.byte	4                               # Abbreviation code
	.long	120                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 4df364bc93af49ae413ec1ae8328f34ac70730c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

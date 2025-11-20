# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s   -o %tmain.o
# RUN: %clang %cflags -gdwarf-5 %tmain.o -o %tmain.exe
# RUN: llvm-bolt %tmain.exe -o %tmain.exe.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-names %tmain.exe.bolt > %tlog.txt
# RUN: cat %tlog.txt | FileCheck -check-prefix=BOLT %s

## This test checks that BOLT correctly generates .debug_names section when there is transative
## DW_AT_name/DW_AT_linkage_name resolution.

# BOLT:       Abbreviations [
# BOLT-NEXT:   Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_subprogram
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:    }
# BOLT-NEXT:    Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_class_type
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:    }
# BOLT-NEXT:    Abbreviation [[ABBREV3:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_inlined_subroutine
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_ref4
# BOLT-NEXT:    }
# BOLT-NEXT:    Abbreviation [[ABBREV4:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_base_type
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 0 [
# BOLT-NEXT:    Name 1 {
# BOLT-NEXT:      Hash: 0xD72418AA
# BOLT-NEXT:      String: {{.+}} "_ZL3fooi"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x000000ba
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 1 [
# BOLT-NEXT:    Name 2 {
# BOLT-NEXT:      Hash: 0x10614A06
# BOLT-NEXT:      String: {{.+}} "State"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV2]]
# BOLT-NEXT:        Tag: DW_TAG_class_type
# BOLT-NEXT:        DW_IDX_die_offset: 0x0000002b
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:      Entry @ [[REF1:0x[0-9a-f]*]] {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x00000089
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV3]]
# BOLT-NEXT:        Tag: DW_TAG_inlined_subroutine
# BOLT-NEXT:        DW_IDX_die_offset: 0x000000a3
# BOLT-NEXT:        DW_IDX_parent: Entry @ [[REF1]]
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 2 [
# BOLT-NEXT:    EMPTY
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 3 [
# BOLT-NEXT:    Name 3 {
# BOLT-NEXT:      Hash: 0xB888030
# BOLT-NEXT:      String: {{.+}} "int"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV4]]
# BOLT-NEXT:        Tag: DW_TAG_base_type
# BOLT-NEXT:        DW_IDX_die_offset: 0x00000085
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:    Name 4 {
# BOLT-NEXT:      Hash: 0x7C9A7F6A
# BOLT-NEXT:      String: {{.+}} "main"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x00000042
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 4 [
# BOLT-NEXT:    EMPTY
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 5 [
# BOLT-NEXT:    Name 5 {
# BOLT-NEXT:      Hash: 0xB887389
# BOLT-NEXT:      String: {{.+}} "foo"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x000000ba
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:    Name 6 {
# BOLT-NEXT:      Hash: 0x7C952063
# BOLT-NEXT:      String: {{.+}} "char"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV4]]
# BOLT-NEXT:        Tag: DW_TAG_base_type
# BOLT-NEXT:        DW_IDX_die_offset: 0x000000d9
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:    Name 7 {
# BOLT-NEXT:      Hash: 0xFBBDC812
# BOLT-NEXT:      String: {{.+}} "_ZN5StateC2Ev"
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x00000089
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:      Entry @ {{.+}}  {
# BOLT-NEXT:        Abbrev: [[ABBREV3]]
# BOLT-NEXT:        Tag: DW_TAG_inlined_subroutine
# BOLT-NEXT:        DW_IDX_die_offset: 0x000000a3
# BOLT-NEXT:        DW_IDX_parent: Entry @ [[REF1]]

## static int foo(int i) {
##   return i ++;
## }
## class State {
## public:
##   State() {[[clang::always_inline]] foo(3);}
## };
##
## int main(int argc, char* argv[]) {
##   State S;
##   return 0;
## }

## Test manually modified to redirect DW_TAG_inlined_subroutine to DW_TAG_subprogram with DW_AT_specification.

	.text
	.file	"main.cpp"
	.file	0 "abstractChainTwo" "main.cpp" md5 0x17ad726b6a1fd49ee59559a1302da539
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.loc	0 9 0                           # main.cpp:9:0
.Ltmp0:
	.loc	0 10 9 prologue_end             # main.cpp:10:9
	callq	_ZN5StateC2Ev
	.loc	0 11 3                          # main.cpp:11:3
	.loc	0 11 3 epilogue_begin is_stmt 0 # main.cpp:11:3
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        # -- End function
	.section	.text._ZN5StateC2Ev,"axG",@progbits,_ZN5StateC2Ev,comdat
	.weak	_ZN5StateC2Ev                   # -- Begin function _ZN5StateC2Ev
	.p2align	4, 0x90
	.type	_ZN5StateC2Ev,@function
_ZN5StateC2Ev:                          # @_ZN5StateC2Ev
.Lfunc_begin1:
	.loc	0 6 0 is_stmt 1                 # main.cpp:6:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -16(%rbp)
	movl	$3, -4(%rbp)
.Ltmp2:
	.loc	0 2 12 prologue_end             # main.cpp:2:12
	movl	-4(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -4(%rbp)
.Ltmp3:
	.loc	0 6 44 epilogue_begin           # main.cpp:6:44
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp4:
.Lfunc_end1:
	.size	_ZN5StateC2Ev, .Lfunc_end1-_ZN5StateC2Ev
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4, 0x90                         # -- Begin function _ZL3fooi
	.type	_ZL3fooi,@function
_ZL3fooi:                               # @_ZL3fooi
.Lfunc_begin2:
	.loc	0 1 0                           # main.cpp:1:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp5:
	.loc	0 2 12 prologue_end             # main.cpp:2:12
	movl	-4(%rbp), %eax
	movl	%eax, %ecx
	addl	$1, %ecx
	movl	%ecx, -4(%rbp)
	.loc	0 2 3 epilogue_begin is_stmt 0  # main.cpp:2:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp6:
.Lfunc_end2:
	.size	_ZL3fooi, .Lfunc_end2-_ZL3fooi
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
	.byte	2                               # DW_TAG_class_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	50                              # DW_AT_accessibility
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
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
	.byte	11                              # Abbreviation Code
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
	.byte	12                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	100                             # DW_AT_object_pointer
	.byte	19                              # DW_FORM_ref4
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	52                              # DW_AT_artificial
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
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
	.byte	15                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	16                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
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
	.byte	1                               # Abbrev [1] 0xc:0xd7 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.byte	0                               # DW_AT_ranges
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2b:0x12 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	3                               # Abbrev [3] 0x31:0xb DW_TAG_subprogram
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	4                               # Abbrev [4] 0x36:0x5 DW_TAG_formal_parameter
	.long	61                              # DW_AT_type
                                        # DW_AT_artificial
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x3d:0x5 DW_TAG_pointer_type
	.long	43                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x42:0x31 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	133                             # DW_AT_type
                                        # DW_AT_external
	.byte	7                               # Abbrev [7] 0x51:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	133                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x5c:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	207                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x67:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	111
	.byte	13                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	43                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x73:0x12 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	133                             # DW_AT_type
                                        # DW_AT_inline
	.byte	10                              # Abbrev [10] 0x7c:0x8 DW_TAG_formal_parameter
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	133                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x85:0x4 DW_TAG_base_type
	.byte	6                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	12                              # Abbrev [12] 0x89:0x31 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	154                             # DW_AT_object_pointer
	.byte	9                               # DW_AT_linkage_name
	.long	49                              # DW_AT_specification
	.byte	13                              # Abbrev [13] 0x9a:0x9 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.byte	14                              # DW_AT_name
	.long	221                             # DW_AT_type
                                        # DW_AT_artificial
	.byte	14                              # Abbrev [14] 0xa3:0x16 DW_TAG_inlined_subroutine
	.long	137                             # DW_AT_abstract_origin Manually Modified
	.byte	2                               # DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   # DW_AT_high_pc
	.byte	0                               # DW_AT_call_file
	.byte	6                               # DW_AT_call_line
	.byte	37                              # DW_AT_call_column
	.byte	15                              # Abbrev [15] 0xb0:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	124                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	16                              # Abbrev [16] 0xba:0x15 DW_TAG_subprogram
	.byte	3                               # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	115                             # DW_AT_abstract_origin
	.byte	15                              # Abbrev [15] 0xc6:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	124                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xcf:0x5 DW_TAG_pointer_type
	.long	212                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xd4:0x5 DW_TAG_pointer_type
	.long	217                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xd9:0x4 DW_TAG_base_type
	.byte	12                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0xdd:0x5 DW_TAG_pointer_type
	.long	43                              # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	1                               # DW_RLE_base_addressx
	.byte	0                               #   base address index
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    #   starting offset
	.uleb128 .Lfunc_end0-.Lfunc_begin0      #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Lfunc_begin2-.Lfunc_begin0    #   starting offset
	.uleb128 .Lfunc_end2-.Lfunc_begin0      #   ending offset
	.byte	3                               # DW_RLE_startx_length
	.byte	1                               #   start index
	.uleb128 .Lfunc_end1-.Lfunc_begin1      #   length
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	64                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 20.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"abstractChainTwo" # string offset=33
.Linfo_string3:
	.asciz	"State"                         # string offset=88
.Linfo_string4:
	.asciz	"main"                          # string offset=94
.Linfo_string5:
	.asciz	"_ZL3fooi"                      # string offset=99
.Linfo_string6:
	.asciz	"foo"                           # string offset=108
.Linfo_string7:
	.asciz	"int"                           # string offset=112
.Linfo_string8:
	.asciz	"i"                             # string offset=116
.Linfo_string9:
	.asciz	"_ZN5StateC2Ev"                 # string offset=118
.Linfo_string10:
	.asciz	"argc"                          # string offset=132
.Linfo_string11:
	.asciz	"argv"                          # string offset=137
.Linfo_string12:
	.asciz	"char"                          # string offset=142
.Linfo_string13:
	.asciz	"S"                             # string offset=147
.Linfo_string14:
	.asciz	"this"                          # string offset=149
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string4
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp2
	.quad	.Lfunc_begin2
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	7                               # Header: bucket count
	.long	7                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	2                               # Bucket 1
	.long	0                               # Bucket 2
	.long	3                               # Bucket 3
	.long	0                               # Bucket 4
	.long	5                               # Bucket 5
	.long	0                               # Bucket 6
	.long	-685500246                      # Hash in Bucket 0
	.long	274811398                       # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 3
	.long	2090499946                      # Hash in Bucket 3
	.long	193491849                       # Hash in Bucket 5
	.long	2090147939                      # Hash in Bucket 5
	.long	-71448558                       # Hash in Bucket 5
	.long	.Linfo_string5                  # String in Bucket 0: _ZL3fooi
	.long	.Linfo_string3                  # String in Bucket 1: State
	.long	.Linfo_string7                  # String in Bucket 3: int
	.long	.Linfo_string4                  # String in Bucket 3: main
	.long	.Linfo_string6                  # String in Bucket 5: foo
	.long	.Linfo_string12                 # String in Bucket 5: char
	.long	.Linfo_string9                  # String in Bucket 5: _ZN5StateC2Ev
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames6-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 5
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	29                              # DW_TAG_inlined_subroutine
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
	.byte	2                               # DW_TAG_class_type
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
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames5:
.L1:
	.byte	1                               # Abbreviation code
	.long	163                             # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
.L0:
	.byte	2                               # Abbreviation code
	.long	186                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _ZL3fooi
.Lnames0:
.L5:
	.byte	3                               # Abbreviation code
	.long	43                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.long	137                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: State
.Lnames2:
.L4:
	.byte	4                               # Abbreviation code
	.long	133                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames1:
.L6:
	.byte	2                               # Abbreviation code
	.long	66                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames4:
	.byte	1                               # Abbreviation code
	.long	163                             # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.long	186                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
.Lnames6:
.L3:
	.byte	4                               # Abbreviation code
	.long	217                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
.Lnames3:
	.byte	2                               # Abbreviation code
	.long	137                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _ZN5StateC2Ev
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

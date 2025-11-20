# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s   -o %tmain.o
# RUN: %clang %cflags -gdwarf-5 %tmain.o -o %tmain.exe
# RUN: llvm-bolt %tmain.exe -o %tmain.exe.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-names %tmain.exe.bolt > %tlog.txt
# RUN: cat %tlog.txt | FileCheck -check-prefix=BOLT %s

## Tests that bolt can correctly generate debug_names when there is an DW_TAG_inlined_subroutine
## with DW_AT_abstract_origin that points to DW_TAG_subprogram that only has DW_AT_linkage_name.

# BOLT:      Name Index @ 0x0 {
# BOLT-NEXT:  Header {
# BOLT-NEXT:    Length: 0xA2
# BOLT-NEXT:    Format: DWARF32
# BOLT-NEXT:    Version: 5
# BOLT-NEXT:    CU count: 1
# BOLT-NEXT:    Local TU count: 0
# BOLT-NEXT:    Foreign TU count: 0
# BOLT-NEXT:    Bucket count: 4
# BOLT-NEXT:    Name count: 4
# BOLT-NEXT:    Abbreviations table size: 0x19
# BOLT-NEXT:    Augmentation: 'BOLT'
# BOLT-NEXT:  }
# BOLT-NEXT:  Compilation Unit offsets [
# BOLT-NEXT:    CU[0]: 0x00000000
# BOLT-NEXT:  ]
# BOLT-NEXT:  Abbreviations [
# BOLT-NEXT:    Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_base_type
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:    }
# BOLT-NEXT:    Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_subprogram
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:    }
# BOLT-NEXT:    Abbreviation [[ABBREV3:0x[0-9a-f]*]] {
# BOLT-NEXT:      Tag: DW_TAG_inlined_subroutine
# BOLT-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:      DW_IDX_parent: DW_FORM_ref4
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 0 [
# BOLT-NEXT:    Name 1 {
# BOLT-NEXT:      Hash: 0xB888030
# BOLT-NEXT:      String: {{.+}} "int"
# BOLT-NEXT:      Entry @ {{.+}} {
# BOLT-NEXT:        Abbrev: 0x1
# BOLT-NEXT:        Tag: DW_TAG_base_type
# BOLT-NEXT:        DW_IDX_die_offset: 0x0000004a
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 1 [
# BOLT-NEXT:    EMPTY
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 2 [
# BOLT-NEXT:    Name 2 {
# BOLT-NEXT:      Hash: 0x7C9A7F6A
# BOLT-NEXT:      String: {{.+}} "main"
# BOLT-NEXT:      Entry @ [[REF1:0x[0-9a-f]*]] {
# BOLT-NEXT:        Abbrev: [[ABBREV2]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x0000004e
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:    Name 3 {
# BOLT-NEXT:      Hash: 0xB5063CFE
# BOLT-NEXT:      String: {{.+}} "_Z3fooi"
# BOLT-NEXT:      Entry @ {{.+}} {
# BOLT-NEXT:        Abbrev: [[ABBREV2]]
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: 0x00000024
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:      }
# BOLT-NEXT:      Entry @ 0x96 {
# BOLT-NEXT:        Abbrev: [[ABBREV3]]
# BOLT-NEXT:        Tag: DW_TAG_inlined_subroutine
# BOLT-NEXT:        DW_IDX_die_offset: 0x0000007e
# BOLT-NEXT:        DW_IDX_parent: Entry @ [[REF1]]
# BOLT-NEXT:      }
# BOLT-NEXT:    }
# BOLT-NEXT:  ]
# BOLT-NEXT:  Bucket 3 [
# BOLT-NEXT:    Name 4 {
# BOLT-NEXT:      Hash: 0x7C952063
# BOLT-NEXT:      String: {{.+}} "char"
# BOLT-NEXT:      Entry @ {{.+}} {
# BOLT-NEXT:        Abbrev: [[ABBREV1]]
# BOLT-NEXT:        Tag: DW_TAG_base_type
# BOLT-NEXT:        DW_IDX_die_offset: 0x0000009f
# BOLT-NEXT:        DW_IDX_parent: <parent not indexed>

## int foo(int i) {
##   return i ++;
## }
## int main(int argc, char* argv[]) {
##   int i = 0;
##   [[clang::always_inline]] i = foo(argc);
##   return i;
## }
## Test was manually modified so that DW_TAG_subprogram only had DW_AT_linkage_name.

	.text
	.file	"main.cpp"
	.globl	_Z3fooi
	.p2align	4, 0x90
	.type	_Z3fooi,@function
_Z3fooi:
.Lfunc_begin0:
	.file	0 "/abstractChain" "main.cpp" md5 0x2e29d55fc1320801a8057a4c50643ea1
	.loc	0 1 0
	.loc	0 2 12 prologue_end
	.loc	0 2 3 epilogue_begin is_stmt 0
	retq
.Lfunc_end0:
	.size	_Z3fooi, .Lfunc_end0-_Z3fooi

	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
.Lfunc_begin1:
	.loc	0 4 0 is_stmt 1
.Ltmp2:
	.loc	0 5 7 prologue_end
	.loc	0 6 36
	movl	-12(%rbp), %eax
.Ltmp3:
	.loc	0 2 12
.Ltmp4:
	.loc	0 6 30
	.loc	0 7 10
	.loc	0 7 3 epilogue_begin is_stmt 0
	retq
.Ltmp5:
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
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	#.byte	3                               # DW_AT_name
	#.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
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
	.byte	11                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xc:0x98 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x15 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	56                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x2f:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	64                              # DW_AT_abstract_origin Manually Modified
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x38:0x12 DW_TAG_subprogram
	.byte	3                               # DW_AT_linkage_name
	#.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	74                              # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	5                               # Abbrev [5] 0x41:0x8 DW_TAG_formal_parameter
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	74                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x4a:0x4 DW_TAG_base_type
	.byte	5                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x4e:0x47 DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	7                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	73                              # DW_AT_type Manually Modified
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x5d:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	8                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	73                              # DW_AT_type Manually Modified
	.byte	8                               # Abbrev [8] 0x68:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	148                             # DW_AT_type  Manually Modified
	.byte	9                               # Abbrev [9] 0x73:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	100
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	73                              # DW_AT_type Manually Modified
	.byte	10                              # Abbrev [10] 0x7e:0x16 DW_TAG_inlined_subroutine
	.long	56                              # DW_AT_abstract_origin
	.byte	2                               # DW_AT_low_pc
	.long	.Ltmp4-.Ltmp3                   # DW_AT_high_pc
	.byte	0                               # DW_AT_call_file
	.byte	6                               # DW_AT_call_line
	.byte	32                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0x8b:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	64                              # DW_AT_abstract_origin Manually Modified
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x95:0x5 DW_TAG_pointer_type
	.long	153                             # DW_AT_type  Manually Modified
	.byte	11                              # Abbrev [11] 0x9a:0x5 DW_TAG_pointer_type
	.long	158                             # DW_AT_type  Manually Modified
	.byte	6                               # Abbrev [6] 0x9f:0x4 DW_TAG_base_type
	.byte	10                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	48                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 20.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/abstractChain" # string offset=33
.Linfo_string3:
	.asciz	"foo"                           # string offset=85
.Linfo_string4:
	.asciz	"_Z3fooi"                       # string offset=89
.Linfo_string5:
	.asciz	"int"                           # string offset=97
.Linfo_string6:
	.asciz	"i"                             # string offset=101
.Linfo_string7:
	.asciz	"main"                          # string offset=103
.Linfo_string8:
	.asciz	"argc"                          # string offset=108
.Linfo_string9:
	.asciz	"argv"                          # string offset=113
.Linfo_string10:
	.asciz	"char"                          # string offset=118
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string4
	.long	.Linfo_string3
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
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp3
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	5                               # Header: bucket count
	.long	5                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	0                               # Bucket 2
	.long	3                               # Bucket 3
	.long	4                               # Bucket 4
	.long	2090499946                      # Hash in Bucket 1
	.long	-1257882370                     # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 3
	.long	193491849                       # Hash in Bucket 4
	.long	2090147939                      # Hash in Bucket 4
	.long	.Linfo_string7                  # String in Bucket 1: main
	.long	.Linfo_string4                  # String in Bucket 1: _Z3fooi
	.long	.Linfo_string5                  # String in Bucket 3: int
	.long	.Linfo_string3                  # String in Bucket 4: foo
	.long	.Linfo_string10                 # String in Bucket 4: char
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 4
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
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
.Lnames3:
.L2:
	.byte	1                               # Abbreviation code
	.long	78                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames1:
.L0:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
.L3:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.long	126                             # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _Z3fooi
.Lnames2:
.L1:
	.byte	3                               # Abbreviation code
	.long	74                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames0:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	2                               # DW_IDX_parent
                                        # Abbreviation code
	.long	126                             # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: foo
.Lnames4:
.L4:
	.byte	3                               # Abbreviation code
	.long	159                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

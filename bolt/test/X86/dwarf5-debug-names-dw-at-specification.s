# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s   -o %tmain.o
# RUN: %clang %cflags -gdwarf-5 %tmain.o -o %tmain.exe
# RUN: llvm-bolt %tmain.exe -o %tmain.exe.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-info -r 0 --debug-names %tmain.exe.bolt > %tlog.txt
# RUN: cat %tlog.txt | FileCheck -check-prefix=BOLT %s

## Tests that BOLT correctly generates entries in .debug_names with DW_AT_specification.

# BOLT: [[OFFSET1:0x[0-9a-f]*]]: Compile Unit
# BOLT:       Name Index @ 0x0
# BOLT-NEXT:    Header {
# BOLT-NEXT:      Length: 0x10F
# BOLT-NEXT:      Format: DWARF32
# BOLT-NEXT:      Version: 5
# BOLT-NEXT:      CU count: 1
# BOLT-NEXT:      Local TU count: 0
# BOLT-NEXT:      Foreign TU count: 0
# BOLT-NEXT:      Bucket count: 9
# BOLT-NEXT:      Name count: 9
# BOLT-NEXT:      Abbreviations table size: 0x21
# BOLT-NEXT:      Augmentation: 'BOLT'
# BOLT-NEXT:    }
# BOLT-NEXT:    Compilation Unit offsets [
# BOLT-NEXT:      CU[0]: [[OFFSET1]]
# BOLT-NEXT:    ]
# BOLT-NEXT:    Abbreviations [
# BOLT-NEXT:      Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
# BOLT-NEXT:        Tag: DW_TAG_variable
# BOLT-NEXT:        DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:        DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:      }
# BOLT-NEXT:      Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
# BOLT-NEXT:        Tag: DW_TAG_structure_type
# BOLT-NEXT:        DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:        DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:      }
# BOLT-NEXT:      Abbreviation [[ABBREV3:0x[0-9a-f]*]] {
# BOLT-NEXT:        Tag: DW_TAG_base_type
# BOLT-NEXT:        DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:        DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:      }
# BOLT-NEXT:      Abbreviation [[ABBREV4:0x[0-9a-f]*]] {
# BOLT-NEXT:        Tag: DW_TAG_subprogram
# BOLT-NEXT:        DW_IDX_die_offset: DW_FORM_ref4
# BOLT-NEXT:        DW_IDX_parent: DW_FORM_flag_present
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 0 [
# BOLT-NEXT:      Name 1 {
# BOLT-NEXT:        Hash: 0x5D3CA9E0
# BOLT-NEXT:        String: {{.+}} "_ZN1A15fully_specifiedE"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV1]]
# BOLT-NEXT:          Tag: DW_TAG_variable
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000024
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:      Name 2 {
# BOLT-NEXT:        Hash: 0x7C9DFC37
# BOLT-NEXT:        String: {{.+}} "smem"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV1]]
# BOLT-NEXT:          Tag: DW_TAG_variable
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000057
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 1 [
# BOLT-NEXT:      Name 3 {
# BOLT-NEXT:        Hash: 0x2B606
# BOLT-NEXT:        String: {{.+}} "A"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV2]]
# BOLT-NEXT:          Tag: DW_TAG_structure_type
# BOLT-NEXT:          DW_IDX_die_offset: 0x0000002d
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 2 [
# BOLT-NEXT:      Name 4 {
# BOLT-NEXT:        Hash: 0xB888030
# BOLT-NEXT:        String: {{.+}} "int"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV3]]
# BOLT-NEXT:          Tag: DW_TAG_base_type
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000044
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 3 [
# BOLT-NEXT:      EMPTY
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 4 [
# BOLT-NEXT:      EMPTY
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 5 [
# BOLT-NEXT:      EMPTY
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 6 [
# BOLT-NEXT:      EMPTY
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 7 [
# BOLT-NEXT:      Name 5 {
# BOLT-NEXT:        Hash: 0x65788E1C
# BOLT-NEXT:        String: {{.+}} "fully_specified"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV1]]
# BOLT-NEXT:          Tag: DW_TAG_variable
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000024
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:      Name 6 {
# BOLT-NEXT:        Hash: 0x7C9A7F6A
# BOLT-NEXT:        String: {{.+}} "main"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV4]]
# BOLT-NEXT:          Tag: DW_TAG_subprogram
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000070
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:    Bucket 8 [
# BOLT-NEXT:      Name 7 {
# BOLT-NEXT:        Hash: 0xCEF4CFB
# BOLT-NEXT:        String: {{.+}} "__ARRAY_SIZE_TYPE__"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV3]]
# BOLT-NEXT:          Tag: DW_TAG_base_type
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000053
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:      Name 8 {
# BOLT-NEXT:        Hash: 0x48684B69
# BOLT-NEXT:        String: {{.+}} "_ZN1A4smemE"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV1]]
# BOLT-NEXT:          Tag: DW_TAG_variable
# BOLT-NEXT:          DW_IDX_die_offset: 0x00000057
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:      Name 9 {
# BOLT-NEXT:        Hash: 0x7C952063
# BOLT-NEXT:        String: {{.+}} "char"
# BOLT-NEXT:        Entry @ {{.+}} {
# BOLT-NEXT:          Abbrev: [[ABBREV3]]
# BOLT-NEXT:          Tag: DW_TAG_base_type
# BOLT-NEXT:          DW_IDX_die_offset: 0x0000009e
# BOLT-NEXT:          DW_IDX_parent: <parent not indexed>
# BOLT-NEXT:        }
# BOLT-NEXT:      }
# BOLT-NEXT:    ]
# BOLT-NEXT:  }

# clang++ main.cpp -O2 -g2 -gdwarf-5 -gpubnames -S
# struct A {
#   static int fully_specified;
#   static int smem[];
# };
#
# int A::fully_specified;
# int A::smem[] = { 0, 1, 2, 3 };
# int main(int argc, char *argv[]) {
#   return 0;
# }
	.text
	.file	"main.cpp"
	.file	0 "/specification" "main.cpp" md5 0x6c1b1c014d300f2e0efd26584acae1a9
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	.loc	0 9 3 prologue_end              # main.cpp:9:3
	xorl	%eax, %eax
	retq
.Ltmp0:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	_ZN1A15fully_specifiedE,@object # @_ZN1A15fully_specifiedE
	.bss
	.globl	_ZN1A15fully_specifiedE
	.p2align	2, 0x0
_ZN1A15fully_specifiedE:
	.long	0                               # 0x0
	.size	_ZN1A15fully_specifiedE, 4

	.type	_ZN1A4smemE,@object             # @_ZN1A4smemE
	.data
	.globl	_ZN1A4smemE
	.p2align	4, 0x0
_ZN1A4smemE:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
	.size	_ZN1A4smemE, 16

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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
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
	.byte	4                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
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
	.byte	6                               # Abbreviation Code
	.byte	1                               # DW_TAG_array_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	71                              # DW_AT_specification
	.byte	19                              # DW_FORM_ref4
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	33                              # DW_TAG_subrange_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	55                              # DW_AT_count
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
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
	.byte	12                              # Abbreviation Code
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
	.byte	13                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xc:0x96 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x9 DW_TAG_variable
	.long	50                              # DW_AT_specification
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	8                               # DW_AT_linkage_name
	.byte	3                               # Abbrev [3] 0x2c:0x17 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x32:0x8 DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	67                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_external
                                        # DW_AT_declaration
	.byte	4                               # Abbrev [4] 0x3a:0x8 DW_TAG_variable
	.byte	5                               # DW_AT_name
	.long	71                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_external
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x43:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x47:0xb DW_TAG_array_type
	.long	67                              # DW_AT_type
	.byte	7                               # Abbrev [7] 0x4c:0x5 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x52:0x4 DW_TAG_base_type
	.byte	6                               # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	7                               # DW_AT_encoding
	.byte	9                               # Abbrev [9] 0x56:0xd DW_TAG_variable
	.long	58                              # DW_AT_specification
	.long	99                              # DW_AT_type
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	1
	.byte	9                               # DW_AT_linkage_name
	.byte	6                               # Abbrev [6] 0x63:0xc DW_TAG_array_type
	.long	67                              # DW_AT_type
	.byte	10                              # Abbrev [10] 0x68:0x6 DW_TAG_subrange_type
	.long	82                              # DW_AT_type
	.byte	4                               # DW_AT_count
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x6f:0x24 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	10                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	67                              # DW_AT_type
                                        # DW_AT_external
	.byte	12                              # Abbrev [12] 0x7e:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	11                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	67                              # DW_AT_type
	.byte	12                              # Abbrev [12] 0x88:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	12                              # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	8                               # DW_AT_decl_line
	.long	147                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0x93:0x5 DW_TAG_pointer_type
	.long	152                             # DW_AT_type
	.byte	13                              # Abbrev [13] 0x98:0x5 DW_TAG_pointer_type
	.long	157                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x9d:0x4 DW_TAG_base_type
	.byte	13                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	60                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git ced1fac8a32e35b63733bda27c7f5b9a2b635403)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=104
.Linfo_string2:
	.asciz	"/specification" # string offset=113
.Linfo_string3:
	.asciz	"A"                             # string offset=165
.Linfo_string4:
	.asciz	"fully_specified"               # string offset=167
.Linfo_string5:
	.asciz	"int"                           # string offset=183
.Linfo_string6:
	.asciz	"smem"                          # string offset=187
.Linfo_string7:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=192
.Linfo_string8:
	.asciz	"_ZN1A15fully_specifiedE"       # string offset=212
.Linfo_string9:
	.asciz	"_ZN1A4smemE"                   # string offset=236
.Linfo_string10:
	.asciz	"main"                          # string offset=248
.Linfo_string11:
	.asciz	"argc"                          # string offset=253
.Linfo_string12:
	.asciz	"argv"                          # string offset=258
.Linfo_string13:
	.asciz	"char"                          # string offset=263
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string3
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	_ZN1A15fully_specifiedE
	.quad	_ZN1A4smemE
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	9                               # Header: bucket count
	.long	9                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	3                               # Bucket 1
	.long	4                               # Bucket 2
	.long	0                               # Bucket 3
	.long	0                               # Bucket 4
	.long	0                               # Bucket 5
	.long	0                               # Bucket 6
	.long	5                               # Bucket 7
	.long	7                               # Bucket 8
	.long	1564256736                      # Hash in Bucket 0
	.long	2090728503                      # Hash in Bucket 0
	.long	177670                          # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 2
	.long	1702399516                      # Hash in Bucket 7
	.long	2090499946                      # Hash in Bucket 7
	.long	217009403                       # Hash in Bucket 8
	.long	1214794601                      # Hash in Bucket 8
	.long	2090147939                      # Hash in Bucket 8
	.long	.Linfo_string8                  # String in Bucket 0: _ZN1A15fully_specifiedE
	.long	.Linfo_string6                  # String in Bucket 0: smem
	.long	.Linfo_string3                  # String in Bucket 1: A
	.long	.Linfo_string5                  # String in Bucket 2: int
	.long	.Linfo_string4                  # String in Bucket 7: fully_specified
	.long	.Linfo_string10                 # String in Bucket 7: main
	.long	.Linfo_string7                  # String in Bucket 8: __ARRAY_SIZE_TYPE__
	.long	.Linfo_string9                  # String in Bucket 8: _ZN1A4smemE
	.long	.Linfo_string13                 # String in Bucket 8: char
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 7
	.long	.Lnames7-.Lnames_entries0       # Offset in Bucket 7
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 8
	.long	.Lnames6-.Lnames_entries0       # Offset in Bucket 8
	.long	.Lnames8-.Lnames_entries0       # Offset in Bucket 8
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
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
	.byte	4                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
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
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _ZN1A15fully_specifiedE
.Lnames5:
.L4:
	.byte	1                               # Abbreviation code
	.long	86                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: smem
.Lnames0:
.L6:
	.byte	2                               # Abbreviation code
	.long	44                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: A
.Lnames1:
.L5:
	.byte	3                               # Abbreviation code
	.long	67                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames3:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: fully_specified
.Lnames7:
.L0:
	.byte	4                               # Abbreviation code
	.long	111                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames2:
.L2:
	.byte	3                               # Abbreviation code
	.long	82                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: __ARRAY_SIZE_TYPE__
.Lnames6:
	.byte	1                               # Abbreviation code
	.long	86                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _ZN1A4smemE
.Lnames8:
.L1:
	.byte	3                               # Abbreviation code
	.long	157                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git ced1fac8a32e35b63733bda27c7f5b9a2b635403)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

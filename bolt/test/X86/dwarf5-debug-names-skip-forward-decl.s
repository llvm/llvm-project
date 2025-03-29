# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-names %t.bolt  FileCheck --check-prefix=POSTCHECK %s

## This test checks that BOLT doesn't set DW_IDX_parent an entry, InnerState, when it's parent is a forward declaration.

# POSTCHECK: debug_names
# POSTCHECK:  Bucket 0 [
# POSTCHECK-NEXT:      Name 1 {
# POSTCHECK-NEXT:        Hash: 0xB888030
# POSTCHECK-NEXT:        String: 0x00000047 "int"
# POSTCHECK-NEXT:        Entry @ 0xfb {
# POSTCHECK-NEXT:          Abbrev: 0x1
# POSTCHECK-NEXT:          Tag: DW_TAG_base_type
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x0000005c
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 1 [
# POSTCHECK-NEXT:      EMPTY
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 2 [
# POSTCHECK-NEXT:      Name 2 {
# POSTCHECK-NEXT:        Hash: 0x7C9A7F6A
# POSTCHECK-NEXT:        String: {{.+}} "main"
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x2
# POSTCHECK-NEXT:          Tag: DW_TAG_subprogram
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000034
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:      Name 3 {
# POSTCHECK-NEXT:        Hash: 0xE0CDC6A2
# POSTCHECK-NEXT:        String: {{.+}} "InnerState"
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x3
# POSTCHECK-NEXT:          Tag: DW_TAG_class_type
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x01
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000030
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 3 [
# POSTCHECK-NEXT:      EMPTY
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 4 [
# POSTCHECK-NEXT:      EMPTY
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 5 [
# POSTCHECK-NEXT:      Name 4 {
# POSTCHECK-NEXT:        Hash: 0x2F94396D
# POSTCHECK-NEXT:        String: {{.+}} "_Z9get_statev"
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x2
# POSTCHECK-NEXT:          Tag: DW_TAG_subprogram
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000024
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:      Name 5 {
# POSTCHECK-NEXT:        Hash: 0xCD86E3E5
# POSTCHECK-NEXT:        String: {{.+}} "get_state"
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x2
# POSTCHECK-NEXT:          Tag: DW_TAG_subprogram
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000024
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 6 [
# POSTCHECK-NEXT:      Name 6 {
# POSTCHECK-NEXT:        Hash: 0x2B606
# POSTCHECK-NEXT:        String: {{.+}} "A"
# POSTCHECK-NEXT:        Entry @ 0x11a {
# POSTCHECK-NEXT:          Abbrev: 0x4
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x00
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000023
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:        Entry @ 0x120 {
# POSTCHECK-NEXT:          Abbrev: 0x4
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x01
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000023
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:        Entry @ 0x126 {
# POSTCHECK-NEXT:          Abbrev: 0x5
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000043
# POSTCHECK-NEXT:          DW_IDX_parent: <parent not indexed>
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:      Name 7 {
# POSTCHECK-NEXT:        Hash: 0x10614A06
# POSTCHECK-NEXT:        String: {{.+}} "State"
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x6
# POSTCHECK-NEXT:          Tag: DW_TAG_structure_type
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x00
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000027
# POSTCHECK-NEXT:          DW_IDX_parent: Entry @ 0x137
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:    Bucket 7 [
# POSTCHECK-NEXT:      Name 8 {
# POSTCHECK-NEXT:        Hash: 0x2B607
# POSTCHECK-NEXT:        String: {{.+}} "B"
# POSTCHECK-NEXT:        Entry @ 0x137 {
# POSTCHECK-NEXT:          Abbrev: 0x7
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x00
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000025
# POSTCHECK-NEXT:          DW_IDX_parent: Entry @ 0x11a
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x7
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_type_unit: 0x01
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000025
# POSTCHECK-NEXT:          DW_IDX_parent: Entry @ 0x120
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:        Entry @ {{.+}} {
# POSTCHECK-NEXT:          Abbrev: 0x8
# POSTCHECK-NEXT:          Tag: DW_TAG_namespace
# POSTCHECK-NEXT:          DW_IDX_die_offset: 0x00000045
# POSTCHECK-NEXT:          DW_IDX_parent: Entry @ 0x126
# POSTCHECK-NEXT:        }
# POSTCHECK-NEXT:      }
# POSTCHECK-NEXT:    ]
# POSTCHECK-NEXT:  }

## clang++ -g2 -O0 -fdebug-types-section -gpubnames -S
## A::B::State::InnerState get_state() { return A::B::State::InnerState(); }
## int main() {
##   return 0;
## }

## Manually modified to fix bug in clang where for TU0 "B" was pointing to CU DIE instead of parent in TU
  .text
	.file	"main.cpp"
	.globl	_Z9get_statev                   # -- Begin function _Z9get_statev
	.p2align	4, 0x90
	.type	_Z9get_statev,@function
_Z9get_statev:                          # @_Z9get_statev
.Lfunc_begin0:
	.file	0 "/skipDecl" "main.cpp" md5 0xd417b4a09217d7c3ec58d64286de7ba4
	.loc	0 2 0                           # main.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 2 39 prologue_end epilogue_begin # main.cpp:2:39
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z9get_statev, .Lfunc_end0-_Z9get_statev
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	0 4 0                           # main.cpp:4:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
.Ltmp2:
	.loc	0 5 3 prologue_end              # main.cpp:5:3
	xorl	%eax, %eax
	.loc	0 5 3 epilogue_begin is_stmt 0  # main.cpp:5:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_info,"G",@progbits,16664150534606561860,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	-1782593539102989756            # Type Signature
	.long	39                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x18 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0xc DW_TAG_namespace
	.byte	5                               # DW_AT_name
	.byte	2                               # Abbrev [2] 0x25:0x9 DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	3                               # Abbrev [3] 0x27:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_info,"G",@progbits,1766745463811827694,comdat
.Ltu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	1766745463811827694             # Type Signature
	.long	48                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x22 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x16 DW_TAG_namespace
	.byte	5                               # DW_AT_name
	.byte	2                               # Abbrev [2] 0x25:0x13 DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	4                               # Abbrev [4] 0x27:0x10 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-1782593539102989756            # DW_AT_signature
	.byte	5                               # Abbrev [5] 0x30:0x6 DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
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
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
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
	.byte	8                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
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
	.byte	9                               # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
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
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	6                               # Abbrev [6] 0xc:0x54 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	7                               # Abbrev [7] 0x23:0x10 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	79                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x33:0xf DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	9                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.long	91                              # DW_AT_type
                                        # DW_AT_external
	.byte	2                               # Abbrev [2] 0x42:0x19 DW_TAG_namespace
	.byte	5                               # DW_AT_name
	.byte	2                               # Abbrev [2] 0x44:0x16 DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	4                               # Abbrev [4] 0x46:0x13 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-1782593539102989756            # DW_AT_signature
	.byte	9                               # Abbrev [9] 0x4f:0x9 DW_TAG_class_type
                                        # DW_AT_declaration
	.quad	1766745463811827694             # DW_AT_signature
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x5b:0x4 DW_TAG_base_type
	.byte	10                              # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_str_offsets,"",@progbits
	.long	48                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/skipDecl" # string offset=33
.Linfo_string3:
	.asciz	"get_state"                     # string offset=80
.Linfo_string4:
	.asciz	"_Z9get_statev"                 # string offset=90
.Linfo_string5:
	.asciz	"main"                          # string offset=104
.Linfo_string6:
	.asciz	"A"                             # string offset=109
.Linfo_string7:
	.asciz	"B"                             # string offset=111
.Linfo_string8:
	.asciz	"State"                         # string offset=113
.Linfo_string9:
	.asciz	"InnerState"                    # string offset=119
.Linfo_string10:
	.asciz	"int"                           # string offset=130
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string4
	.long	.Linfo_string3
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string5
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
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	2                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	8                               # Header: bucket count
	.long	8                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
	.long	.Ltu_begin1                     # Type unit 1
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	2                               # Bucket 2
	.long	0                               # Bucket 3
	.long	0                               # Bucket 4
	.long	4                               # Bucket 5
	.long	6                               # Bucket 6
	.long	8                               # Bucket 7
	.long	193495088                       # Hash in Bucket 0
	.long	2090499946                      # Hash in Bucket 2
	.long	-523385182                      # Hash in Bucket 2
	.long	798243181                       # Hash in Bucket 5
	.long	-846797851                      # Hash in Bucket 5
	.long	177670                          # Hash in Bucket 6
	.long	274811398                       # Hash in Bucket 6
	.long	177671                          # Hash in Bucket 7
	.long	.Linfo_string10                 # String in Bucket 0: int
	.long	.Linfo_string5                  # String in Bucket 2: main
	.long	.Linfo_string9                  # String in Bucket 2: InnerState
	.long	.Linfo_string4                  # String in Bucket 5: _Z9get_statev
	.long	.Linfo_string3                  # String in Bucket 5: get_state
	.long	.Linfo_string6                  # String in Bucket 6: A
	.long	.Linfo_string8                  # String in Bucket 6: State
	.long	.Linfo_string7                  # String in Bucket 7: B
	.long	.Lnames7-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames6-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 5
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 6
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 6
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 7
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
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
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	5                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	6                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	7                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	8                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames7:
.L6:
	.byte	1                               # Abbreviation code
	.long	91                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames2:
.L1:
	.byte	2                               # Abbreviation code
	.long	51                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames6:
.L8:
	.byte	3                               # Abbreviation code
	.byte	1                               # DW_IDX_type_unit
	.long	48                              # DW_IDX_die_offset
	.byte	0                               # End of list: InnerState
.Lnames1:
.L4:
	.byte	2                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z9get_statev
.Lnames0:
	.byte	2                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: get_state
.Lnames3:
.LmanualLabel:
  .byte	4                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
.L3:                                    # DW_IDX_parent
	.byte	4                               # Abbreviation code
	.byte	1                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	5                               # Abbreviation code
	.long	66                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: A
.Lnames5:
.L0:
	.byte	6                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	39                              # DW_IDX_die_offset
	.long	.L5-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: State
.Lnames4:
.L5:
	.byte	7                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	37                              # DW_IDX_die_offset
	.long	.LmanualLabel-.Lnames_entries0  # DW_IDX_parent
.L7:
	.byte	7                               # Abbreviation code
	.byte	1                               # DW_IDX_type_unit
	.long	37                              # DW_IDX_die_offset
	.long	.L3-.Lnames_entries0            # DW_IDX_parent
.L9:
	.byte	8                               # Abbreviation code
	.long	68                              # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: B
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

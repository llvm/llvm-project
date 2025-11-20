# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt > %t.txt
# RUN: llvm-dwarfdump --show-form --verbose --debug-names %t.bolt >> %t.txt
# RUN: cat %t.txt | FileCheck --check-prefix=POSTCHECK %s

## This tests that BOLT doesn't generate entry for a DW_TAG_enumeration_type declaration with DW_AT_name.

# POSTCHECK:       DW_TAG_type_unit
# POSTCHECK:       DW_TAG_enumeration_type [6]
# POSTCHECK-NEXT:    DW_AT_name [DW_FORM_strx1]  (indexed (00000009) string = "InnerState")
# POSTCHECK-NEXT:    DW_AT_byte_size [DW_FORM_data1] (0x04)
# POSTCHECK-NEXT:    DW_AT_declaration [DW_FORM_flag_present]  (true)
# POSTCHECK: Name Index
# POSTCHECK-NOT: "InnerState"

## -g2 -O0 -fdebug-types-section -gpubnames
## namespace B {
##   template <typename Task>
##   class State {
##     public:
##     enum class InnerState { STATE0 };
##     InnerState St;
##   };
## }
##
## int main() {
##   B::State<int> S;
##   return 0;
## }

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	0 "/DW_TAG_enumeration_type" "main.cpp" md5 0x2e8962f8ef4bf6eb6f8bd92966c0848b
	.loc	0 10 0                          # main.cpp:10:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -4(%rbp)
.Ltmp0:
	.loc	0 12 3 prologue_end             # main.cpp:12:3
	xorl	%eax, %eax
	.loc	0 12 3 epilogue_begin is_stmt 0 # main.cpp:12:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_info,"G",@progbits,8822129917070965541,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	8822129917070965541             # Type Signature
	.long	37                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x2d DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x1d DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	3                               # Abbrev [3] 0x25:0x1a DW_TAG_class_type
	.byte	5                               # DW_AT_calling_convention
	.byte	10                              # DW_AT_name
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x2b:0x6 DW_TAG_template_type_parameter
	.long	64                              # DW_AT_type
	.byte	7                               # DW_AT_name
	.byte	5                               # Abbrev [5] 0x31:0xa DW_TAG_member
	.byte	8                               # DW_AT_name
	.long	59                              # DW_AT_type
	.byte	0                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
	.byte	6                               # Abbrev [6] 0x3b:0x3 DW_TAG_enumeration_type
	.byte	9                               # DW_AT_name
	.byte	4                               # DW_AT_byte_size
                                        # DW_AT_declaration
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x40:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
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
	.byte	4                               # Abbreviation Code
	.byte	47                              # DW_TAG_template_type_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	50                              # DW_AT_accessibility
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	4                               # DW_TAG_enumeration_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
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
	.byte	10                              # Abbreviation Code
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
	.byte	11                              # Abbreviation Code
	.byte	2                               # DW_TAG_class_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Abbrev [8] 0xc:0x43 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	9                               # Abbrev [9] 0x23:0x1b DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	62                              # DW_AT_type
                                        # DW_AT_external
	.byte	10                              # Abbrev [10] 0x32:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	11                              # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x3e:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x42:0xc DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	11                              # Abbrev [11] 0x44:0x9 DW_TAG_class_type
                                        # DW_AT_declaration
	.quad	8822129917070965541             # DW_AT_signature
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
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
	.asciz	"/home/ayermolo/local/tasks/T190087639/DW_TAG_enumeration_type" # string offset=33
.Linfo_string3:
	.asciz	"main"                          # string offset=95
.Linfo_string4:
	.asciz	"int"                           # string offset=100
.Linfo_string5:
	.asciz	"S"                             # string offset=104
.Linfo_string6:
	.asciz	"B"                             # string offset=106
.Linfo_string7:
	.asciz	"Task"                          # string offset=108
.Linfo_string8:
	.asciz	"St"                            # string offset=113
.Linfo_string9:
	.asciz	"InnerState"                    # string offset=116
.Linfo_string10:
	.asciz	"State<int>"                    # string offset=127
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
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	1                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	4                               # Header: bucket count
	.long	4                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	2                               # Bucket 2
	.long	3                               # Bucket 3
	.long	193495088                       # Hash in Bucket 0
	.long	2090499946                      # Hash in Bucket 2
	.long	177671                          # Hash in Bucket 3
	.long	624407275                       # Hash in Bucket 3
	.long	.Linfo_string4                  # String in Bucket 0: int
	.long	.Linfo_string3                  # String in Bucket 2: main
	.long	.Linfo_string6                  # String in Bucket 3: B
	.long	.Linfo_string10                 # String in Bucket 3: State<int>
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 3
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
	.byte	36                              # DW_TAG_base_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
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
	.byte	2                               # DW_TAG_class_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
.L0:
	.byte	1                               # Abbreviation code
	.long	62                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	64                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames0:
.L3:
	.byte	3                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames2:
	.byte	4                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
.L1:                                    # DW_IDX_parent
	.byte	5                               # Abbreviation code
	.long	66                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: B
.Lnames3:
.L4:
	.byte	6                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	37                              # DW_IDX_die_offset
	.long	.L3-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: State<int>
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

## Check that DWARF CU with a valid DWOId but missing a dwo_name is correctly detected.
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -split-dwarf-file=main.dwo -o main.o
# RUN: %clang %cflags -O3 -g -gdwarf-5 -gsplit-dwarf -Wl,-q %t/main.o -o main.exe
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections  2>&1 | FileCheck %s --check-prefix=PRECHECK
# PRECHECK: BOLT-ERROR: broken DWARF found in CU at offset 0x3e (DWOId=0x0, missing DW_AT_dwo_name / DW_AT_GNU_dwo_name)

## Checks that Broken dwarf CU is removed
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t/main.exe.bolt | FileCheck %s --check-prefix=POSTCHECK
# POSTCHECK-LABEL: .debug_info contents:
# POSTCHECK: DW_TAG_skeleton_unit
# POSTCHECK-DAG: DW_AT_dwo_name{{.*=.*\.dwo.*}}
# POSTCHECK: NULL
# POSTCHECK-NOT: DW_TAG_skeleton_unit

	.text
	.file	"main.cpp"
	.section	.rodata.cst16,"aM",@progbits,16
.LCPI0_0:
.LCPI0_1:
.LCPI0_2:
.LCPI0_3:
.LCPI0_4:
.LCPI0_5:
.LCPI0_6:
.LCPI0_7:
.LCPI0_8:
.LCPI0_9:
.LCPI0_10:
	.text
	.globl	main
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "." "main.cpp" md5 0x8a68374187457ce14ac0c6c2121349a2
	.loc	1 5 0                           # main.cpp:5:0
# %bb.0:                                # %vector.ph
.Ltmp0:
.Ltmp1:
.LBB0_1:                                # %vector.body
.Ltmp2:
	.file	2 "." "callee.cpp" md5 0x86e19c24983503540b9bb1a6f7bad737
	.loc	2 8 15 prologue_end             # callee.cpp:8:15
.Ltmp3:
	.loc	2 3 15                          # callee.cpp:3:15
.Ltmp4:
	.loc	2 8 15                          # callee.cpp:8:15
.Ltmp5:
	.loc	2 9 19                          # callee.cpp:9:19
.Ltmp6:
	.loc	2 9 13 is_stmt 0                # callee.cpp:9:13
.Ltmp7:
	.loc	2 3 15 is_stmt 1                # callee.cpp:3:15
	.loc	2 3 19 is_stmt 0                # callee.cpp:3:19
.Ltmp8:
	.loc	2 4 19 is_stmt 1                # callee.cpp:4:19
.Ltmp9:
	.loc	2 4 13 is_stmt 0                # callee.cpp:4:13
.Ltmp10:
	.loc	2 4 19                          # callee.cpp:4:19
.Ltmp11:
	.loc	2 4 13                          # callee.cpp:4:13
.Ltmp12:
	.loc	2 2 12 is_stmt 1                # callee.cpp:2:12
	.loc	2 2 17 is_stmt 0                # callee.cpp:2:17
.Ltmp13:
	.loc	2 4 13 is_stmt 1                # callee.cpp:4:13
.Ltmp14:
	.loc	2 0 0 is_stmt 0                 # callee.cpp:0:0
.Ltmp15:
	.loc	1 8 13 is_stmt 1                # main.cpp:8:13
.Ltmp16:
	.loc	2 0 0 is_stmt 0                 # callee.cpp:0:0
.Ltmp17:
	.loc	1 8 13                          # main.cpp:8:13
.Ltmp18:
	.loc	1 7 35 is_stmt 1                # main.cpp:7:35
.Ltmp19:
# %bb.2:                                # %middle.block
	.loc	1 7 5 is_stmt 0                 # main.cpp:7:5
.Ltmp20:
	.loc	1 11 9 is_stmt 1                # main.cpp:11:9
.Ltmp21:
	.loc	1 15 1                          # main.cpp:15:1
	retq
.Ltmp22:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
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
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	16                              # DW_FORM_ref_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	-1861901018463438211
	.byte	1                               # Abbrev [1] 0x14:0x2a DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	3                               # DW_AT_dwo_name
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2c:0x11 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # DW_AT_name
	.byte	3                               # Abbrev [3] 0x33:0x9 DW_TAG_inlined_subroutine
	.long	.debug_info+100                 # DW_AT_abstract_origin
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	8                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	0
	.byte	4                               # Abbrev [4] 0x14:0x15 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	4                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	5                               # DW_AT_name
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	5                               # Abbrev [5] 0x26:0x2 DW_TAG_subprogram
	.byte	1                               # DW_AT_name
                                        # DW_AT_inline
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges1:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp15-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp16-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp17-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"." # string offset=0
.Lskel_string1:
	.asciz	"hotFunction"                   # string offset=45
.Lskel_string2:
	.asciz	"main"                          # string offset=57
.Lskel_string3:
	.asciz	"main.dwo"                      # string offset=62
.Lskel_string4:
	.asciz	"clang version 16.0.6" # string offset=71
.Lskel_string5:
	.asciz	"callee.cpp"                    # string offset=177
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string1
	.long	.Lskel_string2
	.long	.Lskel_string3
	.long	.Lskel_string4
	.long	.Lskel_string5
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	56                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z11hotFunctioni"              # string offset=0
.Linfo_string1:
	.asciz	"hotFunction"                   # string offset=17
.Linfo_string2:
	.asciz	"int"                           # string offset=29
.Linfo_string3:
	.asciz	"x"                             # string offset=33
.Linfo_string4:
	.asciz	"main"                          # string offset=35
.Linfo_string5:
	.asciz	"argc"                          # string offset=40
.Linfo_string6:
	.asciz	"argv"                          # string offset=45
.Linfo_string7:
	.asciz	"char"                          # string offset=50
.Linfo_string8:
	.asciz	"sum"                           # string offset=55
.Linfo_string9:
	.asciz	"i"                             # string offset=59
.Linfo_string10:
	.asciz	"clang version 16.0.6" # string offset=61
.Linfo_string11:
	.asciz	"main.cpp"                      # string offset=167
.Linfo_string12:
	.asciz	"main.dwo"                      # string offset=176
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	17
	.long	29
	.long	33
	.long	35
	.long	40
	.long	45
	.long	50
	.long	55
	.long	59
	.long	61
	.long	167
	.long	176
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-1861901018463438211
	.byte	1                               # Abbrev [1] 0x14:0x71 DW_TAG_compile_unit
	.byte	10                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	11                              # DW_AT_name
	.byte	12                              # DW_AT_dwo_name
	.byte	2                               # Abbrev [2] 0x1a:0x12 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x23:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x2c:0x4 DW_TAG_base_type
	.byte	2                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x30:0x46 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x3f:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x49:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	118                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x53:0x9 DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x5c:0x19 DW_TAG_lexical_block
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp20-.Ltmp2                  # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x62:0x9 DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x6b:0x9 DW_TAG_inlined_subroutine
	.long	26                              # DW_AT_abstract_origin
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	8                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x76:0x5 DW_TAG_pointer_type
	.long	123                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x7b:0x5 DW_TAG_pointer_type
	.long	128                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x80:0x4 DW_TAG_base_type
	.byte	7                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
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
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
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
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_rnglists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 # Length
.Ldebug_list_header_start1:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_dwo_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp15-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp16-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp17-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end1:
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp2
.Ldebug_addr_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	62                              # Compilation Unit Length
	.long	26                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"hotFunction"                   # External Name
	.long	48                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	62                              # Compilation Unit Length
	.long	44                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	128                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end1-.LpubNames_start1 # Length of Public Names Info
.LpubNames_start1:
	.short	2                               # DWARF Version
	.long	.Lcu_begin1                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubNames_end1:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end1-.LpubTypes_start1 # Length of Public Types Info
.LpubTypes_start1:
	.short	2                               # DWARF Version
	.long	.Lcu_begin1                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end1:
	.ident	"clang version 16.0.6"
	.ident	"clang version 16.0.6"
	.section	.GCC.command.line,"MS",@progbits,1
	.zero	1
	.ascii	""
	.zero	1
	.ascii	""
	.zero	1
	.section	.debug_gnu_pubtypes,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

## This test checks updating debuginfo via dwarf4 dwp file
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown --split-dwarf-file=main.exe-main.dwo %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown --split-dwarf-file=main.exe-callee.dwo %t/callee.s -o %t/callee.o
# RUN: %clangxx %cxxflags -gdwarf-4 -gsplit-dwarf=split -Wl,-e,main %t/main.o %t/callee.o -o main.exe
# RUN: llvm-dwp -e %t/main.exe -o %t/main.exe.dwp
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -update-debug-sections  2>&1 | FileCheck %s

# CHECK-NOT: Assertion

#--- main.s
	.file	"main.cpp"
	.globl	main                            // -- Begin function main
	.type	main,@function
main:                                   // @main
.Lfunc_begin0:
	.file	1 "." "main.cpp"
	.loc	1 2 0                           // main.cpp:2:0
	.loc	1 2 21 prologue_end             // main.cpp:2:21
	.loc	1 2 14 epilogue_begin is_stmt 0 // main.cpp:2:14
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.section	.debug_abbrev,"",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	0                               // DW_CHILDREN_no
	.byte	16                              // DW_AT_stmt_list
	.byte	23                              // DW_FORM_sec_offset
	.byte	27                              // DW_AT_comp_dir
	.byte	14                              // DW_FORM_strp
	.ascii	"\264B"                         // DW_AT_GNU_pubnames
	.byte	25                              // DW_FORM_flag_present
	.ascii	"\260B"                         // DW_AT_GNU_dwo_name
	.byte	14                              // DW_FORM_strp
	.ascii	"\261B"                         // DW_AT_GNU_dwo_id
	.byte	7                               // DW_FORM_data8
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.ascii	"\263B"                         // DW_AT_GNU_addr_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	4                               // DWARF version number
	.word	.debug_abbrev                   // Offset Into Abbrev. Section
	.byte	8                               // Address Size (in bytes)
	.byte	1                               // Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
	.word	.Lline_table_start0             // DW_AT_stmt_list
	.word	.Lskel_string0                  // DW_AT_comp_dir
                                        // DW_AT_GNU_pubnames
	.word	.Lskel_string1                  // DW_AT_GNU_dwo_name
	.xword	1465063543908291764             // DW_AT_GNU_dwo_id
	.xword	.Lfunc_begin0                   // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.word	.Laddr_table_base0              // DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"."                             // string offset=0
.Lskel_string1:
	.asciz	"main.exe-main.dwo"             // string offset=2
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"main"                          // string offset=0
.Linfo_string1:
	.asciz	"int"                           // string offset=5
.Linfo_string2:
	.byte	0                               // string offset=9
.Linfo_string3:
	.asciz	"main.cpp"                      // string offset=10
.Linfo_string4:
	.asciz	"main.exe-main.dwo"             // string offset=19
	.section	.debug_str_offsets.dwo,"e",@progbits
	.word	0
	.word	5
	.word	9
	.word	10
	.word	19
	.section	.debug_info.dwo,"e",@progbits
	.word	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 // Length of Unit
.Ldebug_info_dwo_start0:
	.hword	4                               // DWARF version number
	.word	0                               // Offset Into Abbrev. Section
	.byte	8                               // Address Size (in bytes)
	.byte	1                               // Abbrev [1] 0xb:0x22 DW_TAG_compile_unit
	.byte	2                               // DW_AT_producer
	.hword	33                              // DW_AT_language
	.byte	3                               // DW_AT_name
	.byte	4                               // DW_AT_GNU_dwo_name
	.xword	1465063543908291764             // DW_AT_GNU_dwo_id
	.byte	2                               // Abbrev [2] 0x19:0xf DW_TAG_subprogram
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	0                               // DW_AT_name
	.byte	1                               // DW_AT_decl_file
	.byte	2                               // DW_AT_decl_line
	.word	40                              // DW_AT_type
                                        // DW_AT_external
	.byte	3                               // Abbrev [3] 0x28:0x4 DW_TAG_base_type
	.byte	1                               // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	0                               // End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	1                               // DW_CHILDREN_yes
	.byte	37                              // DW_AT_producer
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	19                              // DW_AT_language
	.byte	5                               // DW_FORM_data2
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.ascii	"\260B"                         // DW_AT_GNU_dwo_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.ascii	"\261B"                         // DW_AT_GNU_dwo_id
	.byte	7                               // DW_FORM_data8
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	2                               // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	0                               // DW_CHILDREN_no
	.byte	17                              // DW_AT_low_pc
	.ascii	"\201>"                         // DW_FORM_GNU_addr_index
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.xword	.Lfunc_begin0
	.section	.debug_gnu_pubnames,"",@progbits
	.word	.LpubNames_end0-.LpubNames_start0 // Length of Public Names Info
.LpubNames_start0:
	.hword	2                               // DWARF Version
	.word	.Lcu_begin0                     // Offset of Compilation Unit Info
	.word	48                              // Compilation Unit Length
	.word	25                              // DIE offset
	.byte	48                              // Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          // External Name
	.word	0                               // End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.word	.LpubTypes_end0-.LpubTypes_start0 // Length of Public Types Info
.LpubTypes_start0:
	.hword	2                               // DWARF Version
	.word	.Lcu_begin0                     // Offset of Compilation Unit Info
	.word	48                              // Compilation Unit Length
	.word	40                              // DIE offset
	.byte	144                             // Attributes: TYPE, STATIC
	.asciz	"int"                           // External Name
	.word	0                               // End Mark
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z6calleei
	.section	.debug_line,"",@progbits
.Lline_table_start0:
#--- callee.s
	.file	"callee.cpp"
	.globl	_Z6calleei                      // -- Begin function _Z6calleei
	.type	_Z6calleei,@function
_Z6calleei:                             // @_Z6calleei
.Lfunc_begin0:
	.file	1 "." "callee.cpp"
	.loc	1 1 0                           // callee.cpp:1:0
	.loc	1 1 28 prologue_end             // callee.cpp:1:28
	.loc	1 1 21 epilogue_begin is_stmt 0 // callee.cpp:1:21
	ret
.Lfunc_end0:
	.size	_Z6calleei, .Lfunc_end0-_Z6calleei
	.section	.debug_abbrev,"",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	0                               // DW_CHILDREN_no
	.byte	16                              // DW_AT_stmt_list
	.byte	23                              // DW_FORM_sec_offset
	.byte	27                              // DW_AT_comp_dir
	.byte	14                              // DW_FORM_strp
	.ascii	"\264B"                         // DW_AT_GNU_pubnames
	.byte	25                              // DW_FORM_flag_present
	.ascii	"\260B"                         // DW_AT_GNU_dwo_name
	.byte	14                              // DW_FORM_strp
	.ascii	"\261B"                         // DW_AT_GNU_dwo_id
	.byte	7                               // DW_FORM_data8
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.ascii	"\263B"                         // DW_AT_GNU_addr_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	4                               // DWARF version number
	.word	.debug_abbrev                   // Offset Into Abbrev. Section
	.byte	8                               // Address Size (in bytes)
	.byte	1                               // Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
	.word	.Lline_table_start0             // DW_AT_stmt_list
	.word	.Lskel_string0                  // DW_AT_comp_dir
                                        // DW_AT_GNU_pubnames
	.word	.Lskel_string1                  // DW_AT_GNU_dwo_name
	.xword	7650227797527095061             // DW_AT_GNU_dwo_id
	.xword	.Lfunc_begin0                   // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.word	.Laddr_table_base0              // DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"."                             // string offset=0
.Lskel_string1:
	.asciz	"main.exe-callee.dwo"           // string offset=2
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z6calleei"                    // string offset=0
.Linfo_string1:
	.asciz	"callee"                        // string offset=11
.Linfo_string2:
	.asciz	"int"                           // string offset=18
.Linfo_string3:
	.asciz	"x"                             // string offset=22
.Linfo_string4:
	.byte	0                               // string offset=24
.Linfo_string5:
	.asciz	"callee.cpp"                    // string offset=25
.Linfo_string6:
	.asciz	"main.exe-callee.dwo"           // string offset=36
	.section	.debug_str_offsets.dwo,"e",@progbits
	.word	0
	.word	11
	.word	18
	.word	22
	.word	24
	.word	25
	.word	36
	.section	.debug_info.dwo,"e",@progbits
	.word	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 // Length of Unit
.Ldebug_info_dwo_start0:
	.hword	4                               // DWARF version number
	.word	0                               // Offset Into Abbrev. Section
	.byte	8                               // Address Size (in bytes)
	.byte	1                               // Abbrev [1] 0xb:0x2f DW_TAG_compile_unit
	.byte	4                               // DW_AT_producer
	.hword	33                              // DW_AT_language
	.byte	5                               // DW_AT_name
	.byte	6                               // DW_AT_GNU_dwo_name
	.xword	7650227797527095061             // DW_AT_GNU_dwo_id
	.byte	2                               // Abbrev [2] 0x19:0x1c DW_TAG_subprogram
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	111
	.byte	0                               // DW_AT_linkage_name
	.byte	1                               // DW_AT_name
	.byte	1                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
	.word	53                              // DW_AT_type
                                        // DW_AT_external
	.byte	3                               // Abbrev [3] 0x29:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	12
	.byte	3                               // DW_AT_name
	.byte	1                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
	.word	53                              // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	4                               // Abbrev [4] 0x35:0x4 DW_TAG_base_type
	.byte	2                               // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	0                               // End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	1                               // DW_CHILDREN_yes
	.byte	37                              // DW_AT_producer
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	19                              // DW_AT_language
	.byte	5                               // DW_FORM_data2
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.ascii	"\260B"                         // DW_AT_GNU_dwo_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.ascii	"\261B"                         // DW_AT_GNU_dwo_id
	.byte	7                               // DW_FORM_data8
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	2                               // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.ascii	"\201>"                         // DW_FORM_GNU_addr_index
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	110                             // DW_AT_linkage_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	5                               // DW_TAG_formal_parameter
	.byte	0                               // DW_CHILDREN_no
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	4                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.ascii	"\202>"                         // DW_FORM_GNU_str_index
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.xword	.Lfunc_begin0
	.section	.debug_gnu_pubnames,"",@progbits
	.word	.LpubNames_end0-.LpubNames_start0 // Length of Public Names Info
.LpubNames_start0:
	.hword	2                               // DWARF Version
	.word	.Lcu_begin0                     // Offset of Compilation Unit Info
	.word	48                              // Compilation Unit Length
	.word	25                              // DIE offset
	.byte	48                              // Attributes: FUNCTION, EXTERNAL
	.asciz	"callee"                        // External Name
	.word	0                               // End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.word	.LpubTypes_end0-.LpubTypes_start0 // Length of Public Types Info
.LpubTypes_start0:
	.hword	2                               // DWARF Version
	.word	.Lcu_begin0                     // Offset of Compilation Unit Info
	.word	48                              // Compilation Unit Length
	.word	53                              // DIE offset
	.byte	144                             // Attributes: TYPE, STATIC
	.asciz	"int"                           // External Name
	.word	0                               // End Mark
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

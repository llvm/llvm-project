/// Check that BOLT detects and handles compile units where the DW_AT_comp_dir
/// or DW_AT_name attributes have different values from the entries in the CU's
/// line table.
/// To test this, the .file directive was modified below to point to /foo/bar.c
/// instead of the original /root/main.c.

// REQUIRES: system-linux
// RUN: %clang %cflags %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections 2>&1 | \
// RUN:     FileCheck %s --check-prefix=BOLT -DINPUT=%t.exe
// RUN: llvm-dwarfdump --debug-line %t.exe | FileCheck %s --check-prefix=DEBUGLINE
// RUN: llvm-dwarfdump --debug-line %t.bolt | FileCheck %s --check-prefix=DEBUGLINE

// BOLT: BOLT-WARNING: invalid DWARF detected in [[INPUT]]: the first directory entry in the line table (/foo) does not match the value of DW_AT_comp_dir (/root) for CU at offset 0x0
// BOLT: BOLT-WARNING: invalid DWARF detected in [[INPUT]]: the first file name entry in the line table (bar.c) does not match the value of DW_AT_name (main.c) for CU at offset 0x0

/// Check that the original and BOLTed binaries have the same line table
/// entries.
// DEBUGLINE:      include_directories[  0] = "/foo"
// DEBUGLINE:      file_names[  0]:
// DEBUGLINE-NEXT:            name: "bar.c"
// DEBUGLINE-NEXT:       dir_index: 0
// DEBUGLINE-NEXT:    md5_checksum: 2c7fa9a609df7a2f7e9f545c2571989d

	.file	"main.c"
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin0:
	/// Original .file directive:
	/// .file	0 "/root" "main.c" md5 0x2c7fa9a609df7a2f7e9f545c2571989d
	.file	0 "/foo" "bar.c" md5 0x2c7fa9a609df7a2f7e9f545c2571989d
	.cfi_startproc
// %bb.0:                               // %entry
	.loc	0 1 18 prologue_end             // main.c:1:18
	mov	w0, wzr
	ret
.Ltmp0:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        // -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	1                               // DW_CHILDREN_yes
	.byte	37                              // DW_AT_producer
	.byte	37                              // DW_FORM_strx1
	.byte	19                              // DW_AT_language
	.byte	5                               // DW_FORM_data2
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	114                             // DW_AT_str_offsets_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	16                              // DW_AT_stmt_list
	.byte	23                              // DW_FORM_sec_offset
	.byte	27                              // DW_AT_comp_dir
	.byte	37                              // DW_FORM_strx1
	.ascii	"\341\177"                      // DW_AT_APPLE_optimized
	.byte	25                              // DW_FORM_flag_present
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	115                             // DW_AT_addr_base
	.byte	23                              // DW_FORM_sec_offset
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	2                               // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	0                               // DW_CHILDREN_no
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	122                             // DW_AT_call_all_calls
	.byte	25                              // DW_FORM_flag_present
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	39                              // DW_AT_prototyped
	.byte	25                              // DW_FORM_flag_present
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.ascii	"\341\177"                      // DW_AT_APPLE_optimized
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.word	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.hword	5                               // DWARF version number
	.byte	1                               // DWARF Unit Type
	.byte	8                               // Address Size (in bytes)
	.word	.debug_abbrev                   // Offset Into Abbrev. Section
	.byte	1                               // Abbrev [1] 0xc:0x2b DW_TAG_compile_unit
	.byte	0                               // DW_AT_producer
	.hword	29                              // DW_AT_language
	.byte	1                               // DW_AT_name
	.word	.Lstr_offsets_base0             // DW_AT_str_offsets_base
	.word	.Lline_table_start0             // DW_AT_stmt_list
	.byte	2                               // DW_AT_comp_dir
                                        // DW_AT_APPLE_optimized
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.word	.Laddr_table_base0              // DW_AT_addr_base
	.byte	2                               // Abbrev [2] 0x23:0xf DW_TAG_subprogram
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	111
                                        // DW_AT_call_all_calls
	.byte	3                               // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
                                        // DW_AT_prototyped
	.word	50                              // DW_AT_type
                                        // DW_AT_external
                                        // DW_AT_APPLE_optimized
	.byte	3                               // Abbrev [3] 0x32:0x4 DW_TAG_base_type
	.byte	4                               // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	0                               // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.word	24                              // Length of String Offsets Set
	.hword	5
	.hword	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               // string offset=0 ;
.Linfo_string1:
	.asciz	"main.c"                        // string offset=1 ; main.c
.Linfo_string2:
	.asciz	"/root"                         // string offset=8 ; /root
.Linfo_string3:
	.asciz	"main"                          // string offset=14 ; main
.Linfo_string4:
	.asciz	"int"                           // string offset=19 ; int
	.section	.debug_str_offsets,"",@progbits
	.word	.Linfo_string0
	.word	.Linfo_string1
	.word	.Linfo_string2
	.word	.Linfo_string3
	.word	.Linfo_string4
	.section	.debug_addr,"",@progbits
	.word	.Ldebug_addr_end0-.Ldebug_addr_start0 // Length of contribution
.Ldebug_addr_start0:
	.hword	5                               // DWARF version number
	.byte	8                               // Address size
	.byte	0                               // Segment selector size
.Laddr_table_base0:
	.xword	.Lfunc_begin0
.Ldebug_addr_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

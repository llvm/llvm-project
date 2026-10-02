## Checks that BOLT can correctly decompress, process and recompress zlib
## compressed DWARF debug information.
#
# REQUIRES: zlib
#
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux-gnu %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-dwarfdump --verify %t.exe 2>&1 | FileCheck %s --check-prefix=VERIFY-DWARF
# RUN: llvm-objcopy --compress-debug-sections=zlib %t.exe %t.zlib
# RUN: llvm-readelf -t %t.zlib | FileCheck %s --check-prefix=INPUT-ELF
# INPUT-ELF: .debug_info
# INPUT-ELF: COMPRESSED
# INPUT-ELF-NEXT: ZLIB,
#
# RUN: llvm-bolt %t.zlib -o %t.bolt.exe --update-debug-sections
# RUN: llvm-readelf -t %t.bolt.exe | FileCheck %s --check-prefix=BOLTED-ELF
# BOLTED-ELF: .debug_info
# BOLTED-ELF-NEXT: {{.*}}PROGBITS
# BOLTED-ELF-NEXT: {{.*}}COMPRESSED
# BOLTED-ELF-NEXT: {{.*}}ZLIB,
#
# RUN: llvm-dwarfdump --verify %t.bolt.exe 2>&1 | FileCheck %s --check-prefix=VERIFY-DWARF
# VERIFY-DWARF: No errors.

	.file	"main.c"
	.text
	.globl	square                          // -- Begin function square
	.p2align	2
	.type	square,@function
square:                                 // @square
.Lfunc_begin0:
	.file	0 "." "main.c" md5 0x23769560df9190f53a5be0173a252c67
	.loc	0 1 0                           // main.c:1:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #12]
.Ltmp1:
	.loc	0 1 28 prologue_end             // main.c:1:28
	ldr	w8, [sp, #12]
	.loc	0 1 32 is_stmt 0                // main.c:1:32
	ldr	w9, [sp, #12]
	.loc	0 1 30                          // main.c:1:30
	mul	w0, w8, w9
	.loc	0 1 21 epilogue_begin           // main.c:1:21
	add	sp, sp, #16
	.cfi_def_cfa_offset 0
	ret
.Ltmp2:
.Lfunc_end0:
	.size	square, .Lfunc_end0-square
	.cfi_endproc
                                        // -- End function
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin1:
	.loc	0 2 0 is_stmt 1                 // main.c:2:0
	.cfi_startproc
// %bb.0:                               // %entry
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	stur	wzr, [x29, #-4]
.Ltmp3:
	.loc	0 2 25 prologue_end             // main.c:2:25
	mov	w0, #7                          // =0x7
	bl	square
	.loc	0 2 35 is_stmt 0                // main.c:2:35
	subs	w8, w0, #49
	cset	w0, ne
	.cfi_def_cfa wsp, 32
	.loc	0 2 18 epilogue_begin           // main.c:2:18
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Ltmp4:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
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
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
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
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	5                               // DW_TAG_formal_parameter
	.byte	0                               // DW_CHILDREN_no
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	37                              // DW_FORM_strx1
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	4                               // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	0                               // DW_CHILDREN_no
	.byte	17                              // DW_AT_low_pc
	.byte	27                              // DW_FORM_addrx
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
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
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	5                               // Abbreviation Code
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
	.byte	1                               // Abbrev [1] 0xc:0x46 DW_TAG_compile_unit
	.byte	0                               // DW_AT_producer
	.hword	29                              // DW_AT_language
	.byte	1                               // DW_AT_name
	.word	.Lstr_offsets_base0             // DW_AT_str_offsets_base
	.word	.Lline_table_start0             // DW_AT_stmt_list
	.byte	2                               // DW_AT_comp_dir
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin0       // DW_AT_high_pc
	.word	.Laddr_table_base0              // DW_AT_addr_base
	.byte	2                               // Abbrev [2] 0x23:0x1b DW_TAG_subprogram
	.byte	0                               // DW_AT_low_pc
	.word	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	111
	.byte	3                               // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
                                        // DW_AT_prototyped
	.word	77                              // DW_AT_type
                                        // DW_AT_external
	.byte	3                               // Abbrev [3] 0x32:0xb DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	12
	.byte	6                               // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	1                               // DW_AT_decl_line
	.word	77                              // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	4                               // Abbrev [4] 0x3e:0xf DW_TAG_subprogram
	.byte	1                               // DW_AT_low_pc
	.word	.Lfunc_end1-.Lfunc_begin1       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	109
	.byte	5                               // DW_AT_name
	.byte	0                               // DW_AT_decl_file
	.byte	2                               // DW_AT_decl_line
                                        // DW_AT_prototyped
	.word	77                              // DW_AT_type
                                        // DW_AT_external
	.byte	5                               // Abbrev [5] 0x4d:0x4 DW_TAG_base_type
	.byte	4                               // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	0                               // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.word	32                              // Length of String Offsets Set
	.hword	5
	.hword	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang"                         // string offset=0 ; clang
.Linfo_string1:
	.asciz	"main.c"                        // string offset=6 ; main.c
.Linfo_string2:
	.asciz	"."                             // string offset=13 ; .
.Linfo_string3:
	.asciz	"square"                        // string offset=15 ; square
.Linfo_string4:
	.asciz	"int"                           // string offset=22 ; int
.Linfo_string5:
	.asciz	"main"                          // string offset=26 ; main
.Linfo_string6:
	.asciz	"x"                             // string offset=31 ; x
	.section	.debug_str_offsets,"",@progbits
	.word	.Linfo_string0
	.word	.Linfo_string1
	.word	.Linfo_string2
	.word	.Linfo_string3
	.word	.Linfo_string4
	.word	.Linfo_string5
	.word	.Linfo_string6
	.section	.debug_addr,"",@progbits
	.word	.Ldebug_addr_end0-.Ldebug_addr_start0 // Length of contribution
.Ldebug_addr_start0:
	.hword	5                               // DWARF version number
	.byte	8                               // Address size
	.byte	0                               // Segment selector size
.Laddr_table_base0:
	.xword	.Lfunc_begin0
	.xword	.Lfunc_begin1
.Ldebug_addr_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym square
	.section	.debug_line,"",@progbits
.Lline_table_start0:

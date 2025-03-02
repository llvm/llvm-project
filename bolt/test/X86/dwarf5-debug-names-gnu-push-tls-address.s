# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s   -o %tmain.o
# RUN: %clang %cflags -gdwarf-5 %tmain.o -o %tmain.exe
# RUN: llvm-bolt %tmain.exe -o %tmain.exe.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-names --debug-info %tmain.exe.bolt > %tlog.txt
# RUN: cat %tlog.txt | FileCheck -check-prefix=BOLT %s

## This test checks that BOLT correctly generates .debug_names section when there is DW_TAG_variable
## with DW_OP_GNU_push_tls_address in DW_AT_location.

# BOLT:       [[DIEOFFSET:0x[0-9a-f]*]]:   DW_TAG_variable
# BOLT-NEXT:                 DW_AT_name	("x")
# BOLT-NEXT:                 DW_AT_type	({{.+}} "int")
# BOLT-NEXT:                 DW_AT_external	(true)
# BOLT-NEXT:                 DW_AT_decl_file	("gnu_tls_push/main.cpp")
# BOLT-NEXT:                 DW_AT_decl_line	(1)
# BOLT-NEXT:                 DW_AT_location	(DW_OP_const8u 0x0, DW_OP_GNU_push_tls_address)
# BOLT:       Hash: 0x2B61D
# BOLT-NEXT:       String: {{.+}} "x"
# BOLT-NEXT:       Entry @ {{.+}} {
# BOLT-NEXT:         Abbrev: {{.+}}
# BOLT-NEXT:         Tag: DW_TAG_variable
# BOLT-NEXT:         DW_IDX_die_offset: [[DIEOFFSET]]
# BOLT-NEXT:         DW_IDX_parent: <parent not indexed>

## thread_local int x = 0;
## int main() {
##   x = 10;
##   return x;
## }
	.file	"main.cpp"
	.file	0 "gnu_tls_push" "main.cpp" md5 0x551db97d5e23dc6a81abdc5ade4d9d71
	.globl	main
	.type	main,@function
main:
.Lfunc_begin0:
	.loc	0 2 0
	.loc	0 3 3 prologue_end
	.loc	0 3 5 is_stmt 0
	.loc	0 4 10 is_stmt 1
	.loc	0 4 3 epilogue_begin is_stmt 0
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.hidden	_ZTW1x
	.weak	_ZTW1x
	.type	_ZTW1x,@function
_ZTW1x:
.Lfunc_begin1:
	retq
.Lfunc_end1:
	.size	_ZTW1x, .Lfunc_end1-_ZTW1x

	.type	x,@object
	.section	.tbss,"awT",@nobits
	.globl	x
x:
	.long	0
	.size	x, 4

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
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x3e DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x13 DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	10                              # DW_AT_location
	.byte	14
	.quad	x@DTPOFF
	.byte	224
	.byte	3                               # Abbrev [3] 0x36:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x3a:0xf DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	54                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 17.0.4" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=137
.Linfo_string2:
	.asciz	"gnu_tls_push" # string offset=146
.Linfo_string3:
	.asciz	"x"                             # string offset=184
.Linfo_string4:
	.asciz	"int"                           # string offset=186
.Linfo_string5:
	.asciz	"main"                          # string offset=190
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
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
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	3                               # Header: bucket count
	.long	3                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	2                               # Bucket 1
	.long	3                               # Bucket 2
	.long	177693                          # Hash in Bucket 0
	.long	2090499946                      # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 2
	.long	.Linfo_string3                  # String in Bucket 0: x
	.long	.Linfo_string5                  # String in Bucket 1: main
	.long	.Linfo_string4                  # String in Bucket 2: int
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
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
	.byte	46                              # DW_TAG_subprogram
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
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
.L2:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: x
.Lnames2:
.L0:
	.byte	2                               # Abbreviation code
	.long	58                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames0:
.L1:
	.byte	3                               # Abbreviation code
	.long	54                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 17.0.4 (https://git.internal.tfbnw.net/repos/git/rw/osmeta/external/llvm-project 8d1fd9f463cb31caf429b83cf7a5baea5f67e54a)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

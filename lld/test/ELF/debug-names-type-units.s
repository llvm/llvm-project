# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.tu.s -o a.tu.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.foreign-tu.s -o a.foreign-tu.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

# RUN: ld.lld --debug-names a.tu.o b.o -o out0 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: llvm-dwarfdump --debug-names out0 | FileCheck --check-prefix=DWARF %s

# WARN: warning: a.tu.o:(.debug_names): type units are not implemented

# DWARF:        CU count: 2
# DWARF-NEXT:   Local TU count: 0
# DWARF-NEXT:   Foreign TU count: 0
# DWARF:      Compilation Unit offsets [
# DWARF-NEXT:    CU[0]:
# DWARF-SAME:           0x0000002a
# DWARF-NEXT:    CU[1]:
# DWARF-SAME:           0x0000006a
# DWARF-NEXT:  ]
# DWARF:        String: {{.*}} "t1"
# DWARF-NEXT:   Entry @ 0x8f {
# DWARF-NEXT:     Abbrev: 0x2
# DWARF-NEXT:     Tag: DW_TAG_structure_type
# DWARF-NEXT:     DW_IDX_type_unit: 0x00
# DWARF-NEXT:     DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:     DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:     DW_IDX_compile_unit: 0x00
# DWARF-NEXT:   }
# DWARF-NEXT:   Entry @ 0x96 {
# DWARF-NEXT:     Abbrev: 0x1
# DWARF-NEXT:     Tag: DW_TAG_structure_type
# DWARF-NEXT:     DW_IDX_die_offset: 0x00000036
# DWARF-NEXT:     DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:     DW_IDX_compile_unit: 0x00
# DWARF-NEXT:   }
# DWARF-NEXT:   Entry @ 0x9c {
# DWARF-NEXT:     Abbrev: 0x1
# DWARF-NEXT:     Tag: DW_TAG_structure_type
# DWARF-NEXT:     DW_IDX_die_offset: 0x00000029
# DWARF-NEXT:     DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:     DW_IDX_compile_unit: 0x01
# DWARF-NEXT:   }

# RUN: ld.lld --debug-names a.foreign-tu.o b.o -o out1 2>&1 | FileCheck %s --check-prefix=WARN2
# RUN: llvm-dwarfdump --debug-names out1 | FileCheck --check-prefix=DWARF2 %s

# WARN2: warning: a.foreign-tu.o:(.debug_names): type units are not implemented

# DWARF2:        CU count: 2
# DWARF2-NEXT:   Local TU count: 0
# DWARF2-NEXT:   Foreign TU count: 0
# DWARF2:      Compilation Unit offsets [
# DWARF2-NEXT:    CU[0]:
# DWARF2-SAME:           0x00000000
# DWARF2-NEXT:    CU[1]:
# DWARF2-SAME:           0x00000028
# DWARF2-NEXT:  ]

.ifdef GEN
#--- a.cc
struct t1 {};
extern "C" void _start(t1) {}
#--- b.cc
struct t1 { } vb;
#--- gen
echo '#--- a.tu.s'
clang --target=x86_64-linux -S -O1 -g -gpubnames -fdebug-types-section a.cc -o -
echo '#--- a.foreign-tu.s'
clang --target=x86_64-linux -S -O1 -g -gpubnames -fdebug-types-section -gsplit-dwarf a.cc -o -
echo '#--- b.s'
clang --target=x86_64-linux -S -O1 -g -gpubnames b.cc -o -
.endif
#--- a.tu.s
	.text
	.file	"a.cc"
	.globl	_start                          # -- Begin function _start
	.p2align	4, 0x90
	.type	_start,@function
_start:                                 # @_start
.Lfunc_begin0:
	.file	0 "/proc/self/cwd" "a.cc" md5 0x6835f89a7d36054002b51e54e47d852e
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	0 2 29 prologue_end             # a.cc:2:29
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_start, .Lfunc_end0-_start
	.cfi_endproc
                                        # -- End function
	.section	.debug_info,"G",@progbits,14297044602779165170,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	2                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	-4149699470930386446            # Type Signature
	.long	35                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x12 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	2                               # Abbrev [2] 0x23:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
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
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
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
	.byte	3                               # Abbrev [3] 0xc:0x34 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	4                               # Abbrev [4] 0x23:0x13 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	3                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x2e:0x7 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	54                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x36:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-4149699470930386446            # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	24                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"a.cc"                          # string offset=1
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=6
.Linfo_string3:
	.asciz	"_start"                        # string offset=21
.Linfo_string4:
	.asciz	"t1"                            # string offset=28
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
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
	.long	2                               # Header: bucket count
	.long	2                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	5863786                         # Hash in Bucket 0
	.long	-304389582                      # Hash in Bucket 0
	.long	.Linfo_string4                  # String in Bucket 0: t1
	.long	.Linfo_string3                  # String in Bucket 0: _start
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 0
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
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
.Lnames1:
.L1:
	.byte	1                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
.L0:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.long	54                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames0:
	.byte	3                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _start
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
#--- a.foreign-tu.s
	.text
	.file	"a.cc"
	.globl	_start                          # -- Begin function _start
	.p2align	4, 0x90
	.type	_start,@function
_start:                                 # @_start
.Lfunc_begin0:
	.file	0 "/proc/self/cwd" "a.cc" md5 0x6835f89a7d36054002b51e54e47d852e
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	0 2 29 prologue_end             # a.cc:2:29
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_start, .Lfunc_end0-_start
	.cfi_endproc
                                        # -- End function
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-4149699470930386446            # Type Signature
	.long	33                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0x10 DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_comp_dir
	.byte	2                               # DW_AT_dwo_name
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x21:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
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
	.quad	-2763464185488304260
	.byte	1                               # Abbrev [1] 0x14:0x14 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
	.byte	1                               # DW_AT_dwo_name
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	12                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"/proc/self/cwd"                # string offset=0
.Lskel_string1:
	.asciz	"_start"                        # string offset=15
.Lskel_string2:
	.asciz	"t1"                            # string offset=22
.Lskel_string3:
	.asciz	"a.dwo"                         # string offset=25
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string3
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	28                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_start"                        # string offset=0
.Linfo_string1:
	.asciz	"/proc/self/cwd"                # string offset=7
.Linfo_string2:
	.asciz	"a.dwo"                         # string offset=22
.Linfo_string3:
	.asciz	"t1"                            # string offset=28
.Linfo_string4:
	.byte	0                               # string offset=31
.Linfo_string5:
	.asciz	"a.cc"                          # string offset=32
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	7
	.long	22
	.long	28
	.long	31
	.long	32
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-2763464185488304260
	.byte	3                               # Abbrev [3] 0x14:0x23 DW_TAG_compile_unit
	.byte	4                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	5                               # DW_AT_name
	.byte	2                               # DW_AT_dwo_name
	.byte	4                               # Abbrev [4] 0x1a:0x13 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	0                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x25:0x7 DW_TAG_formal_parameter
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	45                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x2d:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-4149699470930386446            # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
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
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
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
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_line.dwo,"e",@progbits
.Ltmp1:
	.long	.Ldebug_line_end0-.Ldebug_line_start0 # unit length
.Ldebug_line_start0:
	.short	5
	.byte	8
	.byte	0
	.long	.Lprologue_end0-.Lprologue_start0
.Lprologue_start0:
	.byte	1
	.byte	1
	.byte	1
	.byte	-5
	.byte	14
	.byte	1
	.byte	1
	.byte	1
	.byte	8
	.byte	1
	.ascii	"/proc/self/cwd"
	.byte	0
	.byte	3
	.byte	1
	.byte	8
	.byte	2
	.byte	15
	.byte	5
	.byte	30
	.byte	1
	.ascii	"a.cc"
	.byte	0
	.byte	0
	.byte	0x68, 0x35, 0xf8, 0x9a
	.byte	0x7d, 0x36, 0x05, 0x40
	.byte	0x02, 0xb5, 0x1e, 0x54
	.byte	0xe4, 0x7d, 0x85, 0x2e
.Lprologue_end0:
.Ldebug_line_end0:
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
	.long	1                               # Header: foreign type unit count
	.long	2                               # Header: bucket count
	.long	2                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.quad	-4149699470930386446            # Type unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	5863786                         # Hash in Bucket 0
	.long	-304389582                      # Hash in Bucket 0
	.long	.Lskel_string2                  # String in Bucket 0: t1
	.long	.Lskel_string1                  # String in Bucket 0: _start
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 0
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
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
.Lnames1:
.L2:
	.byte	1                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	33                              # DW_IDX_die_offset
.L1:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.long	45                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames0:
.L0:
	.byte	3                               # Abbreviation code
	.long	26                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _start
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
#--- b.s
	.text
	.file	"b.cc"
	.file	0 "/proc/self/cwd" "b.cc" md5 0xe69190c4224a66c796605e3a1324674b
	.type	vb,@object                      # @vb
	.bss
	.globl	vb
vb:
	.zero	1
	.size	vb, 1

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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x24 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x1e:0xb DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	41                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x29:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	4                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	24                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"b.cc"                          # string offset=1
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=6
.Linfo_string3:
	.asciz	"vb"                            # string offset=21
.Linfo_string4:
	.asciz	"t1"                            # string offset=24
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	vb
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	2                               # Header: bucket count
	.long	2                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	2                               # Bucket 1
	.long	5863786                         # Hash in Bucket 0
	.long	5863901                         # Hash in Bucket 1
	.long	.Linfo_string4                  # String in Bucket 0: t1
	.long	.Linfo_string3                  # String in Bucket 1: vb
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
.L1:
	.byte	1                               # Abbreviation code
	.long	41                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames1:
.L0:
	.byte	2                               # Abbreviation code
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: vb
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

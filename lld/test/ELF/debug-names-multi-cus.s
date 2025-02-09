# REQUIRES: x86
## Test name indexes that contain multiple CU offsets due to LTO.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/debug-names-a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bcd.s -o bcd.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ef.s -o ef.o
# RUN: ld.lld --debug-names a.o bcd.o ef.o -o out
# RUN: llvm-dwarfdump --debug-info --debug-names out | FileCheck %s --check-prefix=DWARF

## Place the multiple CU offsets in the second name index in an input file.
# RUN: ld.lld -r a.o bcd.o -o abcd.o
# RUN: ld.lld --debug-names abcd.o ef.o -o out
# RUN: llvm-dwarfdump --debug-info --debug-names out | FileCheck %s --check-prefix=DWARF

# DWARF:      [[CU0:0x[^:]+]]: Compile Unit
# DWARF:      [[CU1:0x[^:]+]]: Compile Unit
# DWARF:      [[CU2:0x[^:]+]]: Compile Unit
# DWARF:      [[CU3:0x[^:]+]]: Compile Unit
# DWARF:      [[CU4:0x[^:]+]]: Compile Unit
# DWARF:      [[CU5:0x[^:]+]]: Compile Unit
# DWARF:      Compilation Unit offsets [
# DWARF-NEXT:   CU[0]: [[CU0]]
# DWARF-NEXT:   CU[1]: [[CU1]]
# DWARF-NEXT:   CU[2]: [[CU2]]
# DWARF-NEXT:   CU[3]: [[CU3]]
# DWARF-NEXT:   CU[4]: [[CU4]]
# DWARF-NEXT:   CU[5]: [[CU5]]
# DWARF-NEXT: ]
# DWARF:        String: {{.*}} "vc"
# DWARF:          DW_IDX_compile_unit: 0x02
# DWARF:        String: {{.*}} "vd"
# DWARF:          DW_IDX_die_offset:
# DWARF-SAME:                        0x00000020
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x03
# DWARF:        String: {{.*}} "ve"
# DWARF:          DW_IDX_die_offset:
# DWARF-SAME:                        0x0000001e
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x04
# DWARF:        String: {{.*}} "vf"
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x05
# DWARF:        String: {{.*}} "vb"
# DWARF:          DW_IDX_compile_unit:
# DWARF-SAME:                          0x01

.ifdef GEN
#--- b.cc
[[gnu::used]] int vb;
#--- c.cc
[[gnu::used]] int vc;
#--- d.cc
namespace ns {
[[gnu::used]] int vd;
}

//--- e.cc
[[gnu::used]] int ve;
//--- f.cc
namespace ns {
[[gnu::used]] int vf;
}

#--- gen
clang --target=x86_64-linux -O1 -g -gpubnames -flto b.cc c.cc d.cc -nostdlib -fuse-ld=lld -Wl,--lto-emit-asm
echo '#--- bcd.s'
cat a.out.lto.s
clang --target=x86_64-linux -O1 -g -gpubnames -flto e.cc f.cc -nostdlib -fuse-ld=lld -Wl,--lto-emit-asm
echo '#--- ef.s'
cat a.out.lto.s
.endif
#--- bcd.s
	.text
	.file	"ld-temp.o"
	.file	1 "/proc/self/cwd" "b.cc" md5 0x78dad32a49063326a4de543198e54944
	.file	2 "/proc/self/cwd" "c.cc" md5 0x7a0f7bf2cb0ec8c297f794908d91ab1b
	.file	3 "/proc/self/cwd" "d.cc" md5 0xf7e2af89615ce48bf9a98fdae55ab5ad
	.type	vb,@object                      # @vb
	.section	.bss.vb,"aw",@nobits
	.globl	vb
	.p2align	2, 0x0
vb:
	.long	0                               # 0x0
	.size	vb, 4

	.type	vc,@object                      # @vc
	.section	.bss.vc,"aw",@nobits
	.globl	vc
	.p2align	2, 0x0
vc:
	.long	0                               # 0x0
	.size	vc, 4

	.type	_ZN2ns2vdE,@object              # @_ZN2ns2vdE
	.section	.bss._ZN2ns2vdE,"aw",@nobits
	.globl	_ZN2ns2vdE
	.p2align	2, 0x0
_ZN2ns2vdE:
	.long	0                               # 0x0
	.size	_ZN2ns2vdE, 4

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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
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
	.byte	5                               # Abbreviation Code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	1                               # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
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
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x29:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x1e DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	5                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	4                               # Abbrev [4] 0x1e:0xb DW_TAG_variable
	.byte	6                               # DW_AT_name
	.long	.debug_info+41                  # DW_AT_type
                                        # DW_AT_external
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	1
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
.Lcu_begin2:
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	7                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	5                               # Abbrev [5] 0x1e:0xf DW_TAG_namespace
	.byte	8                               # DW_AT_name
	.byte	6                               # Abbrev [6] 0x20:0xc DW_TAG_variable
	.byte	9                               # DW_AT_name
	.long	.debug_info+41                  # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	2
	.byte	10                              # DW_AT_linkage_name
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_str_offsets,"",@progbits
	.long	48                              # Length of String Offsets Set
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
	.asciz	"int"                           # string offset=24
.Linfo_string5:
	.asciz	"c.cc"                          # string offset=28
.Linfo_string6:
	.asciz	"vc"                            # string offset=33
.Linfo_string7:
	.asciz	"d.cc"                          # string offset=36
.Linfo_string8:
	.asciz	"ns"                            # string offset=41
.Linfo_string9:
	.asciz	"vd"                            # string offset=44
.Linfo_string10:
	.asciz	"_ZN2ns2vdE"                    # string offset=47
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
	.quad	vb
	.quad	vc
	.quad	_ZN2ns2vdE
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	3                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	6                               # Header: bucket count
	.long	6                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Lcu_begin1                     # Compilation unit 1
	.long	.Lcu_begin2                     # Compilation unit 2
	.long	1                               # Bucket 0
	.long	2                               # Bucket 1
	.long	3                               # Bucket 2
	.long	0                               # Bucket 3
	.long	4                               # Bucket 4
	.long	6                               # Bucket 5
	.long	5863902                         # Hash in Bucket 0
	.long	5863903                         # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 2
	.long	5863654                         # Hash in Bucket 4
	.long	-823734096                      # Hash in Bucket 4
	.long	5863901                         # Hash in Bucket 5
	.long	.Linfo_string6                  # String in Bucket 0: vc
	.long	.Linfo_string9                  # String in Bucket 1: vd
	.long	.Linfo_string4                  # String in Bucket 2: int
	.long	.Linfo_string8                  # String in Bucket 4: ns
	.long	.Linfo_string10                 # String in Bucket 4: _ZN2ns2vdE
	.long	.Linfo_string3                  # String in Bucket 5: vb
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 5
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames2:
.L0:
	.byte	1                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: vc
.Lnames4:
.L4:
	.byte	2                               # Abbreviation code
	.byte	2                               # DW_IDX_compile_unit
	.long	32                              # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: vd
.Lnames0:
.L3:
	.byte	3                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	41                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames3:
.L2:
	.byte	4                               # Abbreviation code
	.byte	2                               # DW_IDX_compile_unit
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: ns
.Lnames5:
	.byte	2                               # Abbreviation code
	.byte	2                               # DW_IDX_compile_unit
	.long	32                              # DW_IDX_die_offset
	.long	.L2-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _ZN2ns2vdE
.Lnames1:
.L1:
	.byte	1                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: vb
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym vb
	.addrsig_sym vc
	.addrsig_sym _ZN2ns2vdE
	.section	.debug_line,"",@progbits
.Lline_table_start0:
#--- ef.s
	.text
	.file	"ld-temp.o"
	.file	1 "/proc/self/cwd" "e.cc" md5 0xa8d6c645998197bd15436f2a351ebd6a
	.file	2 "/proc/self/cwd" "f.cc" md5 0x6ec1ec6b7f003f84cb0bf3409e65b085
	.type	ve,@object                      # @ve
	.section	.bss.ve,"aw",@nobits
	.globl	ve
	.p2align	2, 0x0
ve:
	.long	0                               # 0x0
	.size	ve, 4

	.type	_ZN2ns2vfE,@object              # @_ZN2ns2vfE
	.section	.bss._ZN2ns2vfE,"aw",@nobits
	.globl	_ZN2ns2vfE
	.p2align	2, 0x0
_ZN2ns2vfE:
	.long	0                               # 0x0
	.size	_ZN2ns2vfE, 4

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
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	1                               # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
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
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x29:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	5                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	4                               # Abbrev [4] 0x1e:0xf DW_TAG_namespace
	.byte	6                               # DW_AT_name
	.byte	5                               # Abbrev [5] 0x20:0xc DW_TAG_variable
	.byte	7                               # DW_AT_name
	.long	.debug_info+41                  # DW_AT_type
                                        # DW_AT_external
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	1
	.byte	8                               # DW_AT_linkage_name
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	40                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.byte	0                               # string offset=0
.Linfo_string1:
	.asciz	"e.cc"                          # string offset=1
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=6
.Linfo_string3:
	.asciz	"ve"                            # string offset=21
.Linfo_string4:
	.asciz	"int"                           # string offset=24
.Linfo_string5:
	.asciz	"f.cc"                          # string offset=28
.Linfo_string6:
	.asciz	"ns"                            # string offset=33
.Linfo_string7:
	.asciz	"vf"                            # string offset=36
.Linfo_string8:
	.asciz	"_ZN2ns2vfE"                    # string offset=39
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	ve
	.quad	_ZN2ns2vfE
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	2                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	5                               # Header: bucket count
	.long	5                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Lcu_begin1                     # Compilation unit 1
	.long	1                               # Bucket 0
	.long	2                               # Bucket 1
	.long	0                               # Bucket 2
	.long	3                               # Bucket 3
	.long	4                               # Bucket 4
	.long	5863905                         # Hash in Bucket 0
	.long	-823734030                      # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 3
	.long	5863654                         # Hash in Bucket 4
	.long	5863904                         # Hash in Bucket 4
	.long	.Linfo_string7                  # String in Bucket 0: vf
	.long	.Linfo_string8                  # String in Bucket 1: _ZN2ns2vfE
	.long	.Linfo_string4                  # String in Bucket 3: int
	.long	.Linfo_string6                  # String in Bucket 4: ns
	.long	.Linfo_string3                  # String in Bucket 4: ve
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 4
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	52                              # DW_TAG_variable
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
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
	.byte	1                               # DW_IDX_compile_unit
	.long	32                              # DW_IDX_die_offset
	.long	.L0-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: vf
.Lnames4:
	.byte	1                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	32                              # DW_IDX_die_offset
	.long	.L0-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _ZN2ns2vfE
.Lnames0:
.L3:
	.byte	2                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	41                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames2:
.L0:
	.byte	3                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: ns
.Lnames1:
.L1:
	.byte	4                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	30                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: ve
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym ve
	.addrsig_sym _ZN2ns2vfE
	.section	.debug_line,"",@progbits
.Lline_table_start0:

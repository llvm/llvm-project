# REQUIRES: x86
## Test clang-17-generated DW_TAG_subprogram, which do not contain DW_IDX_parent
## attributes.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --debug-names %t.o -o %t
# RUN: llvm-dwarfdump --debug-info --debug-names %t | FileCheck %s --check-prefix=DWARF

# DWARF: 0x00000023:   DW_TAG_namespace
# DWARF: 0x00000025:     DW_TAG_subprogram

# DWARF:      String: {{.*}} "fa"
# DWARF-NEXT: Entry @ 0x71 {
# DWARF-NEXT:   Abbrev: 0x1
# DWARF-NEXT:   Tag: DW_TAG_subprogram
# DWARF-NEXT:   DW_IDX_die_offset: 0x00000025
# DWARF-NEXT:   DW_IDX_compile_unit: 0x00
# DWARF-NEXT: }
# DWARF:      String: {{.*}} "ns"
# DWARF-NEXT: Entry @ 0x78 {
# DWARF-NEXT:   Abbrev: 0x2
# DWARF-NEXT:   Tag: DW_TAG_namespace
# DWARF-NEXT:   DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:   DW_IDX_compile_unit: 0x00
# DWARF-NEXT: }

.ifdef GEN
//--- a.cc
namespace ns {
void fa() {}
}
//--- gen
clang-17 --target=x86_64-linux -S -O1 -g -gpubnames a.cc -o -
.endif
	.text
	.file	"a.cc"
	.globl	_ZN2ns2faEv                     # -- Begin function _ZN2ns2faEv
	.p2align	4, 0x90
	.type	_ZN2ns2faEv,@function
_ZN2ns2faEv:                            # @_ZN2ns2faEv
.Lfunc_begin0:
	.file	0 "/proc/self/cwd" "a.cc" md5 0xb3281d5b5a0b2997d7d59d49bc912274
	.cfi_startproc
# %bb.0:
	.loc	0 2 12 prologue_end             # a.cc:2:12
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_ZN2ns2faEv, .Lfunc_end0-_ZN2ns2faEv
	.cfi_endproc
                                        # -- End function
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
	.byte	57                              # DW_TAG_namespace
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x27 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0xf DW_TAG_namespace
	.byte	3                               # DW_AT_name
	.byte	3                               # Abbrev [3] 0x25:0xc DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28                              # Length of String Offsets Set
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
	.asciz	"ns"                            # string offset=21
.Linfo_string4:
	.asciz	"fa"                            # string offset=24
.Linfo_string5:
	.asciz	"_ZN2ns2faEv"                   # string offset=27
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string5
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
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	3                               # Header: bucket count
	.long	3                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	3                               # Bucket 2
	.long	5863372                         # Hash in Bucket 1
	.long	5863654                         # Hash in Bucket 1
	.long	-1413999533                     # Hash in Bucket 2
	.long	.Linfo_string4                  # String in Bucket 1: fa
	.long	.Linfo_string3                  # String in Bucket 1: ns
	.long	.Linfo_string5                  # String in Bucket 2: _ZN2ns2faEv
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 2
.Lnames_abbrev_start0:
	.byte	46                              # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	57                              # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
	.byte	46                              # Abbreviation code
	.long	37                              # DW_IDX_die_offset
	.byte	0                               # End of list: fa
.Lnames0:
	.byte	57                              # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # End of list: ns
.Lnames2:
	.byte	46                              # Abbreviation code
	.long	37                              # DW_IDX_die_offset
	.byte	0                               # End of list: _ZN2ns2faEv
	.p2align	2, 0x0
.Lnames_end0:
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

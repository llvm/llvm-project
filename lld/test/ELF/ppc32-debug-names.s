# ppc32-debug-names.s was generated with:

# clang++ --target=powerpc -g -gpubnames \
#    -fdebug-compilation-dir='/self/proc/cwd' -S a.cpp -o a.s

# a.cpp contents:

# struct t1 { };
# void f1(t1) { }

# REQUIRES: ppc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=powerpc a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=powerpc b.s -o b.o

# RUN: ld.lld --debug-names --no-debug-names a.o b.o -o out0
# RUN: llvm-readelf -SW out0 | FileCheck %s --check-prefix=NO_MERGE
	
# NO_MERGE: Name              Type     Address          Off      Size   ES Flg Lk Inf Al
# NO_MERGE: .debug_names      PROGBITS 00000000 [[#%x,]] 000110 00      0   0  4
	
# RUN: ld.lld --debug-names a.o b.o -o out

# RUN: llvm-dwarfdump -debug-names out | FileCheck %s --check-prefix=DWARF
# RUN: llvm-readelf -SW out | FileCheck %s --check-prefix=READELF

# READELF: Name              Type     Address          Off      Size   ES Flg Lk Inf Al
# READELF: .debug_names      PROGBITS 00000000 [[#%x,]] 0000cc 00      0   0  4

# DWARF:      file format elf32-powerpc
# DWARF:      .debug_names contents:
# DWARF:      Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length: 0xC8
# DWARF-NEXT:     Format: DWARF32
# DWARF-NEXT:     Version: 5
# DWARF-NEXT:     CU count: 2
# DWARF-NEXT:     Local TU count: 0
# DWARF-NEXT:     Foreign TU count: 0
# DWARF-NEXT:     Bucket count: 5
# DWARF-NEXT:     Name count: 5
# DWARF-NEXT:     Abbreviations table size: 0x1F
# DWARF-NEXT:     Augmentation: 'LLVM0700'
# DWARF:        Compilation Unit offsets [
# DWARF-NEXT:     CU[0]: 0x00000000
# DWARF-NEXT:     CU[1]: 0x0000000c
# DWARF:          Abbreviations [
# DWARF-NEXT:     Abbreviation 0x1 {
# DWARF:            Tag: DW_TAG_structure_type
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x2 {
# DWARF-NEXT:       Tag: DW_TAG_subprogram
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x3 {
# DWARF-NEXT:       Tag: DW_TAG_base_type
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:        Bucket 0 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 1 [
# DWARF-NEXT:     Name 1 {
# DWARF-NEXT:       Hash: 0x59796A
# DWARF-NEXT:       String: 0x00000089 "t1"
# DWARF-NEXT:       Entry @ 0xaa {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_structure_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x0000003a
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0xb0 {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_structure_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000042
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 2 {
# DWARF-NEXT:       Hash: 0x5355B2BE
# DWARF-NEXT:       String: 0x00000080 "_Z2f12t1"
# DWARF-NEXT:       Entry @ 0xbe {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 3 {
# DWARF-NEXT:       Hash: 0x7C9A7F6A
# DWARF-NEXT:       String: 0x0000010d "main"
# DWARF-NEXT:       Entry @ 0xc5 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 2 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 3 [
# DWARF-NEXT:     Name 4 {
# DWARF-NEXT:       Hash: 0xB888030
# DWARF-NEXT:       String: 0x00000112 "int"
# DWARF-NEXT:       Entry @ 0xb7 {
# DWARF-NEXT:         Abbrev: 0x3
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x0000003e
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 4 [
# DWARF-NEXT:     Name 5 {
# DWARF-NEXT:       Hash: 0x59779C
# DWARF-NEXT:       String: 0x0000007d "f1"
# DWARF-NEXT:       Entry @ 0xa3 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]

#--- a.s
	.text
	.globl	_Z2f12t1                        # -- Begin function _Z2f12t1
	.p2align	2
	.type	_Z2f12t1,@function
_Z2f12t1:                               # @_Z2f12t1
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: f1: <- [$r3+0]
	stwu 1, -16(1)
	stw 31, 12(1)
	.cfi_def_cfa_offset 16
	.cfi_offset r31, -4
	mr	31, 1
	.cfi_def_cfa_register r31
.Ltmp0:
	lwz 31, 12(1)
	addi 1, 1, 16
	blr
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f12t1, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)" # string offset=0
.Linfo_string1:
	.asciz	"a.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"f1"                            # string offset=125
.Linfo_string4:
	.asciz	"_Z2f12t1"                      # string offset=128
.Linfo_string5:
	.asciz	"t1"                            # string offset=137
.Laddr_table_base0:
	.long	.Lfunc_begin0
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
	.long	5863324                         # Hash in Bucket 1
	.long	5863786                         # Hash in Bucket 1
	.long	1398125246                      # Hash in Bucket 2
	.long	.Linfo_string3                  # String in Bucket 1: f1
	.long	.Linfo_string5                  # String in Bucket 1: t1
	.long	.Linfo_string4                  # String in Bucket 2: _Z2f12t1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 2
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
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
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
.L1:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: f1
.Lnames2:
.L0:
	.byte	2                               # Abbreviation code
	.long	58                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames1:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z2f12t1
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

#--- b.s
# Generated with:
# - clang++ --target=powerpc -g -O0 -gpubnames \
#     -fdebug-compilation-dir='/self/proc/cwd' -S b.cpp -o b.s

# b.cpp contents:

# struct t1 { };
# int main() {
#   t1 v1;
# }
#
	.text
	.globl	main                            # -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	stwu 1, -16(1)
	stw 31, 12(1)
	.cfi_def_cfa_offset 16
	.cfi_offset r31, -4
	mr	31, 1
	.cfi_def_cfa_register r31
	li 3, 0
.Ltmp0:
	lwz 31, 12(1)
	addi 1, 1, 16
	blr
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	32                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)" # string offset=0
.Linfo_string1:
	.asciz	"b.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"main"                          # string offset=125
.Linfo_string4:
	.asciz	"int"                           # string offset=130
.Linfo_string5:
	.asciz	"v1"                            # string offset=134
.Linfo_string6:
	.asciz	"t1"                            # string offset=137
.Laddr_table_base0:
	.long	.Lfunc_begin0
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
	.long	5863786                         # Hash in Bucket 1
	.long	2090499946                      # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 2
	.long	.Linfo_string6                  # String in Bucket 1: t1
	.long	.Linfo_string3                  # String in Bucket 1: main
	.long	.Linfo_string4                  # String in Bucket 2: int
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 2
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
.Lnames2:
.L1:
	.byte	1                               # Abbreviation code
	.long	66                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames0:
.L2:
	.byte	2                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames1:
.L0:
	.byte	3                               # Abbreviation code
	.long	62                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

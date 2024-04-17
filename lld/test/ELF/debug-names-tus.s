# The .s files were generated with:

# clang++ -g -gpubnames -fdebug-compilation-dir='/proc/self/cwd' \
#    -fdebug-types-section -S a.cpp -o a.tu.s

# clang++ -g -gpubnames -fdebug-compilation-dir='/proc/self/cwd' \
#    -fdebug-types-section -gsplit-dwarf -S a.cpp -o a.foreign-tu.s

# clang++ -g -gpubnames -fdebug-compilation-dir='/proc/self/cwd' \
#     -S b.cpp -o b.s

# a.cpp contents:

# struct t1 { };
# void f1(t1) { }

# b.cpp contents:

# struct t1 { };
# int main() {
#   t1 v1;
# }
#

# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.tu.s -o a.tu.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.foreign-tu.s -o a.foreign-tu.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

# RUN: ld.lld --debug-names  a.tu.o b.o -o out0 2>&1 \ 
# RUN:     | FileCheck  --check-prefix=WARN %s
# RUN: llvm-dwarfdump --debug-names out0 | FileCheck --check-prefix=DWARF %s

# WARN: warning: a.tu.o:(.debug_names): type units are not implemented

# DWARF:Name Index @ 0x0 {
# DWARF:  Compilation Unit offsets [
# DWARF-NEXT:    CU[0]:
# DWARF-SAME:           0x00000004
# DWARF-NEXT:    CU[1]:
# DWARF-SAME:           0x00000008
# DWARF-NEXT:  ]
# DWARF-NEXT:  Abbreviations [
# DWARF-NEXT:    Abbreviation 0x1 {
# DWARF-NEXT:      Tag: DW_TAG_structure_type
# DWARF-NEXT:      DW_IDX_type_unit: DW_FORM_data1
# DWARF-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:      DW_IDX_compile_unit: DW_FORM_data1 
# DWARF-NEXT:   }

	
# RUN: ld.lld --debug-names a.foreign-tu.o b.o -o out1 2>&1 \
# RUN:     | FileCheck %s --check-prefix=WARN2 
# RUN: llvm-dwarfdump --debug-names out1 | FileCheck --check-prefix=DWARF2 %s

# WARN2: warning: a.foreign-tu.o:(.debug_names): type units are not implemented

# DWARF2:Name Index @ 0x0 {
# DWARF2:  Compilation Unit offsets [
# DWARF2-NEXT:    CU[0]:
# DWARF2-SAME:           0x00000000
# DWARF2-NEXT:    CU[1]:
# DWARF2-SAME:           0x00000001
# DWARF2-NEXT:  ]
# DWARF2-NEXT:  Abbreviations [
# DWARF2-NEXT:    Abbreviation 0x1 {
# DWARF2-NEXT:      Tag: DW_TAG_structure_type
# DWARF2-NEXT:      DW_IDX_type_unit: DW_FORM_data1
# DWARF2-NEXT:      DW_IDX_die_offset: DW_FORM_ref4
# DWARF2-NEXT:      DW_IDX_parent: DW_FORM_flag_present
# DWARF2-NEXT:      DW_IDX_compile_unit: DW_FORM_data1
# DWARF2-NEXT:    }
 

#--- a.tu.s
	.text
	.globl	_Z2f12t1                        # -- Begin function _Z2f12t1
	.p2align	4, 0x90
	.type	_Z2f12t1,@function
_Z2f12t1:                               # @_Z2f12t1
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f12t1, .Lfunc_end0-_Z2f12t1
	.cfi_endproc
                                        # -- End function
	.section	.debug_info,"G",@progbits,14297044602779165170,comdat
.Ltu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
.Ldebug_info_end0:
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
.Ldebug_info_end1:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git d12762cdb6357915e0f6f6dbfc09c2c75d746ee7)" # string offset=0
.Linfo_string1:
	.asciz	"a.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"debug-names-test"              # string offset=110
.Linfo_string3:
	.asciz	"f1"                            # string offset=127
.Linfo_string4:
	.asciz	"_Z2f12t1"                      # string offset=130
.Linfo_string5:
	.asciz	"t1"                            # string offset=139
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	1                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	3                               # Header: bucket count
	.long	3                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Ltu_begin0                     # Type unit 0
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
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
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
	.byte	2                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	35                              # DW_IDX_die_offset
.L0:                                    # DW_IDX_parent
	.byte	3                               # Abbreviation code
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
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git d12762cdb6357915e0f6f6dbfc09c2c75d746ee7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:


#--- a.foreign-tu.s
	.text
	.globl	_Z2f12t1                        # -- Begin function _Z2f12t1
	.p2align	4, 0x90
	.type	_Z2f12t1,@function
_Z2f12t1:                               # @_Z2f12t1
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f12t1, .Lfunc_end0-_Z2f12t1
	.cfi_endproc
                                        # -- End function
	.section	.debug_info.dwo,"e",@progbits
	.byte 0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.byte 0
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"debug-names-test"              # string offset=0
.Lskel_string1:
	.asciz	"f1"                            # string offset=17
.Lskel_string2:
	.asciz	"_Z2f12t1"                      # string offset=20
.Lskel_string3:
	.asciz	"t1"                            # string offset=29
.Lskel_string4:
	.asciz	"a.dwo"                         # string offset=32
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z2f12t1"                      # string offset=0
.Linfo_string1:
	.asciz	"f1"                            # string offset=9
.Linfo_string2:
	.asciz	"debug-names-test"              # string offset=12
.Linfo_string3:
	.asciz	"a.dwo"                         # string offset=29
.Linfo_string4:
	.asciz	"t1"                            # string offset=35
.Linfo_string5:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git d12762cdb6357915e0f6f6dbfc09c2c75d746ee7)" # string offset=38
.Linfo_string6:
	.asciz	"a.cpp"                         # string offset=142
	.section	.debug_info.dwo,"e",@progbits
	.byte 0
	.section	.debug_line.dwo,"e",@progbits
.Ltmp2:
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
	.ascii	"debug-names-test"
	.byte	0
	.byte	3
	.byte	1
	.byte	8
	.byte	2
	.byte	15
	.byte	5
	.byte	30
	.byte	1
	.ascii	"a.cpp"
	.byte	0
	.byte	0
	.byte	0x5e, 0xc4, 0x4d, 0x3b
	.byte	0x78, 0xd0, 0x2a, 0x57
	.byte	0xd2, 0x75, 0xc1, 0x22
	.byte	0x36, 0xb7, 0x17, 0xbf
.Lprologue_end0:
.Ldebug_line_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	1                               # Header: foreign type unit count
	.long	3                               # Header: bucket count
	.long	3                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.quad	-4149699470930386446            # Type unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	3                               # Bucket 2
	.long	5863324                         # Hash in Bucket 1
	.long	5863786                         # Hash in Bucket 1
	.long	1398125246                      # Hash in Bucket 2
	.long	.Lskel_string1                  # String in Bucket 1: f1
	.long	.Lskel_string3                  # String in Bucket 1: t1
	.long	.Lskel_string2                  # String in Bucket 2: _Z2f12t1
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
	.byte	2                               # DW_IDX_type_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
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
.L0:
	.byte	1                               # Abbreviation code
	.long	26                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: f1
.Lnames2:
.L1:
	.byte	2                               # Abbreviation code
	.byte	0                               # DW_IDX_type_unit
	.long	33                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	3                               # Abbreviation code
	.long	49                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames1:
	.byte	1                               # Abbreviation code
	.long	26                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z2f12t1
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git d12762cdb6357915e0f6f6dbfc09c2c75d746ee7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	
#--- b.s

	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
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
	.byte	8                               # Address Size (in bytes)
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

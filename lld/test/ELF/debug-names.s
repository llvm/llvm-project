# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/debug-names-2.s -o %t2.o
# RUN: ld.lld --debug-names %t1.o %t2.o -o %t

# RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=DISASM
# RUN: llvm-dwarfdump -debug-names %t | FileCheck %s --check-prefix=DWARF
# RUN: llvm-readelf -SW %t | FileCheck %s --check-prefix=READELF
	
# DISASM:       Disassembly of section .text:
# DISASM-EMPTY:
# DISASM:       <_Z2f12t1>:
# DISASM-CHECK:   201180: 55       pushq %rbp
# DISASM-CHECK:   201181: 48 89 e5 movq  %rsp, %rbp
# DISASM-CHECK:   201184: 5d       popq  %rbp
# DISASM-CHECK:   201185: c3       retq
# DISASM-CHECK:   201186: cc       int3
# DISASM-CHECK:   201187: cc       int3
# DISASM-CHECK:   201188: cc       int3
# DISASM-CHECK:   201189: cc       int3
# DISASM-CHECK:   20118a: cc       int3
# DISASM-CHECK:   20118b: cc       int3
# DISASM-CHECK:   20118c: cc       int3
# DISASM-CHECK:   20118d: cc       int3
# DISASM-CHECK:   20118e: cc       int3
# DISASM-CHECK:   20118f: cc       int3
# DISASM:       <main>:
# DISASM-CHECK:   201190: 55       pushq %rbp
# DISASM-CHECK:   201191: 48 89 e5 movq  %rsp, %rbp
# DISASM-CHECK:   201194: 31 c0    xorl  %eax, %eax
# DISASM-CHECK:   201196: 5d       popq  %rbp
# DISASM-CHECK:   201197: c3       retq

# DWARF:      .debug_names contents:
# DWARF:      Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length: 0xCC
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
# DWARF-NEXT:     CU[1]: 0x00000041
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
# DWARF:        Bucket 1 [
# DWARF:            String: 0x00000089 "f1"
# DWARF-NEXT:       Entry @ 0xa3 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF:            String: 0x00000095 "t1"
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
# DWARF:            String: 0x00000130 "int"
# DWARF-NEXT:       Entry @ 0xb7 {
# DWARF-NEXT:         Abbrev: 0x3
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x0000003e
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF:        Bucket 2 [
# DWARF:        Bucket 3 [
# DWARF:            String: 0x0000008c "_Z2f12t1"
# DWARF-NEXT:       Entry @ 0xbe {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF:        Bucket 4 [
# DWARF:            String: 0x0000012b "main"
# DWARF-NEXT:       Entry @ 0xc5 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01

# READELF: .debug_names PROGBITS 0000000000000000 0003eb 0000d0

# RUN: ld.lld --debug-names --no-debug-names %t1.o %t2.o -o %t
# RUN: llvm-readelf -SW %t | FileCheck %s --check-prefix=NO_DBG_NAMES
	

# NO_DBG_NAMES: .debug_names  PROGBITS  0000000000000000 00037c 000110
	
#-- input file: debug-names.cpp
## Generated with:
##
## - clang++ -g -O0 -gpubnames -fdebug-compilation-dir='debug-names-test' \
##     -S debug-names.cpp -o debug-names.s
##
## debug-names.cpp contents:
##
## struct t1 { };
## void f1(t1) { }
##
##
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
	.byte	1                               # Abbrev [1] 0xc:0x35 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x17 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	3                               # DW_AT_linkage_name
	.byte	4                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x2f:0xa DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	127
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	58                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x3a:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	5                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 4df364bc93af49ae413ec1ae8328f34ac70730c4)" # string offset=0
.Linfo_string1:
	.asciz	"debug-names.cpp"               # string offset=104
.Linfo_string2:
	.asciz	"debug-names-test"              # string offset=120
.Linfo_string3:
	.asciz	"f1"                            # string offset=137
.Linfo_string4:
	.asciz	"_Z2f12t1"                      # string offset=140
.Linfo_string5:
	.asciz	"t1"                            # string offset=149
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
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 4df364bc93af49ae413ec1ae8328f34ac70730c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:


# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-4 %t1.o -o %t.exe
# RUN: llvm-objcopy --remove-section=.debug_loc %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections &> file
# RUN: cat file | FileCheck --check-prefix=CHECK %s

# Making sure we handle error returned by visitLocationList correctly.

# CHECK: BOLT-WARNING: empty location list detected at
# CHECK-NEXT: BOLT-WARNING: empty location list detected at
# CHECK-NEXT: BOLT:

# clang++ main.cpp -g -ffunction-sections -Os -gdwarf-4 -S
# int doStuff2(int val) {
#   return val += 3;
# }
#
# int main(int argc, const char** argv) {
#   return doStuff2(argc);
# }

	.text
	.file	"main.cpp"
	.section	.text._Z8doStuff2i,"ax",@progbits
	.globl	_Z8doStuff2i                    # -- Begin function _Z8doStuff2i
	.type	_Z8doStuff2i,@function
_Z8doStuff2i:                           # @_Z8doStuff2i
.Lfunc_begin0:
	.file	1 "." "main.cpp"
	.loc	1 2 0                           # main.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: doStuff2:val <- $edi
                                        # kill: def $edi killed $edi def $rdi
	.loc	1 3 14 prologue_end             # main.cpp:3:14
	leal	3(%rdi), %eax
.Ltmp0:
	#DEBUG_VALUE: doStuff2:val <- $eax
	.loc	1 3 3 is_stmt 0                 # main.cpp:3:3
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z8doStuff2i, .Lfunc_end0-_Z8doStuff2i
	.cfi_endproc
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.loc	1 6 0 is_stmt 1                 # main.cpp:6:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: doStuff2:val <- $edi
                                        # kill: def $edi killed $edi def $rdi
	.loc	1 3 14 prologue_end             # main.cpp:3:14
	leal	3(%rdi), %eax
.Ltmp2:
	#DEBUG_VALUE: doStuff2:val <- $eax
	.loc	1 7 3                           # main.cpp:7:3
	retq
.Ltmp3:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	-1
	.quad	.Lfunc_begin0                   #   base address
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	-1
	.quad	.Lfunc_begin1                   #   base address
	.quad	.Lfunc_begin1-.Lfunc_begin1
	.quad	.Ltmp2-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp2-.Lfunc_begin1
	.quad	.Lfunc_end1-.Lfunc_begin1
	.short	1                               # Loc expr size
	.byte	80                              # super-register DW_OP_reg0
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
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
	.byte	8                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	38                              # DW_TAG_const_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xc8 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x1d DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	71                              # DW_AT_abstract_origin
	.byte	3                               # Abbrev [3] 0x3d:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	87                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x47:0x1c DW_TAG_subprogram
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	99                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_inline
	.byte	5                               # Abbrev [5] 0x57:0xb DW_TAG_formal_parameter
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	99                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x63:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x6a:0x52 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	99                              # DW_AT_type
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0x83:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	99                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x90:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	188                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x9d:0x1e DW_TAG_inlined_subroutine
	.long	71                              # DW_AT_abstract_origin
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Ltmp2-.Lfunc_begin1            # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	7                               # DW_AT_call_line
	.byte	10                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0xb1:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	87                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xbc:0x5 DW_TAG_pointer_type
	.long	193                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0xc1:0x5 DW_TAG_pointer_type
	.long	198                             # DW_AT_type
	.byte	11                              # Abbrev [11] 0xc6:0x5 DW_TAG_const_type
	.long	203                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xcb:0x7 DW_TAG_base_type
	.long	.Linfo_string10                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 15.0.0" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=134
.Linfo_string2:
	.asciz	"." # string offset=143
.Linfo_string3:
	.asciz	"_Z8doStuff2i"                  # string offset=181
.Linfo_string4:
	.asciz	"doStuff2"                      # string offset=194
.Linfo_string5:
	.asciz	"int"                           # string offset=203
.Linfo_string6:
	.asciz	"val"                           # string offset=207
.Linfo_string7:
	.asciz	"main"                          # string offset=211
.Linfo_string8:
	.asciz	"argc"                          # string offset=216
.Linfo_string9:
	.asciz	"argv"                          # string offset=221
.Linfo_string10:
	.asciz	"char"                          # string offset=226
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

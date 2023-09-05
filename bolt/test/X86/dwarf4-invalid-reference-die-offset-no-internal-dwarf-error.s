# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-4 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections -v 1 &> %tlog.txt
# RUN: cat %tlog.txt | FileCheck --check-prefix=CHECKBOLT %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=CHECK %s

# Tests BOLT does not assert when DIE reference is invalid.

# CHECKBOLT: Referenced DIE offsets not in .debug_info
# CHECKBOLT-NEXT: 91
# CHECK: DW_TAG_variable
# CHECK-NEXT: DW_AT_name
# CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x0091 => {0x00000091})

# Assembly manually modified
# struct pair {int i; int j; };
# static pair p;
# int load() {
#     return p.i + p.j;
# }
# void store(int i, int j) {
#     p.i = i;
#     p.j = j;
# }
# int main() {
# return 0;
# }

	.text
	.file	"main.cpp"
	.file	1 "/invalidWithin" "main.cpp"
	.section	.text._Z4loadv,"ax",@progbits
	.globl	_Z4loadv                        # -- Begin function _Z4loadv
	.p2align	4, 0x90
	.type	_Z4loadv,@function
_Z4loadv:                               # @_Z4loadv
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	1 4 20 prologue_end             # main.cpp:4:20
	movl	_ZL1p.1(%rip), %eax
	.loc	1 4 16 is_stmt 0                # main.cpp:4:16
	addl	_ZL1p.0(%rip), %eax
	.loc	1 4 5                           # main.cpp:4:5
	retq
.Ltmp0:
.Lfunc_end0:
	.size	_Z4loadv, .Lfunc_end0-_Z4loadv
	.cfi_endproc
                                        # -- End function
	.section	.text._Z5storeii,"ax",@progbits
	.globl	_Z5storeii                      # -- Begin function _Z5storeii
	.p2align	4, 0x90
	.type	_Z5storeii,@function
_Z5storeii:                             # @_Z5storeii
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: store:i <- $edi
	#DEBUG_VALUE: store:j <- $esi
	.loc	1 7 9 prologue_end is_stmt 1    # main.cpp:7:9
	movl	%edi, _ZL1p.0(%rip)
	.loc	1 8 9                           # main.cpp:8:9
	movl	%esi, _ZL1p.1(%rip)
	.loc	1 9 1                           # main.cpp:9:1
	retq
.Ltmp1:
.Lfunc_end1:
	.size	_Z5storeii, .Lfunc_end1-_Z5storeii
	.cfi_endproc
                                        # -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin2:
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	1 11 1 prologue_end             # main.cpp:11:1
	xorl	%eax, %eax
	retq
.Ltmp2:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.type	_ZL1p.0,@object                 # @_ZL1p.0
	.local	_ZL1p.0
	.comm	_ZL1p.0,4,4
	.type	_ZL1p.1,@object                 # @_ZL1p.1
	.local	_ZL1p.1
	.comm	_ZL1p.1,4,4
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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	1                               # DW_CHILDREN_yes
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	13                              # DW_TAG_member
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	56                              # DW_AT_data_member_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	6                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
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
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
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
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0xd9 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x26 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	145                              # DW_AT_type --> Modified manually s/80/145
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	22                              # DW_AT_location
	.byte	3
	.quad	_ZL1p.0
	.byte	147
	.byte	4
	.byte	3
	.quad	_ZL1p.1
	.byte	147
	.byte	4
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.byte	3                               # Abbrev [3] 0x50:0x22 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.long	.Linfo_string7                  # DW_AT_name
	.byte	8                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	4                               # Abbrev [4] 0x59:0xc DW_TAG_member
	.long	.Linfo_string4                  # DW_AT_name
	.long	114                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # DW_AT_data_member_location
	.byte	4                               # Abbrev [4] 0x65:0xc DW_TAG_member
	.long	.Linfo_string6                  # DW_AT_name
	.long	114                             # DW_AT_type
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	4                               # DW_AT_data_member_location
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x72:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0x79:0x1d DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string9                  # DW_AT_linkage_name
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
                                        # DW_AT_external
	.byte	7                               # Abbrev [7] 0x96:0x34 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string11                 # DW_AT_linkage_name
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	8                               # Abbrev [8] 0xaf:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0xbc:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	114                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0xca:0x19 DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	114                             # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 37d6c1cc7d4dd3a8a47ba62254bc88521bd50d66)" # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=101
.Linfo_string2:
	.asciz	"/invalidWithin" # string offset=110
.Linfo_string3:
	.asciz	"p"                             # string offset=162
.Linfo_string4:
	.asciz	"i"                             # string offset=164
.Linfo_string5:
	.asciz	"int"                           # string offset=166
.Linfo_string6:
	.asciz	"j"                             # string offset=170
.Linfo_string7:
	.asciz	"pair"                          # string offset=172
.Linfo_string8:
	.asciz	"_ZL1p"                         # string offset=177
.Linfo_string9:
	.asciz	"_Z4loadv"                      # string offset=183
.Linfo_string10:
	.asciz	"load"                          # string offset=192
.Linfo_string11:
	.asciz	"_Z5storeii"                    # string offset=197
.Linfo_string12:
	.asciz	"store"                         # string offset=208
.Linfo_string13:
	.asciz	"main"                          # string offset=214
	.ident	"clang version 18.0.0 (git@github.com:llvm/llvm-project.git 37d6c1cc7d4dd3a8a47ba62254bc88521bd50d66)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

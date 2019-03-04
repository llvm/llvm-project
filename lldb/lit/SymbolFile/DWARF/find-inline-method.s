# REQUIRES: lld
# XFAIL: system-netbsd

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj > %t.o
# RUN: ld.lld %t.o -o %t
# RUN: lldb-test symbols --find=function --name=inl --function-flags=method %t \
# RUN:   | FileCheck %s

# CHECK: Function: {{.*}} mangled = "_Z8externali"
# CHECK: Blocks: {{.*}} range = [0x00201000-0x00201002)
# CHECK-NEXT: range = [0x00201000-0x00201002), name = "inl", mangled = _ZN1S3inlEi


# Generated via:
#   clang -O2 -g -S

# from file:
#   int forward(int);
#   struct S {
#     static int inl(int a) { return forward(a); }
#   };
#   int external(int a) { return S::inl(a); }

# and then simplified.

	.text
_Z8externali:
.Lfunc_begin0:
	jmp	_Z7forwardi
.Lfunc_end0:

.globl _start
_start:
_Z7forwardi:
        ret

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 7.0.0 (trunk 332830) (llvm/trunk 332835) with manual modifications"
.Linfo_string3:
	.asciz	"_ZN1S3inlEi"
.Linfo_string4:
	.asciz	"inl"
.Linfo_string6:
	.asciz	"S"
.Linfo_string8:
	.asciz	"_Z8externali"
.Linfo_string9:
	.asciz	"external"
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	19                      # DW_TAG_structure_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	71                      # DW_AT_specification
	.byte	19                      # DW_FORM_ref4
	.byte	32                      # DW_AT_inline
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	1                       # DW_CHILDREN_yes
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	4                       # DW_AT_language
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x1f DW_TAG_structure_type
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_byte_size
.Linl_spec:
	.byte	3                       # Abbrev [3] 0x33:0x15 DW_TAG_subprogram
	.long	.Linfo_string3          # DW_AT_linkage_name
	.long	.Linfo_string4          # DW_AT_name
	.byte	0                       # End Of Children Mark
.Linl_abstract:
	.byte	6                       # Abbrev [6] 0x50:0x12 DW_TAG_subprogram
	.long	.Linl_spec              # DW_AT_specification
	.byte	1                       # DW_AT_inline
.Linl_a_abstract:
	.byte	8                       # Abbrev [8] 0x62:0x46 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.long	.Linfo_string8          # DW_AT_linkage_name
	.long	.Linfo_string9          # DW_AT_name
	.byte	10                      # Abbrev [10] 0x8c:0x1b DW_TAG_inlined_subroutine
	.long	.Linl_abstract          # DW_AT_abstract_origin
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
.Lcu_end0:

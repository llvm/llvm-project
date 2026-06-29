# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -dwarf-5 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections 2>&1 | \
# RUN:   FileCheck %s --check-prefix CHECK-BOLT
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt | FileCheck %s

## Verify BOLT preserves DW_FORM_ref_udata (CU-relative ULEB128 DIE reference),
## a form GCC may emit instead of DW_FORM_ref4.

# CHECK:      DW_TAG_subprogram
# CHECK:      DW_AT_type [DW_FORM_ref_udata]
# CHECK-SAME: "int"

# CHECK-BOLT-NOT: BOLT-WARNING

	.text
	.file	0 "." "main.cpp"
	.globl	main
main:
.Lfunc_begin0:
	.loc	0 1 0
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

## Force relocations against .text
.reloc 0, R_X86_64_NONE

	.section	.debug_abbrev,"",@progbits
	.byte	1, 17, 1                # CU, has children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	16, 23                  # DW_AT_stmt_list, DW_FORM_sec_offset
	.byte	0, 0
	.byte	2, 46, 0                # subprogram, no children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	73, 21                  # DW_AT_type, DW_FORM_ref_udata
	.byte	0, 0
	.byte	3, 36, 0                # base_type, no children
	.byte	3, 8                    # DW_AT_name, DW_FORM_string
	.byte	0, 0
	.byte	0

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	5                       # DWARF version
	.byte	1                       # DW_UT_compile
	.byte	8                       # Address size
	.long	.debug_abbrev           # Abbrev offset
	.byte	1                       # CU
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	2                       # subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc
	.uleb128 .Ltype_int-.Lcu_begin0    # DW_AT_type (DW_FORM_ref_udata)
.Ltype_int:
	.byte	3                       # base_type
	.asciz	"int"                   # DW_AT_name
	.byte	0                       # End children of CU
.Ldebug_info_end0:
	.section	.debug_line,"",@progbits
.Lline_table_start0:

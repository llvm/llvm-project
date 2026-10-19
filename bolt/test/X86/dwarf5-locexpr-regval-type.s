# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -dwarf-5 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections 2>&1 | \
# RUN:   FileCheck %s --check-prefix CHECK-BOLT
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt | FileCheck %s

## Verify BOLT correctly handles DW_OP_regval_type. Its operands are
## (ULEB128 register, ULEB128 base type DIE offset). The base type
## reference must be updated when DIEs are relocated. Use a register
## number that requires multi-byte ULEB128 encoding to exercise the
## first-operand byte-copy path.

# CHECK:      DW_TAG_variable
# CHECK:      DW_AT_location [DW_FORM_exprloc]
# CHECK-SAME: DW_OP_regval_type 0xc8 (0x[[#%.8x,TYPE:]] ->
# CHECK:      0x[[#TYPE]]: DW_TAG_base_type

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
	.byte	2, 46, 1                # subprogram, has children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	0, 0
	.byte	3, 52, 0                # variable, no children
	.byte	2, 24                   # DW_AT_location, DW_FORM_exprloc
	.byte	0, 0
	.byte	4, 36, 0                # base_type, no children
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
	.byte	3                       # variable
	.byte	.Lloc_end-.Lloc_start   # exprloc length
.Lloc_start:
	.byte	0xa5                    # DW_OP_regval_type
	.uleb128 200                    # register 200 (multi-byte ULEB128)
	.uleb128 .Ltype_int-.Lcu_begin0 # base type DIE offset
.Lloc_end:
	.byte	0                       # End children of subprogram
.Ltype_int:
	.byte	4                       # base_type
	.asciz	"int"                   # DW_AT_name
	.byte	0                       # End children of CU
.Ldebug_info_end0:
	.section	.debug_line,"",@progbits
.Lline_table_start0:

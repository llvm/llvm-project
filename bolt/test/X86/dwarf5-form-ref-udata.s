# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -dwarf-5 %t.o -o %t.exe -Wl,-q
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | \
# RUN:   FileCheck %s --check-prefix CHECK-INPUT
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections 2>&1 | \
# RUN:   FileCheck %s --check-prefix CHECK-BOLT
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not=warning:
# RUN: llvm-dwarfdump --verify %t.bolt | FileCheck %s --check-prefix CHECK-VERIFY

## Verify BOLT handles DW_FORM_ref_udata (CU-relative ULEB128 DIE reference),
## a form GNU as emits instead of DW_FORM_ref4. BOLT rewrites these
## references as DW_FORM_ref4: the size of a ULEB128 reference depends on the
## output offset of the referenced DIE, which is not known yet when a DIE with
## a forward reference is laid out.
##
## Until a DIE is laid out, BOLT keeps its input offset, which is relative to
## the start of .debug_info. The DIE referenced by the subprogram in the second
## CU is at least 0x80 bytes into the section, so that offset needs a two-byte
## ULEB128, while its CU-relative output offset needs only one byte. Sizing the
## forward reference with the former and emitting the latter used to leave the
## unit length and all subsequent DIE offsets of the unit off by one.

## Check the input satisfies these conditions.
# CHECK-INPUT: DW_AT_type [DW_FORM_ref_udata] (cu + 0x{{[0-7]?[0-9a-f]}} => {0x000000{{[89a-f][0-9a-f]}}} "long")

# CHECK:      DW_TAG_compile_unit
# CHECK:      DW_TAG_variable
# CHECK:      DW_AT_type [DW_FORM_ref4]
# CHECK-SAME: "int"
# CHECK:      DW_TAG_compile_unit
# CHECK:      DW_TAG_subprogram
# CHECK:      DW_AT_type [DW_FORM_ref4]
# CHECK-SAME: "long"

# CHECK-VERIFY: No errors.

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
	.byte	37, 8                   # DW_AT_producer, DW_FORM_string
	.byte	3, 8                    # DW_AT_name, DW_FORM_string
	.byte	0, 0
	.byte	2, 52, 0                # variable, no children
	.byte	3, 8                    # DW_AT_name, DW_FORM_string
	.byte	73, 21                  # DW_AT_type, DW_FORM_ref_udata
	.byte	0, 0
	.byte	3, 36, 0                # base_type, no children
	.byte	3, 8                    # DW_AT_name, DW_FORM_string
	.byte	0, 0
	.byte	4, 17, 1                # CU, has children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	16, 23                  # DW_AT_stmt_list, DW_FORM_sec_offset
	.byte	0, 0
	.byte	5, 46, 0                # subprogram, no children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	73, 21                  # DW_AT_type, DW_FORM_ref_udata
	.byte	0, 0
	.byte	0

	.section	.debug_info,"",@progbits
## A CU without code, which moves the second CU further into .debug_info.
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	5                       # DWARF version
	.byte	1                       # DW_UT_compile
	.byte	8                       # Address size
	.long	.debug_abbrev           # Abbrev offset
	.byte	1                       # CU
	.asciz	"GNU C17 13.2.0 -mtune=generic -march=x86-64 -g -O2" # DW_AT_producer
	.asciz	"/src/lib/global_data.c" # DW_AT_name
	.byte	2                       # variable
	.asciz	"data"                  # DW_AT_name
	.uleb128 .Ltype_int-.Lcu_begin0    # DW_AT_type (DW_FORM_ref_udata)
.Ltype_int:
	.byte	3                       # base_type
	.asciz	"int"                   # DW_AT_name
	.byte	0                       # End children of CU
.Ldebug_info_end0:

.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1
.Ldebug_info_start1:
	.short	5                       # DWARF version
	.byte	1                       # DW_UT_compile
	.byte	8                       # Address size
	.long	.debug_abbrev           # Abbrev offset
	.byte	4                       # CU
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	5                       # subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0  # DW_AT_high_pc
	.uleb128 .Ltype_long-.Lcu_begin1   # DW_AT_type (DW_FORM_ref_udata)
.Ltype_long:
	.byte	3                       # base_type
	.asciz	"long"                  # DW_AT_name
	.byte	0                       # End children of CU
.Ldebug_info_end1:
	.section	.debug_line,"",@progbits
.Lline_table_start0:

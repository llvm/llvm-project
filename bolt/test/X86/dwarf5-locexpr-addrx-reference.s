# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -dwarf-5 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --verify %t.bolt
# RUN: llvm-dwarfdump --debug-info %t.bolt | FileCheck %s

## Verify BOLT keeps the new DW_OP_addrx index in a location expression that
## also references a base type DIE. The index of var goes from 128, which takes
## two ULEB128 bytes, to 1, which takes one byte.

# CHECK: DW_AT_location (DW_OP_addrx 0x1, DW_OP_convert (0x{{[0-9a-f]+}}) "int")

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

	.data
var:
	.long	0

	.section	.debug_abbrev,"",@progbits
	.byte	1, 17, 1                # CU, has children
	.byte	17, 1                   # DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6                   # DW_AT_high_pc, DW_FORM_data4
	.byte	16, 23                  # DW_AT_stmt_list, DW_FORM_sec_offset
	.byte	0x73, 23                # DW_AT_addr_base, DW_FORM_sec_offset
	.byte	0, 0
	.byte	2, 52, 0                # variable, no children
	.byte	2, 24                   # DW_AT_location, DW_FORM_exprloc
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
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.byte	2                       # variable
	.byte	.Lloc_end-.Lloc_start   # exprloc length
.Lloc_start:
	.byte	0xa1                    # DW_OP_addrx
	.uleb128 128                    # var
	.byte	0xa8                    # DW_OP_convert
	.uleb128 .Ltype_int-.Lcu_begin0 # base type DIE offset
.Lloc_end:
.Ltype_int:
	.byte	3                       # base_type
	.asciz	"int"                   # DW_AT_name
	.byte	0                       # End children of CU
.Ldebug_info_end0:

	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0
.Ldebug_addr_start0:
	.short	5                       # DWARF version
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
.Laddr_table_base0:
	.fill	128, 8, 0               # Indices 0-127
	.quad	var                     # Index 128
.Ldebug_addr_end0:

	.section	.debug_line,"",@progbits
.Lline_table_start0:

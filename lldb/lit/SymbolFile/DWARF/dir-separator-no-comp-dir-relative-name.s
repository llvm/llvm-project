# Test that parsing of line tables works reasonably. In this case the debug info
# does not have enough information for our heuristics to determine the path
# style, so we will just treat them as native host paths.

# REQUIRES: lld
# XFAIL: system-netbsd

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj > %t.o
# RUN: ld.lld %t.o -o %t
# RUN: %lldb %t -s %S/Inputs/dir-separator-no-comp-dir-relative-name.lldbinit -o exit | FileCheck %s

# CHECK-LABEL: image dump line-table a.c
# CHECK: Line table for foo{{.}}a.c
# CHECK-NEXT: 0x0000000000201000: foo{{.}}a.c:1
# CHECK-NEXT: 0x0000000000201001: foo{{.}}b.c:1
# CHECK-NEXT: 0x0000000000201002: foo{{.}}b.c:1
# CHECK-EMPTY:

# CHECK-LABEL: breakpoint set -f a.c -l 1
# CHECK: Breakpoint 1: {{.*}}`_start,

# CHECK-LABEL: breakpoint set -f foo/b.c -l 1
# CHECK: Breakpoint 2: {{.*}}`_start + 1,

	.text
	.globl	_start
_start:
	.file	1 "foo/a.c"
	.loc	1 1 0
        nop
	.file	2 "foo/b.c"
	.loc	2 1 0
        nop

	.section	.debug_str,"MS",@progbits,1
.Linfo_string1:
	.asciz	"foo/a.c"
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
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
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
.Lcu_end0:
	.section	.debug_line,"",@progbits
.Lline_table_start0:

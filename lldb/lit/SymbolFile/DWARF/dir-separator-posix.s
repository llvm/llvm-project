# Test that parsing of line tables works reasonably, even if the host directory
# separator does not match the separator of the compile unit.

# REQUIRES: lld, x86

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj > %t.o
# RUN: ld.lld %t.o -o %t
# RUN: %lldb %t -s %S/Inputs/dir-separator-posix.lldbinit -o exit | FileCheck %s

# CHECK-LABEL: image dump line-table a.c
# CHECK: Line table for /tmp/a.c
# CHECK-NEXT: 0x0000000000201000: /tmp/a.c:1
# CHECK-NEXT: 0x0000000000201001: /tmp/b.c:1
# CHECK-NEXT: 0x0000000000201002: /tmp/b.c:1
# CHECK-EMPTY:

# CHECK-LABEL: breakpoint set -f a.c -l 1
# CHECK: Breakpoint 1: {{.*}}`_start,

# CHECK-LABEL: breakpoint set -f /tmp/b.c -l 1
# CHECK: Breakpoint 2: {{.*}}`_start + 1,

	.text
	.globl	_start
_start:
	.file	1 "/tmp/a.c"
	.loc	1 1 0
        nop
	.file	2 "/tmp/b.c"
	.loc	2 1 0
        nop

	.section	.debug_str,"MS",@progbits,1
.Linfo_string1:
	.asciz	"a.c"
.Linfo_string2:
	.asciz	"/tmp"
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
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
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
	.long	.Linfo_string2          # DW_AT_comp_dir
.Lcu_end0:
	.section	.debug_line,"",@progbits
.Lline_table_start0:

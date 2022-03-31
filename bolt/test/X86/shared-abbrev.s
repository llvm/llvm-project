# RUN: rm -rf %t
# RUN: mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 -dwarf-version=4 %s -o %t/shared-abbrev.o
# RUN: %clang %cflags %t/shared-abbrev.o -o %t/shared-abbrev.exe
# RUN: llvm-bolt %t/shared-abbrev.exe -o %t/shared-abbrev.exe.bolt -update-debug-sections
# RUN: llvm-dwarfdump --debug-info %t/shared-abbrev.exe.bolt | FileCheck %s

# CHECK: 0x00000000:
# CHECK-SAME: abbr_offset = 0x0000
# CHECK-EMPTY:
# CHECK-NEXT: 		DW_TAG_compile_unit
# CHECK-NEXT: 		DW_AT_stmt_list
# CHECK-NEXT: 		DW_AT_low_pc
# CHECK-NEXT: 		DW_AT_ranges
# CHECK: 0x0000001d:
# CHECK-SAME: abbr_offset = 0x0017
# CHECK-EMPTY:
# CHECK:		DW_TAG_compile_unit
# CHECK-NEXT: 		DW_AT_stmt_list
# CHECK-NEXT: 		DW_AT_low_pc
# CHECK-NEXT: 		DW_AT_ranges
# CHECK: 0x0000003a:
# CHECK-SAME: abbr_offset = 0x0000
# CHECK-EMPTY:
# CHECK-NEXT: 		DW_TAG_compile_unit
# CHECK-NEXT: 		DW_AT_stmt_list
# CHECK-NEXT: 		DW_AT_low_pc
# CHECK-NEXT: 		DW_AT_ranges

	.text
	.file	"main.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "test" "main.cpp"
	.loc	1 1 0                           # main.cpp:1:0
	.cfi_startproc
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
  .section	.debug_abbrev,"",@progbits
  .byte	1                               # Abbreviation Code
  .byte	17                              # DW_TAG_compile_unit
  .byte	0                               # DW_CHILDREN_no
  .byte	16                              # DW_AT_stmt_list
  .byte	23                              # DW_FORM_sec_offset
  .byte	17                              # DW_AT_low_pc
  .byte	1                               # DW_FORM_addr
  .byte	18                              # DW_AT_high_pc
  .byte	7                               # DW_FORM_data8
  .byte	0                               # EOM(1)
  .byte	0                               # EOM(2)
  .byte 2                               # Abbreviation Code
  .byte 17                              # DW_TAG_compile_unit
  .byte 0                               # DW_CHILDREN_no
  .byte 16                              # DW_AT_stmt_list
  .byte 23                              # DW_FORM_sec_offset
  .byte 17                              # DW_AT_low_pc
  .byte 1                               # DW_FORM_addr
  .byte 85                              # DW_AT_ranges
  .byte 23                              # DW_FORM_sec_offset
  .byte 0                               # EOM(1)
  .byte 0                               # EOM(2)
  .byte	0                               # EOM(3)
  .section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short  4                               # DWARF version number
  .long .debug_abbrev                   # Offset Into Abbrev. Section
  .byte 8                               # Address Size (in bytes)
  .byte 2                               # Abbrev [2] DW_TAG_compile_unit
  .long .Lline_table_start0             # DW_AT_stmt_list
  .quad 0                               # DW_AT_low_pc
  .byte 0                               # End Of Children Mark
  .long .Ldebug_ranges0                 # DW_AT_ranges --- end manual --
.Ldebug_info_end0:

   # Second CU table.
   .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
  .short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.quad	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
  .byte 0                               # End Of Children Mark
.Ldebug_info_end1:

   # Third CU table.
   .long   .Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short  4                               # DWARF version number
  .long .debug_abbrev                   # Offset Into Abbrev. Section
  .byte 8                               # Address Size (in bytes)
  .byte 2                               # Abbrev [2] DW_TAG_compile_unit
  .long .Lline_table_start0             # DW_AT_stmt_list
  .quad 0                               # DW_AT_low_pc
  .byte 0                               # End Of Children Mark
  .long .Ldebug_ranges0                 # DW_AT_ranges --- end manual --
.Ldebug_info_end2:
  .section  .debug_ranges,"",@progbits
.Ldebug_ranges0:
  .quad .Lfunc_begin0
  .quad .Lfunc_end0
  .quad .Lfunc_begin0
  .quad .Lfunc_end0
  .quad 0
  .quad 0
  .section	".note.GNU-stack","",@progbits
  .addrsig
  .section	.debug_line,"",@progbits
.Lline_table_start0:

# REQUIRES: x86
## Regression test that we don't crash on DWARF v5 .debug_loclists

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gdb-index %t.o -o /dev/null

.section .debug_abbrev,"",@progbits
  .byte 1            # Abbreviation Code
  .byte 17           # DW_TAG_compile_unit
  .byte 0            # DW_CHILDREN_no
  .ascii "\214\001"  # DW_AT_loclists_base
  .byte 23           # DW_FORM_sec_offset
  .byte 0            # EOM(1)
  .byte 0            # EOM(2)
  .byte 0

.section .debug_info,"",@progbits
.Lcu_begin0:
  .long .Lcu_end0-.Lcu_begin0-4  # Length of Unit
  .short 5                       # DWARF version number
  .byte  1                       # DWARF Unit Type
  .byte  8                       # Address Size
  .long  0                       # Offset Into Abbrev. Section
  .byte  1                       # Abbrev [1] DW_TAG_compile_unit
  .long  .Lloclists_table_base0  # DW_AT_loclists_base
.Lcu_end0:

.section .debug_loclists,"",@progbits
  .long .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
  .short 5                # Version
  .byte  8                # Address size
  .byte  0                # Segment selector size
  .long  0                # Offset entry count
.Lloclists_table_base0:
  .byte  0                # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:

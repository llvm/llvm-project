# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
# RUN: ld.lld --gdb-index %t1.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: warning: {{.*}}:{{(\(\.debug_info\):)?}} invalid reference to or invalid content in .debug_str_offsets[.dwo]: insufficient space for 32 bit header prefix

.section .debug_abbrev,"",@progbits
  .byte  1                           # Abbreviation Code
  .byte  17                          # DW_TAG_compile_unit
  .byte  0                           # DW_CHILDREN_no
  .byte  114                         # DW_AT_str_offsets_base
  .byte  23                          # DW_FORM_sec_offset
  .byte  0                           # EOM(1)
  .byte  0                           # EOM(2)
  .byte  0                           # EOM(3)

.section .debug_info,"",@progbits
  .long  .Lunit_end0-.Lunit_begin0   # Length of Unit
.Lunit_begin0:
  .short 5                           # DWARF version number
  .byte  1                           # DWARF Unit Type
  .byte  8                           # Address Size (in bytes)
  .long  .debug_abbrev               # Offset Into Abbrev. Section
  
  .byte  1                           # Abbrev [1] 0xc:0x43 DW_TAG_compile_unit
  .long  0                           # DW_AT_str_offsets_base
.Lunit_end0:


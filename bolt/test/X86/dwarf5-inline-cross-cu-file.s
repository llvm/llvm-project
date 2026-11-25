## Check that when BOLT inlines code from another CU, the file that it adds to
## the line table of the caller's CU keeps its directory and MD5 checksum.
## helper() is in inc/helper.h, which is file 1 with directory 1 in the DWARF 5
## line table of the helper.c CU. BOLT inlines it into main() in the main.c CU.

# REQUIRES: system-linux

# RUN: split-file %s %t
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux \
# RUN:   %t/main.s -o %t/main.o
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux \
# RUN:   %t/helper.s -o %t/helper.o
# RUN: %clang %cflags %t/main.o %t/helper.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections --force-inline=helper
# RUN: llvm-dwarfdump --debug-line %t.bolt | FileCheck %s

## The line table of the main.c CU, with helper.h added for the inlined code.
# CHECK:      debug_line[0x00000000]
# CHECK:      include_directories[  0] = "."
# CHECK-NEXT: include_directories[  1] = "./inc"
# CHECK-NEXT: file_names[  0]:
# CHECK-NEXT:            name: "main.c"
# CHECK-NEXT:       dir_index: 0
# CHECK-NEXT:    md5_checksum: c8a54c720f7bf83bc9481131f0080087
# CHECK-NEXT: file_names[  1]:
# CHECK-NEXT:            name: "helper.h"
# CHECK-NEXT:       dir_index: 1
# CHECK-NEXT:    md5_checksum: a1a1e1a2e4fdb00b6c88da2e9a733b99
# CHECK-NOT:  file_names[
## The inlined instruction comes from helper.h:1:30.
# CHECK:      0x{{[0-9a-f]+}} 1 30 1 0 0 0 is_stmt

## Each CU has only what BOLT needs: a compile unit DIE with DW_AT_stmt_list
## and the address range of the function in it.
#--- main.s
  .text
  .globl main
  .type main,@function
main:
  .file 0 "." "main.c" md5 0xc8a54c720f7bf83bc9481131f0080087
  .loc 0 2 42
  callq helper
  retq
.Lmain_end:
  .size main, .Lmain_end-main

  .section .debug_abbrev,"",@progbits
  .byte 1       # Abbreviation Code
  .byte 0x11    # DW_TAG_compile_unit
  .byte 0       # DW_CHILDREN_no
  .byte 0x10    # DW_AT_stmt_list
  .byte 0x17    # DW_FORM_sec_offset
  .byte 0x11    # DW_AT_low_pc
  .byte 0x01    # DW_FORM_addr
  .byte 0x12    # DW_AT_high_pc
  .byte 0x06    # DW_FORM_data4
  .byte 0, 0
  .byte 0

  .section .debug_info,"",@progbits
  .long .Lcu_end-.Lcu_start   # Length of Unit
.Lcu_start:
  .short 5                    # DWARF version number
  .byte 1                     # DW_UT_compile
  .byte 8                     # Address Size
  .long .debug_abbrev         # Offset Into Abbrev. Section
  .byte 1                     # DW_TAG_compile_unit
  .long .Lline_table_start    # DW_AT_stmt_list
  .quad main                  # DW_AT_low_pc
  .long .Lmain_end-main       # DW_AT_high_pc
.Lcu_end:

  .section .debug_line,"",@progbits
.Lline_table_start:

#--- helper.s
  .text
  .globl helper
  .type helper,@function
helper:
  .file 0 "." "helper.c" md5 0x80de5335325950b494ed130b633e7bcb
  .file 1 "./inc" "helper.h" md5 0xa1a1e1a2e4fdb00b6c88da2e9a733b99
  .loc 1 1 30
  leal 1(%rdi), %eax
  retq
.Lhelper_end:
  .size helper, .Lhelper_end-helper

  .section .debug_abbrev,"",@progbits
  .byte 1       # Abbreviation Code
  .byte 0x11    # DW_TAG_compile_unit
  .byte 0       # DW_CHILDREN_no
  .byte 0x10    # DW_AT_stmt_list
  .byte 0x17    # DW_FORM_sec_offset
  .byte 0x11    # DW_AT_low_pc
  .byte 0x01    # DW_FORM_addr
  .byte 0x12    # DW_AT_high_pc
  .byte 0x06    # DW_FORM_data4
  .byte 0, 0
  .byte 0

  .section .debug_info,"",@progbits
  .long .Lcu_end-.Lcu_start   # Length of Unit
.Lcu_start:
  .short 5                    # DWARF version number
  .byte 1                     # DW_UT_compile
  .byte 8                     # Address Size
  .long .debug_abbrev         # Offset Into Abbrev. Section
  .byte 1                     # DW_TAG_compile_unit
  .long .Lline_table_start    # DW_AT_stmt_list
  .quad helper                # DW_AT_low_pc
  .long .Lhelper_end-helper   # DW_AT_high_pc
.Lcu_end:

  .section .debug_line,"",@progbits
.Lline_table_start:

# This tests that lldb is able to process DW_OP_addrx tags introduced in dwarf5.

# REQUIRES: lld, x86

# RUN: llvm-mc -g -dwarf-version=5 -triple x86_64-unknown-linux-gnu %s -filetype=obj > %t.o
# RUN: ld.lld -m elf_x86_64 %t.o -o %t 
# RUN: lldb-test symbols %t | FileCheck %s

# CHECK: Variable{0x7fffffff00000011}, name = "color"
# CHECK-SAME: location = DW_OP_addrx 0x0

        .text
        .section        .debug_str,"MS",@progbits,1
.Lstr_offsets_base0:
        .asciz  "color"

        .section        .debug_str_offsets,"",@progbits
        .long   .Lstr_offsets_base0

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   114                     # DW_AT_str_offsets_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .byte   0                       # EOM(4)

        .section        .debug_info,"",@progbits
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
        .long   .Lstr_offsets_base0     # DW_AT_str_offsets_base
        .byte   2                       # Abbrev [2] 0x1e:0xb DW_TAG_variable
        .byte   0                       # DW_AT_name
                                        # DW_AT_external
        .byte   2                       # DW_AT_location
        .byte   161
        .byte   0
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .quad   color
.Ldebug_addr_end0:


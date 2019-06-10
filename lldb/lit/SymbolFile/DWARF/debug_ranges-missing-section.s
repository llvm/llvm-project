# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s lookup_ranges" -o exit 2>&1 | FileCheck %s

# CHECK: DIE has DW_AT_ranges(0x47) attribute, but range extraction failed (No debug_ranges section),
# CHECK:  Function: id = {0x7fffffff0000001c}, name = "ranges", range = [0x0000000000000000-0x0000000000000004)
# CHECK:    Blocks: id = {0x7fffffff0000001c}, range = [0x00000000-0x00000004)

        .text
        .p2align 12
        .globl  ranges
        .type   ranges,@function
ranges:                                    # @ranges
        nop
lookup_ranges:
        nop
        nop
        nop
.Lranges_end:
        .size   ranges, .Lranges_end-ranges
                                        # -- End function
        .section        .debug_str,"MS",@progbits,1
.Lproducer:
        .asciz  "Hand-written DWARF"
.Lranges:
        .asciz  "ranges"

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   11                      # DW_TAG_lexical_block
        .byte   0                       # DW_CHILDREN_no
        .byte   85                      # DW_AT_ranges
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x7b DW_TAG_compile_unit
        .long   .Lproducer              # DW_AT_producer
        .quad   ranges                  # DW_AT_low_pc
        .long   .Lranges_end-ranges     # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x4d DW_TAG_subprogram
        .quad   ranges                  # DW_AT_low_pc
        .long   .Lranges_end-ranges     # DW_AT_high_pc
        .long   .Lranges                # DW_AT_name
        .byte   5                       # Abbrev [5] 0x61:0x15 DW_TAG_lexical_block
        .long   0x47                    # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

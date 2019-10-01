# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s f1" -o exit | FileCheck %s

# CHECK:  Function: id = {0x7fffffff0000003c}, name = "f1", range = [0x0000000000000000-0x0000000000000001)
# CHECK:    Blocks: id = {0x7fffffff0000003c}, range = [0x00000000-0x00000001)


        .text
        .globl  f1
        .type   f1,@function
f1:
        nop
.Lfunc_end0:
        .size   f1, .Lfunc_end0-f1
                                        # -- End function
        .section        .debug_str,"MS",@progbits,1
.Lproducer:
        .asciz  "Hand-written DWARF"
.Lf1:
        .asciz  "f1"
.Le1:
        .asciz  "e1"

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
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Ltu_begin0:
        .long   .Ltu_end0-.Ltu_start0   # Length of Unit
.Ltu_start0:
        .short  5                       # DWARF version number
        .byte   2                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   47                      # Type Signature
        .long   .Ltype-.Ltu_begin0      # Type DIE Offset
        .byte   3                       # Abbrev [1] 0x18:0x1d DW_TAG_type_unit
.Ltype:
        .byte   5                       # Abbrev [5] 0x2e:0x9 DW_TAG_enumeration_type
        .long   .Le1                    # DW_AT_name
        .byte   0                       # End Of Children Mark
.Ltu_end0:

.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .long   .Lproducer              # DW_AT_producer
        .quad   f1                      # DW_AT_low_pc
        .long   .Lfunc_end0-f1          # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .quad   f1                      # DW_AT_low_pc
        .long   .Lfunc_end0-f1          # DW_AT_high_pc
        .long   .Lf1                    # DW_AT_name
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

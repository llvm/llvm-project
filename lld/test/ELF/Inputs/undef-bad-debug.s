.section .text,"ax"
sym:
    .quad zed6
    
.section .debug_info,"",@progbits
    .long   .Lcu_end - .Lcu_start   # Length of Unit
.Lcu_start:
    .short  4                       # DWARF version number
    .long   .Lsection_abbrev        # Offset Into Abbrev. Section
    .byte   8                       # Address Size (in bytes)
    .byte   1                       # Abbrev [1] 0xb:0x79 DW_TAG_compile_unit
    .byte   2                       # Abbrev [2] 0x2a:0x15 DW_TAG_variable
    .long   .Linfo_string           # DW_AT_name
                                        # DW_AT_external
    .byte   1                       # DW_AT_decl_file
    .byte   3                       # DW_AT_decl_line
    .byte   0                       # End Of Children Mark
.Lcu_end:

.section .debug_abbrev,"",@progbits
.Lsection_abbrev:
    .byte   1                       # Abbreviation Code
    .byte   17                      # DW_TAG_compile_unit
    .byte   1                       # DW_CHILDREN_yes
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    .byte   2                       # Abbreviation Code
    .byte   52                      # DW_TAG_variable
    .byte   0                       # DW_CHILDREN_no
    .byte   3                       # DW_AT_name
    .byte   14                      # DW_FORM_strp
    .byte   63                      # DW_AT_external
    .byte   25                      # DW_FORM_flag_present
    .byte   58                      # DW_AT_decl_file
    .byte   11                      # DW_FORM_data1
    .byte   59                      # DW_AT_decl_line
    .byte   11                      # DW_FORM_data1
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    .byte   0                       # EOM(3)

.section .debug_str,"MS",@progbits,1
.Linfo_string:
    .asciz "sym"

# Test handling of the situation (including the error message) where a structure
# has a incomplete member.

# REQUIRES: x86

# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: %lldb %t -o "target var b" -b 2>&1 | FileCheck %s

# CHECK: error: {{.*}} DWARF DIE at 0x0000002b (class B) has a member variable 0x00000030 (a) whose type is a forward declaration, not a complete definition.
# CHECK-NEXT: Please file a bug against the compiler and include the preprocessed output for /tmp/a.cc

# CHECK: b = (a = A @ 0x0000000000000001)

        .type   b,@object               # @b
        .comm   b,1,1
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "Hand-written DWARF"
.Lcu_name:
        .asciz  "/tmp/a.cc"
.Lcu_compdir:
        .asciz  "/foo/bar"
.Lb:
        .asciz  "b"
.La:
        .asciz  "a"
.LA:
        .asciz  "A"
.LB:
        .asciz  "B"

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   27                      # DW_AT_comp_dir
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   0                       # DW_CHILDREN_no
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
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
        .byte   1                       # Abbrev [1] 0xb:0x46 DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .long   .Lcu_name               # DW_AT_name
        .long   .Lcu_compdir            # DW_AT_comp_dir
        .byte   2                       # Abbrev [2] 0x1e:0x15 DW_TAG_variable
        .long   .Lb                     # DW_AT_name
        .long   .LB_die-.Lcu_begin0     # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   b
.LB_die:
        .byte   3                       # Abbrev [3] 0x33:0x15 DW_TAG_structure_type
        .long   .LB                     # DW_AT_name
        .byte   4                       # Abbrev [4] 0x3b:0xc DW_TAG_member
        .long   .La                     # DW_AT_name
        .long   .LA_die-.Lcu_begin0     # DW_AT_type
        .byte   0                       # End Of Children Mark
.LA_die:
        .byte   5                       # Abbrev [5] 0x48:0x8 DW_TAG_structure_type
                                        # DW_AT_declaration
        .long   .LA                     # DW_AT_name
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

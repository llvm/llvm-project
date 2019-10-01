# Check address lookup works correctly in the presence of type units.
# Specifically check that we don't use the line table pointed to by the
# DW_AT_stmt_list of the type unit (which used only for the file names) to
# compute address range for the type unit as type units don't describe any
# addresses. The addresses should always resolve to the relevant compile units.

# REQUIRES: lld, x86

# RUN: llvm-mc -dwarf-version=5 -triple x86_64-pc-linux %s -filetype=obj >%t.o
# RUN: ld.lld %t.o -o %t -image-base=0x47000 -z separate-code
# RUN: %lldb %t -o "image lookup -a 0x48000 -v" -o exit | FileCheck %s

# CHECK:   CompileUnit: id = {0x00000001}, file = "/tmp/a.cc", language = "c++"
# CHECK:      Function: id = {0x7fffffff0000006a}, name = "::_start({{.*}})", range = [0x0000000000048000-0x000000000004800c)
# CHECK:     LineEntry: [0x0000000000048000-0x000000000004800a): /tmp/a.cc:4
# CHECK:        Symbol: id = {0x00000002}, range = [0x0000000000048000-0x000000000004800c), name="_start"
# CHECK:      Variable: id = {0x7fffffff00000075}, name = "v1", {{.*}} decl = a.cc:4
# CHECK:      Variable: id = {0x7fffffff00000080}, name = "v2", {{.*}} decl = a.cc:4


# Output generated via
# clang -g -fdebug-types-section -gdwarf-5 -S
# from
# enum E1 { e1 };
# enum E2 { e2 };
# extern "C" void _start(E1 v1, E2 v2) {}
# The output was modified to place the compile unit in between the two type
# units.

        .text
        .file   "a.cc"
        .file   0 "/tmp" "a.cc"

        .text
        .globl  _start                  # -- Begin function _start
        .p2align        4, 0x90
        .type   _start,@function
_start:                                 # @_start
.Lfunc_begin0:
        .loc    0 4 0                   # /tmp/a.cc:4:0
        .cfi_startproc
# %bb.0:                                # %entry
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register %rbp
        movl    %edi, -4(%rbp)
        movl    %esi, -8(%rbp)
.Ltmp0:
        .loc    0 4 23 prologue_end     # /tmp/a.cc:4:23
        popq    %rbp
        .cfi_def_cfa %rsp, 8
        retq
.Ltmp1:
.Lfunc_end0:
        .size   _start, .Lfunc_end0-_start
        .cfi_endproc
                                        # -- End function
        .section        .debug_str_offsets,"",@progbits
        .long   52
        .short  5
        .short  0
.Lstr_offsets_base0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 9.0.0 (trunk 360907) (llvm/trunk 360908)"
.Linfo_string1:
        .asciz  "a.cc"
.Linfo_string2:
        .asciz  "/tmp"
.Linfo_string3:
        .asciz  "unsigned int"
.Linfo_string4:
        .asciz  "e1"
.Linfo_string5:
        .asciz  "E1"
.Linfo_string6:
        .asciz  "e2"
.Linfo_string7:
        .asciz  "E2"
.Linfo_string8:
        .asciz  "_start"
.Linfo_string9:
        .asciz  "f"
.Linfo_string10:
        .asciz  "v1"
.Linfo_string11:
        .asciz  "v2"
        .section        .debug_str_offsets,"",@progbits
        .long   .Linfo_string0
        .long   .Linfo_string1
        .long   .Linfo_string2
        .long   .Linfo_string3
        .long   .Linfo_string4
        .long   .Linfo_string5
        .long   .Linfo_string6
        .long   .Linfo_string7
        .long   .Linfo_string8
        .long   .Linfo_string9
        .long   .Linfo_string10
        .long   .Linfo_string11
        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   114                     # DW_AT_str_offsets_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   40                      # DW_TAG_enumerator
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   28                      # DW_AT_const_value
        .byte   15                      # DW_FORM_udata
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   37                      # DW_FORM_strx1
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   114                     # DW_AT_str_offsets_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   37                      # DW_FORM_strx1
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   0                       # DW_CHILDREN_no
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   105                     # DW_AT_signature
        .byte   32                      # DW_FORM_ref_sig8
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   7                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   64                      # DW_AT_frame_base
        .byte   24                      # DW_FORM_exprloc
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   8                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   2                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   -6180787752776176174    # Type Signature
        .long   35                      # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x18:0x1d DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0     # DW_AT_stmt_list
        .long   .Lstr_offsets_base0     # DW_AT_str_offsets_base
        .byte   2                       # Abbrev [2] 0x23:0xd DW_TAG_enumeration_type
        .long   48                      # DW_AT_type
        .byte   5                       # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # DW_AT_decl_file
        .byte   1                       # DW_AT_decl_line
        .byte   3                       # Abbrev [3] 0x2c:0x3 DW_TAG_enumerator
        .byte   4                       # DW_AT_name
        .byte   0                       # DW_AT_const_value
        .byte   0                       # End Of Children Mark
        .byte   4                       # Abbrev [4] 0x30:0x4 DW_TAG_base_type
        .byte   3                       # DW_AT_name
        .byte   7                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .long   .Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   5                       # Abbrev [5] 0xc:0x4d DW_TAG_compile_unit
        .byte   0                       # DW_AT_producer
        .short  4                       # DW_AT_language
        .byte   1                       # DW_AT_name
        .long   .Lstr_offsets_base0     # DW_AT_str_offsets_base
        .long   .Lline_table_start0     # DW_AT_stmt_list
        .byte   2                       # DW_AT_comp_dir
        .byte   0                       # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .byte   6                       # Abbrev [6] 0x23:0x9 DW_TAG_enumeration_type
                                        # DW_AT_declaration
        .quad   -6180787752776176174    # DW_AT_signature
        .byte   6                       # Abbrev [6] 0x2c:0x9 DW_TAG_enumeration_type
                                        # DW_AT_declaration
        .quad   7818257750321376053     # DW_AT_signature
        .byte   7                       # Abbrev [7] 0x35:0x23 DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .byte   1                       # DW_AT_frame_base
        .byte   86
        .byte   8                       # DW_AT_name
        .byte   0                       # DW_AT_decl_file
        .byte   4                       # DW_AT_decl_line
                                        # DW_AT_external
        .byte   8                       # Abbrev [8] 0x41:0xb DW_TAG_formal_parameter
        .byte   2                       # DW_AT_location
        .byte   145
        .byte   124
        .byte   10                      # DW_AT_name
        .byte   0                       # DW_AT_decl_file
        .byte   4                       # DW_AT_decl_line
        .long   35                      # DW_AT_type
        .byte   8                       # Abbrev [8] 0x4c:0xb DW_TAG_formal_parameter
        .byte   2                       # DW_AT_location
        .byte   145
        .byte   120
        .byte   11                      # DW_AT_name
        .byte   0                       # DW_AT_decl_file
        .byte   4                       # DW_AT_decl_line
        .long   44                      # DW_AT_type
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end2:

        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   2                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   7818257750321376053     # Type Signature
        .long   35                      # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x18:0x1d DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0     # DW_AT_stmt_list
        .long   .Lstr_offsets_base0     # DW_AT_str_offsets_base
        .byte   2                       # Abbrev [2] 0x23:0xd DW_TAG_enumeration_type
        .long   48                      # DW_AT_type
        .byte   7                       # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # DW_AT_decl_file
        .byte   2                       # DW_AT_decl_line
        .byte   3                       # Abbrev [3] 0x2c:0x3 DW_TAG_enumerator
        .byte   6                       # DW_AT_name
        .byte   0                       # DW_AT_const_value
        .byte   0                       # End Of Children Mark
        .byte   4                       # Abbrev [4] 0x30:0x4 DW_TAG_base_type
        .byte   3                       # DW_AT_name
        .byte   7                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
        .section        .debug_macinfo,"",@progbits
        .byte   0                       # End Of Macro List Mark
        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   .Lfunc_begin0
.Ldebug_addr_end0:

        .section        .debug_line,"",@progbits
.Lline_table_start0:

## Check that BOLT correctly handles two CUs with DWARF-5 debug info (does not crash), when
## a function from one CU is forced to be inlined into another.

# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t-main.o
# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %p/Inputs/dwarf5_helper.s -o %thelper.o
# RUN: %clang %cflags -gdwarf-5 -Wl,-q %t-main.o %thelper.o -o %t.exe
# RUN: llvm-bolt %t.exe --update-debug-sections --force-inline=_Z3fooi \
# RUN:   -o %t.bolt | FileCheck %s

# CHECK-NOT: BOLT-ERROR
# CHECK-NOT: BOLT-WARNING
# CHECK: BOLT-INFO: inlined {{[0-9]+}} calls at {{[1-9][0-9]*}} call sites

# extern int foo(int);
# int main(){
#     foo(10);
#     return 0;
# }
        .file   "main.cpp"
        .text
        .globl  main                            # -- Begin function main
        .p2align        4
        .type   main,@function
main:                                   # @main
.Lfunc_begin0:
        .file   0 "/home/gpastukhov/tmp2" "main.cpp" md5 0x5c930f5d3a068b09fd18ece59c58bdcf
        .loc    0 2 0                           # main.cpp:2:0
        .cfi_startproc
# %bb.0:
        pushq   %rax
        .cfi_def_cfa_offset 16
.Ltmp0:
        .loc    0 3 5 prologue_end              # main.cpp:3:5
        movl    $10, %edi
        callq   _Z3fooi
.Ltmp1:
        .loc    0 4 5                           # main.cpp:4:5
        xorl    %eax, %eax
        .loc    0 4 5 epilogue_begin is_stmt 0  # main.cpp:4:5
        popq    %rcx
        .cfi_def_cfa_offset 8
        retq
.Ltmp2:
.Lfunc_end0:
        .size   main, .Lfunc_end0-main
        .cfi_endproc
                                        # -- End function
        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   37                              # DW_FORM_strx1
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   114                             # DW_AT_str_offsets_base
        .byte   23                              # DW_FORM_sec_offset
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   27                              # DW_AT_comp_dir
        .byte   37                              # DW_FORM_strx1
        .byte   17                              # DW_AT_low_pc
        .byte   27                              # DW_FORM_addrx
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   115                             # DW_AT_addr_base
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   17                              # DW_AT_low_pc
        .byte   27                              # DW_FORM_addrx
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .byte   122                             # DW_AT_call_all_calls
        .byte   25                              # DW_FORM_flag_present
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   72                              # DW_TAG_call_site
        .byte   1                               # DW_CHILDREN_yes
        .byte   127                             # DW_AT_call_origin
        .byte   19                              # DW_FORM_ref4
        .byte   125                             # DW_AT_call_return_pc
        .byte   27                              # DW_FORM_addrx
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   73                              # DW_TAG_call_site_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   126                             # DW_AT_call_value
        .byte   24                              # DW_FORM_exprloc
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   37                              # DW_FORM_strx1
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev [1] 0xc:0x47 DW_TAG_compile_unit
        .byte   0                               # DW_AT_producer
        .short  33                              # DW_AT_language
        .byte   1                               # DW_AT_name
        .long   .Lstr_offsets_base0             # DW_AT_str_offsets_base
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .byte   2                               # DW_AT_comp_dir
        .byte   0                               # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .long   .Laddr_table_base0              # DW_AT_addr_base
        .byte   2                               # Abbrev [2] 0x23:0x1c DW_TAG_subprogram
        .byte   0                               # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
                                        # DW_AT_call_all_calls
        .byte   6                               # DW_AT_name
        .byte   0                               # DW_AT_decl_file
        .byte   2                               # DW_AT_decl_line
        .long   78                              # DW_AT_type
                                        # DW_AT_external
        .byte   3                               # Abbrev [3] 0x32:0xc DW_TAG_call_site
        .long   63                              # DW_AT_call_origin
        .byte   1                               # DW_AT_call_return_pc
        .byte   4                               # Abbrev [4] 0x38:0x5 DW_TAG_call_site_parameter
        .byte   1                               # DW_AT_location
        .byte   85
        .byte   1                               # DW_AT_call_value
        .byte   58
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   5                               # Abbrev [5] 0x3f:0xf DW_TAG_subprogram
        .byte   3                               # DW_AT_linkage_name
        .byte   4                               # DW_AT_name
        .byte   0                               # DW_AT_decl_file
        .byte   1                               # DW_AT_decl_line
        .long   78                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   6                               # Abbrev [6] 0x48:0x5 DW_TAG_formal_parameter
        .long   78                              # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   7                               # Abbrev [7] 0x4e:0x4 DW_TAG_base_type
        .byte   5                               # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_str_offsets,"",@progbits
        .long   32                              # Length of String Offsets Set
        .short  5
        .short  0
.Lstr_offsets_base0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 20.1.8 (CentOS 20.1.8-1.el9)" # string offset=0
.Linfo_string1:
        .asciz  "main.cpp"                      # string offset=43
.Linfo_string2:
        .asciz  "/home/gpastukhov/tmp2"         # string offset=52
.Linfo_string3:
        .asciz  "_Z3fooi"                       # string offset=74
.Linfo_string4:
        .asciz  "foo"                           # string offset=82
.Linfo_string5:
        .asciz  "int"                           # string offset=86
.Linfo_string6:
        .asciz  "main"                          # string offset=90
        .section        .debug_str_offsets,"",@progbits
        .long   .Linfo_string0
        .long   .Linfo_string1
        .long   .Linfo_string2
        .long   .Linfo_string3
        .long   .Linfo_string4
        .long   .Linfo_string5
        .long   .Linfo_string6
        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                               # DWARF version number
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
.Laddr_table_base0:
        .quad   .Lfunc_begin0
        .quad   .Ltmp1
.Ldebug_addr_end0:
        .ident  "clang version 20.1.8 (CentOS 20.1.8-1.el9)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .section        .debug_line,"",@progbits
.Lline_table_start0:

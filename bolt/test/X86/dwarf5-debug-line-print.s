# REQUIRES: system-linux

## Check that BOLT correctly prints debug line comments for DWARF-5.


# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --update-debug-sections --print-debug-info \
# RUN:   --print-after-lowering -o %t.bolt | FileCheck %s

# CHECK: xorq    %rdi, %rdi # debug line main.c:2:5

# __attribute__((naked)) void _start() {
#     __asm__(
#         "xor %rdi, %rdi\n"   // exit code 0
#         "mov $60, %rax\n"    // syscall number for exit
#         "syscall\n"
#     );
# }

        .file   "main.c"
        .text
        .globl  _start                          # -- Begin function _start
        .p2align        4
        .type   _start,@function
_start:                                 # @_start
.Lfunc_begin0:
        .file   0 "/home/gpastukhov/tmp2" "main.c" md5 0x94c0e54a615c2a21415ddb904991abd8
        .cfi_startproc
# %bb.0:
        .loc    0 2 5 prologue_end              # main.c:2:5
        #APP
        xorq    %rdi, %rdi
        movq    $60, %rax
        syscall

        #NO_APP
.Ltmp0:
.Lfunc_end0:
        .size   _start, .Lfunc_end0-_start
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
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   27                              # DW_FORM_addrx
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
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
        .byte   1                               # Abbrev [1] 0xc:0x23 DW_TAG_compile_unit
        .byte   0                               # DW_AT_producer
        .short  29                              # DW_AT_language
        .byte   1                               # DW_AT_name
        .long   .Lstr_offsets_base0             # DW_AT_str_offsets_base
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .byte   2                               # DW_AT_comp_dir
        .byte   0                               # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .long   .Laddr_table_base0              # DW_AT_addr_base
        .byte   2                               # Abbrev [2] 0x23:0xb DW_TAG_subprogram
        .byte   0                               # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
        .byte   3                               # DW_AT_name
        .byte   0                               # DW_AT_decl_file
        .byte   1                               # DW_AT_decl_line
                                        # DW_AT_external
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_str_offsets,"",@progbits
        .long   20                              # Length of String Offsets Set
        .short  5
        .short  0
.Lstr_offsets_base0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 20.1.8 (CentOS 20.1.8-1.el9)" # string offset=0
.Linfo_string1:
        .asciz  "main.c"                        # string offset=43
.Linfo_string2:
        .asciz  "/home/gpastukhov/tmp2"         # string offset=50
.Linfo_string3:
        .asciz  "_start"                        # string offset=72
        .section        .debug_str_offsets,"",@progbits
        .long   .Linfo_string0
        .long   .Linfo_string1
        .long   .Linfo_string2
        .long   .Linfo_string3
        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                               # DWARF version number
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
.Laddr_table_base0:
        .quad   .Lfunc_begin0
.Ldebug_addr_end0:
        .ident  "clang version 20.1.8 (CentOS 20.1.8-1.el9)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .section        .debug_line,"",@progbits
.Lline_table_start0:

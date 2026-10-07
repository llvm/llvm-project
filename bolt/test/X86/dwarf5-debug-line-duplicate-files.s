# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-dwarfdump --show-form --verbose --debug-line %t.exe | FileCheck --check-prefix=PRE %s
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --verify %t.bolt
# RUN: llvm-dwarfdump --show-form --verbose --debug-line --debug-info %t.bolt | FileCheck --check-prefix=POST %s

## BOLT rebuilds processed line tables through MC, which intentionally interns
## duplicate file entries. Check that references to input file indexes are
## remapped to the indexes in the rebuilt table.

# PRE: file_names[  0]:
# PRE-NEXT: name: "main.cpp"
# PRE: file_names[  1]:
# PRE-NEXT: name: "main.cpp"
# PRE: file_names[  2]:
# PRE-NEXT: name: "header.h"

# POST: DW_AT_name [DW_FORM_string] ("main")
# POST: DW_AT_decl_file [DW_FORM_data1] ("./header.h")
# POST: DW_AT_call_file [DW_FORM_data1] ("./header.h")
# POST: file_names[  0]:
# POST-NEXT: name: "main.cpp"
# POST: file_names[  1]:
# POST-NEXT: name: "header.h"
# POST-NOT: file_names[  2]:
# POST: {{0x[0-9a-f]+}}{{ +}}7{{ +}}0{{ +}}1{{ +}}0

        .text
        .globl  main
        .p2align 4, 0x90
        .type   main,@function
main:
.Lfunc_begin0:
        xorl    %eax, %eax
        retq
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

        .section .debug_abbrev,"",@progbits
        .byte   1                       # Abbrev code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   27                      # DW_AT_comp_dir
        .byte   8                       # DW_FORM_string
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0
        .byte   0
        .byte   2                       # Abbrev code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   88                      # DW_AT_call_file
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0
        .byte   0

        .section .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
        .short  5                       # DWARF version
        .byte   1                       # DW_UT_compile
        .byte   8                       # Address size
        .long   .debug_abbrev
        .byte   1                       # DW_TAG_compile_unit
        .asciz  "test producer"
        .short  33                      # DW_LANG_C_plus_plus_14
        .asciz  "main.cpp"
        .asciz  "."
        .long   .Lline_table_start0
        .quad   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
        .byte   2                       # DW_TAG_subprogram
        .quad   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
        .asciz  "main"
        .byte   2                       # Input file index for header.h
        .byte   7
        .byte   2                       # Input call file index for header.h
        .byte   0                       # End children
.Ldebug_info_end0:

        .section .debug_line,"",@progbits
.Lline_table_start0:
        .long   .Lline_table_end0-.Lline_table_start1
.Lline_table_start1:
        .short  5                       # DWARF version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   .Lline_prologue_end0-.Lline_prologue_start0
.Lline_prologue_start0:
        .byte   1                       # Minimum instruction length
        .byte   1                       # Maximum operations per instruction
        .byte   1                       # Default is_stmt
        .byte   -5                      # Line base
        .byte   14                      # Line range
        .byte   13                      # Opcode base
        .byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
        .byte   1                       # Directory format count
        .byte   1                       # DW_LNCT_path
        .byte   8                       # DW_FORM_string
        .uleb128 1                      # Directory count
        .asciz  "."
        .byte   2                       # File format count
        .byte   1                       # DW_LNCT_path
        .byte   8                       # DW_FORM_string
        .byte   2                       # DW_LNCT_directory_index
        .byte   11                      # DW_FORM_data1
        .uleb128 3                      # File count
        .asciz  "main.cpp"
        .byte   0
        .asciz  "main.cpp"             # Duplicate root entry
        .byte   0
        .asciz  "header.h"
        .byte   0
.Lline_prologue_end0:
        .byte   0                       # DW_LNE_set_address
        .uleb128 9
        .byte   2
        .quad   .Lfunc_begin0
        .byte   4                       # DW_LNS_set_file
        .uleb128 2                      # Input file index for header.h
        .byte   3                       # DW_LNS_advance_line
        .sleb128 6
        .byte   1                       # DW_LNS_copy
        .byte   2                       # DW_LNS_advance_pc
        .uleb128 .Lfunc_end0-.Lfunc_begin0
        .byte   0                       # DW_LNE_end_sequence
        .uleb128 1
        .byte   1
.Lline_table_end0:

        .section ".note.GNU-stack","",@progbits

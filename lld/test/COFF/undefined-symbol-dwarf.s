# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.o %s
# RUN: not lld-link /lldmingw /out:%t.exe %t.o /entry:entry 2>&1 | FileCheck %s

# CHECK: error: undefined symbol: bar()
# CHECK-NEXT: >>> referenced by /path/to/src{{[/\\]}}undef.cpp:17
# CHECK-NEXT: >>>               {{.*}}.o:(entry)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: foo()
# CHECK-NEXT: >>> referenced by /path/to/src{{[/\\]}}undef.cpp:7
# CHECK-NEXT: >>>               {{.*}}.o:(A::afunc())

        .text
        .file   "undef.cpp"
        .file   1 "/path/to/src" "undef.cpp"
        .globl  entry                   # -- Begin function entry
entry:                                  # @entry
.Lfunc_begin0:
        .loc    1 14 0                  # undef.cpp:14:0
        subq    $40, %rsp
.Ltmp0:
        leaq    32(%rsp), %rcx
.Ltmp1:
        .loc    1 16 4 prologue_end     # undef.cpp:16:4
        callq   _ZN1A5afuncEv
        .loc    1 17 2                  # undef.cpp:17:2
        callq   _Z3barv
        .loc    1 18 1                  # undef.cpp:18:1
        addq    $40, %rsp
        retq
.Ltmp2:
.Lfunc_end0:

        .def     _ZN1A5afuncEv;
        .scl    2;
        .type   32;
        .endef
        .section        .text$_ZN1A5afuncEv,"xr",discard,_ZN1A5afuncEv
        .globl  _ZN1A5afuncEv           # -- Begin function _ZN1A5afuncEv
        .p2align        1, 0x90
_ZN1A5afuncEv:                          # @_ZN1A5afuncEv
.Lfunc_begin1:
        .loc    1 6 0                   # undef.cpp:6:0
        .loc    1 7 3 prologue_end      # undef.cpp:7:3
        jmp     _Z3foov                 # TAILCALL
.Ltmp3:
.Lfunc_end1:

        .section        .debug_abbrev,"dr"
.Lsection_abbrev:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   37                      # DW_FORM_strx1
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   37                      # DW_FORM_strx1
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   37                      # DW_FORM_strx1
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   85                      # DW_AT_ranges
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"dr"
.Lsection_info:
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .secrel32       .Lsection_abbrev # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0xb0 DW_TAG_compile_unit
        .byte   0                       # DW_AT_producer
        .short  4                       # DW_AT_language
        .byte   0                       # DW_AT_name
        .secrel32       .Lline_table_start0 # DW_AT_stmt_list
        .byte   0                       # DW_AT_comp_dir
        .quad   0                       # DW_AT_low_pc
        .secrel32       .Ldebug_ranges0 # DW_AT_ranges
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_ranges,"dr"
.Ldebug_range:
.Ldebug_ranges0:
        .quad   .Lfunc_begin0
        .quad   .Lfunc_end0
        .quad   .Lfunc_begin1
        .quad   .Lfunc_end1
        .quad   0
        .quad   0

        .section        .debug_line,"dr"
.Lline_table_start0:

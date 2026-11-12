# REQUIRES: x86
# RUN: llvm-mc -triple=i686-windows-gnu -filetype=obj -o %t.o %s
# RUN: cp %t.o %t.dupl.o
# RUN: not lld-link -lldmingw -out:%t.exe %t.o %t.dupl.o -entry:_Z4funcv 2>&1 | FileCheck %s

# CHECK: error: duplicate symbol: func()
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:6
# CHECK-NEXT: >>>            {{.*}}.o
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:6
# CHECK-NEXT: >>>            {{.*}}.o
# CHECK-EMPTY:
# CHECK-NEXT: error: duplicate symbol: _var
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:1
# CHECK-NEXT: >>>            {{.*}}.o
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:1
# CHECK-NEXT: >>>            {{.*}}.o
# CHECK-EMPTY:
# CHECK-NEXT: error: duplicate symbol: A::namespaceVar
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:3
# CHECK-NEXT: >>>            {{.*}}.o
# CHECK-NEXT: >>> defined at /path/to/src{{[/\\]}}dupl.cpp:3
# CHECK-NEXT: >>>            {{.*}}.o

        .text
        .file   "dupl.cpp"
        .file   1 "/path/to/src" "dupl.cpp"
        .def     __Z4funcv;
        .globl  __Z4funcv               # -- Begin function _Z4funcv
__Z4funcv:                              # @_Z4funcv
Lfunc_begin0:
        .loc    1 5 0                   # dupl.cpp:5:0
# %bb.0:                                # %entry
        .loc    1 6 1 prologue_end      # dupl.cpp:6:1
        retl
Lfunc_end0:
                                        # -- End function
        .bss
        .globl  _var                    # @var
_var:
        .long   0                       # 0x0

        .globl  __ZN1A12namespaceVarE   # @_ZN1A12namespaceVarE
__ZN1A12namespaceVarE:
        .long   0                       # 0x0

        .section        .debug_str,"dr"
Linfo_string:
Linfo_string0:
        .asciz  "var"
Linfo_string1:
        .asciz  "int"
Linfo_string2:
        .asciz  "A"
Linfo_string3:
        .asciz  "namespaceVar"
Linfo_string4:
        .asciz  "_ZN1A12namespaceVarE"
Linfo_string5:
        .asciz  "_Z4funcv"
Linfo_string6:
        .asciz  "func"
        .section        .debug_abbrev,"dr"
Lsection_abbrev:
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
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   57                      # DW_TAG_namespace
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   110                     # DW_AT_linkage_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   64                      # DW_AT_frame_base
        .byte   24                      # DW_FORM_exprloc
        .byte   110                     # DW_AT_linkage_name
        .byte   14                      # DW_FORM_strp
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"dr"
Lsection_info:
Lcu_begin0:
        .long   Ldebug_info_end0-Ldebug_info_start0 # Length of Unit
Ldebug_info_start0:
        .short  4                       # DWARF version number
        .secrel32       Lsection_abbrev # Offset Into Abbrev. Section
        .byte   4                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x64 DW_TAG_compile_unit
        .byte   0                       # DW_AT_producer
        .short  33                      # DW_AT_language
        .byte   0                       # DW_AT_name
        .secrel32       Lline_table_start0 # DW_AT_stmt_list
        .long   Lfunc_begin0            # DW_AT_low_pc
        .long   Lfunc_end0-Lfunc_begin0 # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x26:0x11 DW_TAG_variable
        .secrel32       Linfo_string0   # DW_AT_name
        .secrel32       Linfo_type_int  # DW_AT_type
                                        # DW_AT_external
        .byte   1                       # DW_AT_decl_file
        .byte   1                       # DW_AT_decl_line
        .byte   5                       # DW_AT_location
        .byte   3
        .long   _var
Linfo_type_int:
        .byte   3                       # Abbrev [3] 0x37:0x7 DW_TAG_base_type
        .secrel32       Linfo_string1   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   4                       # Abbrev [4] 0x3e:0x1b DW_TAG_namespace
        .secrel32       Linfo_string2   # DW_AT_name
        .byte   5                       # Abbrev [5] 0x43:0x15 DW_TAG_variable
        .secrel32       Linfo_string3   # DW_AT_name
        .secrel32       Linfo_type_int  # DW_AT_type
                                        # DW_AT_external
        .byte   1                       # DW_AT_decl_file
        .byte   3                       # DW_AT_decl_line
        .byte   5                       # DW_AT_location
        .byte   3
        .long   __ZN1A12namespaceVarE
        .secrel32       Linfo_string4   # DW_AT_linkage_name
        .byte   0                       # End Of Children Mark
        .byte   6                       # Abbrev [6] 0x59:0x15 DW_TAG_subprogram
        .long   Lfunc_begin0            # DW_AT_low_pc
        .long   Lfunc_end0-Lfunc_begin0 # DW_AT_high_pc
        .byte   1                       # DW_AT_frame_base
        .byte   84
        .secrel32       Linfo_string5   # DW_AT_linkage_name
        .secrel32       Linfo_string6   # DW_AT_name
        .byte   1                       # DW_AT_decl_file
        .byte   5                       # DW_AT_decl_line
                                        # DW_AT_external
        .byte   0                       # End Of Children Mark
Ldebug_info_end0:

        .section        .debug_line,"dr"
Lline_table_start0:

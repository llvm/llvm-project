  .globaltype __stack_pointer, i32
  .text
  .file  "signature-mismatch-debug-info-a.ll"
  .functype  foo (i32) -> ()
  .functype  test0 () -> ()
  .section  .text.foo,"",@
  .weak  foo                             # -- Begin function foo
  .type  foo,@function
foo:                                    # @foo
.Lfunc_begin0:
  .functype  foo (i32) -> ()
# %bb.0:
                                        # fallthrough-return
  end_function
.Lfunc_end0:
                                        # -- End function
  .section  .text.test0,"",@
  .globl  test0                           # -- Begin function test0
  .type  test0,@function
test0:                                  # @test0
.Lfunc_begin1:
  .functype  test0 () -> ()
# %bb.0:
  i32.const  3
  call  foo
                                        # fallthrough-return
  end_function
.Lfunc_end1:
                                        # -- End function
  .file  1 "a.c"
  .section  .debug_abbrev,"",@
  .int8  1                               # Abbreviation Code
  .int8  17                              # DW_TAG_compile_unit
  .int8  1                               # DW_CHILDREN_yes
  .int8  37                              # DW_AT_producer
  .int8  14                              # DW_FORM_strp
  .int8  19                              # DW_AT_language
  .int8  5                               # DW_FORM_data2
  .int8  3                               # DW_AT_name
  .int8  14                              # DW_FORM_strp
  .int8  16                              # DW_AT_stmt_list
  .int8  23                              # DW_FORM_sec_offset
  .int8  17                              # DW_AT_low_pc
  .int8  1                               # DW_FORM_addr
  .int8  85                              # DW_AT_ranges
  .int8  23                              # DW_FORM_sec_offset
  .int8  0                               # EOM(1)
  .int8  0                               # EOM(2)
  .int8  2                               # Abbreviation Code
  .int8  46                              # DW_TAG_subprogram
  .int8  0                               # DW_CHILDREN_no
  .int8  17                              # DW_AT_low_pc
  .int8  1                               # DW_FORM_addr
  .int8  18                              # DW_AT_high_pc
  .int8  6                               # DW_FORM_data4
  .int8  64                              # DW_AT_frame_base
  .int8  24                              # DW_FORM_exprloc
  .int8  3                               # DW_AT_name
  .int8  14                              # DW_FORM_strp
  .int8  58                              # DW_AT_decl_file
  .int8  11                              # DW_FORM_data1
  .int8  59                              # DW_AT_decl_line
  .int8  11                              # DW_FORM_data1
  .int8  39                              # DW_AT_prototyped
  .int8  25                              # DW_FORM_flag_present
  .int8  63                              # DW_AT_external
  .int8  25                              # DW_FORM_flag_present
  .int8  0                               # EOM(1)
  .int8  0                               # EOM(2)
  .int8  3                               # Abbreviation Code
  .int8  46                              # DW_TAG_subprogram
  .int8  0                               # DW_CHILDREN_no
  .int8  17                              # DW_AT_low_pc
  .int8  1                               # DW_FORM_addr
  .int8  18                              # DW_AT_high_pc
  .int8  6                               # DW_FORM_data4
  .int8  64                              # DW_AT_frame_base
  .int8  24                              # DW_FORM_exprloc
  .int8  3                               # DW_AT_name
  .int8  14                              # DW_FORM_strp
  .int8  58                              # DW_AT_decl_file
  .int8  11                              # DW_FORM_data1
  .int8  59                              # DW_AT_decl_line
  .int8  11                              # DW_FORM_data1
  .int8  63                              # DW_AT_external
  .int8  25                              # DW_FORM_flag_present
  .int8  0                               # EOM(1)
  .int8  0                               # EOM(2)
  .int8  0                               # EOM(3)
  .section  .debug_info,"",@
.Lcu_begin0:
  .int32  .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
  .int16  4                               # DWARF version number
  .int32  .debug_abbrev0                  # Offset Into Abbrev. Section
  .int8  4                               # Address Size (in bytes)
  .int8  1                               # Abbrev [1] 0xb:0x46 DW_TAG_compile_unit
  .int32  .Linfo_string0                  # DW_AT_producer
  .int16  29                              # DW_AT_language
  .int32  .Linfo_string1                  # DW_AT_name
  .int32  .Lline_table_start0             # DW_AT_stmt_list
  .int32  0                               # DW_AT_low_pc
  .int32  .Ldebug_ranges0                 # DW_AT_ranges
  .int8  2                               # Abbrev [2] 0x22:0x17 DW_TAG_subprogram
  .int32  .Lfunc_begin0                   # DW_AT_low_pc
  .int32  .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
  .int8  7                               # DW_AT_frame_base
  .int8  237
  .int8  3
  .int32  __stack_pointer
  .int8  159
  .int32  .Linfo_string2                  # DW_AT_name
  .int8  1                               # DW_AT_decl_file
  .int8  3                               # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_external
  .int8  3                               # Abbrev [3] 0x39:0x17 DW_TAG_subprogram
  .int32  .Lfunc_begin1                   # DW_AT_low_pc
  .int32  .Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
  .int8  7                               # DW_AT_frame_base
  .int8  237
  .int8  3
  .int32  __stack_pointer
  .int8  159
  .int32  .Linfo_string3                  # DW_AT_name
  .int8  1                               # DW_AT_decl_file
  .int8  7                               # DW_AT_decl_line
                                        # DW_AT_external
  .int8  0                               # End Of Children Mark
.Ldebug_info_end0:
  .section  .debug_ranges,"",@
.Ldebug_ranges0:
  .int32  .Lfunc_begin0
  .int32  .Lfunc_end0
  .int32  .Lfunc_begin1
  .int32  .Lfunc_end1
  .int32  0
  .int32  0
  .section  .debug_str,"S",@
.Linfo_string0:
  .asciz  "clang version 19.0.0git"       # string offset=0
.Linfo_string1:
  .asciz  "a.c"                           # string offset=24
.Linfo_string2:
  .asciz  "foo"                           # string offset=28
.Linfo_string3:
  .asciz  "test0"                         # string offset=32
  .ident  "clang version 19.0.0git"
  .section  .custom_section.producers,"",@
  .int8  2
  .int8  8
  .ascii  "language"
  .int8  1
  .int8  3
  .ascii  "C11"
  .int8  0
  .int8  12
  .ascii  "processed-by"
  .int8  1
  .int8  5
  .ascii  "clang"
  .int8  9
  .ascii  "19.0.0git"
  .section  .debug_str,"S",@
  .section  .custom_section.target_features,"",@
  .int8  2
  .int8  43
  .int8  15
  .ascii  "mutable-globals"
  .int8  43
  .int8  8
  .ascii  "sign-ext"
  .section  .debug_str,"S",@
  .section  .debug_line,"",@
.Lline_table_start0:

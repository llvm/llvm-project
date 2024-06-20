  .globaltype __stack_pointer, i32
  .functype  __original_main () -> (i32)
  .functype  test0 () -> ()
  .functype  test1 () -> ()
  .functype  main (i32, i32) -> (i32)
  .section  .text.__original_main,"",@
  .globl  __original_main                 # -- Begin function __original_main
  .type  __original_main,@function
__original_main:                        # @__original_main
.Lfunc_begin0:
  .functype  __original_main () -> (i32)
# %bb.0:
  call  test0
  call  test1
  i32.const  0
                                        # fallthrough-return
  end_function
.Lfunc_end0:
                                        # -- End function
  .section  .text.main,"",@
  .globl  main                            # -- Begin function main
  .type  main,@function
main:                                   # @main
.Lfunc_begin1:
  .functype  main (i32, i32) -> (i32)
# %bb.0:                                # %body
  call  __original_main
                                        # fallthrough-return
  end_function
.Lfunc_end1:
                                        # -- End function
  .file  1 "main.c"
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
  .int8  18                              # DW_AT_high_pc
  .int8  6                               # DW_FORM_data4
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
  .int8  73                              # DW_AT_type
  .int8  19                              # DW_FORM_ref4
  .int8  63                              # DW_AT_external
  .int8  25                              # DW_FORM_flag_present
  .int8  0                               # EOM(1)
  .int8  0                               # EOM(2)
  .int8  3                               # Abbreviation Code
  .int8  36                              # DW_TAG_base_type
  .int8  0                               # DW_CHILDREN_no
  .int8  3                               # DW_AT_name
  .int8  14                              # DW_FORM_strp
  .int8  62                              # DW_AT_encoding
  .int8  11                              # DW_FORM_data1
  .int8  11                              # DW_AT_byte_size
  .int8  11                              # DW_FORM_data1
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
  .int8  1                               # Abbrev [1] 0xb:0x3a DW_TAG_compile_unit
  .int32  .Linfo_string0                  # DW_AT_producer
  .int16  29                              # DW_AT_language
  .int32  .Linfo_string1                  # DW_AT_name
  .int32  .Lline_table_start0             # DW_AT_stmt_list
  .int32  .Lfunc_begin0                   # DW_AT_low_pc
  .int32  .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
  .int8  2                               # Abbrev [2] 0x22:0x1b DW_TAG_subprogram
  .int32  .Lfunc_begin0                   # DW_AT_low_pc
  .int32  .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
  .int8  7                               # DW_AT_frame_base
  .int8  237
  .int8  3
  .int32  __stack_pointer
  .int8  159
  .int32  .Linfo_string2                  # DW_AT_name
  .int8  1                               # DW_AT_decl_file
  .int8  4                               # DW_AT_decl_line
  .int32  61                              # DW_AT_type
                                        # DW_AT_external
  .int8  3                               # Abbrev [3] 0x3d:0x7 DW_TAG_base_type
  .int32  .Linfo_string3                  # DW_AT_name
  .int8  5                               # DW_AT_encoding
  .int8  4                               # DW_AT_byte_size
  .int8  0                               # End Of Children Mark
.Ldebug_info_end0:

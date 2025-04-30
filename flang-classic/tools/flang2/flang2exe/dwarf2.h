/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Manifest constants, etc. used in for DWARF2/DWARF3 emission. DWARF
 * versions 2 and 3
 */

#define DWARF_VERSION 2
#define DWARF_VERSION3 3
#define DWARF_VERSION4 4

/*
 * Functions converting numerical tags to textual tags are in dwarf_names.c.
 */
extern const char *dwarf_tag_name(unsigned);
extern const char *dwarf_attr_name(unsigned);
extern const char *dwarf_form_name(unsigned);
extern const char *dwarf_stack_op_name(unsigned);
extern const char *dwarf_virtuality_name(unsigned);
extern const char *dwarf_lang_name(unsigned);
extern const char *dwarf_encoding_name(unsigned);

void emit_dwf2_ftn_func_begin(int sptr);

#define DW_TAG_padding 0x00
#define DW_TAG_array_type 0x01
#define DW_TAG_class_type 0x02
#define DW_TAG_entry_point 0x03
#define DW_TAG_enumeration_type 0x04
#define DW_TAG_formal_parameter 0x05
#define DW_TAG_imported_declaration 0x08
#define DW_TAG_label 0x0a
#define DW_TAG_lexical_block 0x0b
#define DW_TAG_member 0x0d
#define DW_TAG_pointer_type 0x0f
#define DW_TAG_reference_type 0x10
#define DW_TAG_compile_unit 0x11
#define DW_TAG_string_type 0x12
#define DW_TAG_structure_type 0x13
#define DW_TAG_subroutine_type 0x15
#define DW_TAG_typedef 0x16
#define DW_TAG_union_type 0x17
#define DW_TAG_unspecified_parameters 0x18
#define DW_TAG_variant 0x19
#define DW_TAG_common_block 0x1a
#define DW_TAG_common_inclusion 0x1b
#define DW_TAG_inheritance 0x1c
#define DW_TAG_inlined_subroutine 0x1d
#define DW_TAG_module 0x1e
#define DW_TAG_ptr_to_member_type 0x1f
#define DW_TAG_set_type 0x20
#define DW_TAG_subrange_type 0x21
#define DW_TAG_with_stmt 0x22
#define DW_TAG_access_declaration 0x23
#define DW_TAG_base_type 0x24
#define DW_TAG_catch_block 0x25
#define DW_TAG_const_type 0x26
#define DW_TAG_constant 0x27
#define DW_TAG_enumerator 0x28
#define DW_TAG_file_type 0x29
#define DW_TAG_friend 0x2a
#define DW_TAG_namelist 0x2b
#define DW_TAG_namelist_item 0x2c
#define DW_TAG_packed_type 0x2d
#define DW_TAG_subprogram 0x2e
#define DW_TAG_template_type_param 0x2f
#define DW_TAG_template_value_param 0x30
#define DW_TAG_thrown_type 0x31
#define DW_TAG_try_block 0x32
#define DW_TAG_variant_part 0x33
#define DW_TAG_variable 0x34
#define DW_TAG_volatile_type 0x35
/* dwarf3 additions */
#define DW_TAG_dwarf_procedure 0x36
#define DW_TAG_restrict_type 0x37
#define DW_TAG_interface_type 0x38
#define DW_TAG_namespace 0x39
#define DW_TAG_imported_module 0x3a
#define DW_TAG_unspecified_type 0x3b
#define DW_TAG_partial_unit 0x3c
#define DW_TAG_imported_unit 0x3d
#define DW_TAG_condition 0x3f
#define DW_TAG_shared_type 0x40

/* LLVM extension: Only valid in LLVM metadata. */
#define DW_TAG_auto_variable 0x100
#define DW_TAG_arg_variable 0x101

/* SGI/MIPS Extension */
#define DW_TAG_MIPS_loop 0x4081
/* HP extensions */
#define DW_TAG_HP_array_descriptor 0x4090
/* GNU extensions */
#define DW_TAG_format_label 0x4101
#define DW_TAG_function_template 0x4102
#define DW_TAG_class_template 0x4103
#define DW_TAG_GNU_BINCL 0x4104
#define DW_TAG_GNU_EINCL 0x4105
/* UPC extensions */
#define DW_TAG_upc_shared_type 0x8765
#define DW_TAG_upc_strict_type 0x8766
#define DW_TAG_upc_relaxed_type 0x8767
/* PGI extensions */
#define DW_TAG_kanji_type 0xA000
#define DW_TAG_interface_block 0xA020

#define DW_TAG_lo_user 0x4080
#define DW_TAG_hi_user 0xffff

#define DW_children_no 0
#define DW_children_yes 1

#define DW_FORM_addr 0x01
#define DW_FORM_block2 0x03
#define DW_FORM_block4 0x04
#define DW_FORM_data2 0x05
#define DW_FORM_data4 0x06
#define DW_FORM_data8 0x07
#define DW_FORM_string 0x08
#define DW_FORM_block 0x09
#define DW_FORM_block1 0x0a
#define DW_FORM_data1 0x0b
#define DW_FORM_flag 0x0c
#define DW_FORM_sdata 0x0d
#define DW_FORM_strp 0x0e
#define DW_FORM_udata 0x0f
#define DW_FORM_ref_addr 0x10
#define DW_FORM_ref1 0x11
#define DW_FORM_ref2 0x12
#define DW_FORM_ref4 0x13
#define DW_FORM_ref8 0x14
#define DW_FORM_ref_udata 0x15
#define DW_FORM_indirect 0x16
#define DW_FORM_sec_offset 0x17   /* v. 4 */
#define DW_FORM_exprloc 0x18      /* v. 4 */
#define DW_FORM_flag_present 0x19 /* v. 4 */
#define DW_FORM_ref_sig8 0x20     /* v. 4 */

#define DW_AT_sibling 0x01
#define DW_AT_location 0x02
#define DW_AT_name 0x03
#define DW_AT_ordering 0x09
#define DW_AT_subscr_data 0x0a
#define DW_AT_byte_size 0x0b
#define DW_AT_bit_offset 0x0c
#define DW_AT_bit_size 0x0d
#define DW_AT_element_list 0x0f
#define DW_AT_stmt_list 0x10
#define DW_AT_low_pc 0x11
#define DW_AT_high_pc 0x12
#define DW_AT_language 0x13
#define DW_AT_member 0x14
#define DW_AT_discr 0x15
#define DW_AT_discr_value 0x16
#define DW_AT_visibility 0x17
#define DW_AT_import 0x18
#define DW_AT_string_length 0x19
#define DW_AT_common_reference 0x1a
#define DW_AT_comp_dir 0x1b
#define DW_AT_const_value 0x1c
#define DW_AT_containing_type 0x1d
#define DW_AT_default_value 0x1e
#define DW_AT_inline 0x20
#define DW_AT_is_optional 0x21
#define DW_AT_lower_bound 0x22
#define DW_AT_producer 0x25
#define DW_AT_prototyped 0x27
#define DW_AT_return_addr 0x2a
#define DW_AT_start_scope 0x2c
#define DW_AT_stride_size 0x2e
#define DW_AT_upper_bound 0x2f
#define DW_AT_abstract_origin 0x31
#define DW_AT_accessibility 0x32
#define DW_AT_address_class 0x33
#define DW_AT_artificial 0x34
#define DW_AT_base_types 0x35
#define DW_AT_calling_convention 0x36
#define DW_AT_count 0x37
#define DW_AT_data_member_location 0x38
#define DW_AT_decl_column 0x39
#define DW_AT_decl_file 0x3a
#define DW_AT_decl_line 0x3b
#define DW_AT_declaration 0x3c
#define DW_AT_discr_list 0x3d
#define DW_AT_encoding 0x3e
#define DW_AT_external 0x3f
#define DW_AT_frame_base 0x40
#define DW_AT_friend 0x41
#define DW_AT_identifier_case 0x42
#define DW_AT_macro_info 0x43
#define DW_AT_namelist_item 0x44
#define DW_AT_priority 0x45
#define DW_AT_segment 0x46
#define DW_AT_specification 0x47
#define DW_AT_static_link 0x48
#define DW_AT_type 0x49
#define DW_AT_use_location 0x4a
#define DW_AT_variable_parameter 0x4b
#define DW_AT_virtuality 0x4c
#define DW_AT_vtable_elem_location 0x4d
/* dwarf3 additions */
#define DW_AT_bit_stride 0x2e /* official name */
#define DW_AT_allocated 0x4e
#define DW_AT_associated 0x4f
#define DW_AT_data_location 0x50
#define DW_AT_stride 0x51
#define DW_AT_byte_stride 0x51 /* official name */
#define DW_AT_entry_pc 0x52
#define DW_AT_use_UTF8 0x53
#define DW_AT_extension 0x54
#define DW_AT_ranges 0x55
#define DW_AT_trampoline 0x56
#define DW_AT_call_column 0x57
#define DW_AT_call_file 0x58
#define DW_AT_call_line 0x59
#define DW_AT_description 0x5a
#define DW_AT_binary_scale 0x5b
#define DW_AT_decimal_scale 0x5c
#define DW_AT_small 0x5d
#define DW_AT_decimal_sign 0x5e
#define DW_AT_digit_count 0x5f
#define DW_AT_picture_string 0x60
#define DW_AT_mutable 0x61
#define DW_AT_threads_scaled 0x62
#define DW_AT_explicit 0x63
#define DW_AT_object_pointer 0x64
#define DW_AT_endianity 0x65
#define DW_AT_elemental 0x66
#define DW_AT_pure 0x67
#define DW_AT_recursive 0x68
/* SGI/MIPS extensions */
#define DW_AT_MIPS_fde 0x2001
#define DW_AT_MIPS_loop_begin 0x2002
#define DW_AT_MIPS_tail_loop_begin 0x2003
#define DW_AT_MIPS_epilog_begin 0x2004
#define DW_AT_MIPS_loop_unroll_factor 0x2005
#define DW_AT_MIPS_software_pipeline_depth 0x2006
#define DW_AT_MIPS_linkage_name 0x2007 /* supported by PGI */
#define DW_AT_MIPS_stride 0x2008
#define DW_AT_MIPS_abstract_name 0x2009
#define DW_AT_MIPS_clone_origin 0x200a
#define DW_AT_MIPS_has_inlines 0x200b
/* HP extensions */
#define DW_AT_HP_block_index 0x2000
#define DW_AT_HP_unmodifiable 0x2001
#define DW_AT_HP_actuals_stmt_list 0x2010
#define DW_AT_HP_proc_per_section 0x2011
#define DW_AT_HP_raw_data_ptr 0x2012
#define DW_AT_HP_pass_by_reference 0x2013
#define DW_AT_HP_opt_level 0x2014
#define DW_AT_HP_prof_version_id 0x2015
#define DW_AT_HP_opt_flags 0x2016
#define DW_AT_HP_cold_region_low_pc 0x2017
#define DW_AT_HP_cold_region_high_pc 0x2018
#define DW_AT_HP_all_variables_modifiable 0x2019
#define DW_AT_HP_linkage_name 0x201a
#define DW_AT_HP_prof_flags 0x201b
/* GNU extensions */
#define DW_AT_sf_names 0x2101
#define DW_AT_src_info 0x2102
#define DW_AT_mac_info 0x2103
#define DW_AT_src_coords 0x2104
#define DW_AT_body_begin 0x2105
#define DW_AT_body_end 0x2106
#define DW_AT_GNU_vector 0x2107
/* VMS extension */
#define DW_AT_VMS_rtnbeg_pd_address 0x2201
/* UPC extension */
#define DW_AT_upc_threads_scaled 0x3210
/* astplab extensions */
#define DW_AT_lbase 0x3a00
#define DW_AT_soffset 0x3a01
#define DW_AT_lstride 0x3a02
/* Apple extensions */
#define DW_AT_APPLE_optimized 0x3fe1
#define DW_AT_APPLE_flags 0x3fe2
#define DW_AT_APPLE_isa 0x3fe3
#define DW_AT_APPLE_block 0x3fe4
#define DW_AT_APPLE_major_runtime_vers 0x3fe5
#define DW_AT_APPLE_runtime_class 0x3fe6
#define DW_AT_APPLE_omit_frame_ptr 0x3fe7
#define DW_AT_APPLE_property_name 0x3fe8
#define DW_AT_APPLE_property_getter 0x3fe9
#define DW_AT_APPLE_property_setter 0x3fea
#define DW_AT_APPLE_property_attribute 0x3feb
#define DW_AT_APPLE_objc_complete_type 0x3fec
#define DW_AT_APPLE_property 0x3fed

#define DW_AT_lo_user 0x2000
#define DW_AT_hi_user 0x3fff

#define DW_OP_addr 0x03
#define DW_OP_deref 0x06
#define DW_OP_const1u 0x08
#define DW_OP_const1s 0x09
#define DW_OP_const2u 0x0a
#define DW_OP_const2s 0x0b
#define DW_OP_const4u 0x0c
#define DW_OP_const4s 0x0d
#define DW_OP_const8u 0x0e
#define DW_OP_const8s 0x0f
#define DW_OP_constu 0x10
#define DW_OP_consts 0x11
#define DW_OP_dup 0x12
#define DW_OP_drop 0x13
#define DW_OP_over 0x14
#define DW_OP_pick 0x15
#define DW_OP_swap 0x16
#define DW_OP_rot 0x17
#define DW_OP_xderef 0x18
#define DW_OP_abs 0x19
#define DW_OP_and 0x1a
#define DW_OP_div 0x1b
#define DW_OP_minus 0x1c
#define DW_OP_mod 0x1d
#define DW_OP_mul 0x1e
#define DW_OP_neg 0x1f
#define DW_OP_not 0x20
#define DW_OP_or 0x21
#define DW_OP_plus 0x22
#define DW_OP_plus_uconst 0x23
#define DW_OP_shl 0x24
#define DW_OP_shr 0x25
#define DW_OP_shra 0x26
#define DW_OP_xor 0x27
#define DW_OP_bra 0x28
#define DW_OP_eq 0x29
#define DW_OP_ge 0x2a
#define DW_OP_gt 0x2b
#define DW_OP_le 0x2c
#define DW_OP_lt 0x2d
#define DW_OP_ne 0x2e
#define DW_OP_skip 0x2f
#define DW_OP_lit0 0x30
#define DW_OP_lit1 0x31
#define DW_OP_lit2 0x32
#define DW_OP_lit3 0x33
#define DW_OP_lit4 0x34
#define DW_OP_lit5 0x35
#define DW_OP_lit6 0x36
#define DW_OP_lit7 0x37
#define DW_OP_lit8 0x38
#define DW_OP_lit9 0x39
#define DW_OP_lit10 0x3a
#define DW_OP_lit11 0x3b
#define DW_OP_lit12 0x3c
#define DW_OP_lit13 0x3d
#define DW_OP_lit14 0x3e
#define DW_OP_lit15 0x3f
#define DW_OP_lit16 0x40
#define DW_OP_lit17 0x41
#define DW_OP_lit18 0x42
#define DW_OP_lit19 0x43
#define DW_OP_lit20 0x44
#define DW_OP_lit21 0x45
#define DW_OP_lit22 0x46
#define DW_OP_lit23 0x47
#define DW_OP_lit24 0x48
#define DW_OP_lit25 0x49
#define DW_OP_lit26 0x4a
#define DW_OP_lit27 0x4b
#define DW_OP_lit28 0x4c
#define DW_OP_lit29 0x4d
#define DW_OP_lit30 0x4e
#define DW_OP_lit31 0x4f
#define DW_OP_reg0 0x50
#define DW_OP_reg1 0x51
#define DW_OP_reg2 0x52
#define DW_OP_reg3 0x53
#define DW_OP_reg4 0x54
#define DW_OP_reg5 0x55
#define DW_OP_reg6 0x56
#define DW_OP_reg7 0x57
#define DW_OP_reg8 0x58
#define DW_OP_reg9 0x59
#define DW_OP_reg10 0x5a
#define DW_OP_reg11 0x5b
#define DW_OP_reg12 0x5c
#define DW_OP_reg13 0x5d
#define DW_OP_reg14 0x5e
#define DW_OP_reg15 0x5f
#define DW_OP_reg16 0x60
#define DW_OP_reg17 0x61
#define DW_OP_reg18 0x62
#define DW_OP_reg19 0x63
#define DW_OP_reg20 0x64
#define DW_OP_reg21 0x65
#define DW_OP_reg22 0x66
#define DW_OP_reg23 0x67
#define DW_OP_reg24 0x68
#define DW_OP_reg25 0x69
#define DW_OP_reg26 0x6a
#define DW_OP_reg27 0x6b
#define DW_OP_reg28 0x6c
#define DW_OP_reg29 0x6d
#define DW_OP_reg30 0x6e
#define DW_OP_reg31 0x6f
#define DW_OP_breg0 0x70
#define DW_OP_breg1 0x71
#define DW_OP_breg2 0x72
#define DW_OP_breg3 0x73
#define DW_OP_breg4 0x74
#define DW_OP_breg5 0x75
#define DW_OP_breg6 0x76
#define DW_OP_breg7 0x77
#define DW_OP_breg8 0x78
#define DW_OP_breg9 0x79
#define DW_OP_breg10 0x7a
#define DW_OP_breg11 0x7b
#define DW_OP_breg12 0x7c
#define DW_OP_breg13 0x7d
#define DW_OP_breg14 0x7e
#define DW_OP_breg15 0x7f
#define DW_OP_breg16 0x80
#define DW_OP_breg17 0x81
#define DW_OP_breg18 0x82
#define DW_OP_breg19 0x83
#define DW_OP_breg20 0x84
#define DW_OP_breg21 0x85
#define DW_OP_breg22 0x86
#define DW_OP_breg23 0x87
#define DW_OP_breg24 0x88
#define DW_OP_breg25 0x89
#define DW_OP_breg26 0x8a
#define DW_OP_breg27 0x8b
#define DW_OP_breg28 0x8c
#define DW_OP_breg29 0x8d
#define DW_OP_breg30 0x8e
#define DW_OP_breg31 0x8f
#define DW_OP_regx 0x90
#define DW_OP_fbreg 0x91
#define DW_OP_bregx 0x92
#define DW_OP_piece 0x93
#define DW_OP_deref_size 0x94
#define DW_OP_xderef_size 0x95
#define DW_OP_nop 0x96
/* dwarf3 additions */
#define DW_OP_push_object_address 0x97
#define DW_OP_call2 0x98
#define DW_OP_call4 0x99
#define DW_OP_call_ref 0x9a
#define DW_OP_form_tls_address 0x9b
#define DW_OP_call_frame_cfa 0x9c
#define DW_OP_bit_piece 0x9d
/* dwarf4 additions */
#define DW_OP_implicit_value 0x9e
#define DW_OP_stack_value 0x9f
/* GNU extensions */
#define DW_OP_GNU_push_tls_address 0xe0
#define DW_OP_GNU_uninit 0xf0
/* HP extensions */
#define DW_OP_HP_unknown 0xe0
#define DW_OP_HP_is_value 0xe1
#define DW_OP_HP_fltconst4 0xe2
#define DW_OP_HP_fltconst8 0xe3
#define DW_OP_HP_mod_range 0xe4
#define DW_OP_HP_unmod_range 0xe5
#define DW_OP_HP_tls 0xe6
/* PGI extensions */
#define DW_OP_PGI_omp_thread_num 0xf8

#define DW_OP_lo_user 0xe0
#define DW_OP_hi_user 0xff

#define DW_ATE_address 0x1
#define DW_ATE_boolean 0x2
#define DW_ATE_complex_float 0x3
#define DW_ATE_float 0x4
#define DW_ATE_signed 0x5
#define DW_ATE_signed_char 0x6
#define DW_ATE_unsigned 0x7
#define DW_ATE_unsigned_char 0x8
/* dwarf3 additions */
#define DW_ATE_imaginary_float 0x09
#define DW_ATE_packed_decimal 0x0a
#define DW_ATE_numeric_string 0x0b
#define DW_ATE_edited 0x0c
#define DW_ATE_signed_fixed 0x0d
#define DW_ATE_unsigned_fixed 0x0e
#define DW_ATE_decimal_float 0x0f
/* HP extensions */
#define DW_ATE_HP_float80 0x80
#define DW_ATE_HP_complex_float80 0x81
#define DW_ATE_HP_float128 0x82
#define DW_ATE_HP_complex_float128 0x83
#define DW_ATE_HP_floathpintel 0x84
#define DW_ATE_HP_imaginary_float80 0x85
#define DW_ATE_HP_imaginary_float128 0x86

#define DW_ATE_lo_user 0x80
#define DW_ATE_hi_user 0xff

#define DW_ACCESS_public 1
#define DW_ACCESS_protected 2
#define DW_ACCESS_private 3

#define DW_VIS_local 1
#define DW_VIS_exported 2
#define DW_VIS_qualified 3

#define DW_VIRTUALITY_none 0
#define DW_VIRTUALITY_virtual 1
#define DW_VIRTUALITY_pure_virtual 2

#define DW_LANG_C89 0x0001
#define DW_LANG_C 0x0002
#define DW_LANG_Ada83 0x0003
#define DW_LANG_C_plus_plus 0x0004
#define DW_LANG_Cobol74 0x0005
#define DW_LANG_Cobol85 0x0006
#define DW_LANG_Fortran77 0x0007
#define DW_LANG_Fortran90 0x0008
#define DW_LANG_Pascal83 0x0009
#define DW_LANG_Modula2 0x000a
/* dwarf3 additions */
#define DW_LANG_Java 0x000b
#define DW_LANG_C99 0x000c
#define DW_LANG_Ada95 0x000d
#define DW_LANG_Fortran95 0x000e
#define DW_LANG_PLI 0x000f
#define DW_LANG_ObjC 0x0010
#define DW_LANG_ObjC_plus_plus 0x0011
#define DW_LANG_UPC 0x0012
#define DW_LANG_D 0x0013
/* MIPS extension */
#define DW_LANG_Mips_Assembler 0x8001
/* UPC extension */
#define DW_LANG_Upc 0x8765

#define DW_LANG_lo_user 0x8000
#define DW_LANG_hi_user 0xffff

#define DW_ID_case_sensitive 0
#define DW_ID_up_case 1
#define DW_ID_down_case 2
#define DW_ID_case_insensitive 3

#define DW_CC_normal 0x1
#define DW_CC_program 0x2
#define DW_CC_nocall 0x3
/* GNU extension */
#define DW_CC_GNU_renesas_sh 0x40

#define DW_CC_lo_user 0x40
#define DW_CC_hi_user 0xff

#define DW_INL_not_inlined 0
#define DW_INL_inlined 1
#define DW_INL_declared_not_inlined 2
#define DW_INL_declared_inlined 3

#define DW_ORD_row_major 0
#define DW_ORD_col_major 1

#define DW_DSC_label 0
#define DW_DSC_range 1

#define DW_LNS_copy 0x01
#define DW_LNS_advance_pc 0x02
#define DW_LNS_advance_line 0x03
#define DW_LNS_set_file 0x04
#define DW_LNS_set_column 0x05
#define DW_LNS_negate_stmt 0x06
#define DW_LNS_set_basic_block 0x07
#define DW_LNS_const_add_pc 0x08
#define DW_LNS_fixed_advance_pc 0x09
/* dwarf3 additions */
#define DW_LNS_set_prologue_end 0x0a
#define DW_LNS_set_epilogue_begin 0x0b
#define DW_LNS_set_isa 0x0c

#define DW_LNE_end_sequence 1
#define DW_LNE_set_address 2
#define DW_LNE_define_file 3
/* dwarf4 extended op */
#define DW_LNE_set_discriminator 4

/* HP extensions */
#define DW_LNE_HP_negate_is_UV_update 0x11
#define DW_LNE_HP_push_context 0x12
#define DW_LNE_HP_pop_context 0x13
#define DW_LNE_HP_set_file_line_column 0x14
#define DW_LNE_HP_set_routine_name 0x15
#define DW_LNE_HP_set_sequence 0x16
#define DW_LNE_HP_negate_post_semantics 0x17
#define DW_LNE_HP_negate_function_exit 0x18
#define DW_LNE_HP_negate_front_end_logical 0x19
#define DW_LNE_HP_define_proc 0x20

#define DW_LNE_lo_user 0x80
#define DW_LNE_hi_user 0xff

#define DW_MACINFO_define 1
#define DW_MACINFO_undef 2
#define DW_MACINFO_start_file 3
#define DW_MACINFO_end_file 4
#define DW_MACINFO_vendor_ext 255

#define DW_CFA_advance_loc 0x40
#define DW_CFA_offset 0x80
#define DW_CFA_restore 0xc0
#define DW_CFA_extended 0

#define DW_CFA_nop 0x00
#define DW_CFA_set_loc 0x01
#define DW_CFA_advance_loc1 0x02
#define DW_CFA_advance_loc2 0x03
#define DW_CFA_advance_loc4 0x04
#define DW_CFA_offset_extended 0x05
#define DW_CFA_restore_extended 0x06
#define DW_CFA_undefined 0x07
#define DW_CFA_same_value 0x08
#define DW_CFA_register 0x09
#define DW_CFA_remember_state 0x0a
#define DW_CFA_restore_state 0x0b
#define DW_CFA_def_cfa 0x0c
#define DW_CFA_def_cfa_register 0x0d
#define DW_CFA_def_cfa_offset 0x0e
/* dwarf3 additions */
#define DW_CFA_def_cfa_expression 0x0f
#define DW_CFA_expression 0x10
#define DW_CFA_offset_extended_sf 0x11
#define DW_CFA_def_cfa_sf 0x12
#define DW_CFA_def_cfa_offset_sf 0x13
#define DW_CFA_val_offset 0x14
#define DW_CFA_val_offset_sf 0x15
#define DW_CFA_val_expression 0x16
/* SGI/MIPS extension */
#define DW_CFA_MIPS_advance_loc8 0x1d
/* GNU extensions */
#define DW_CFA_GNU_window_save 0x2d
#define DW_CFA_GNU_args_size 0x2e
#define DW_CFA_GNU_negative_offset_extended 0x2f

#define DW_CFA_low_user 0x1c
#define DW_CFA_high_user 0x3f

/* Mapping from machine registers and pseudo-regs into the .debug_frame table.
   DW_FRAME entries are machine specific.
*/

#define DW_CHILDREN_no 0x00
#define DW_CHILDREN_yes 0x01

#define DW_ADDR_none 0

/* for pgc++ only, not seen in actual output */
/* use this tag to define nameless structs, etc in our debug file only */
#define DW_TAG_nameless_struct (0x0050)
#define DW_TAG_nameless_union (0x0051)
#define DW_TAG_nameless_class (0x0052)
#define DW_TAG_nameless_enumeration (0x0053)
#define DW_TAG_include_table (0x0054)
/* don't  use this in actual output, but needed for throw lists */
#define DW_TAG_void_type (0x0055)
/* end for pgc++ only, not seen in actual output */

/* dwarf_i.c */
void input_dwarf_info(char *filename);
void dwarf_get_fn(char *filename);


#ifdef GET_DIAG_ARRAYS
static const int16_t DiagArrays[] = {
  /* Empty */ -1,
  /* DiagArray1 */ diag::warn_pragma_message, -1,
  /* DiagArray2 */ diag::pp_hash_warning, -1,
  /* DiagArray3 */ diag::warn_cfstring_truncated, -1,
  /* DiagArray5 */ diag::warn_independentclass_attribute, diag::warn_ptr_independentclass_attribute, -1,
  /* DiagArray6 */ diag::warn_nsobject_attribute, -1,
  /* DiagArray7 */ diag::warn_function_always_inline_attribute_mismatch, diag::warn_sme_locally_streaming_has_vl_args_returns, diag::warn_sme_streaming_pass_return_vl_to_non_streaming, -1,
  /* DiagArray9 */ diag::warn_abs_too_small, diag::warn_pointer_abs, diag::warn_unsigned_abs, diag::warn_wrong_absolute_value_type, -1,
  /* DiagArray10 */ diag::warn_abstract_final_class, -1,
  /* DiagArray11 */ diag::warn_abstract_vbase_init_ignored, -1,
  /* DiagArray13 */ diag::warn_taking_address_of_packed_member, -1,
  /* DiagArray14 */ diag::ext_typecheck_addrof_temporary, -1,
  /* DiagArray16 */ diag::warn_not_xl_compatible, diag::warn_pragma_align_not_xl_compatible, -1,
  /* DiagArray17 */ diag::ext_alias_template_in_declarative_nns, -1,
  /* DiagArray18 */ diag::warn_param_mismatched_alignment, -1,
  /* DiagArray20 */ diag::warn_alloca, -1,
  /* DiagArray21 */ diag::warn_alloca_align_alignof, -1,
  /* DiagArray22 */ diag::warn_always_inline_coroutine, -1,
  /* DiagArray23 */ diag::warn_ambiguous_suitable_delete_function_found, -1,
  /* DiagArray24 */ diag::warn_misplaced_ellipsis_vararg, -1,
  /* DiagArray25 */ diag::warn_pp_ambiguous_macro, -1,
  /* DiagArray26 */ diag::ext_nested_name_member_ref_lookup_ambiguous, -1,
  /* DiagArray27 */ diag::ext_ovl_ambiguous_oper_binary_reversed, -1,
  /* DiagArray28 */ diag::warn_incompatible_analyzer_plugin_api, -1,
  /* DiagArray29 */ diag::warn_android_unversioned_fallback, -1,
  /* DiagArray30 */ diag::warn_arith_conv_mixed_anon_enum_types, -1,
  /* DiagArray31 */ diag::ext_abstract_pack_declarator_parens, -1,
  /* DiagArray32 */ diag::warn_apinotes_message, -1,
  /* DiagArray34 */ diag::warn_arc_bridge_cast_nonarc, -1,
  /* DiagArray35 */ diag::warn_arc_possible_repeated_use_of_weak, -1,
  /* DiagArray36 */ diag::warn_arc_object_memaccess, -1,
  /* DiagArray37 */ diag::warn_arc_perform_selector_leaks, -1,
  /* DiagArray38 */ diag::warn_arc_repeated_use_of_weak, -1,
  /* DiagArray39 */ diag::warn_arc_retain_cycle, -1,
  /* DiagArray40 */ diag::warn_arc_literal_assign, diag::warn_arc_retained_assign, diag::warn_arc_retained_property_assign, -1,
  /* DiagArray41 */ diag::warn_argument_invalid_range, -1,
  /* DiagArray42 */ diag::warn_argument_undefined_behaviour, -1,
  /* DiagArray43 */ diag::warn_arm_interrupt_save_fp_without_vfp_unit, -1,
  /* DiagArray44 */ diag::warn_arm_interrupt_vfp_clobber, -1,
  /* DiagArray45 */ diag::warn_array_index_exceeds_bounds, diag::warn_array_index_exceeds_max_addressable_bounds, diag::warn_array_index_precedes_bounds, diag::warn_ptr_arith_exceeds_max_addressable_bounds, diag::warn_static_array_too_small, -1,
  /* DiagArray46 */ diag::warn_ptr_arith_exceeds_bounds, diag::warn_ptr_arith_precedes_bounds, -1,
  /* DiagArray47 */ diag::warn_array_comparison, -1,
  /* DiagArray48 */ diag::warn_array_comparison_cxx26, -1,
  /* DiagArray49 */ diag::warn_inconsistent_array_form, -1,
  /* DiagArray51 */ diag::warn_asm_mismatched_size_modifier, -1,
  /* DiagArray52 */ diag::warn_not_in_enum_assignment, -1,
  /* DiagArray53 */ diag::warn_assume_side_effects, -1,
  /* DiagArray55 */ diag::warn_atimport_in_framework_header, -1,
  /* DiagArray56 */ diag::warn_atomic_member_access, -1,
  /* DiagArray57 */ diag::warn_atomic_op_misaligned, diag::warn_atomic_op_oversized, -1,
  /* DiagArray58 */ diag::warn_atomic_implicit_seq_cst, -1,
  /* DiagArray59 */ diag::warn_atomic_op_has_invalid_memory_order, -1,
  /* DiagArray61 */ diag::warn_atomic_property_rule, -1,
  /* DiagArray62 */ diag::warn_attribute_packed_for_bitfield, -1,
  /* DiagArray63 */ diag::warn_attr_min_eq_max, -1,
  /* DiagArray64 */ diag::warn_fe_backend_warning_attr, diag::warn_toc_unsupported_type, -1,
  /* DiagArray66 */ diag::ext_c23_auto_non_plain_identifier, -1,
  /* DiagArray67 */ diag::warn_drv_disabling_vptr_no_rtti_default, -1,
  /* DiagArray69 */ diag::ext_auto_storage_class, -1,
  /* DiagArray70 */ diag::warn_auto_var_is_id, -1,
  /* DiagArray71 */ diag::warn_availability_and_unavailable, diag::warn_availability_fuchsia_unavailable_minor, diag::warn_availability_on_static_initializer, diag::warn_availability_swift_unavailable_deprecated_only, diag::warn_availability_unknown_environment, diag::warn_availability_unknown_platform, diag::warn_availability_version_ordering, diag::warn_expected_consistent_version_separator, diag::warn_mismatched_availability, diag::warn_mismatched_availability_override, diag::warn_mismatched_availability_override_unavail, -1,
  /* DiagArray72 */ diag::warn_drv_avr_family_linking_stdlibs_not_implemented, diag::warn_drv_avr_libc_not_found, diag::warn_drv_avr_linker_section_addresses_not_implemented, diag::warn_drv_avr_mcu_not_specified, diag::warn_drv_avr_stdlib_not_linked, -1,
  /* DiagArray73 */ diag::warn_fe_backend_plugin, diag::warn_fe_backend_resource_limit, -1,
  /* DiagArray74 */ diag::backslash_newline_space, -1,
  /* DiagArray75 */ diag::warn_bad_function_cast, -1,
  /* DiagArray77 */ diag::ext_rvalue_to_reference_access_ctor, diag::ext_rvalue_to_reference_temp_copy_no_viable, -1,
  /* DiagArray78 */ diag::ext_bit_int, diag::ext_cxx_bitint_suffix, -1,
  /* DiagArray79 */ diag::warn_impcast_bitfield_precision_constant, -1,
  /* DiagArray80 */ diag::warn_bitfield_too_small_for_enum, diag::warn_signed_bitfield_enum_conversion, diag::warn_unsigned_bitfield_assigned_signed_enum, -1,
  /* DiagArray81 */ diag::warn_bitfield_width_exceeds_type_width, -1,
  /* DiagArray82 */ diag::warn_precedence_bitwise_conditional, -1,
  /* DiagArray83 */ diag::warn_bitwise_instead_of_logical, -1,
  /* DiagArray84 */ diag::warn_bitwise_op_in_bitwise_op, -1,
  /* DiagArray85 */ diag::warn_block_capture_autoreleasing, -1,
  /* DiagArray86 */ diag::warn_impcast_bool_to_null_pointer, -1,
  /* DiagArray88 */ diag::warn_bitwise_negation_bool, -1,
  /* DiagArray89 */ diag::warn_counted_by_attr_elt_type_unknown_size, -1,
  /* DiagArray90 */ diag::warn_braces_around_init, -1,
  /* DiagArray91 */ diag::warn_incompatible_branch_protection_option, diag::warn_target_unsupported_branch_protection_attribute, diag::warn_unsupported_branch_protection, diag::warn_unsupported_branch_protection_spec, -1,
  /* DiagArray92 */ diag::warn_objc_invalid_bridge, diag::warn_objc_invalid_bridge_to_cf, -1,
  /* DiagArray93 */ diag::warn_assume_aligned_too_great, -1,
  /* DiagArray94 */ diag::ext_pp_redef_builtin_macro, diag::ext_pp_undef_builtin_macro, -1,
  /* DiagArray95 */ diag::warn_builtin_chk_overflow, -1,
  /* DiagArray96 */ diag::warn_implicit_decl_requires_sysheader, -1,
  /* DiagArray97 */ diag::warn_zero_size_struct_union_compat, -1,
  /* DiagArray101 */ diag::ext_old_implicitly_unsigned_long_cxx, diag::warn_auto_storage_class, diag::warn_cxx11_compat_user_defined_literal, diag::warn_cxx11_keyword, diag::warn_cxx11_right_shift_in_template_arg, diag::warn_explicit_instantiation_inline_0x, diag::warn_explicit_instantiation_must_be_global_0x, diag::warn_explicit_instantiation_out_of_scope_0x, diag::warn_explicit_instantiation_unqualified_wrong_namespace_0x, diag::warn_old_implicitly_unsigned_long_cxx, -1,
  /* DiagArray102 */ diag::warn_deprecated_string_literal_conversion, -1,
  /* DiagArray104 */ diag::warn_cxx11_compat_reserved_user_defined_literal, -1,
  /* DiagArray105 */ diag::compat_pre_cxx11_enum_fixed_underlying_type, diag::compat_pre_cxx11_nonclass_type_friend, diag::compat_pre_cxx11_static_data_member_in_union, diag::compat_pre_cxx11_templ_default_in_function_templ, diag::compat_pre_cxx11_template_arg_extra_parens, diag::compat_pre_cxx11_typename_outside_of_template, diag::ext_alias_declaration, diag::ext_array_size_conversion, diag::ext_auto_type_specifier, diag::ext_defaulted_deleted_function, diag::ext_enumerator_list_comma_cxx, diag::ext_explicit_conversion_functions, diag::ext_extern_template, diag::ext_for_range, diag::ext_generalized_initializer_lists, diag::ext_lambda, diag::ext_nested_name_spec_is_enum, diag::ext_nonstatic_member_init, diag::ext_override_control_keyword, diag::ext_ref_qualifier, diag::ext_rvalue_reference, diag::ext_scoped_enum, diag::ext_template_arg_object_internal, diag::ext_template_outside_of_template, diag::ext_unelaborated_friend_type, diag::ext_variadic_templates, diag::warn_ext_cxx11_attributes, -1,
  /* DiagArray106 */ diag::ext_extra_semi_cxx11, -1,
  /* DiagArray107 */ diag::ext_inline_namespace, -1,
  /* DiagArray108 */ diag::ext_cxx11_longlong, -1,
  /* DiagArray109 */ diag::ext_cce_narrowing, diag::ext_init_list_constant_narrowing, diag::ext_init_list_type_narrowing, diag::ext_init_list_variable_narrowing, diag::warn_init_list_constant_narrowing, diag::warn_init_list_type_narrowing, diag::warn_init_list_variable_narrowing, -1,
  /* DiagArray110 */ diag::ext_init_list_constant_narrowing_const_reference, diag::ext_init_list_type_narrowing_const_reference, diag::ext_init_list_variable_narrowing_const_reference, -1,
  /* DiagArray111 */ diag::ext_cxx14_attr, -1,
  /* DiagArray112 */ diag::ext_binary_literal_cxx14, -1,
  /* DiagArray115 */ diag::compat_pre_cxx14_constexpr_body_multiple_return, diag::compat_pre_cxx14_constexpr_local_var, diag::compat_pre_cxx14_constexpr_type_definition, diag::compat_pre_cxx14_cxx14_constexpr_body_invalid_stmt, diag::compat_pre_cxx14_variable_template, diag::ext_decltype_auto_type_specifier, diag::ext_init_capture, -1,
  /* DiagArray116 */ diag::ext_cxx17_attr, -1,
  /* DiagArray118 */ diag::warn_cxx17_compat_exception_spec_in_signature, -1,
  /* DiagArray120 */ diag::compat_pre_cxx17_decomp_decl, diag::compat_pre_cxx17_inline_variable, diag::ext_auto_new_list_init, diag::ext_constexpr_if, diag::ext_constexpr_on_lambda_cxx17, diag::ext_cxx_static_assert_no_message, diag::ext_fold_expression, diag::ext_for_range_begin_end_types_differ, diag::ext_hex_literal_invalid, diag::ext_init_statement, diag::ext_multi_using_declaration, diag::ext_nested_namespace_definition, diag::ext_ns_enum_attribute, diag::ext_star_this_lambda_capture_cxx17, diag::ext_template_template_param_typename, diag::ext_using_attribute_ns, diag::ext_using_declaration_pack, -1,
  /* DiagArray125 */ diag::ext_cxx20_attr, -1,
  /* DiagArray126 */ diag::warn_cxx17_compat_implicit_typename, diag::warn_cxx20_compat_aggregate_init_with_ctors, diag::warn_cxx20_compat_consteval, diag::warn_cxx20_compat_constinit, diag::warn_cxx20_compat_explicit_bool, diag::warn_cxx20_compat_requires_explicit_init_non_aggregate, diag::warn_cxx20_compat_spaceship, diag::warn_cxx20_compat_use_of_unaddressable_function, diag::warn_cxx20_compat_utf8_string, diag::warn_cxx20_keyword, -1,
  /* DiagArray128 */ diag::ext_cxx_designated_init, -1,
  /* DiagArray129 */ diag::compat_pre_cxx20_adl_only_template_id, diag::compat_pre_cxx20_constexpr_ctor_missing_init, diag::compat_pre_cxx20_constexpr_function_try_block, diag::compat_pre_cxx20_constexpr_local_var_no_init, diag::compat_pre_cxx20_constexpr_union_ctor_no_init, diag::compat_pre_cxx20_ctad_for_alias_templates, diag::compat_pre_cxx20_cxx20_constexpr_body_invalid_stmt, diag::compat_pre_cxx20_decomp_decl_spec, diag::ext_bitfield_member_init, diag::ext_capture_binding, diag::ext_defaulted_comparison, diag::ext_equals_this_lambda_capture_cxx20, diag::ext_explicit_bool, diag::ext_for_range_init_stmt, diag::ext_implicit_typename, diag::ext_init_capture_pack, diag::ext_inline_nested_namespace_definition, diag::ext_lambda_template_parameter_list, diag::ext_pointer_to_const_ref_member_on_rvalue, diag::ext_using_decl_scoped_enumerator, diag::ext_using_enum_declaration, diag::warn_cxx17_compat_aggregate_init_paren_list, -1,
  /* DiagArray130 */ diag::ext_cxx23_attr, -1,
  /* DiagArray132 */ diag::compat_pre_cxx23_constexpr_static_var, diag::compat_pre_cxx23_cxx23_constexpr_body_invalid_stmt, diag::err_static_lambda, diag::ext_alias_in_init_statement, diag::ext_consteval_if, diag::ext_cxx23_pp_directive, diag::ext_cxx23_size_t_suffix, diag::ext_cxx_label_end_of_compound_statement, diag::ext_lambda_missing_parens, diag::ext_operator_overload_static, -1,
  /* DiagArray133 */ diag::ext_decl_attrs_on_lambda, -1,
  /* DiagArray134 */ diag::compat_pre_cxx26_decomp_decl_cond, diag::ext_cxx26_raw_string_literal_character_set, diag::ext_cxx26_type_aware_allocators, diag::ext_cxx_binding_pack, diag::ext_cxx_static_assert_user_generated_message, diag::ext_decl_attrs_on_binding, diag::ext_delete_with_message, diag::ext_pack_indexing, diag::ext_placeholder_var_definition, diag::ext_variadic_friends, -1,
  /* DiagArray146 */ diag::warn_cxx11_compat_binary_literal, -1,
  /* DiagArray148 */ diag::compat_cxx11_enum_fixed_underlying_type, diag::compat_cxx11_nonclass_type_friend, diag::compat_cxx11_static_data_member_in_union, diag::compat_cxx11_templ_default_in_function_templ, diag::compat_cxx11_template_arg_extra_parens, diag::compat_cxx11_typename_outside_of_template, diag::warn_cxx98_compat_alias_declaration, diag::warn_cxx98_compat_alignas, diag::warn_cxx98_compat_alignof, diag::warn_cxx98_compat_attribute, diag::warn_cxx98_compat_auto_type_specifier, diag::warn_cxx98_compat_constexpr, diag::warn_cxx98_compat_ctor_list_init, diag::warn_cxx98_compat_decltype, diag::warn_cxx98_compat_defaulted_deleted_function, diag::warn_cxx98_compat_delegating_ctor, diag::warn_cxx98_compat_empty_scalar_initializer, diag::warn_cxx98_compat_empty_sizeless_initializer, diag::warn_cxx98_compat_enum_nested_name_spec, diag::warn_cxx98_compat_explicit_conversion_functions, diag::warn_cxx98_compat_for_range, diag::warn_cxx98_compat_friend_is_member, diag::warn_cxx98_compat_generalized_initializer_lists, diag::warn_cxx98_compat_goto_into_protected_scope, diag::warn_cxx98_compat_indirect_goto_in_protected_scope, diag::warn_cxx98_compat_initializer_list_init, diag::warn_cxx98_compat_inline_namespace, diag::warn_cxx98_compat_lambda, diag::warn_cxx98_compat_less_colon_colon, diag::warn_cxx98_compat_literal_operator, diag::warn_cxx98_compat_literal_ucn_control_character, diag::warn_cxx98_compat_literal_ucn_escape_basic_scs, diag::warn_cxx98_compat_noexcept_decl, diag::warn_cxx98_compat_noexcept_expr, diag::warn_cxx98_compat_non_static_member_use, diag::warn_cxx98_compat_nonstatic_member_init, diag::warn_cxx98_compat_nontrivial_union_or_anon_struct_member, diag::warn_cxx98_compat_nullptr, diag::warn_cxx98_compat_override_control_keyword, diag::warn_cxx98_compat_pass_non_pod_arg_to_vararg, diag::warn_cxx98_compat_raw_string_literal, diag::warn_cxx98_compat_ref_qualifier, diag::warn_cxx98_compat_reference_list_init, diag::warn_cxx98_compat_rvalue_reference, diag::warn_cxx98_compat_scoped_enum, diag::warn_cxx98_compat_sfinae_access_control, diag::warn_cxx98_compat_static_assert, diag::warn_cxx98_compat_switch_into_protected_scope, diag::warn_cxx98_compat_template_arg_null, diag::warn_cxx98_compat_template_arg_object_internal, diag::warn_cxx98_compat_template_outside_of_template, diag::warn_cxx98_compat_trailing_return_type, diag::warn_cxx98_compat_two_right_angle_brackets, diag::warn_cxx98_compat_unelaborated_friend_type, diag::warn_cxx98_compat_unicode_literal, diag::warn_cxx98_compat_unicode_type, diag::warn_cxx98_compat_using_decl_constructor, diag::warn_cxx98_compat_variadic_templates, -1,
  /* DiagArray149 */ diag::warn_cxx98_compat_temp_copy, -1,
  /* DiagArray150 */ diag::warn_cxx98_compat_top_level_semi, -1,
  /* DiagArray151 */ diag::warn_cxx98_compat_template_arg_local_type, -1,
  /* DiagArray152 */ diag::warn_cxx98_compat_array_size_conversion, diag::warn_cxx98_compat_cast_fn_obj, diag::warn_cxx98_compat_empty_fnmacro_arg, diag::warn_cxx98_compat_enumerator_list_comma, diag::warn_cxx98_compat_extern_template, diag::warn_cxx98_compat_longlong, diag::warn_cxx98_compat_pp_line_too_big, diag::warn_cxx98_compat_variadic_macro, -1,
  /* DiagArray153 */ diag::warn_cxx98_compat_template_arg_unnamed_type, -1,
  /* DiagArray154 */ diag::ext_anonymous_union, diag::ext_c11_anonymous_struct, diag::ext_c11_feature, diag::ext_typecheck_compare_complete_incomplete_pointers, -1,
  /* DiagArray155 */ diag::warn_c23_compat_utf8_string, diag::warn_c23_keyword, -1,
  /* DiagArray156 */ diag::ext_binary_literal, diag::ext_c23_bitint_suffix, diag::ext_c23_enum_fixed_underlying_type, diag::ext_c23_enum_value_not_int, diag::ext_c23_non_local_variable_decl_in_for, diag::ext_c23_non_variable_decl_in_for, diag::ext_c23_pp_directive, diag::ext_c_empty_initializer, diag::ext_c_label_end_of_compound_statement, diag::ext_c_label_followed_by_declaration, diag::ext_c_nullptr, diag::ext_c_static_assert_no_message, diag::ext_parameter_name_omitted_c23, diag::ext_pp_embed_directive, diag::ext_restrict_on_array_of_pointers_c23, diag::ext_type_defined_in_offsetof, diag::warn_ext_c23_attributes, -1,
  /* DiagArray159 */ diag::ext_assoc_type_incomplete, diag::ext_c2y_alignof_incomplete_array, diag::ext_c2y_case_range, diag::ext_c2y_feature, diag::ext_c2y_generic_with_type_arg, diag::ext_c2y_imaginary_constant, diag::ext_c2y_increment_complex, diag::ext_octal_literal, -1,
  /* DiagArray160 */ diag::warn_c99_compat_unicode_id, diag::warn_c99_compat_unicode_literal, diag::warn_c99_keyword, diag::warn_old_implicitly_unsigned_long, -1,
  /* DiagArray161 */ diag::ext_designated_init, diag::ext_designated_init_array, diag::ext_designated_init_brace_elision, diag::ext_designated_init_mixed, diag::ext_designated_init_nested, -1,
  /* DiagArray162 */ diag::ext_aggregate_init_not_constant, diag::ext_c99_array_usage, diag::ext_c99_compound_literal, diag::ext_c99_feature, diag::ext_c99_flexible_array_member, diag::ext_c99_variable_decl_in_for_loop, diag::ext_c99_whitespace_required_after_macro_name, diag::ext_empty_fnmacro_arg, diag::ext_enumerator_list_comma_c, diag::ext_hex_constant_invalid, -1,
  /* DiagArray163 */ diag::warn_call_to_pure_virtual_member_function_from_ctor_dtor, -1,
  /* DiagArray164 */ diag::warn_called_once_gets_called_twice, diag::warn_called_once_never_called, diag::warn_called_once_never_called_when, -1,
  /* DiagArray165 */ diag::warn_cast_align, -1,
  /* DiagArray166 */ diag::warn_cast_calling_conv, -1,
  /* DiagArray168 */ diag::warn_cast_function_type, -1,
  /* DiagArray169 */ diag::warn_cast_function_type_strict, -1,
  /* DiagArray170 */ diag::warn_cast_pointer_from_sel, -1,
  /* DiagArray171 */ diag::warn_cast_qual, diag::warn_cast_qual2, -1,
  /* DiagArray172 */ diag::ext_bad_cxx_cast_qualifiers_away_incoherent, -1,
  /* DiagArray174 */ diag::warn_omp_section_is_char, diag::warn_subscript_is_char, -1,
  /* DiagArray175 */ diag::warn_drv_yc_multiple_inputs_clang_cl, diag::warn_drv_ycyu_different_arg_clang_cl, diag::warn_pp_hdrstop_filename_ignored, diag::warn_pp_macro_def_mismatch_with_pch, -1,
  /* DiagArray176 */ diag::warn_conv_to_base_not_used, diag::warn_conv_to_self_not_used, diag::warn_conv_to_void_not_used, -1,
  /* DiagArray177 */ diag::warn_pass_class_arg_to_vararg, -1,
  /* DiagArray178 */ diag::warn_cmse_nonsecure_union, -1,
  /* DiagArray179 */ diag::warn_comma_operator, -1,
  /* DiagArray180 */ diag::escaped_newline_block_comment_end, diag::ext_line_comment, diag::ext_multi_line_line_comment, diag::warn_nested_block_comment, -1,
  /* DiagArray182 */ diag::ext_typecheck_comparison_of_distinct_pointers, -1,
  /* DiagArray183 */ diag::warn_completion_handler_called_twice, diag::warn_completion_handler_never_called, diag::warn_completion_handler_never_called_when, -1,
  /* DiagArray184 */ diag::ext_complex_component_init, -1,
  /* DiagArray186 */ diag::warn_compound_token_split_by_macro, -1,
  /* DiagArray187 */ diag::warn_compound_token_split_by_whitespace, -1,
  /* DiagArray188 */ diag::ext_typecheck_cond_pointer_integer_mismatch, -1,
  /* DiagArray189 */ diag::warn_maybe_uninit_var, -1,
  /* DiagArray190 */ diag::warn_module_config_macro_undef, -1,
  /* DiagArray191 */ diag::warn_impcast_integer_precision_constant, -1,
  /* DiagArray192 */ diag::warn_is_constant_evaluated_always_true_constexpr, -1,
  /* DiagArray193 */ diag::warn_logical_instead_of_bitwise, -1,
  /* DiagArray194 */ diag::warn_cxx14_compat_constexpr_not_const, -1,
  /* DiagArray195 */ diag::warn_attr_on_unconsumable_class, diag::warn_loop_state_mismatch, diag::warn_param_return_typestate_mismatch, diag::warn_param_typestate_mismatch, diag::warn_return_typestate_for_unconsumable_type, diag::warn_return_typestate_mismatch, diag::warn_use_in_invalid_state, diag::warn_use_of_temp_in_invalid_state, -1,
  /* DiagArray196 */ diag::warn_hlsl_impcast_vector_truncation, diag::warn_impcast_complex_scalar, diag::warn_impcast_vector_scalar, diag::warn_opencl_generic_address_space_arg, diag::warn_template_arg_negative, diag::warn_template_arg_too_large, -1,
  /* DiagArray198 */ diag::warn_non_aligned_allocation_function, -1,
  /* DiagArray199 */ diag::warn_coroutine_type_aware_allocator_ignored, -1,
  /* DiagArray200 */ diag::warn_coroutine_handle_address_invalid_return_type, -1,
  /* DiagArray201 */ diag::warn_coroutine_promise_unhandled_exception_required_with_exceptions, -1,
  /* DiagArray202 */ diag::warn_unreachable_default, -1,
  /* DiagArray204 */ diag::warn_objc_cdirective_format_string, -1,
  /* DiagArray205 */ diag::warn_ctad_maybe_unsupported, -1,
  /* DiagArray207 */ diag::warn_ctu_incompat_triple, -1,
  /* DiagArray208 */ diag::warn_attribute_argument_n_negative, diag::warn_cuda_attr_lambda_position, diag::warn_kern_is_inline, diag::warn_kern_is_method, diag::warn_pragma_unroll_cuda_value_in_parens, -1,
  /* DiagArray209 */ diag::warn_default_atomic_custom_getter_setter, -1,
  /* DiagArray210 */ diag::ext_cxx11_attr_placement, -1,
  /* DiagArray211 */ diag::warn_dangling_variable, -1,
  /* DiagArray212 */ diag::warn_dangling_pointer_assignment, -1,
  /* DiagArray213 */ diag::warn_dangling_lifetime_pointer_assignment, -1,
  /* DiagArray214 */ diag::warn_dangling_reference_captured, diag::warn_dangling_reference_captured_by_unknown, -1,
  /* DiagArray215 */ diag::warn_dangling_else, -1,
  /* DiagArray216 */ diag::warn_bind_ref_member_to_parameter, diag::warn_dangling_member, diag::warn_init_ptr_member_to_parameter_addr, diag::warn_new_dangling_reference, -1,
  /* DiagArray217 */ diag::warn_dangling_lifetime_pointer, diag::warn_dangling_lifetime_pointer_member, -1,
  /* DiagArray218 */ diag::warn_new_dangling_initializer_list, -1,
  /* DiagArray219 */ diag::warn_drv_darwin_sdk_invalid_settings, -1,
  /* DiagArray220 */ diag::warn_pp_date_time, -1,
  /* DiagArray221 */ diag::warn_dealloc_in_category, -1,
  /* DiagArray222 */ diag::warn_debug_compression_unavailable, -1,
  /* DiagArray223 */ diag::ext_mixed_decls_code, diag::warn_mixed_decls_code, -1,
  /* DiagArray224 */ diag::warn_decls_in_multiple_modules, -1,
  /* DiagArray225 */ diag::warn_defaulted_comparison_deleted, diag::warn_defaulted_method_deleted, -1,
  /* DiagArray226 */ diag::warn_drv_delayed_template_parsing_after_cxx20, -1,
  /* DiagArray227 */ diag::warn_delegating_ctor_cycle, -1,
  /* DiagArray228 */ diag::warn_delete_abstract_non_virtual_dtor, -1,
  /* DiagArray229 */ diag::ext_delete_void_ptr_operand, diag::warn_delete_incomplete, -1,
  /* DiagArray230 */ diag::warn_delete_non_virtual_dtor, -1,
  /* DiagArray232 */ diag::ext_delimited_escape_sequence, -1,
  /* DiagArray233 */ diag::warn_deprecated_lax_vec_conv_all, -1,
  /* DiagArray234 */ diag::warn_O4_is_O3, diag::warn_access_decl_deprecated, diag::warn_drv_deprecated_arg, diag::warn_drv_deprecated_custom, diag::warn_drv_treating_input_as_cxx, diag::warn_omp_depend_in_ordered_deprecated, diag::warn_omp_minus_in_reduction_deprecated, diag::warn_option_invalid_ocl_version, diag::warn_vector_long_decl_spec_combination, -1,
  /* DiagArray235 */ diag::warn_deprecated_altivec_src_compat, -1,
  /* DiagArray236 */ diag::warn_arith_conv_mixed_anon_enum_types_cxx20, -1,
  /* DiagArray237 */ diag::warn_depr_array_comparison, -1,
  /* DiagArray238 */ diag::warn_deprecated_noreturn_spelling, diag::warn_type_attribute_deprecated_on_decl, diag::warn_vector_mode_deprecated, -1,
  /* DiagArray239 */ diag::warn_deprecated_builtin, -1,
  /* DiagArray240 */ diag::warn_deprecated_comma_subscript, -1,
  /* DiagArray241 */ diag::warn_deprecated_copy, -1,
  /* DiagArray243 */ diag::warn_deprecated_copy_with_dtor, -1,
  /* DiagArray244 */ diag::warn_deprecated_copy_with_user_provided_copy, -1,
  /* DiagArray245 */ diag::warn_deprecated_copy_with_user_provided_dtor, -1,
  /* DiagArray246 */ diag::warn_deprecated_for_co_await, -1,
  /* DiagArray247 */ diag::warn_atl_uuid_deprecated, diag::warn_cstyle_param, diag::warn_deprecated, diag::warn_deprecated_fwdclass_message, diag::warn_deprecated_message, diag::warn_property_method_deprecated, -1,
  /* DiagArray248 */ diag::warn_exception_spec_deprecated, -1,
  /* DiagArray249 */ diag::warn_comparison_mixed_enum_types_cxx20, -1,
  /* DiagArray250 */ diag::warn_conditional_mixed_enum_types_cxx20, -1,
  /* DiagArray251 */ diag::warn_arith_conv_mixed_enum_types_cxx20, -1,
  /* DiagArray252 */ diag::warn_arith_conv_enum_float_cxx20, -1,
  /* DiagArray253 */ diag::warn_deprecated_def, diag::warn_unavailable_def, -1,
  /* DiagArray254 */ diag::warn_increment_bool, -1,
  /* DiagArray255 */ diag::warn_deprecated_literal_operator_id, -1,
  /* DiagArray256 */ diag::warn_deprecated_missing_comma_before_ellipsis, -1,
  /* DiagArray257 */ diag::warn_deprecated_module_dot_map, -1,
  /* DiagArray258 */ diag::warn_non_prototype_changes_behavior, diag::warn_strict_uses_without_prototype, -1,
  /* DiagArray259 */ diag::warn_objc_isa_assign, diag::warn_objc_isa_use, -1,
  /* DiagArray260 */ diag::warn_objc_pointer_masking, -1,
  /* DiagArray261 */ diag::warn_objc_pointer_masking_performSelector, -1,
  /* DiagArray262 */ diag::warn_unprefixed_octal_deprecated, -1,
  /* DiagArray263 */ diag::warn_drv_deprecated_arg_ofast, diag::warn_drv_deprecated_arg_ofast_for_flang, -1,
  /* DiagArray264 */ diag::warn_pragma_deprecated_macro_use, -1,
  /* DiagArray265 */ diag::warn_deprecated_redundant_constexpr_static_def, -1,
  /* DiagArray266 */ diag::warn_deprecated_register, -1,
  /* DiagArray267 */ diag::warn_deprecated_this_capture, -1,
  /* DiagArray268 */ diag::warn_ext_int_deprecated, -1,
  /* DiagArray269 */ diag::warn_deprecated_increment_decrement_volatile, diag::warn_deprecated_simple_assign_volatile, diag::warn_deprecated_volatile_param, diag::warn_deprecated_volatile_return, diag::warn_deprecated_volatile_structured_binding, -1,
  /* DiagArray271 */ diag::warn_direct_ivar_access, -1,
  /* DiagArray272 */ diag::pp_disabled_macro_expansion, -1,
  /* DiagArray275 */ diag::warn_conflicting_param_modifiers, diag::warn_conflicting_ret_type_modifiers, -1,
  /* DiagArray277 */ diag::warn_remainder_division_by_zero, -1,
  /* DiagArray278 */ diag::warn_attribute_dll_redeclaration, -1,
  /* DiagArray279 */ diag::warn_attribute_dllexport_explicit_instantiation_decl, -1,
  /* DiagArray280 */ diag::warn_attribute_dllimport_static_field_definition, -1,
  /* DiagArray281 */ diag::warn_doc_api_container_decl_mismatch, diag::warn_doc_block_command_duplicate, diag::warn_doc_block_command_empty_paragraph, diag::warn_doc_container_decl_mismatch, diag::warn_doc_function_method_decl_mismatch, diag::warn_doc_html_start_tag_expected_ident_or_greater, diag::warn_doc_html_start_tag_expected_quoted_string, diag::warn_doc_inline_command_not_enough_arguments, diag::warn_doc_param_duplicate, diag::warn_doc_param_invalid_direction, diag::warn_doc_param_not_attached_to_a_function_decl, diag::warn_doc_param_not_found, diag::warn_doc_returns_attached_to_a_void_function, diag::warn_doc_returns_not_attached_to_a_function_decl, diag::warn_doc_tparam_duplicate, diag::warn_doc_tparam_not_attached_to_a_template_decl, diag::warn_doc_tparam_not_found, diag::warn_not_a_doxygen_trailing_member_comment, diag::warn_splice_in_doxygen_comment, diag::warn_verbatim_block_end_without_start, -1,
  /* DiagArray282 */ diag::warn_doc_deprecated_not_sync, -1,
  /* DiagArray283 */ diag::warn_doc_html_end_forbidden, diag::warn_doc_html_end_unbalanced, diag::warn_doc_html_missing_end_tag, diag::warn_doc_html_start_end_mismatch, -1,
  /* DiagArray284 */ diag::warn_doc_param_spaces_in_direction, -1,
  /* DiagArray285 */ diag::warn_correct_comment_command_name, diag::warn_unknown_comment_command_name, -1,
  /* DiagArray286 */ diag::ext_dollar_in_identifier, -1,
  /* DiagArray287 */ diag::warn_impcast_double_promotion, -1,
  /* DiagArray288 */ diag::ext_dtor_name_ambiguous, diag::ext_dtor_named_in_wrong_scope, diag::ext_qualified_dtor_named_in_lexical_scope, -1,
  /* DiagArray289 */ diag::ext_destructor_typedef_name, -1,
  /* DiagArray290 */ diag::ext_duplicate_declspec, diag::ext_warn_duplicate_declspec, diag::warn_attribute_address_multiple_identical_qualifiers, diag::warn_duplicate_declspec, -1,
  /* DiagArray291 */ diag::warn_duplicate_enum_values, -1,
  /* DiagArray292 */ diag::warn_method_param_declaration, -1,
  /* DiagArray293 */ diag::warn_duplicate_method_decl, -1,
  /* DiagArray294 */ diag::warn_duplicate_protocol_def, -1,
  /* DiagArray295 */ diag::warn_drv_dxc_missing_dxv, -1,
  /* DiagArray296 */ diag::warn_dyn_class_memaccess, -1,
  /* DiagArray297 */ diag::ext_dynamic_exception_spec, -1,
  /* DiagArray298 */ diag::warn_eagerly_load_for_standard_cplusplus_modules, -1,
  /* DiagArray300 */ diag::ext_enum_base_in_type_specifier, -1,
  /* DiagArray301 */ diag::ext_elaborated_enum_class, -1,
  /* DiagArray302 */ diag::ext_embedded_directive, -1,
  /* DiagArray303 */ diag::warn_empty_for_body, diag::warn_empty_if_body, diag::warn_empty_range_based_for_body, diag::warn_empty_switch_body, diag::warn_empty_while_body, -1,
  /* DiagArray304 */ diag::ext_decomp_decl_empty, -1,
  /* DiagArray305 */ diag::warn_empty_init_statement, -1,
  /* DiagArray306 */ diag::ext_empty_translation_unit, -1,
  /* DiagArray307 */ diag::warn_incomplete_encoded_type, -1,
  /* DiagArray309 */ diag::warn_comparison_mixed_enum_types, -1,
  /* DiagArray310 */ diag::warn_conditional_mixed_enum_types, -1,
  /* DiagArray311 */ diag::warn_comparison_of_mixed_enum_types_switch, -1,
  /* DiagArray312 */ diag::warn_impcast_different_enum_types, -1,
  /* DiagArray313 */ diag::warn_arith_conv_mixed_enum_types, diag::warn_conv_mixed_enum_types_cxx26, -1,
  /* DiagArray314 */ diag::warn_arith_conv_enum_float, -1,
  /* DiagArray315 */ diag::ext_enum_too_large, diag::ext_enumerator_increment_too_large, -1,
  /* DiagArray316 */ diag::warn_cdtor_function_try_handler_mem_expr, diag::warn_exception_caught_by_earlier_handler, diag::warn_throw_in_noexcept_func, -1,
  /* DiagArray317 */ diag::ext_excess_initializers, diag::ext_excess_initializers_for_sizeless_type, diag::ext_excess_initializers_in_char_array_initializer, diag::ext_initializer_string_for_char_array_too_long, -1,
  /* DiagArray318 */ diag::warn_anyx86_excessive_regsave, -1,
  /* DiagArray319 */ diag::warn_exit_time_destructor, -1,
  /* DiagArray320 */ diag::warn_defined_in_function_type_macro, diag::warn_defined_in_object_type_macro, -1,
  /* DiagArray321 */ diag::warn_experimental_header_unit, -1,
  /* DiagArray322 */ diag::warn_openmp_experimental, -1,
  /* DiagArray323 */ diag::warn_direct_initialize_call, diag::warn_direct_super_initialize_call, -1,
  /* DiagArray324 */ diag::warn_arc_strong_pointer_objc_pointer, -1,
  /* DiagArray325 */ diag::ext_explicit_specialization_storage_class, -1,
  /* DiagArray327 */ diag::warn_zero_size_struct_union_in_extern_c, -1,
  /* DiagArray328 */ diag::warn_extern_init, -1,
  /* DiagArray330 */ diag::warn_namespace_member_extra_qualification, -1,
  /* DiagArray331 */ diag::ext_extra_semi, diag::warn_extra_semi_after_mem_fn_def, -1,
  /* DiagArray332 */ diag::warn_null_statement, -1,
  /* DiagArray333 */ diag::ext_pp_extra_tokens_at_eol, diag::warn_omp_extra_tokens_at_eol, -1,
  /* DiagArray334 */ diag::warn_missing_symbol_graph_dir, -1,
  /* DiagArray335 */ diag::ext_template_spec_extra_headers, -1,
  /* DiagArray336 */ diag::warn_final_dtor_non_final_class, -1,
  /* DiagArray337 */ diag::warn_pragma_final_macro, -1,
  /* DiagArray338 */ diag::warn_fixedpoint_constant_overflow, -1,
  /* DiagArray339 */ diag::warn_flag_enum_constant_out_of_range, -1,
  /* DiagArray340 */ diag::ext_flexible_array_in_array, diag::ext_flexible_array_in_struct, -1,
  /* DiagArray341 */ diag::warn_impcast_float_integer, -1,
  /* DiagArray342 */ diag::warn_floatingpoint_eq, -1,
  /* DiagArray343 */ diag::warn_impcast_float_to_integer, diag::warn_impcast_float_to_integer_out_of_range, -1,
  /* DiagArray344 */ diag::warn_impcast_float_to_integer_zero, -1,
  /* DiagArray345 */ diag::warn_redundant_loop_iteration, diag::warn_variables_not_in_loop_body, -1,
  /* DiagArray346 */ diag::warn_format_P_no_precision, diag::warn_format_P_with_objc_pointer, diag::warn_format_argument_needs_cast, diag::warn_format_bool_as_character, diag::warn_format_cmp_modifierfor_mismatch, diag::warn_format_cmp_role_mismatch, diag::warn_format_cmp_sensitivity_mismatch, diag::warn_format_cmp_specifier_mismatch, diag::warn_format_cmp_specifier_sign_mismatch, diag::warn_format_conversion_argument_type_mismatch, diag::warn_format_invalid_annotation, diag::warn_format_invalid_positional_specifier, diag::warn_format_mix_positional_nonpositional_args, diag::warn_format_nonsensical_length, diag::warn_format_string_is_wide_literal, diag::warn_format_string_type_incompatible, diag::warn_format_zero_positional_specifier, diag::warn_missing_format_string, diag::warn_printf_ObjCflags_without_ObjCConversion, diag::warn_printf_asterisk_missing_arg, diag::warn_printf_asterisk_wrong_type, diag::warn_printf_empty_objc_flag, diag::warn_printf_format_string_contains_null_char, diag::warn_printf_format_string_not_null_terminated, diag::warn_printf_ignored_flag, diag::warn_printf_incomplete_specifier, diag::warn_printf_invalid_objc_flag, diag::warn_printf_narg_not_supported, diag::warn_printf_nonsensical_flag, diag::warn_printf_nonsensical_optional_amount, diag::warn_printf_positional_arg_exceeds_data_args, diag::warn_scanf_nonzero_width, diag::warn_scanf_scanlist_incomplete, -1,
  /* DiagArray347 */ diag::warn_printf_data_arg_not_used, -1,
  /* DiagArray348 */ diag::warn_format_cmp_specifier_arity, diag::warn_printf_insufficient_data_args, -1,
  /* DiagArray349 */ diag::warn_format_invalid_conversion, -1,
  /* DiagArray350 */ diag::warn_format_non_standard, diag::warn_format_non_standard_conversion_spec, diag::warn_format_non_standard_positional_arg, -1,
  /* DiagArray351 */ diag::warn_format_nonliteral, -1,
  /* DiagArray352 */ diag::warn_format_overflow, -1,
  /* DiagArray353 */ diag::warn_format_overflow_non_kprintf, -1,
  /* DiagArray354 */ diag::warn_format_argument_needs_cast_pedantic, diag::warn_format_cmp_specifier_mismatch_pedantic, diag::warn_format_conversion_argument_type_mismatch_pedantic, -1,
  /* DiagArray355 */ diag::warn_format_nonliteral_noargs, -1,
  /* DiagArray356 */ diag::warn_format_conversion_argument_type_mismatch_signedness, -1,
  /* DiagArray357 */ diag::warn_format_truncation, -1,
  /* DiagArray358 */ diag::warn_format_truncation_non_kprintf, -1,
  /* DiagArray359 */ diag::warn_format_conversion_argument_type_mismatch_confusion, -1,
  /* DiagArray361 */ diag::warn_empty_format_string, -1,
  /* DiagArray363 */ diag::warn_fortify_scanf_overflow, diag::warn_fortify_source_overflow, diag::warn_fortify_source_size_mismatch, diag::warn_fortify_strlen_overflow, -1,
  /* DiagArray364 */ diag::warn_four_char_character_literal, -1,
  /* DiagArray365 */ diag::warn_frame_address, -1,
  /* DiagArray366 */ diag::warn_fe_backend_frame_larger_than, diag::warn_fe_frame_larger_than, -1,
  /* DiagArray368 */ diag::warn_framework_include_private_from_public, -1,
  /* DiagArray369 */ diag::warn_free_nonheap_object, -1,
  /* DiagArray370 */ diag::ext_enum_friend, -1,
  /* DiagArray371 */ diag::warn_function_def_in_objc_container, -1,
  /* DiagArray372 */ diag::warn_conflicting_func_effects, diag::warn_func_effect_calls_expr_without_effect, diag::warn_func_effect_calls_func_without_effect, diag::warn_func_effect_violation, diag::warn_invalid_add_func_effects, diag::warn_mismatched_func_effect_override, diag::warn_mismatched_func_effect_redeclaration, -1,
  /* DiagArray373 */ diag::warn_dispatch_body_ignored, diag::warn_multiversion_duplicate_entries, diag::warn_target_clone_duplicate_options, diag::warn_target_clone_no_impact_options, -1,
  /* DiagArray374 */ diag::warn_drv_fuse_ld_path, -1,
  /* DiagArray377 */ diag::ext_clang_diagnose_if, diag::ext_clang_enable_if, diag::ext_warn_gnu_final, diag::warn_attribute_on_function_definition, diag::warn_break_binds_to_switch, diag::warn_cleanup_ext, diag::warn_gcc_attribute_location, diag::warn_gcc_ignores_type_attr, diag::warn_gcc_requires_variadic_function, diag::warn_gcc_variable_decl_in_for_loop, diag::warn_loop_ctrl_binds_to_inner, -1,
  /* DiagArray378 */ diag::warn_global_constructor, diag::warn_global_destructor, -1,
  /* DiagArray379 */ diag::warn_drv_global_isel_incomplete, diag::warn_drv_global_isel_incomplete_opt, -1,
  /* DiagArray381 */ diag::ext_alignof_expr, -1,
  /* DiagArray382 */ diag::ext_gnu_anonymous_struct, -1,
  /* DiagArray383 */ diag::ext_array_init_parens, -1,
  /* DiagArray384 */ diag::ext_auto_type, -1,
  /* DiagArray386 */ diag::ext_gnu_case_range, -1,
  /* DiagArray387 */ diag::ext_integer_complex, -1,
  /* DiagArray388 */ diag::ext_array_init_copy, -1,
  /* DiagArray389 */ diag::ext_gnu_conditional_expr, -1,
  /* DiagArray390 */ diag::ext_gnu_array_range, diag::ext_gnu_missing_equal_designator, diag::ext_gnu_old_style_field_designator, -1,
  /* DiagArray392 */ diag::ext_empty_struct_union, diag::ext_flexible_array_empty_aggregate_gnu, diag::ext_no_named_members_in_struct_union, -1,
  /* DiagArray393 */ diag::ext_flexible_array_init, -1,
  /* DiagArray394 */ diag::ext_flexible_array_union_gnu, -1,
  /* DiagArray395 */ diag::ext_expr_not_ice, diag::ext_in_class_initializer_non_constant, diag::ext_vla_folded_to_constant, -1,
  /* DiagArray396 */ diag::ext_gnu_imaginary_constant, -1,
  /* DiagArray397 */ diag::ext_pp_include_next_directive, -1,
  /* DiagArray398 */ diag::warn_gnu_inline_cplusplus_without_extern, -1,
  /* DiagArray399 */ diag::ext_gnu_address_of_label, diag::ext_gnu_indirect_goto, -1,
  /* DiagArray400 */ diag::ext_pp_gnu_line_directive, -1,
  /* DiagArray401 */ diag::warn_gnu_null_ptr_arith, -1,
  /* DiagArray403 */ diag::ext_gnu_ptr_func_arith, diag::ext_gnu_subscript_void_type, diag::ext_gnu_void_ptr, -1,
  /* DiagArray404 */ diag::ext_forward_ref_enum_def, -1,
  /* DiagArray405 */ diag::ext_gnu_statement_expr, -1,
  /* DiagArray406 */ diag::ext_gnu_statement_expr_macro, -1,
  /* DiagArray407 */ diag::ext_in_class_initializer_float_type, -1,
  /* DiagArray408 */ diag::ext_string_literal_operator_template, -1,
  /* DiagArray409 */ diag::ext_typecheck_cast_to_union, -1,
  /* DiagArray410 */ diag::ext_variable_sized_type_in_struct, -1,
  /* DiagArray411 */ diag::ext_pp_line_zero, -1,
  /* DiagArray412 */ diag::ext_paste_comma, -1,
  /* DiagArray413 */ diag::warn_maybe_capture_bad_target_this_ptr, -1,
  /* DiagArray414 */ diag::warn_header_guard, -1,
  /* DiagArray415 */ diag::warn_using_directive_in_header, -1,
  /* DiagArray416 */ diag::warn_excess_precision_not_supported, -1,
  /* DiagArray417 */ diag::warn_hip_omp_target_directives, -1,
  /* DiagArray418 */ diag::warn_ignored_hip_only_option, -1,
  /* DiagArray419 */ diag::ext_hlsl_auto_type_specifier, -1,
  /* DiagArray420 */ diag::warn_hlsl_availability, diag::warn_hlsl_availability_unavailable, -1,
  /* DiagArray421 */ diag::warn_hlsl_langstd_minimal, -1,
  /* DiagArray422 */ diag::ext_hlsl_access_specifiers, diag::ext_hlsl_lambda, -1,
  /* DiagArray423 */ diag::warn_hlsl_implicit_binding, -1,
  /* DiagArray424 */ diag::warn_condition_is_idiomatic_assignment, -1,
  /* DiagArray425 */ diag::ext_cannot_use_trivial_abi, diag::warn_alias_to_weak_alias, diag::warn_alias_with_section, diag::warn_aligned_attr_underaligned, diag::warn_attr_abi_tag_namespace, diag::warn_attribute_after_definition_ignored, diag::warn_attribute_cmse_entry_static, diag::warn_attribute_dllexport_explicit_instantiation_def, diag::warn_attribute_exclude_from_explicit_instantiation_local_class, diag::warn_attribute_form_ignored, diag::warn_attribute_has_no_effect_on_compile_time_if, diag::warn_attribute_has_no_effect_on_infinite_loop, diag::warn_attribute_hybrid_patchable_non_extern, diag::warn_attribute_iboutlet, diag::warn_attribute_ignored, diag::warn_attribute_ignored_for_field_of_type, diag::warn_attribute_ignored_no_calls_in_stmt, diag::warn_attribute_ignored_non_function_pointer, diag::warn_attribute_ignored_on_inline, diag::warn_attribute_ignored_on_non_definition, diag::warn_attribute_invalid_on_definition, diag::warn_attribute_needs_aggregate, diag::warn_attribute_no_decl, diag::warn_attribute_nonnull_no_pointers, diag::warn_attribute_nonnull_parm_no_args, diag::warn_attribute_not_on_decl, diag::warn_attribute_on_void_param, diag::warn_attribute_pointer_or_reference_only, diag::warn_attribute_pointers_only, diag::warn_attribute_precede_definition, diag::warn_attribute_return_pointers_only, diag::warn_attribute_return_pointers_refs_only, diag::warn_attribute_sentinel_named_arguments, diag::warn_attribute_sentinel_not_variadic, diag::warn_attribute_type_not_supported, diag::warn_attribute_type_not_supported_global, diag::warn_attribute_unknown_visibility, diag::warn_attribute_void_function_method, diag::warn_attribute_weak_on_field, diag::warn_attribute_weak_on_local, diag::warn_attribute_wrong_decl_type, diag::warn_attribute_wrong_decl_type_str, diag::warn_attributes_likelihood_ifstmt_conflict, diag::warn_block_literal_attributes_on_omitted_return_type, diag::warn_cconv_unsupported, diag::warn_const_attr_with_pure_attr, diag::warn_cuda_maxclusterrank_sm_90, diag::warn_cxx11_gnu_attribute_on_type, diag::warn_declspec_allocator_nonpointer, diag::warn_declspec_attribute_ignored, diag::warn_deprecated_anonymous_namespace, diag::warn_deprecated_ignored_on_using, diag::warn_dllimport_dropped_from_inline_function, diag::warn_duplicate_attribute, diag::warn_duplicate_attribute_exact, diag::warn_function_attribute_ignored_in_stmt, diag::warn_function_stmt_attribute_precedence, diag::warn_gc_attribute_weak_on_local, diag::warn_gnu_inline_attribute_requires_inline, diag::warn_ignored_ms_inheritance, diag::warn_ignored_objc_externally_retained, diag::warn_import_on_definition, diag::warn_internal_linkage_local_storage, diag::warn_interrupt_signal_attribute_invalid, diag::warn_microsoft_qualifiers_ignored, diag::warn_mig_server_routine_does_not_return_kern_return_t, diag::warn_mismatched_import, diag::warn_mmap_unknown_attribute, diag::warn_nocf_check_attribute_ignored, diag::warn_noderef_on_non_pointer_or_array, diag::warn_nothrow_attribute_ignored, diag::warn_ns_attribute_wrong_parameter_type, diag::warn_ns_attribute_wrong_return_type, diag::warn_objc_direct_ignored, diag::warn_objc_direct_property_ignored, diag::warn_opencl_attr_deprecated_ignored, diag::warn_pure_function_returns_void, diag::warn_require_const_init_added_too_late, diag::warn_riscv_repeated_interrupt_attribute, diag::warn_sycl_kernel_invalid_template_param_type, diag::warn_sycl_kernel_num_of_function_params, diag::warn_sycl_kernel_num_of_template_params, diag::warn_sycl_kernel_return_type, diag::warn_transparent_union_attribute_field_size_align, diag::warn_transparent_union_attribute_floating, diag::warn_transparent_union_attribute_not_definition, diag::warn_transparent_union_attribute_zero_fields, diag::warn_type_attribute_wrong_type, diag::warn_unhandled_ms_attribute_ignored, diag::warn_unsupported_target_attribute, diag::warn_unused_result_typedef_unsupported_spelling, diag::warn_wrong_clang_attr_namespace, -1,
  /* DiagArray426 */ diag::warn_missing_sdksettings_for_availability_checking, -1,
  /* DiagArray427 */ diag::warn_qual_base_type, -1,
  /* DiagArray428 */ diag::warn_drv_pch_ignoring_gch_dir, diag::warn_drv_pch_ignoring_gch_file, -1,
  /* DiagArray429 */ diag::warn_drv_unsupported_opt_for_target, diag::warn_ignored_gcc_optimization, -1,
  /* DiagArray430 */ diag::warn_pragma_intrinsic_builtin, -1,
  /* DiagArray432 */ diag::warn_clause_expected_string, diag::warn_pragma_align_expected_equal, diag::warn_pragma_align_invalid_option, diag::warn_pragma_comment_ignored, diag::warn_pragma_debug_dependent_argument, diag::warn_pragma_debug_missing_argument, diag::warn_pragma_debug_missing_command, diag::warn_pragma_debug_unable_to_find_module, diag::warn_pragma_debug_unexpected_argument, diag::warn_pragma_debug_unexpected_command, diag::warn_pragma_debug_unknown_module, diag::warn_pragma_expected_action_or_r_paren, diag::warn_pragma_expected_colon, diag::warn_pragma_expected_colon_r_paren, diag::warn_pragma_expected_comma, diag::warn_pragma_expected_identifier, diag::warn_pragma_expected_init_seg, diag::warn_pragma_expected_integer, diag::warn_pragma_expected_lparen, diag::warn_pragma_expected_non_wide_string, diag::warn_pragma_expected_predicate, diag::warn_pragma_expected_punc, diag::warn_pragma_expected_rparen, diag::warn_pragma_expected_section_label_or_name, diag::warn_pragma_expected_section_name, diag::warn_pragma_expected_section_push_pop_or_name, diag::warn_pragma_expected_string, diag::warn_pragma_extra_tokens_at_eol, diag::warn_pragma_force_cuda_host_device_bad_arg, diag::warn_pragma_fp_ignored, diag::warn_pragma_init_seg_unsupported_target, diag::warn_pragma_invalid_action, diag::warn_pragma_invalid_argument, diag::warn_pragma_invalid_specific_action, diag::warn_pragma_missing_argument, diag::warn_pragma_ms_fenv_access, diag::warn_pragma_ms_struct, diag::warn_pragma_options_align_reset_failed, diag::warn_pragma_options_expected_align, diag::warn_pragma_pack_invalid_alignment, diag::warn_pragma_pack_malformed, diag::warn_pragma_pop_failed, diag::warn_pragma_pop_macro_no_push, diag::warn_pragma_unknown_extension, diag::warn_pragma_unsupported_action, diag::warn_pragma_unsupported_extension, diag::warn_pragma_unused_expected_var, diag::warn_pragma_unused_expected_var_arg, diag::warn_pragma_unused_undeclared_var, diag::warn_stdc_unknown_rounding_mode, -1,
  /* DiagArray433 */ diag::ext_typecheck_function_qualifiers_unspecified, diag::warn_arc_lifetime_result_type, diag::warn_block_literal_qualifiers_on_omitted_return_type, diag::warn_qual_return_type, diag::warn_typecheck_function_qualifiers_ignored, -1,
  /* DiagArray434 */ diag::warn_typecheck_reference_qualifiers, -1,
  /* DiagArray436 */ diag::warn_auto_implicit_atomic_property, diag::warn_implicit_atomic_property, -1,
  /* DiagArray437 */ diag::warn_impcast_integer_float_precision_constant, -1,
  /* DiagArray438 */ diag::warn_impcast_floating_point_to_bool, -1,
  /* DiagArray439 */ diag::ext_implicit_exception_spec_mismatch, -1,
  /* DiagArray440 */ diag::warn_unannotated_fallthrough, -1,
  /* DiagArray441 */ diag::warn_unannotated_fallthrough_per_function, -1,
  /* DiagArray442 */ diag::warn_impcast_fixed_point_range, -1,
  /* DiagArray443 */ diag::warn_impcast_float_precision, diag::warn_impcast_float_result_precision, -1,
  /* DiagArray444 */ diag::ext_implicit_function_decl_c99, diag::ext_implicit_lib_function_decl, diag::ext_implicit_lib_function_decl_c99, diag::warn_builtin_unknown, diag::warn_implicit_function_decl, -1,
  /* DiagArray445 */ diag::ext_missing_type_specifier, diag::ext_param_not_declared, diag::warn_missing_type_specifier, -1,
  /* DiagArray446 */ diag::warn_impcast_high_order_zero_bits, diag::warn_impcast_integer_precision, -1,
  /* DiagArray447 */ diag::warn_impcast_integer_float_precision, -1,
  /* DiagArray448 */ diag::warn_implicitly_retains_self, -1,
  /* DiagArray449 */ diag::ext_integer_literal_too_large_for_signed, -1,
  /* DiagArray451 */ diag::warn_import_implementation_partition_unit_in_interface_unit, -1,
  /* DiagArray452 */ diag::ext_pp_import_directive, -1,
  /* DiagArray453 */ diag::warn_inaccessible_base_class, -1,
  /* DiagArray454 */ diag::warn_pp_include_angled_in_module_purview, -1,
  /* DiagArray455 */ diag::pp_include_next_absolute_path, -1,
  /* DiagArray456 */ diag::pp_include_next_in_primary, -1,
  /* DiagArray457 */ diag::warn_deep_exception_specs_differ, diag::warn_incompatible_exception_specs, -1,
  /* DiagArray458 */ diag::ext_typecheck_convert_incompatible_function_pointer, -1,
  /* DiagArray459 */ diag::warn_typecheck_convert_incompatible_function_pointer_strict, -1,
  /* DiagArray460 */ diag::warn_redecl_library_builtin, -1,
  /* DiagArray461 */ diag::warn_section_msvc_compat, -1,
  /* DiagArray462 */ diag::warn_cxx_ms_struct, diag::warn_npot_ms_struct, -1,
  /* DiagArray463 */ diag::ext_typecheck_convert_incompatible_pointer, -1,
  /* DiagArray464 */ diag::ext_nested_pointer_qualifier_mismatch, diag::ext_typecheck_convert_discards_qualifiers, diag::warn_bad_cxx_cast_nested_pointer_addr_space, -1,
  /* DiagArray465 */ diag::warn_property_types_are_incompatible, -1,
  /* DiagArray466 */ diag::warn_incompatible_sysroot, -1,
  /* DiagArray467 */ diag::warn_mmap_incomplete_framework_module_declaration, -1,
  /* DiagArray468 */ diag::warn_undef_method_impl, -1,
  /* DiagArray470 */ diag::warn_implicit_decl_no_jmp_buf, -1,
  /* DiagArray471 */ diag::warn_missing_submodule, diag::warn_mmap_umbrella_dir_not_found, diag::warn_uncovered_module_header, -1,
  /* DiagArray472 */ diag::warn_redeclaration_without_attribute_prev_attribute_ignored, diag::warn_redeclaration_without_import_attribute, -1,
  /* DiagArray473 */ diag::warn_inconsistent_destructor_marked_not_override_overriding, -1,
  /* DiagArray474 */ diag::warn_inconsistent_function_marked_not_override_overriding, -1,
  /* DiagArray475 */ diag::ext_increment_bool, -1,
  /* DiagArray476 */ diag::warn_infinite_recursive_function, -1,
  /* DiagArray477 */ diag::warn_init_priority_reserved, -1,
  /* DiagArray479 */ diag::ext_initializer_overrides, diag::ext_initializer_union_overrides, diag::warn_initializer_overrides, -1,
  /* DiagArray480 */ diag::ext_out_of_line_qualified_id_type_names_constructor, -1,
  /* DiagArray482 */ diag::warn_fe_inline_asm, -1,
  /* DiagArray483 */ diag::warn_inline_namespace_reopened_noninline, -1,
  /* DiagArray484 */ diag::ext_operator_new_delete_declared_inline, -1,
  /* DiagArray485 */ diag::warn_dylib_symbol_flags_mismatch, diag::warn_glob_did_not_match, diag::warn_header_availability_mismatch, diag::warn_header_hidden_symbol, diag::warn_header_symbol_flags_mismatch, diag::warn_header_symbol_missing, diag::warn_library_hidden_symbol, diag::warn_library_missing_symbol, diag::warn_no_such_excluded_header_file, diag::warn_platform_mismatch, diag::warn_rpaths_mismatch, diag::warn_rpaths_missing, diag::warn_target, -1,
  /* DiagArray486 */ diag::warn_explicit_instantiation_after_specialization, -1,
  /* DiagArray487 */ diag::ext_typecheck_convert_int_pointer, diag::ext_typecheck_convert_pointer_int, -1,
  /* DiagArray489 */ diag::warn_enum_constant_in_bool_context, diag::warn_left_shift_in_bool_context, -1,
  /* DiagArray490 */ diag::warn_int_to_pointer_cast, -1,
  /* DiagArray491 */ diag::warn_int_to_void_pointer_cast, -1,
  /* DiagArray492 */ diag::warn_integer_constant_overflow, -1,
  /* DiagArray493 */ diag::warn_drv_missing_plugin_arg, diag::warn_drv_missing_plugin_name, diag::warn_drv_msp430_hwmult_mismatch, diag::warn_drv_msp430_hwmult_no_device, diag::warn_drv_msp430_hwmult_unsupported, diag::warn_drv_object_size_disabled_O0, diag::warn_drv_optimization_value, diag::warn_fe_backend_invalid_feature_flag, diag::warn_fe_backend_readonly_feature_flag, diag::warn_target_unrecognized_env, diag::warn_target_unsupported_extension, -1,
  /* DiagArray494 */ diag::ext_constexpr_function_never_constant_expr, -1,
  /* DiagArray495 */ diag::warn_invalid_feature_combination, -1,
  /* DiagArray496 */ diag::warn_invalid_asm_cast_lvalue, -1,
  /* DiagArray497 */ diag::warn_iboutlet_object_type, diag::warn_iboutletcollection_property_assign, -1,
  /* DiagArray498 */ diag::warn_invalid_initializer_from_system_header, -1,
  /* DiagArray499 */ diag::warn_invalid_ios_deployment_target, -1,
  /* DiagArray500 */ diag::warn_attribute_no_builtin_invalid_builtin_name, -1,
  /* DiagArray501 */ diag::warn_noreturn_coroutine, diag::warn_noreturn_function_has_return_expr, diag::warn_noreturn_has_return_expr, -1,
  /* DiagArray502 */ diag::ext_offsetof_non_pod_type, diag::ext_offsetof_non_standardlayout_type, -1,
  /* DiagArray503 */ diag::warn_drv_unable_to_find_directory_expected, -1,
  /* DiagArray504 */ diag::ext_partial_spec_not_more_specialized_than_primary, -1,
  /* DiagArray506 */ diag::ext_empty_character, diag::ext_unterminated_char_or_string, -1,
  /* DiagArray507 */ diag::warn_bad_character_encoding, diag::warn_bad_string_encoding, -1,
  /* DiagArray508 */ diag::warn_invalid_specialization, -1,
  /* DiagArray509 */ diag::warn_user_defined_msg_constexpr, -1,
  /* DiagArray510 */ diag::ext_pp_bad_paste_ms, -1,
  /* DiagArray511 */ diag::warn_unevaluated_string_prefix, -1,
  /* DiagArray512 */ diag::warn_invalid_utf8_in_comment, -1,
  /* DiagArray513 */ diag::warn_jump_out_of_seh_finally, -1,
  /* DiagArray514 */ diag::ext_keyword_as_ident, -1,
  /* DiagArray515 */ diag::warn_pp_macro_hides_keyword, -1,
  /* DiagArray516 */ diag::ext_param_promoted_not_compatible_with_prototype, -1,
  /* DiagArray517 */ diag::ext_token_used, -1,
  /* DiagArray518 */ diag::warn_parameter_size, diag::warn_return_value_size, -1,
  /* DiagArray519 */ diag::warn_hlsl_deprecated_register_type_b, diag::warn_hlsl_deprecated_register_type_i, diag::warn_hlsl_register_type_c_packoffset, diag::warn_hlsl_user_defined_type_missing_member, -1,
  /* DiagArray521 */ diag::warn_fe_linking_module, -1,
  /* DiagArray522 */ diag::warn_impcast_literal_float_to_integer, diag::warn_impcast_literal_float_to_integer_out_of_range, -1,
  /* DiagArray523 */ diag::warn_float_compare_literal, diag::warn_float_overflow, diag::warn_float_underflow, -1,
  /* DiagArray524 */ diag::ext_template_arg_local_type, -1,
  /* DiagArray525 */ diag::warn_logical_not_on_lhs_of_check, -1,
  /* DiagArray526 */ diag::warn_logical_and_in_logical_or, -1,
  /* DiagArray527 */ diag::ext_c99_longlong, -1,
  /* DiagArray529 */ diag::ext_pp_macro_redef, -1,
  /* DiagArray530 */ diag::ext_main_invalid_linkage_specification, diag::ext_main_used, diag::ext_noreturn_main, diag::ext_variadic_main, diag::warn_main_one_arg, diag::warn_main_redefined, diag::warn_main_returns_bool_literal, diag::warn_static_main, -1,
  /* DiagArray531 */ diag::ext_main_no_return, diag::ext_main_returns_nonint, -1,
  /* DiagArray532 */ diag::warn_has_warning_invalid_option, -1,
  /* DiagArray533 */ diag::ext_many_braces_around_init, -1,
  /* DiagArray534 */ diag::warn_drv_math_errno_enabled_after_veclib, -1,
  /* DiagArray535 */ diag::ext_mathematical_notation, -1,
  /* DiagArray536 */ diag::warn_max_tokens, diag::warn_max_tokens_total, -1,
  /* DiagArray537 */ diag::warn_max_unsigned_zero, -1,
  /* DiagArray538 */ diag::warn_suspicious_sizeof_memset, -1,
  /* DiagArray539 */ diag::warn_memsize_comparison, -1,
  /* DiagArray540 */ diag::warn_non_contravariant_param_types, diag::warn_non_covariant_ret_types, -1,
  /* DiagArray542 */ diag::ext_ms_abstract_keyword, -1,
  /* DiagArray543 */ diag::ext_anonymous_record_with_type, diag::ext_ms_anonymous_record, -1,
  /* DiagArray544 */ diag::ext_ms_cast_fn_obj, diag::ext_ms_impcast_fn_obj, -1,
  /* DiagArray545 */ diag::ext_charize_microsoft, -1,
  /* DiagArray546 */ diag::ext_comment_paste_microsoft, -1,
  /* DiagArray547 */ diag::ext_default_init_const, -1,
  /* DiagArray548 */ diag::ext_pp_operator_used_as_macro_name, -1,
  /* DiagArray549 */ diag::ext_param_default_argument_redefinition, -1,
  /* DiagArray550 */ diag::warn_attribute_section_drectve, -1,
  /* DiagArray551 */ diag::ext_ctrl_z_eof_microsoft, -1,
  /* DiagArray552 */ diag::ext_ms_forward_ref_enum, -1,
  /* DiagArray553 */ diag::ext_enumerator_too_large, -1,
  /* DiagArray554 */ diag::ext_ellipsis_exception_spec, diag::ext_incomplete_in_exception_spec, diag::ext_mismatched_exception_spec, diag::ext_mismatched_exception_spec_explicit_instantiation, diag::ext_override_exception_spec, -1,
  /* DiagArray555 */ diag::warn_microsoft_dependent_exists, -1,
  /* DiagArray556 */ diag::ext_ms_explicit_constructor_call, -1,
  /* DiagArray557 */ diag::warn_member_extra_qualification, -1,
  /* DiagArray558 */ diag::ext_ms_c_enum_fixed_underlying_type, -1,
  /* DiagArray559 */ diag::ext_flexible_array_empty_aggregate_ms, diag::ext_flexible_array_union_ms, -1,
  /* DiagArray560 */ diag::ext_goto_into_protected_scope, -1,
  /* DiagArray561 */ diag::ext_ms_ambiguous_direct_base, -1,
  /* DiagArray562 */ diag::ext_pp_include_search_ms, -1,
  /* DiagArray563 */ diag::ext_init_from_predefined, -1,
  /* DiagArray564 */ diag::warn_ms_inline_non_function, -1,
  /* DiagArray565 */ diag::ext_mutable_reference, -1,
  /* DiagArray566 */ diag::ext_pure_function_definition, -1,
  /* DiagArray567 */ diag::ext_static_non_static, -1,
  /* DiagArray568 */ diag::ext_ms_sealed_keyword, -1,
  /* DiagArray569 */ diag::ext_ms_static_assert, -1,
  /* DiagArray570 */ diag::ext_string_literal_from_predefined, -1,
  /* DiagArray571 */ diag::ext_explicit_instantiation_duplicate, diag::ext_found_in_dependent_base, diag::ext_found_later_in_class, diag::ext_ms_delayed_template_argument, diag::ext_ms_deref_template_argument, diag::ext_ms_template_spec_redecl_out_of_scope, diag::ext_ms_template_type_arg_missing_typename, diag::ext_static_out_of_line, diag::ext_undeclared_unqual_id_with_dependent_base, diag::ext_unqualified_base_class, -1,
  /* DiagArray572 */ diag::ext_template_param_shadow, -1,
  /* DiagArray573 */ diag::ext_union_member_of_reference_type, -1,
  /* DiagArray574 */ diag::ext_friend_tag_redecl_outside_namespace, -1,
  /* DiagArray575 */ diag::ext_ms_using_declaration_inaccessible, -1,
  /* DiagArray576 */ diag::ext_pseudo_dtor_on_void, -1,
  /* DiagArray577 */ diag::warn_profile_data_misexpect, -1,
  /* DiagArray578 */ diag::warn_misleading_indentation, -1,
  /* DiagArray579 */ diag::warn_mismatched_delete_new, -1,
  /* DiagArray580 */ diag::warn_conflicting_param_types, -1,
  /* DiagArray581 */ diag::warn_conflicting_ret_types, -1,
  /* DiagArray582 */ diag::warn_struct_class_previous_tag_mismatch, diag::warn_struct_class_tag_mismatch, -1,
  /* DiagArray583 */ diag::warn_missing_braces, -1,
  /* DiagArray584 */ diag::ext_constinit_missing, -1,
  /* DiagArray585 */ diag::ext_no_declarators, diag::ext_standalone_specifier, diag::ext_typedef_without_a_name, diag::warn_standalone_specifier, -1,
  /* DiagArray586 */ diag::warn_missing_designated_field_initializers, -1,
  /* DiagArray587 */ diag::ext_missing_exception_specification, -1,
  /* DiagArray588 */ diag::warn_missing_field_initializers, -1,
  /* DiagArray590 */ diag::warn_missing_include_dirs, -1,
  /* DiagArray591 */ diag::warn_missing_method_return_type, -1,
  /* DiagArray592 */ diag::warn_drv_missing_multilib, -1,
  /* DiagArray593 */ diag::warn_overriding_method_missing_noescape, -1,
  /* DiagArray594 */ diag::warn_suggest_noreturn_block, diag::warn_suggest_noreturn_function, -1,
  /* DiagArray595 */ diag::warn_cconv_knr, -1,
  /* DiagArray596 */ diag::warn_missing_prototype, -1,
  /* DiagArray597 */ diag::warn_missing_selector_name, -1,
  /* DiagArray598 */ diag::warn_missing_sysroot, -1,
  /* DiagArray599 */ diag::missing_template_arg_list_after_template_kw, -1,
  /* DiagArray600 */ diag::warn_missing_variable_declarations, -1,
  /* DiagArray601 */ diag::warn_omp_assume_attribute_string_unknown_suggested, -1,
  /* DiagArray602 */ diag::warn_hlsl_packoffset_mix, -1,
  /* DiagArray603 */ diag::remark_module_build, diag::remark_module_build_done, diag::remark_module_lock_failure, diag::remark_module_lock_timeout, -1,
  /* DiagArray604 */ diag::warn_module_conflict, diag::warn_module_system_bit_conflict, -1,
  /* DiagArray605 */ diag::warn_module_config_mismatch, -1,
  /* DiagArray606 */ diag::warn_duplicate_module_file_extension, -1,
  /* DiagArray607 */ diag::remark_module_import, -1,
  /* DiagArray608 */ diag::ext_module_import_in_extern_c, -1,
  /* DiagArray609 */ diag::remark_pp_include_directive_modular_translation, -1,
  /* DiagArray610 */ diag::remark_module_lock, -1,
  /* DiagArray611 */ diag::ext_equivalent_internal_linkage_decl_in_modules, -1,
  /* DiagArray612 */ diag::ext_module_import_not_at_top_level_noop, -1,
  /* DiagArray616 */ diag::warn_drv_msvc_not_found, -1,
  /* DiagArray617 */ diag::warn_drv_multi_gpu_arch, -1,
  /* DiagArray618 */ diag::warn_multichar_character_literal, -1,
  /* DiagArray619 */ diag::warn_vbase_moved_multiple_times, -1,
  /* DiagArray620 */ diag::warn_fp_nan_inf_when_disabled, -1,
  /* DiagArray622 */ diag::ext_anonymous_record_with_anonymous_type, -1,
  /* DiagArray624 */ diag::warn_operator_new_returns_null, -1,
  /* DiagArray625 */ diag::warn_no_newline_eof, -1,
  /* DiagArray626 */ diag::warn_dereference_of_noderef_type, diag::warn_dereference_of_noderef_type_no_decl, diag::warn_noderef_to_dereferenceable_pointer, -1,
  /* DiagArray628 */ diag::ext_non_c_like_anon_struct_in_typedef, -1,
  /* DiagArray630 */ diag::warn_non_literal_null_pointer, -1,
  /* DiagArray631 */ diag::warn_non_modular_include_in_framework_module, -1,
  /* DiagArray632 */ diag::warn_non_modular_include_in_module, -1,
  /* DiagArray633 */ diag::warn_cannot_pass_non_pod_arg_to_vararg, diag::warn_non_pod_vararg_with_format_string, diag::warn_second_parameter_to_va_arg_not_pod, diag::warn_second_parameter_to_va_arg_ownership_qualified, -1,
  /* DiagArray634 */ diag::warn_alignment_not_power_of_two, -1,
  /* DiagArray635 */ diag::warn_non_virtual_dtor, -1,
  /* DiagArray636 */ diag::warn_null_arg, diag::warn_null_ret, -1,
  /* DiagArray638 */ diag::pp_nonportable_path, -1,
  /* DiagArray639 */ diag::warn_apinotes_private_case, -1,
  /* DiagArray640 */ diag::warn_apinotes_private_case_system, -1,
  /* DiagArray641 */ diag::warn_sycl_kernel_name_not_a_class_type, -1,
  /* DiagArray642 */ diag::pp_nonportable_system_path, -1,
  /* DiagArray643 */ diag::warn_neon_vector_initializer_non_portable, -1,
  /* DiagArray644 */ diag::warn_cstruct_memaccess, -1,
  /* DiagArray645 */ diag::warn_cxxstruct_memaccess, -1,
  /* DiagArray646 */ diag::warn_nsconsumed_attribute_mismatch, -1,
  /* DiagArray647 */ diag::warn_nsreturns_retained_attribute_mismatch, -1,
  /* DiagArray648 */ diag::warn_null_in_arithmetic_operation, diag::warn_null_in_comparison_operation, -1,
  /* DiagArray649 */ diag::null_in_char_or_string, diag::null_in_file, -1,
  /* DiagArray650 */ diag::warn_impcast_null_pointer_to_integer, -1,
  /* DiagArray651 */ diag::warn_binding_null_to_reference, diag::warn_indirection_through_null, -1,
  /* DiagArray652 */ diag::warn_pointer_arith_null_ptr, -1,
  /* DiagArray653 */ diag::warn_pointer_sub_null_ptr, -1,
  /* DiagArray654 */ diag::warn_conflicting_nullability_attr_overriding_param_types, diag::warn_conflicting_nullability_attr_overriding_ret_types, diag::warn_mismatched_nullability_attr, diag::warn_null_resettable_setter, diag::warn_nullability_duplicate, -1,
  /* DiagArray655 */ diag::warn_nullability_missing, -1,
  /* DiagArray656 */ diag::warn_nullability_missing_array, -1,
  /* DiagArray657 */ diag::warn_nullability_declspec, -1,
  /* DiagArray658 */ diag::ext_nullability, -1,
  /* DiagArray659 */ diag::warn_nullability_inferred_on_nested_type, -1,
  /* DiagArray660 */ diag::warn_nullability_lost, -1,
  /* DiagArray661 */ diag::warn_offload_incompatible_redeclare, -1,
  /* DiagArray662 */ diag::warn_autosynthesis_property_ivar_match, -1,
  /* DiagArray663 */ diag::warn_impcast_constant_value_to_objc_bool, -1,
  /* DiagArray664 */ diag::warn_objc_boxing_invalid_utf8_string, -1,
  /* DiagArray665 */ diag::warn_objc_circular_container, -1,
  /* DiagArray667 */ diag::warn_objc_designated_init_missing_super_call, diag::warn_objc_designated_init_non_designated_init_call, diag::warn_objc_designated_init_non_super_designated_init_call, diag::warn_objc_implementation_missing_designated_init_override, diag::warn_objc_secondary_init_missing_init_call, diag::warn_objc_secondary_init_super_init_call, -1,
  /* DiagArray668 */ diag::warn_nsdictionary_duplicate_key, -1,
  /* DiagArray669 */ diag::warn_dup_category_def, -1,
  /* DiagArray670 */ diag::warn_superclass_variable_sized_type_not_at_end, diag::warn_variable_sized_ivar_visibility, -1,
  /* DiagArray671 */ diag::warn_forward_class_redefinition, -1,
  /* DiagArray672 */ diag::warn_ivars_in_interface, -1,
  /* DiagArray673 */ diag::warn_objc_literal_comparison, -1,
  /* DiagArray674 */ diag::warn_impcast_objective_c_literal_to_bool, diag::warn_objc_collection_literal_element, -1,
  /* DiagArray675 */ diag::warn_pp_objc_macro_redef_ignored, -1,
  /* DiagArray676 */ diag::warn_messaging_unqualified_id, -1,
  /* DiagArray677 */ diag::warn_class_method_not_found, diag::warn_class_method_not_found_with_typo, diag::warn_inst_method_not_found, diag::warn_instance_method_not_found_with_typo, diag::warn_instance_method_on_class_found, diag::warn_root_inst_method_not_found, -1,
  /* DiagArray678 */ diag::warn_missing_explicit_synthesis, -1,
  /* DiagArray679 */ diag::warn_objc_missing_super_call, -1,
  /* DiagArray680 */ diag::warn_multiple_method_decl, -1,
  /* DiagArray681 */ diag::warn_objc_property_retain_of_block, -1,
  /* DiagArray682 */ diag::warn_objc_pointer_cxx_catch_fragile, -1,
  /* DiagArray683 */ diag::warn_objc_property_assign_on_object, -1,
  /* DiagArray684 */ diag::warn_impl_required_for_class_property, diag::warn_impl_required_in_category_for_class_property, diag::warn_setter_getter_impl_required, diag::warn_setter_getter_impl_required_in_category, -1,
  /* DiagArray685 */ diag::warn_property_implicitly_mismatched, -1,
  /* DiagArray686 */ diag::warn_cocoa_naming_owned_rule, -1,
  /* DiagArray687 */ diag::warn_objc_property_default_assign_on_object, diag::warn_objc_property_no_assignment_attribute, -1,
  /* DiagArray688 */ diag::warn_autosynthesis_property_in_superclass, diag::warn_no_autosynthesis_property, diag::warn_no_autosynthesis_shared_ivar_property, -1,
  /* DiagArray689 */ diag::warn_category_method_impl_match, -1,
  /* DiagArray690 */ diag::warn_auto_synthesizing_protocol_property, -1,
  /* DiagArray691 */ diag::warn_objc_redundant_qualified_class_type, -1,
  /* DiagArray692 */ diag::warn_objc_readonly_property_has_setter, -1,
  /* DiagArray694 */ diag::warn_objc_redundant_literal_use, -1,
  /* DiagArray695 */ diag::warn_objc_root_class_missing, -1,
  /* DiagArray697 */ diag::warn_impcast_float_to_objc_signed_char_bool, -1,
  /* DiagArray698 */ diag::warn_impcast_int_to_objc_signed_char_bool, -1,
  /* DiagArray699 */ diag::warn_objc_string_literal_comparison, -1,
  /* DiagArray700 */ diag::warn_concatenated_nsarray_literal, -1,
  /* DiagArray701 */ diag::warn_objc_unsafe_perform_selector, -1,
  /* DiagArray702 */ diag::ext_cpp_octal_literal, -1,
  /* DiagArray703 */ diag::warn_odr_different_num_template_parameters, diag::warn_odr_different_template_parameter_kind, diag::warn_odr_field_type_inconsistent, diag::warn_odr_function_type_inconsistent, diag::warn_odr_ivar_type_inconsistent, diag::warn_odr_non_type_parameter_type_inconsistent, diag::warn_odr_objc_method_num_params_inconsistent, diag::warn_odr_objc_method_param_type_inconsistent, diag::warn_odr_objc_method_result_type_inconsistent, diag::warn_odr_objc_method_variadic_inconsistent, diag::warn_odr_objc_property_impl_kind_inconsistent, diag::warn_odr_objc_property_type_inconsistent, diag::warn_odr_objc_superclass_inconsistent, diag::warn_odr_objc_synthesize_ivar_inconsistent, diag::warn_odr_parameter_pack_non_pack, diag::warn_odr_tag_type_inconsistent, diag::warn_odr_variable_multiple_def, diag::warn_odr_variable_type_inconsistent, -1,
  /* DiagArray704 */ diag::warn_old_style_cast, -1,
  /* DiagArray707 */ diag::warn_acc_confusing_routine_name, -1,
  /* DiagArray708 */ diag::warn_acc_deprecated_alias_name, -1,
  /* DiagArray709 */ diag::warn_acc_if_self_conflict, -1,
  /* DiagArray710 */ diag::ext_opencl_ext_vector_type_rgba_selector, -1,
  /* DiagArray712 */ diag::ext_omp_attributes, -1,
  /* DiagArray713 */ diag::warn_omp_alignment_not_power_of_two, diag::warn_omp_allocate_thread_on_task_target_directive, diag::warn_omp_ctx_incompatible_property_for_selector, diag::warn_omp_ctx_incompatible_score_for_property, diag::warn_omp_ctx_incompatible_selector_for_set, diag::warn_omp_ctx_selector_without_properties, diag::warn_omp_declare_variant_ctx_mutiple_use, diag::warn_omp_declare_variant_ctx_not_a_property, diag::warn_omp_declare_variant_ctx_not_a_selector, diag::warn_omp_declare_variant_ctx_not_a_set, diag::warn_omp_declare_variant_expected, diag::warn_omp_declare_variant_string_literal_or_identifier, diag::warn_omp_linear_step_zero, diag::warn_omp_more_one_device_type_clause, diag::warn_omp_more_one_interop_type, diag::warn_omp_more_one_omp_all_memory, diag::warn_omp_unknown_assumption_clause_missing_id, diag::warn_omp_unknown_assumption_clause_without_args, diag::warn_omp_used_different_allocator, -1,
  /* DiagArray714 */ diag::warn_omp_invalid_attribute_for_ompx_attributes, -1,
  /* DiagArray715 */ diag::ext_omp_loop_not_canonical_init, diag::warn_omp_loop_64_bit_var, -1,
  /* DiagArray716 */ diag::warn_omp_non_trivial_type_mapped, -1,
  /* DiagArray717 */ diag::warn_drv_omp_offload_target_duplicate, diag::warn_omp_declare_target_after_first_use, diag::warn_omp_not_in_target_context, -1,
  /* DiagArray718 */ diag::warn_throw_not_valid_on_target, diag::warn_try_not_valid_on_target, -1,
  /* DiagArray719 */ diag::warn_drv_fine_grained_bitfield_accesses_ignored, diag::warn_drv_fjmc_for_elf_only, diag::warn_drv_jmc_requires_debuginfo, diag::warn_drv_loongarch_conflicting_implied_val, diag::warn_drv_moutline_atomics_unsupported_opt, diag::warn_drv_moutline_unsupported_opt, diag::warn_drv_needs_hvx, diag::warn_drv_ps_force_pic, diag::warn_drv_unsupported_diag_option_for_flang, diag::warn_drv_unsupported_longcalls, diag::warn_drv_unsupported_openmp_library, diag::warn_drv_unsupported_option_for_flang, diag::warn_drv_unsupported_option_for_offload_arch_req_feature, diag::warn_drv_unsupported_option_for_processor, diag::warn_drv_unsupported_option_for_target, diag::warn_drv_unsupported_pic_with_mabicalls, diag::warn_target_override_arm64ec, -1,
  /* DiagArray720 */ diag::ext_typecheck_ordered_comparison_of_function_pointers, diag::warn_typecheck_ordered_comparison_of_function_pointers, -1,
  /* DiagArray721 */ diag::ext_out_of_line_declaration, -1,
  /* DiagArray722 */ diag::ext_use_out_of_scope_declaration, -1,
  /* DiagArray723 */ diag::warn_overaligned_type, -1,
  /* DiagArray725 */ diag::ext_string_too_long, -1,
  /* DiagArray726 */ diag::warn_overloaded_shift_in_comparison, -1,
  /* DiagArray727 */ diag::warn_overloaded_virtual, -1,
  /* DiagArray729 */ diag::warn_fe_override_module, -1,
  /* DiagArray730 */ diag::warn_conflicting_overriding_param_modifiers, diag::warn_conflicting_overriding_param_types, diag::warn_conflicting_overriding_ret_type_modifiers, diag::warn_conflicting_overriding_ret_types, diag::warn_conflicting_overriding_variadic, diag::warn_non_contravariant_overriding_param_types, diag::warn_non_covariant_overriding_ret_types, -1,
  /* DiagArray731 */ diag::warn_drv_overriding_option, -1,
  /* DiagArray732 */ diag::warn_unnecessary_packed, -1,
  /* DiagArray733 */ diag::warn_unpacked_field, -1,
  /* DiagArray734 */ diag::warn_padded_struct_anon_field, diag::warn_padded_struct_field, diag::warn_padded_struct_size, -1,
  /* DiagArray735 */ diag::warn_padded_struct_anon_bitfield, diag::warn_padded_struct_bitfield, -1,
  /* DiagArray736 */ diag::warn_condition_is_assignment, diag::warn_consecutive_comparison, diag::warn_precedence_bitwise_rel, diag::warn_precedence_conditional, -1,
  /* DiagArray737 */ diag::warn_equality_with_extra_parens, -1,
  /* DiagArray739 */ diag::remark_fe_backend_optimization_remark, -1,
  /* DiagArray740 */ diag::remark_fe_backend_optimization_remark_analysis, diag::remark_fe_backend_optimization_remark_analysis_aliasing, diag::remark_fe_backend_optimization_remark_analysis_fpcommute, -1,
  /* DiagArray741 */ diag::warn_fe_backend_optimization_failure, -1,
  /* DiagArray742 */ diag::remark_fe_backend_optimization_remark_missed, -1,
  /* DiagArray743 */ diag::warn_module_uses_date_time, -1,
  /* DiagArray744 */ diag::warn_pch_vfsoverlay_mismatch, -1,
  /* DiagArray745 */ diag::compat_pre_cxx11_enum_fixed_underlying_type, diag::ext_aggregate_init_not_constant, diag::ext_anonymous_record_with_type, diag::ext_anonymous_struct_union_qualified, diag::ext_array_size_conversion, diag::ext_auto_new_list_init, diag::ext_binary_literal, diag::ext_c23_enum_fixed_underlying_type, diag::ext_c23_enum_value_not_int, diag::ext_c23_non_local_variable_decl_in_for, diag::ext_c23_non_variable_decl_in_for, diag::ext_c99_array_usage, diag::ext_c99_compound_literal, diag::ext_c99_feature, diag::ext_c99_flexible_array_member, diag::ext_c99_variable_decl_in_for_loop, diag::ext_c_empty_initializer, diag::ext_c_missing_varargs_arg, diag::ext_c_nullptr, diag::ext_cast_fn_obj, diag::ext_clang_diagnose_if, diag::ext_clang_enable_if, diag::ext_cxx26_raw_string_literal_character_set, diag::ext_cxx_missing_varargs_arg, diag::ext_delete_with_message, diag::ext_designated_init, diag::ext_dtor_name_ambiguous, diag::ext_dtor_named_in_wrong_scope, diag::ext_duplicate_declspec, diag::ext_ellipsis_exception_spec, diag::ext_empty_fnmacro_arg, diag::ext_enumerator_list_comma_c, diag::ext_enumerator_list_comma_cxx, diag::ext_explicit_instantiation_without_qualified_id, diag::ext_expr_not_ice, diag::ext_extern_template, diag::ext_extra_semi, diag::ext_forward_ref_enum, diag::ext_freestanding_complex, diag::ext_gnu_array_range, diag::ext_hex_constant_invalid, diag::ext_hex_literal_invalid, diag::ext_ident_list_in_param, diag::ext_in_class_initializer_non_constant, diag::ext_integer_complement_complex, diag::ext_internal_in_extern_inline_quiet, diag::ext_line_comment, diag::ext_main_no_return, diag::ext_main_used, diag::ext_mixed_decls_code, diag::ext_multi_line_line_comment, diag::ext_named_variadic_macro, diag::ext_nonstandard_escape, diag::ext_ns_enum_attribute, diag::ext_opencl_double_without_pragma, diag::ext_paste_comma, diag::ext_pointer_to_const_ref_member_on_rvalue, diag::ext_pp_bad_vaargs_use, diag::ext_pp_comma_expr, diag::ext_pp_ident_directive, diag::ext_pp_line_too_big, diag::ext_pp_opencl_variadic_macros, diag::ext_pp_warning_directive, diag::ext_restrict_on_array_of_pointers_c23, diag::ext_return_has_void_expr, diag::ext_rvalue_to_reference_access_ctor, diag::ext_rvalue_to_reference_temp_copy_no_viable, diag::ext_sizeof_alignof_function_type, diag::ext_sizeof_alignof_void_type, diag::ext_subscript_non_lvalue, diag::ext_thread_before, diag::ext_type_defined_in_offsetof, diag::ext_typecheck_addrof_void, diag::ext_typecheck_cast_nonscalar, diag::ext_typecheck_comparison_of_fptr_to_void, diag::ext_typecheck_cond_one_void, diag::ext_typecheck_convert_pointer_void_func, diag::ext_typecheck_ordered_comparison_of_pointer_and_zero, diag::ext_variadic_macro, diag::ext_vla, diag::ext_vla_cxx_in_gnu_mode, diag::ext_vla_cxx_in_gnu_mode_static_assert, diag::warn_defined_in_function_type_macro, diag::warn_ext_c23_attributes, diag::warn_ext_cxx11_attributes, diag::warn_format_cmp_specifier_mismatch_pedantic, diag::warn_format_conversion_argument_type_mismatch_pedantic, diag::warn_kern_is_method, diag::warn_strict_prototypes, -1,
  /* DiagArray746 */ diag::warn_opencl_unsupported_core_feature, diag::warn_pragma_extension_is_core, -1,
  /* DiagArray748 */ diag::warn_perf_constraint_implies_noexcept, -1,
  /* DiagArray749 */ diag::warn_pessimizing_move_on_initialization, diag::warn_pessimizing_move_on_return, -1,
  /* DiagArray750 */ diag::ext_sizeof_alignof_function_type, diag::ext_sizeof_alignof_void_type, diag::warn_sub_ptr_zero_size_types, -1,
  /* DiagArray751 */ diag::warn_cast_nonnull_to_bool, diag::warn_impcast_pointer_to_bool, -1,
  /* DiagArray752 */ diag::warn_pointer_compare, -1,
  /* DiagArray753 */ diag::ext_typecheck_comparison_of_pointer_integer, -1,
  /* DiagArray754 */ diag::ext_typecheck_convert_incompatible_pointer_sign, -1,
  /* DiagArray755 */ diag::warn_pointer_to_enum_cast, -1,
  /* DiagArray756 */ diag::warn_pointer_to_int_cast, -1,
  /* DiagArray757 */ diag::ext_typecheck_cond_incompatible_pointers, -1,
  /* DiagArray758 */ diag::warn_poison_system_directories, -1,
  /* DiagArray759 */ diag::warn_potentially_direct_selector_expression, -1,
  /* DiagArray760 */ diag::warn_side_effects_typeid, -1,
  /* DiagArray761 */ diag::warn_pragma_attribute_unused, -1,
  /* DiagArray762 */ diag::pp_pragma_once_in_main_file, -1,
  /* DiagArray763 */ diag::warn_pragma_pack_modified_after_include, diag::warn_pragma_pack_no_pop_eof, -1,
  /* DiagArray764 */ diag::warn_pragma_pack_non_default_at_include, -1,
  /* DiagArray765 */ diag::pp_pragma_sysheader_in_main_file, -1,
  /* DiagArray766 */ diag::warn_no_support_for_eval_method_source_on_m32, diag::warn_redefine_extname_not_applied, -1,
  /* DiagArray767 */ diag::compat_cxx14_constexpr_body_multiple_return, diag::compat_cxx14_constexpr_local_var, diag::compat_cxx14_constexpr_type_definition, diag::compat_cxx14_cxx14_constexpr_body_invalid_stmt, diag::compat_cxx14_variable_template, diag::warn_cxx11_compat_constexpr_body_no_return, diag::warn_cxx11_compat_decltype_auto_type_specifier, diag::warn_cxx11_compat_deduced_return_type, diag::warn_cxx11_compat_digit_separator, diag::warn_cxx11_compat_generic_lambda, diag::warn_cxx11_compat_init_capture, -1,
  /* DiagArray769 */ diag::compat_cxx17_decomp_decl, diag::compat_cxx17_inline_variable, diag::warn_cxx14_compat_class_template_argument_deduction, diag::warn_cxx14_compat_constexpr_if, diag::warn_cxx14_compat_constexpr_on_lambda, diag::warn_cxx14_compat_fold_expression, diag::warn_cxx14_compat_init_statement, diag::warn_cxx14_compat_nested_namespace_definition, diag::warn_cxx14_compat_star_this_lambda_capture, diag::warn_cxx14_compat_static_assert_no_message, diag::warn_cxx14_compat_template_nontype_parm_auto_type, diag::warn_cxx14_compat_template_template_param_typename, diag::warn_cxx14_compat_u8_character_literal, diag::warn_cxx14_compat_using_attribute_ns, diag::warn_cxx17_compat_multi_using_declaration, diag::warn_cxx17_compat_using_declaration_pack, diag::warn_for_range_begin_end_types_differ, -1,
  /* DiagArray770 */ diag::warn_cxx14_compat_ns_enum_attribute, diag::warn_cxx17_hex_literal, -1,
  /* DiagArray771 */ diag::compat_cxx20_adl_only_template_id, diag::compat_cxx20_constexpr_ctor_missing_init, diag::compat_cxx20_constexpr_function_try_block, diag::compat_cxx20_constexpr_local_var_no_init, diag::compat_cxx20_constexpr_union_ctor_no_init, diag::compat_cxx20_ctad_for_alias_templates, diag::compat_cxx20_cxx20_constexpr_body_invalid_stmt, diag::compat_cxx20_decomp_decl_spec, diag::warn_cxx17_compat_bitfield_member_init, diag::warn_cxx17_compat_capture_binding, diag::warn_cxx17_compat_constexpr_virtual, diag::warn_cxx17_compat_defaulted_comparison, diag::warn_cxx17_compat_defaulted_method_type_mismatch, diag::warn_cxx17_compat_equals_this_lambda_capture, diag::warn_cxx17_compat_explicit_bool, diag::warn_cxx17_compat_for_range_init_stmt, diag::warn_cxx17_compat_init_capture_pack, diag::warn_cxx17_compat_inline_nested_namespace_definition, diag::warn_cxx17_compat_lambda_def_ctor_assign, diag::warn_cxx17_compat_lambda_template_parameter_list, diag::warn_cxx17_compat_missing_varargs_arg, diag::warn_cxx17_compat_spaceship, diag::warn_cxx17_compat_template_nontype_parm_type, diag::warn_cxx17_compat_unicode_type, diag::warn_cxx17_compat_using_decl_class_member_enumerator, diag::warn_cxx17_compat_using_decl_non_member_enumerator, diag::warn_cxx17_compat_using_decl_scoped_enumerator, diag::warn_cxx17_compat_using_enum_declaration, -1,
  /* DiagArray772 */ diag::warn_cxx17_compat_designated_init, diag::warn_cxx17_compat_pointer_to_const_ref_member_on_rvalue, -1,
  /* DiagArray773 */ diag::compat_cxx23_constexpr_static_var, diag::compat_cxx23_cxx23_constexpr_body_invalid_stmt, diag::ext_subscript_overload, diag::warn_cxx20_alias_in_init_statement, diag::warn_cxx20_compat_auto_expr, diag::warn_cxx20_compat_consteval_if, diag::warn_cxx20_compat_constexpr_var, diag::warn_cxx20_compat_decl_attrs_on_lambda, diag::warn_cxx20_compat_label_end_of_compound_statement, diag::warn_cxx20_compat_operator_overload_static, diag::warn_cxx20_compat_size_t_suffix, diag::warn_cxx20_compat_static_lambda, diag::warn_cxx23_compat_pp_directive, diag::warn_cxx23_compat_warning_directive, diag::warn_cxx23_delimited_escape_sequence, -1,
  /* DiagArray775 */ diag::compat_cxx26_decomp_decl_cond, diag::warn_cxx20_compat_static_assert_user_generated_message, diag::warn_cxx23_compat_binding_pack, diag::warn_cxx23_compat_decl_attrs_on_binding, diag::warn_cxx23_delete_with_message, diag::warn_cxx23_pack_indexing, diag::warn_cxx23_placeholder_var_definition, diag::warn_cxx23_variadic_friends, diag::warn_cxx26_compat_raw_string_literal_character_set, diag::warn_cxx26_type_aware_allocators, diag::warn_pre_cxx26_ambiguous_pack_indexing_type, -1,
  /* DiagArray779 */ diag::warn_c11_compat_keyword, -1,
  /* DiagArray781 */ diag::warn_c17_compat_ellipsis_only_parameter, diag::warn_c17_compat_enum_fixed_underlying_type, diag::warn_c17_compat_enum_value_not_int, diag::warn_c17_compat_static_assert_no_message, diag::warn_c17_compat_u8_character_literal, diag::warn_c17_compat_va_start_one_arg, diag::warn_c17_non_local_variable_decl_in_for, diag::warn_c17_non_variable_decl_in_for, diag::warn_c23_compat_binary_literal, diag::warn_c23_compat_bitint_suffix, diag::warn_c23_compat_digit_separator, diag::warn_c23_compat_empty_initializer, diag::warn_c23_compat_keyword, diag::warn_c23_compat_label_end_of_compound_statement, diag::warn_c23_compat_label_followed_by_declaration, diag::warn_c23_compat_literal_ucn_control_character, diag::warn_c23_compat_literal_ucn_escape_basic_scs, diag::warn_c23_compat_pp_directive, diag::warn_c23_compat_restrict_on_array_of_pointers, diag::warn_c23_compat_warning_directive, diag::warn_compat_pp_embed_directive, diag::warn_pre_c23_compat_attributes, -1,
  /* DiagArray785 */ diag::warn_c2y_compat_alignof_incomplete_array, diag::warn_c2y_compat_assoc_type_incomplete, diag::warn_c2y_compat_generic_with_type_arg, diag::warn_c2y_compat_increment_complex, diag::warn_c2y_compat_keyword, diag::warn_c2y_compat_octal_literal, diag::warn_c2y_delimited_escape_sequence, diag::warn_c23_compat_case_range, diag::warn_c23_compat_imaginary_constant, -1,
  /* DiagArray787 */ diag::warn_omp51_compat_attributes, -1,
  /* DiagArray788 */ diag::ext_predef_outside_function, -1,
  /* DiagArray789 */ diag::warn_preferred_type_bitfield_too_small_for_enum, diag::warn_preferred_type_signed_bitfield_enum_conversion, diag::warn_preferred_type_unsigned_bitfield_assigned_signed_enum, -1,
  /* DiagArray791 */ diag::warn_private_extern, -1,
  /* DiagArray792 */ diag::warn_use_of_private_header_outside_module, -1,
  /* DiagArray793 */ diag::warn_mmap_mismatched_private_module_name, diag::warn_mmap_mismatched_private_submodule, diag::warn_mmap_redundant_export_as, diag::warn_no_priv_submodule_use_toplevel, -1,
  /* DiagArray794 */ diag::warn_profile_data_missing, -1,
  /* DiagArray795 */ diag::warn_profile_data_out_of_date, -1,
  /* DiagArray796 */ diag::warn_profile_data_unprofiled, -1,
  /* DiagArray797 */ diag::warn_property_access_suggest, -1,
  /* DiagArray798 */ diag::warn_property_attr_mismatch, diag::warn_property_attribute, diag::warn_property_redecl_getter_mismatch, diag::warn_readonly_property, -1,
  /* DiagArray799 */ diag::warn_unimplemented_protocol_method, -1,
  /* DiagArray800 */ diag::warn_protocol_property_mismatch, -1,
  /* DiagArray801 */ diag::warn_avx_calling_convention, -1,
  /* DiagArray802 */ diag::warn_ptrauth_auth_null_pointer, diag::warn_ptrauth_sign_null_pointer, -1,
  /* DiagArray803 */ diag::err_func_returning_qualified_void, -1,
  /* DiagArray804 */ diag::warn_quoted_include_in_framework_header, -1,
  /* DiagArray806 */ diag::warn_for_range_ref_binds_ret_temp, -1,
  /* DiagArray807 */ diag::warn_for_range_const_ref_binds_temp_built_from_ref, diag::warn_for_range_copy, -1,
  /* DiagArray808 */ diag::warn_var_decl_not_read_only, -1,
  /* DiagArray809 */ diag::warn_auto_readonly_iboutlet_property, -1,
  /* DiagArray810 */ diag::warn_bad_receiver_type, -1,
  /* DiagArray811 */ diag::warn_receiver_forward_class, diag::warn_receiver_forward_instance, -1,
  /* DiagArray812 */ diag::ext_member_redeclared, -1,
  /* DiagArray813 */ diag::err_drv_reduced_module_output_overrided, -1,
  /* DiagArray814 */ diag::warn_sycl_entry_point_redundant_declaration, -1,
  /* DiagArray815 */ diag::warn_consteval_if_always_true, -1,
  /* DiagArray817 */ diag::warn_redundant_move_on_return, -1,
  /* DiagArray818 */ diag::warn_redundant_parens_around_declarator, -1,
  /* DiagArray819 */ diag::ext_register_storage_class, -1,
  /* DiagArray820 */ diag::warn_reinterpret_different_from_static, -1,
  /* DiagArray821 */ diag::remark_fe_backend_plugin, -1,
  /* DiagArray823 */ diag::warn_initializer_out_of_order, diag::warn_some_initializers_out_of_order, -1,
  /* DiagArray824 */ diag::ext_designated_init_reordered, -1,
  /* DiagArray825 */ diag::warn_objc_requires_super_protocol, -1,
  /* DiagArray826 */ diag::warn_pp_macro_is_reserved_attribute_id, -1,
  /* DiagArray828 */ diag::warn_reserved_extern_symbol, -1,
  /* DiagArray829 */ diag::warn_pp_macro_is_reserved_id, -1,
  /* DiagArray830 */ diag::warn_reserved_module_name, -1,
  /* DiagArray831 */ diag::ext_ms_reserved_user_defined_literal, diag::ext_reserved_user_defined_literal, -1,
  /* DiagArray832 */ diag::warn_pragma_restrict_expansion_macro_use, -1,
  /* DiagArray833 */ diag::ext_retained_language_linkage, -1,
  /* DiagArray835 */ diag::ext_return_has_expr, diag::ext_return_missing_expr, diag::warn_return_missing_expr, -1,
  /* DiagArray836 */ diag::warn_musttail_local_temp_addr_ref, diag::warn_ret_addr_label, diag::warn_ret_local_temp_addr_ref, diag::warn_ret_stack_addr_ref, -1,
  /* DiagArray838 */ diag::warn_falloff_nonvoid, -1,
  /* DiagArray839 */ diag::warn_return_value_udt, diag::warn_return_value_udt_incomplete, -1,
  /* DiagArray840 */ diag::ext_ovl_rewrite_equalequal_not_bool, -1,
  /* DiagArray841 */ diag::remark_cc1_round_trip_generated, -1,
  /* DiagArray842 */ diag::warn_no_dynamic_cast_with_rtti_disabled, diag::warn_no_typeid_with_rtti_disabled, -1,
  /* DiagArray843 */ diag::remark_sanitize_address_insert_extra_padding_accepted, diag::remark_sanitize_address_insert_extra_padding_rejected, -1,
  /* DiagArray844 */ diag::warn_drv_sarif_format_unstable, -1,
  /* DiagArray845 */ diag::remark_pp_search_path_usage, -1,
  /* DiagArray846 */ diag::warn_attribute_section_on_redeclaration, diag::warn_duplicate_codeseg_attribute, diag::warn_mismatched_section, -1,
  /* DiagArray847 */ diag::warn_unimplemented_selector, -1,
  /* DiagArray848 */ diag::warn_multiple_selectors, -1,
  /* DiagArray849 */ diag::warn_self_assignment_builtin, -1,
  /* DiagArray850 */ diag::warn_identity_field_assign, -1,
  /* DiagArray851 */ diag::warn_self_assignment_overloaded, -1,
  /* DiagArray852 */ diag::warn_self_move, -1,
  /* DiagArray853 */ diag::warn_semicolon_before_method_body, -1,
  /* DiagArray854 */ diag::warn_missing_sentinel, diag::warn_not_enough_argument, -1,
  /* DiagArray856 */ diag::warn_fe_serialized_diag_failure, diag::warn_fe_serialized_diag_failure_during_finalization, diag::warn_fe_serialized_diag_merge_failure, -1,
  /* DiagArray857 */ diag::warn_decl_shadow, -1,
  /* DiagArray859 */ diag::warn_shadow_field, -1,
  /* DiagArray860 */ diag::warn_ctor_parm_shadows_field, -1,
  /* DiagArray861 */ diag::warn_modifying_shadowing_decl, -1,
  /* DiagArray862 */ diag::warn_ivar_use_hidden, -1,
  /* DiagArray863 */ diag::warn_decl_shadow_uncaptured_local, -1,
  /* DiagArray864 */ diag::warn_shift_bool, -1,
  /* DiagArray865 */ diag::warn_shift_negative, -1,
  /* DiagArray866 */ diag::warn_shift_gt_typewidth, -1,
  /* DiagArray867 */ diag::warn_shift_lhs_negative, -1,
  /* DiagArray868 */ diag::warn_addition_in_bitshift, -1,
  /* DiagArray869 */ diag::warn_shift_result_gt_typewidth, -1,
  /* DiagArray870 */ diag::warn_shift_result_sets_sign_bit, -1,
  /* DiagArray871 */ diag::warn_impcast_integer_64_32, -1,
  /* DiagArray872 */ diag::warn_mixed_sign_comparison, -1,
  /* DiagArray873 */ diag::warn_impcast_integer_sign, diag::warn_impcast_integer_sign_conditional, diag::warn_impcast_nonnegative_result, -1,
  /* DiagArray875 */ diag::warn_no_underlying_type_specified_for_enum_bitfield, -1,
  /* DiagArray876 */ diag::ext_wchar_t_sign_spec, -1,
  /* DiagArray877 */ diag::warn_impcast_single_bit_bitield_precision_constant, -1,
  /* DiagArray878 */ diag::warn_sizeof_array_param, -1,
  /* DiagArray879 */ diag::warn_sizeof_array_decay, -1,
  /* DiagArray880 */ diag::warn_division_sizeof_array, -1,
  /* DiagArray881 */ diag::warn_division_sizeof_ptr, -1,
  /* DiagArray882 */ diag::warn_sizeof_pointer_expr_memaccess, diag::warn_sizeof_pointer_type_memaccess, -1,
  /* DiagArray883 */ diag::warn_slash_u_filename, -1,
  /* DiagArray884 */ diag::warn_slh_does_not_support_asm_goto, -1,
  /* DiagArray885 */ diag::remark_sloc_usage, -1,
  /* DiagArray886 */ diag::warn_sometimes_uninit_var, -1,
  /* DiagArray887 */ diag::warn_fe_source_mgr, -1,
  /* DiagArray888 */ diag::warn_acc_routine_unimplemented, diag::warn_drv_openacc_without_cir, diag::warn_pragma_acc_ignored, -1,
  /* DiagArray889 */ diag::warn_omp_declare_variant_after_emitted, diag::warn_omp_declare_variant_after_used, diag::warn_omp_declare_variant_marked_as_declare_variant, diag::warn_omp_declare_variant_score_not_constant, diag::warn_omp_nesting_simd, diag::warn_omp_unterminated_declare_target, diag::warn_pragma_omp_ignored, diag::warn_unknown_declare_variant_isa_trait, -1,
  /* DiagArray890 */ diag::warn_sampler_initializer_invalid_bits, -1,
  /* DiagArray892 */ diag::warn_stack_exhausted, -1,
  /* DiagArray893 */ diag::warn_stack_clash_protection_inline_asm, -1,
  /* DiagArray894 */ diag::ext_in_class_initializer_float_type_cxx11, -1,
  /* DiagArray895 */ diag::ext_internal_in_extern_inline, diag::ext_internal_in_extern_inline_quiet, -1,
  /* DiagArray896 */ diag::warn_static_inline_explicit_inst_ignored, -1,
  /* DiagArray897 */ diag::warn_static_local_in_extern_inline, -1,
  /* DiagArray898 */ diag::warn_static_self_reference_in_init, -1,
  /* DiagArray899 */ diag::warn_drv_libstdcxx_not_found, -1,
  /* DiagArray911 */ diag::warn_strict_potentially_direct_selector_expression, -1,
  /* DiagArray912 */ diag::ext_compat_template_param_shadow, -1,
  /* DiagArray913 */ diag::warn_strict_prototypes, -1,
  /* DiagArray914 */ diag::warn_strict_multiple_method_decl, -1,
  /* DiagArray915 */ diag::warn_stringcompare, -1,
  /* DiagArray916 */ diag::warn_concatenated_literal_array_init, -1,
  /* DiagArray917 */ diag::warn_impcast_string_literal_to_bool, -1,
  /* DiagArray918 */ diag::warn_string_plus_char, -1,
  /* DiagArray919 */ diag::warn_string_plus_int, -1,
  /* DiagArray920 */ diag::warn_strlcpycat_wrong_size, -1,
  /* DiagArray921 */ diag::warn_strncat_large_size, diag::warn_strncat_src_size, diag::warn_strncat_wrong_size, -1,
  /* DiagArray922 */ diag::warn_suggest_destructor_marked_not_override_overriding, -1,
  /* DiagArray923 */ diag::warn_suggest_function_marked_not_override_overriding, -1,
  /* DiagArray924 */ diag::ext_typecheck_base_super, -1,
  /* DiagArray925 */ diag::warn_suspicious_bzero_size, -1,
  /* DiagArray927 */ diag::warn_attr_swift_name_decl_kind, diag::warn_attr_swift_name_decl_missing_params, diag::warn_attr_swift_name_function, diag::warn_attr_swift_name_getter_parameters, diag::warn_attr_swift_name_invalid_identifier, diag::warn_attr_swift_name_missing_parameters, diag::warn_attr_swift_name_multiple_selfs, diag::warn_attr_swift_name_num_params, diag::warn_attr_swift_name_setter_parameters, diag::warn_attr_swift_name_subscript_getter_newValue, diag::warn_attr_swift_name_subscript_invalid_parameter, diag::warn_attr_swift_name_subscript_setter_multiple_newValues, diag::warn_attr_swift_name_subscript_setter_no_newValue, -1,
  /* DiagArray928 */ diag::warn_case_value_overflow, diag::warn_missing_case, diag::warn_not_in_enum, -1,
  /* DiagArray929 */ diag::warn_bool_switch_condition, -1,
  /* DiagArray930 */ diag::warn_switch_default, -1,
  /* DiagArray931 */ diag::warn_def_missing_case, -1,
  /* DiagArray932 */ diag::warn_sync_op_misaligned, -1,
  /* DiagArray933 */ diag::warn_sync_fetch_and_nand_semantics_change, -1,
  /* DiagArray935 */ diag::warn_target_clone_mixed_values, -1,
  /* DiagArray936 */ diag::warn_comparison_bitwise_always, diag::warn_comparison_bitwise_or, -1,
  /* DiagArray937 */ diag::warn_alignment_builtin_useless, diag::warn_comparison_always, -1,
  /* DiagArray938 */ diag::warn_integer_constants_in_conditional_always_true, diag::warn_left_shift_always, diag::warn_tautological_bool_compare, -1,
  /* DiagArray940 */ diag::warn_out_of_range_compare, -1,
  /* DiagArray941 */ diag::warn_tautological_negation_and_compare, diag::warn_tautological_negation_or_compare, -1,
  /* DiagArray942 */ diag::warn_tautological_compare_objc_bool, -1,
  /* DiagArray943 */ diag::warn_tautological_overlap_comparison, -1,
  /* DiagArray944 */ diag::warn_nonnull_expr_compare, diag::warn_null_pointer_compare, -1,
  /* DiagArray945 */ diag::warn_tautological_constant_compare, -1,
  /* DiagArray946 */ diag::warn_address_of_reference_null_compare, diag::warn_this_null_compare, -1,
  /* DiagArray947 */ diag::warn_unsigned_char_always_true_comparison, -1,
  /* DiagArray948 */ diag::warn_unsigned_enum_always_true_comparison, -1,
  /* DiagArray949 */ diag::warn_unsigned_always_true_comparison, -1,
  /* DiagArray950 */ diag::warn_tautological_compare_value_range, -1,
  /* DiagArray951 */ diag::warn_tcb_enforcement_violation, -1,
  /* DiagArray952 */ diag::ext_template_after_declarative_nns, -1,
  /* DiagArray953 */ diag::warn_tentative_incomplete_array, -1,
  /* DiagArray954 */ diag::ext_typecheck_decl_incomplete_type, -1,
  /* DiagArray956 */ diag::warn_acquired_before, diag::warn_acquired_before_after_cycle, diag::warn_cannot_resolve_lock, diag::warn_double_lock, diag::warn_expect_fewer_underlying_mutexes, diag::warn_expect_more_underlying_mutexes, diag::warn_expecting_lock_held_on_loop, diag::warn_expecting_locked, diag::warn_fun_excludes_mutex, diag::warn_fun_requires_lock, diag::warn_fun_requires_negative_cap, diag::warn_lock_exclusive_and_shared, diag::warn_lock_some_predecessors, diag::warn_no_unlock, diag::warn_unlock_but_no_lock, diag::warn_unlock_kind_mismatch, diag::warn_unmatched_underlying_mutexes, diag::warn_var_deref_requires_any_lock, diag::warn_var_deref_requires_lock, diag::warn_variable_requires_any_lock, diag::warn_variable_requires_lock, -1,
  /* DiagArray957 */ diag::warn_thread_attribute_argument_not_lockable, diag::warn_thread_attribute_decl_not_lockable, diag::warn_thread_attribute_decl_not_pointer, diag::warn_thread_attribute_ignored, diag::warn_thread_attribute_not_on_capability_member, diag::warn_thread_attribute_not_on_non_static_member, diag::warn_thread_attribute_not_on_scoped_lockable_param, -1,
  /* DiagArray958 */ diag::warn_thread_safety_beta, -1,
  /* DiagArray959 */ diag::warn_acquire_requires_negative_cap, -1,
  /* DiagArray960 */ diag::warn_guarded_pass_pointer, diag::warn_guarded_return_pointer, diag::warn_pt_guarded_pass_pointer, diag::warn_pt_guarded_return_pointer, -1,
  /* DiagArray961 */ diag::warn_fun_requires_lock_precise, diag::warn_var_deref_requires_lock_precise, diag::warn_variable_requires_lock_precise, -1,
  /* DiagArray962 */ diag::warn_guarded_pass_by_reference, diag::warn_pt_guarded_pass_by_reference, -1,
  /* DiagArray963 */ diag::warn_guarded_return_by_reference, diag::warn_pt_guarded_return_by_reference, -1,
  /* DiagArray964 */ diag::warn_thread_safety_verbose, -1,
  /* DiagArray965 */ diag::trigraph_converted, diag::trigraph_ends_block_comment, diag::trigraph_ignored, diag::trigraph_ignored_block_comment, -1,
  /* DiagArray967 */ diag::warn_type_safety_null_pointer_required, diag::warn_type_safety_type_mismatch, diag::warn_type_tag_for_datatype_wrong_kind, -1,
  /* DiagArray968 */ diag::ext_redefinition_of_typedef, -1,
  /* DiagArray969 */ diag::ext_typename_missing, -1,
  /* DiagArray970 */ diag::warn_fe_unable_to_open_stats_file, -1,
  /* DiagArray971 */ diag::warn_unaligned_access, -1,
  /* DiagArray972 */ diag::warn_imp_cast_drops_unaligned, -1,
  /* DiagArray973 */ diag::warn_unavailable_fwdclass_message, -1,
  /* DiagArray974 */ diag::warn_undeclared_selector, diag::warn_undeclared_selector_with_typo, -1,
  /* DiagArray975 */ diag::warn_pp_undef_identifier, -1,
  /* DiagArray976 */ diag::warn_pp_undef_prefix, -1,
  /* DiagArray977 */ diag::warn_pp_undef_true_identifier, -1,
  /* DiagArray978 */ diag::warn_attribute_arm_za_builtin_no_za_state, -1,
  /* DiagArray979 */ diag::warn_attribute_arm_zt0_builtin_no_zt0_state, -1,
  /* DiagArray980 */ diag::warn_address_of_reference_bool_conversion, diag::warn_this_bool_conversion, -1,
  /* DiagArray981 */ diag::warn_func_template_missing, -1,
  /* DiagArray982 */ diag::warn_undefined_inline, -1,
  /* DiagArray983 */ diag::warn_undefined_internal, -1,
  /* DiagArray984 */ diag::ext_undefined_internal_type, -1,
  /* DiagArray985 */ diag::warn_pointer_indirection_from_incompatible_type, diag::warn_undefined_reinterpret_cast, -1,
  /* DiagArray986 */ diag::warn_var_template_missing, -1,
  /* DiagArray987 */ diag::warn_throw_underaligned_obj, -1,
  /* DiagArray988 */ diag::warn_side_effects_unevaluated_context, -1,
  /* DiagArray989 */ diag::warn_unguarded_availability, diag::warn_unguarded_availability_unavailable, -1,
  /* DiagArray990 */ diag::warn_unguarded_availability_new, diag::warn_unguarded_availability_unavailable_new, -1,
  /* DiagArray991 */ diag::warn_delimited_ucn_empty, diag::warn_delimited_ucn_incomplete, diag::warn_ucn_escape_incomplete, diag::warn_ucn_escape_no_digits, diag::warn_ucn_escape_surrogate, diag::warn_ucn_not_valid_in_c89, diag::warn_ucn_not_valid_in_c89_literal, -1,
  /* DiagArray992 */ diag::warn_utf8_symbol_homoglyph, -1,
  /* DiagArray993 */ diag::ext_unicode_whitespace, -1,
  /* DiagArray994 */ diag::warn_utf8_symbol_zero_width, -1,
  /* DiagArray995 */ diag::warn_base_class_is_uninit, diag::warn_field_is_uninit, diag::warn_no_constructor_for_refconst, diag::warn_reference_field_is_uninit, diag::warn_uninit_byref_blockvar_captured_by_block, diag::warn_uninit_self_reference_in_init, diag::warn_uninit_self_reference_in_reference_init, diag::warn_uninit_var, -1,
  /* DiagArray996 */ diag::warn_uninit_const_reference, -1,
  /* DiagArray997 */ diag::warn_field_requires_explicit_init, -1,
  /* DiagArray998 */ diag::warn_possible_object_duplication_init, diag::warn_possible_object_duplication_mutable, -1,
  /* DiagArray999 */ diag::warn_drv_potentially_misspelled_joined_argument, diag::warn_drv_unknown_argument_clang_cl, diag::warn_drv_unknown_argument_clang_cl_with_suggestion, -1,
  /* DiagArray1000 */ diag::warn_omp_assume_attribute_string_unknown, -1,
  /* DiagArray1001 */ diag::warn_unknown_attribute_ignored, -1,
  /* DiagArray1002 */ diag::warn_drv_new_cuda_version, diag::warn_drv_partially_supported_cuda_version, -1,
  /* DiagArray1003 */ diag::warn_pp_invalid_directive, -1,
  /* DiagArray1004 */ diag::ext_unknown_escape, -1,
  /* DiagArray1005 */ diag::ext_on_off_switch_syntax, diag::ext_pragma_syntax_eod, diag::ext_stdc_pragma_ignored, diag::warn_pragma_diagnostic_cannot_pop, diag::warn_pragma_diagnostic_invalid, diag::warn_pragma_diagnostic_invalid_option, diag::warn_pragma_diagnostic_invalid_token, diag::warn_pragma_exec_charset_expected, diag::warn_pragma_exec_charset_push_invalid, diag::warn_pragma_exec_charset_spec_invalid, diag::warn_pragma_ignored, diag::warn_pragma_include_alias_expected, diag::warn_pragma_include_alias_expected_filename, diag::warn_pragma_include_alias_mismatch_angle, diag::warn_pragma_include_alias_mismatch_quote, diag::warn_pragma_warning_expected, diag::warn_pragma_warning_expected_number, diag::warn_pragma_warning_push_level, diag::warn_pragma_warning_spec_invalid, diag::warn_stdc_fenv_round_not_supported, -1,
  /* DiagArray1006 */ diag::warn_unknown_sanitizer_ignored, -1,
  /* DiagArray1007 */ diag::warn_pragma_diagnostic_unknown_warning, diag::warn_unknown_diag_option, diag::warn_unknown_warning_specifier, -1,
  /* DiagArray1008 */ diag::ext_template_arg_unnamed_type, -1,
  /* DiagArray1009 */ diag::warn_unnecessary_virtual_specifier, -1,
  /* DiagArray1010 */ diag::warn_unneeded_internal_decl, diag::warn_unneeded_static_internal_decl, -1,
  /* DiagArray1011 */ diag::warn_unneeded_member_function, -1,
  /* DiagArray1012 */ diag::warn_unqualified_call_to_std_cast_function, -1,
  /* DiagArray1013 */ diag::warn_unreachable, -1,
  /* DiagArray1015 */ diag::warn_unreachable_break, -1,
  /* DiagArray1016 */ diag::warn_unreachable_fallthrough_attr, -1,
  /* DiagArray1017 */ diag::warn_unreachable_association, -1,
  /* DiagArray1018 */ diag::warn_unreachable_loop_increment, -1,
  /* DiagArray1019 */ diag::warn_unreachable_return, -1,
  /* DiagArray1020 */ diag::warn_unsafe_buffer_operation, diag::warn_unsafe_buffer_variable, -1,
  /* DiagArray1021 */ diag::warn_unsafe_buffer_usage_in_container, -1,
  /* DiagArray1022 */ diag::warn_unsafe_buffer_libc_call, -1,
  /* DiagArray1023 */ diag::warn_unsequenced_mod_mod, diag::warn_unsequenced_mod_use, -1,
  /* DiagArray1024 */ diag::warn_drv_no_floating_point_registers, diag::warn_drv_unsupported_float_abi_by_lib, -1,
  /* DiagArray1025 */ diag::warn_target_unsupported_abs2008, diag::warn_target_unsupported_abslegacy, -1,
  /* DiagArray1026 */ diag::warn_at_available_unchecked_use, -1,
  /* DiagArray1027 */ diag::warn_target_unsupported_compact_branches, -1,
  /* DiagArray1028 */ diag::warn_attribute_dll_instantiated_base_class, -1,
  /* DiagArray1029 */ diag::warn_fe_backend_unsupported_fp_exceptions, diag::warn_fe_backend_unsupported_fp_rounding, -1,
  /* DiagArray1030 */ diag::warn_template_qualified_friend_ignored, diag::warn_template_qualified_friend_unsupported, -1,
  /* DiagArray1031 */ diag::warn_drv_unsupported_gpopt, -1,
  /* DiagArray1032 */ diag::warn_target_unsupported_nan2008, diag::warn_target_unsupported_nanlegacy, -1,
  /* DiagArray1033 */ diag::warn_drv_dwarf_version_limited_by_target, diag::warn_drv_unsupported_debug_info_opt_for_target, -1,
  /* DiagArray1034 */ diag::warn_attribute_protected_visibility, -1,
  /* DiagArray1035 */ diag::ext_partial_specs_not_deducible, -1,
  /* DiagArray1038 */ diag::warn_unused_but_set_parameter, -1,
  /* DiagArray1039 */ diag::warn_unused_but_set_variable, -1,
  /* DiagArray1040 */ diag::warn_drv_diagnostics_hotness_requires_pgo, diag::warn_drv_diagnostics_misexpect_requires_pgo, diag::warn_drv_empty_joined_argument, diag::warn_drv_fraw_string_literals_in_cxx11, diag::warn_drv_input_file_unused, diag::warn_drv_input_file_unused_by_cpp, diag::warn_drv_invalid_arch_name_with_suggestion, diag::warn_drv_large_data_threshold_invalid_code_model, diag::warn_drv_preprocessed_input_file_unused, diag::warn_drv_unused_argument, diag::warn_drv_unused_x, diag::warn_ignored_clang_option, diag::warn_ignoring_fdiscard_for_bitcode, diag::warn_ignoring_verify_debuginfo_preserve_export, -1,
  /* DiagArray1041 */ diag::warn_unused_comparison, -1,
  /* DiagArray1042 */ diag::warn_unused_const_variable, -1,
  /* DiagArray1043 */ diag::warn_unused_exception_param, -1,
  /* DiagArray1044 */ diag::warn_unused_function, -1,
  /* DiagArray1045 */ diag::warn_unused_property_expr, -1,
  /* DiagArray1046 */ diag::warn_unused_label, -1,
  /* DiagArray1047 */ diag::warn_unused_lambda_capture, -1,
  /* DiagArray1048 */ diag::warn_unused_local_typedef, -1,
  /* DiagArray1050 */ diag::pp_macro_not_used, -1,
  /* DiagArray1051 */ diag::warn_unused_member_function, -1,
  /* DiagArray1052 */ diag::warn_unused_parameter, -1,
  /* DiagArray1053 */ diag::warn_unused_private_field, -1,
  /* DiagArray1054 */ diag::warn_unused_property_backing_ivar, -1,
  /* DiagArray1055 */ diag::warn_unused_result, -1,
  /* DiagArray1056 */ diag::warn_unused_template, -1,
  /* DiagArray1057 */ diag::warn_unused_call, diag::warn_unused_comma_left_operand, diag::warn_unused_constructor, diag::warn_unused_container_subscript_expr, diag::warn_unused_expr, diag::warn_unused_return_type, diag::warn_unused_voidptr, -1,
  /* DiagArray1058 */ diag::warn_unused_variable, -1,
  /* DiagArray1059 */ diag::warn_unused_volatile, -1,
  /* DiagArray1060 */ diag::warn_used_but_marked_unused, -1,
  /* DiagArray1061 */ diag::warn_user_literal_reserved, -1,
  /* DiagArray1062 */ diag::warn_diagnose_if_succeeded, -1,
  /* DiagArray1063 */ diag::warn_second_arg_of_va_start_not_last_non_variadic_param, diag::warn_second_parameter_to_va_arg_array, diag::warn_second_parameter_to_va_arg_never_compatible, diag::warn_va_start_type_is_undefined, -1,
  /* DiagArray1064 */ diag::ext_c_missing_varargs_arg, diag::ext_cxx_missing_varargs_arg, diag::warn_c17_compat_missing_varargs_arg, -1,
  /* DiagArray1065 */ diag::ext_named_variadic_macro, diag::ext_pp_bad_vaopt_use, diag::ext_variadic_macro, -1,
  /* DiagArray1066 */ diag::warn_typecheck_vector_element_sizes_not_equal, -1,
  /* DiagArray1067 */ diag::warn_incompatible_vectors, -1,
  /* DiagArray1069 */ diag::warn_empty_parens_are_function_decl, diag::warn_parens_disambiguated_as_function_declaration, diag::warn_parens_disambiguated_as_variable_declaration, -1,
  /* DiagArray1070 */ diag::warn_decl_in_param_list, diag::warn_redefinition_in_param_list, -1,
  /* DiagArray1071 */ diag::warn_vla_used, -1,
  /* DiagArray1072 */ diag::ext_vla_cxx, diag::ext_vla_cxx_in_gnu_mode, -1,
  /* DiagArray1073 */ diag::ext_vla, -1,
  /* DiagArray1074 */ diag::ext_vla_cxx_in_gnu_mode_static_assert, diag::ext_vla_cxx_static_assert, -1,
  /* DiagArray1075 */ diag::warn_void_pointer_to_enum_cast, -1,
  /* DiagArray1076 */ diag::warn_void_pointer_to_int_cast, -1,
  /* DiagArray1077 */ diag::ext_typecheck_indirection_through_void_pointer, -1,
  /* DiagArray1079 */ diag::warn_wasm_dynamic_exception_spec_ignored, -1,
  /* DiagArray1080 */ diag::warn_weak_template_vtable, -1,
  /* DiagArray1081 */ diag::warn_weak_vtable, -1,
  /* DiagArray1082 */ diag::ext_deprecated_string_literal_conversion, -1,
  /* DiagArray1084 */ diag::warn_xor_used_as_pow, diag::warn_xor_used_as_pow_base, diag::warn_xor_used_as_pow_base_extra, -1,
  /* DiagArray1085 */ diag::warn_zero_as_null_pointer_constant, -1,
  /* DiagArray1086 */ diag::ext_typecheck_zero_array_size, -1,
};

static const int16_t DiagSubGroups[] = {
  /* Empty */ -1,
  /* DiagSubGroup0 */ 329, -1,
  /* DiagSubGroup4 */ 19, 329, -1,
  /* DiagSubGroup12 */ 751, 915, 944, -1,
  /* DiagSubGroup19 */ 613, 736, 928, 929, 578, 733, 1072, 748, -1,
  /* DiagSubGroup30 */ 236, -1,
  /* DiagSubGroup33 */ 40, 39, 36, -1,
  /* DiagSubGroup38 */ 35, -1,
  /* DiagSubGroup50 */ 51, -1,
  /* DiagSubGroup60 */ 436, 209, -1,
  /* DiagSubGroup65 */ 1001, 425, -1,
  /* DiagSubGroup76 */ 112, 146, -1,
  /* DiagSubGroup77 */ 149, -1,
  /* DiagSubGroup79 */ 877, -1,
  /* DiagSubGroup86 */ 751, 980, -1,
  /* DiagSubGroup87 */ 86, -1,
  /* DiagSubGroup88 */ 83, -1,
  /* DiagSubGroup98 */ 101, -1,
  /* DiagSubGroup99 */ 105, -1,
  /* DiagSubGroup100 */ 109, -1,
  /* DiagSubGroup101 */ 109, 104, 102, 767, 769, 771, 773, 775, -1,
  /* DiagSubGroup103 */ 101, 768, 770, 772, 774, 776, -1,
  /* DiagSubGroup105 */ 106, 107, 108, -1,
  /* DiagSubGroup109 */ 110, -1,
  /* DiagSubGroup113 */ 769, 771, 773, 775, -1,
  /* DiagSubGroup114 */ 113, 770, 772, 774, 776, -1,
  /* DiagSubGroup115 */ 112, 111, -1,
  /* DiagSubGroup117 */ 266, 254, 118, 771, 773, 775, -1,
  /* DiagSubGroup119 */ 117, 772, 774, 776, -1,
  /* DiagSubGroup120 */ 116, -1,
  /* DiagSubGroup121 */ 115, -1,
  /* DiagSubGroup122 */ 117, -1,
  /* DiagSubGroup123 */ 118, -1,
  /* DiagSubGroup124 */ 120, -1,
  /* DiagSubGroup126 */ 773, 775, -1,
  /* DiagSubGroup127 */ 126, 774, 776, -1,
  /* DiagSubGroup129 */ 128, 125, 1064, -1,
  /* DiagSubGroup131 */ 775, -1,
  /* DiagSubGroup132 */ 133, -1,
  /* DiagSubGroup135 */ 126, -1,
  /* DiagSubGroup136 */ 127, -1,
  /* DiagSubGroup137 */ 129, -1,
  /* DiagSubGroup138 */ 132, -1,
  /* DiagSubGroup139 */ 229, -1,
  /* DiagSubGroup140 */ 134, -1,
  /* DiagSubGroup141 */ 771, -1,
  /* DiagSubGroup142 */ 772, -1,
  /* DiagSubGroup143 */ 769, -1,
  /* DiagSubGroup144 */ 770, -1,
  /* DiagSubGroup145 */ 767, -1,
  /* DiagSubGroup147 */ 768, -1,
  /* DiagSubGroup148 */ 151, 153, 767, 769, 771, 773, 775, -1,
  /* DiagSubGroup152 */ 148, 149, 150, 768, 770, 772, 774, 776, -1,
  /* DiagSubGroup156 */ 1064, -1,
  /* DiagSubGroup157 */ 155, -1,
  /* DiagSubGroup158 */ 156, -1,
  /* DiagSubGroup161 */ 128, 479, 824, -1,
  /* DiagSubGroup162 */ 161, -1,
  /* DiagSubGroup164 */ 183, -1,
  /* DiagSubGroup167 */ 169, 168, -1,
  /* DiagSubGroup177 */ 633, -1,
  /* DiagSubGroup181 */ 180, -1,
  /* DiagSubGroup185 */ 186, 187, -1,
  /* DiagSubGroup191 */ 79, 663, -1,
  /* DiagSubGroup196 */ 86, 191, 312, 80, 341, 487, 446, 443, 522, 630, 650, 674, 873, 917, -1,
  /* DiagSubGroup197 */ 650, -1,
  /* DiagSubGroup200 */ 201, 246, 22, 198, 199, -1,
  /* DiagSubGroup203 */ 2, -1,
  /* DiagSubGroup211 */ 212, 213, 214, 216, 218, 217, 836, -1,
  /* DiagSubGroup231 */ 230, 228, -1,
  /* DiagSubGroup234 */ 236, 237, 238, 240, 241, 243, 247, 248, 249, 250, 251, 252, 239, 254, 255, 264, 266, 263, 267, 268, 269, 270, 265, 256, 262, -1,
  /* DiagSubGroup241 */ 244, -1,
  /* DiagSubGroup242 */ 243, -1,
  /* DiagSubGroup243 */ 245, -1,
  /* DiagSubGroup260 */ 261, -1,
  /* DiagSubGroup270 */ 102, -1,
  /* DiagSubGroup276 */ 277, -1,
  /* DiagSubGroup281 */ 283, 282, -1,
  /* DiagSubGroup284 */ 285, -1,
  /* DiagSubGroup297 */ 248, -1,
  /* DiagSubGroup299 */ 635, -1,
  /* DiagSubGroup308 */ 333, -1,
  /* DiagSubGroup309 */ 311, 249, -1,
  /* DiagSubGroup310 */ 250, -1,
  /* DiagSubGroup312 */ 313, 314, 310, -1,
  /* DiagSubGroup313 */ 251, -1,
  /* DiagSubGroup314 */ 252, -1,
  /* DiagSubGroup329 */ 241, 588, 433, 479, 853, 591, 872, 1052, 1038, 652, 653, 305, 916, 374, 168, -1,
  /* DiagSubGroup331 */ 150, 106, -1,
  /* DiagSubGroup332 */ 305, -1,
  /* DiagSubGroup341 */ 343, 344, -1,
  /* DiagSubGroup346 */ 347, 361, 636, 355, 360, 349, 348, 352, 357, -1,
  /* DiagSubGroup352 */ 353, -1,
  /* DiagSubGroup357 */ 358, -1,
  /* DiagSubGroup362 */ 351, 355, 360, -1,
  /* DiagSubGroup363 */ 352, 357, -1,
  /* DiagSubGroup367 */ 366, -1,
  /* DiagSubGroup373 */ 935, -1,
  /* DiagSubGroup375 */ 111, 116, 125, 130, -1,
  /* DiagSubGroup380 */ 381, 382, 384, 385, 386, 387, 388, 389, 390, 392, 1073, 393, 394, 395, 396, 397, 399, 400, 401, 402, 403, 812, 404, 405, 407, 408, 409, 410, 1086, 411, 412, -1,
  /* DiagSubGroup405 */ 406, -1,
  /* DiagSubGroup412 */ 1064, -1,
  /* DiagSubGroup422 */ 419, -1,
  /* DiagSubGroup427 */ 433, -1,
  /* DiagSubGroup432 */ 430, 431, -1,
  /* DiagSubGroup433 */ 434, -1,
  /* DiagSubGroup435 */ 444, 445, -1,
  /* DiagSubGroup440 */ 441, -1,
  /* DiagSubGroup443 */ 447, 697, -1,
  /* DiagSubGroup446 */ 871, 698, -1,
  /* DiagSubGroup447 */ 437, -1,
  /* DiagSubGroup463 */ 464, 458, -1,
  /* DiagSubGroup469 */ 471, 632, -1,
  /* DiagSubGroup475 */ 254, -1,
  /* DiagSubGroup488 */ 487, -1,
  /* DiagSubGroup490 */ 491, -1,
  /* DiagSubGroup493 */ 429, -1,
  /* DiagSubGroup524 */ 151, -1,
  /* DiagSubGroup527 */ 108, -1,
  /* DiagSubGroup528 */ 345, 805, -1,
  /* DiagSubGroup541 */ 545, 550, 562, 548, 558, 568, 542, 574, 554, 575, 565, 566, 573, 556, 553, 549, 571, 567, 552, 560, 559, 557, 544, 547, 576, 543, 546, 551, 569, 563, 570, 472, 564, -1,
  /* DiagSubGroup571 */ 572, -1,
  /* DiagSubGroup588 */ 586, -1,
  /* DiagSubGroup613 */ 49, 88, 174, 180, 231, 346, 345, 365, 435, 476, 489, 582, 583, 614, 618, 807, 822, 838, 849, 852, 878, 879, 919, 937, 965, 995, 1005, 1036, 1078, 679, 667, 670, 727, 791, 170, 327, 1062, -1,
  /* DiagSubGroup614 */ 749, 817, 837, 852, -1,
  /* DiagSubGroup615 */ 562, -1,
  /* DiagSubGroup621 */ 109, -1,
  /* DiagSubGroup627 */ 118, -1,
  /* DiagSubGroup629 */ 872, 196, 523, -1,
  /* DiagSubGroup632 */ 631, -1,
  /* DiagSubGroup644 */ 645, -1,
  /* DiagSubGroup652 */ 401, -1,
  /* DiagSubGroup655 */ 656, -1,
  /* DiagSubGroup666 */ 693, -1,
  /* DiagSubGroup673 */ 699, -1,
  /* DiagSubGroup693 */ 694, -1,
  /* DiagSubGroup696 */ 698, 697, 663, 942, -1,
  /* DiagSubGroup706 */ 888, -1,
  /* DiagSubGroup711 */ 889, 713, 715, 717, 716, 712, 714, 718, -1,
  /* DiagSubGroup717 */ 716, -1,
  /* DiagSubGroup728 */ 479, -1,
  /* DiagSubGroup732 */ 733, -1,
  /* DiagSubGroup734 */ 735, -1,
  /* DiagSubGroup736 */ 526, 525, 82, 84, 868, 726, 737, 215, -1,
  /* DiagSubGroup738 */ 989, -1,
  /* DiagSubGroup745 */ 78, 159, 154, 106, 108, 111, 112, 116, 125, 128, 340, 375, 382, 384, 386, 387, 388, 389, 392, 393, 394, 396, 397, 399, 400, 401, 403, 404, 405, 409, 411, 515, 527, 545, 546, 548, 551, 553, 558, 559, 567, 725, 1086, 286, 517, 512, 232, 702, 452, 302, 306, 658, 66, 984, 833, 184, 954, 622, -1,
  /* DiagSubGroup747 */ 264, 529, 94, 832, 337, -1,
  /* DiagSubGroup750 */ 403, -1,
  /* DiagSubGroup755 */ 1075, -1,
  /* DiagSubGroup756 */ 755, 1076, -1,
  /* DiagSubGroup763 */ 764, -1,
  /* DiagSubGroup766 */ 1005, 432, 761, 763, -1,
  /* DiagSubGroup768 */ 767, 146, -1,
  /* DiagSubGroup770 */ 769, -1,
  /* DiagSubGroup772 */ 771, -1,
  /* DiagSubGroup774 */ 773, -1,
  /* DiagSubGroup776 */ 775, -1,
  /* DiagSubGroup777 */ 775, -1,
  /* DiagSubGroup778 */ 776, -1,
  /* DiagSubGroup780 */ 779, -1,
  /* DiagSubGroup781 */ 1064, -1,
  /* DiagSubGroup782 */ 781, -1,
  /* DiagSubGroup783 */ 781, -1,
  /* DiagSubGroup784 */ 782, -1,
  /* DiagSubGroup786 */ 785, -1,
  /* DiagSubGroup805 */ 807, 806, -1,
  /* DiagSubGroup819 */ 266, -1,
  /* DiagSubGroup822 */ 823, 824, -1,
  /* DiagSubGroup827 */ 829, -1,
  /* DiagSubGroup828 */ 829, 830, 1061, 826, -1,
  /* DiagSubGroup831 */ 104, -1,
  /* DiagSubGroup834 */ 836, -1,
  /* DiagSubGroup838 */ 839, 835, -1,
  /* DiagSubGroup847 */ 848, -1,
  /* DiagSubGroup849 */ 851, 850, -1,
  /* DiagSubGroup855 */ 1023, -1,
  /* DiagSubGroup857 */ 861, 862, -1,
  /* DiagSubGroup858 */ 857, 860, 863, 859, -1,
  /* DiagSubGroup860 */ 861, -1,
  /* DiagSubGroup891 */ 890, -1,
  /* DiagSubGroup894 */ 407, -1,
  /* DiagSubGroup911 */ 759, -1,
  /* DiagSubGroup913 */ 258, -1,
  /* DiagSubGroup926 */ 882, 296, 644, 538, 925, -1,
  /* DiagSubGroup937 */ 938, 944, 943, 936, 946, 942, 941, -1,
  /* DiagSubGroup938 */ 940, -1,
  /* DiagSubGroup939 */ 966, 950, -1,
  /* DiagSubGroup955 */ 957, 956, 961, 962, -1,
  /* DiagSubGroup962 */ 963, -1,
  /* DiagSubGroup966 */ 945, 949, 947, 948, -1,
  /* DiagSubGroup975 */ 977, -1,
  /* DiagSubGroup988 */ 760, -1,
  /* DiagSubGroup989 */ 990, -1,
  /* DiagSubGroup995 */ 886, 898, 996, -1,
  /* DiagSubGroup1008 */ 153, -1,
  /* DiagSubGroup1013 */ 1018, 1016, 1017, -1,
  /* DiagSubGroup1014 */ 1013, 1015, 1019, -1,
  /* DiagSubGroup1020 */ 1021, 1022, -1,
  /* DiagSubGroup1036 */ 1037, 1044, 1046, 1053, 1047, 1048, 1057, 1058, 1039, 1054, -1,
  /* DiagSubGroup1044 */ 1010, -1,
  /* DiagSubGroup1049 */ 1048, -1,
  /* DiagSubGroup1051 */ 1011, -1,
  /* DiagSubGroup1056 */ 1010, -1,
  /* DiagSubGroup1057 */ 1041, 1055, 988, -1,
  /* DiagSubGroup1058 */ 1042, -1,
  /* DiagSubGroup1068 */ 1067, -1,
  /* DiagSubGroup1071 */ 1073, -1,
  /* DiagSubGroup1072 */ 1074, -1,
  /* DiagSubGroup1073 */ 1072, -1,
  /* DiagSubGroup1076 */ 1075, -1,
  /* DiagSubGroup1082 */ 270, -1,
  /* DiagSubGroup1083 */ 1082, -1,
};


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
static constexpr char DiagGroupNamesStorage[] = 
  "\0"
  "#pragma-messages\0"
  "#warnings\0"
  "CFString-literal\0"
  "CL4\0"
  "IndependentClass-attribute\0"
  "NSObject-attribute\0"
  "aarch64-sme-attributes\0"
  "abi\0"
  "absolute-value\0"
  "abstract-final-class\0"
  "abstract-vbase-init\0"
  "address\0"
  "address-of-packed-member\0"
  "address-of-temporary\0"
  "aggregate-return\0"
  "aix-compat\0"
  "alias-template-in-declaration-name\0"
  "align-mismatch\0"
  "all\0"
  "alloca\0"
  "alloca-with-align-alignof\0"
  "always-inline-coroutine\0"
  "ambiguous-delete\0"
  "ambiguous-ellipsis\0"
  "ambiguous-macro\0"
  "ambiguous-member-template\0"
  "ambiguous-reversed-operator\0"
  "analyzer-incompatible-plugin\0"
  "android-unversioned-fallback\0"
  "anon-enum-enum-conversion\0"
  "anonymous-pack-parens\0"
  "apinotes\0"
  "arc\0"
  "arc-bridge-casts-disallowed-in-nonarc\0"
  "arc-maybe-repeated-use-of-weak\0"
  "arc-non-pod-memaccess\0"
  "arc-performSelector-leaks\0"
  "arc-repeated-use-of-weak\0"
  "arc-retain-cycles\0"
  "arc-unsafe-retained-assign\0"
  "argument-outside-range\0"
  "argument-undefined-behaviour\0"
  "arm-interrupt-save-fp-no-vfp-unit\0"
  "arm-interrupt-vfp-clobber\0"
  "array-bounds\0"
  "array-bounds-pointer-arithmetic\0"
  "array-compare\0"
  "array-compare-cxx26\0"
  "array-parameter\0"
  "asm\0"
  "asm-operand-widths\0"
  "assign-enum\0"
  "assume\0"
  "at-protocol\0"
  "atimport-in-framework-header\0"
  "atomic-access\0"
  "atomic-alignment\0"
  "atomic-implicit-seq-cst\0"
  "atomic-memory-ordering\0"
  "atomic-properties\0"
  "atomic-property-with-user-defined-accessor\0"
  "attribute-packed-for-bitfield\0"
  "attribute-statement\0"
  "attribute-warning\0"
  "attributes\0"
  "auto-decl-extensions\0"
  "auto-disable-vptr-sanitizer\0"
  "auto-import\0"
  "auto-storage-class\0"
  "auto-var-id\0"
  "availability\0"
  "avr-rtlib-linking-quirks\0"
  "backend-plugin\0"
  "backslash-newline-escape\0"
  "bad-function-cast\0"
  "binary-literal\0"
  "bind-to-temporary-copy\0"
  "bit-int-extension\0"
  "bitfield-constant-conversion\0"
  "bitfield-enum-conversion\0"
  "bitfield-width\0"
  "bitwise-conditional-parentheses\0"
  "bitwise-instead-of-logical\0"
  "bitwise-op-parentheses\0"
  "block-capture-autoreleasing\0"
  "bool-conversion\0"
  "bool-conversions\0"
  "bool-operation\0"
  "bounds-safety-counted-by-elt-type-unknown-size\0"
  "braced-scalar-init\0"
  "branch-protection\0"
  "bridge-cast\0"
  "builtin-assume-aligned-alignment\0"
  "builtin-macro-redefined\0"
  "builtin-memcpy-chk-size\0"
  "builtin-requires-header\0"
  "c++-compat\0"
  "c++0x-compat\0"
  "c++0x-extensions\0"
  "c++0x-narrowing\0"
  "c++11-compat\0"
  "c++11-compat-deprecated-writable-strings\0"
  "c++11-compat-pedantic\0"
  "c++11-compat-reserved-user-defined-literal\0"
  "c++11-extensions\0"
  "c++11-extra-semi\0"
  "c++11-inline-namespace\0"
  "c++11-long-long\0"
  "c++11-narrowing\0"
  "c++11-narrowing-const-reference\0"
  "c++14-attribute-extensions\0"
  "c++14-binary-literal\0"
  "c++14-compat\0"
  "c++14-compat-pedantic\0"
  "c++14-extensions\0"
  "c++17-attribute-extensions\0"
  "c++17-compat\0"
  "c++17-compat-mangling\0"
  "c++17-compat-pedantic\0"
  "c++17-extensions\0"
  "c++1y-extensions\0"
  "c++1z-compat\0"
  "c++1z-compat-mangling\0"
  "c++1z-extensions\0"
  "c++20-attribute-extensions\0"
  "c++20-compat\0"
  "c++20-compat-pedantic\0"
  "c++20-designator\0"
  "c++20-extensions\0"
  "c++23-attribute-extensions\0"
  "c++23-compat\0"
  "c++23-extensions\0"
  "c++23-lambda-attributes\0"
  "c++26-extensions\0"
  "c++2a-compat\0"
  "c++2a-compat-pedantic\0"
  "c++2a-extensions\0"
  "c++2b-extensions\0"
  "c++2c-compat\0"
  "c++2c-extensions\0"
  "c++98-c++11-c++14-c++17-compat\0"
  "c++98-c++11-c++14-c++17-compat-pedantic\0"
  "c++98-c++11-c++14-compat\0"
  "c++98-c++11-c++14-compat-pedantic\0"
  "c++98-c++11-compat\0"
  "c++98-c++11-compat-binary-literal\0"
  "c++98-c++11-compat-pedantic\0"
  "c++98-compat\0"
  "c++98-compat-bind-to-temporary-copy\0"
  "c++98-compat-extra-semi\0"
  "c++98-compat-local-type-template-args\0"
  "c++98-compat-pedantic\0"
  "c++98-compat-unnamed-type-template-args\0"
  "c11-extensions\0"
  "c23-compat\0"
  "c23-extensions\0"
  "c2x-compat\0"
  "c2x-extensions\0"
  "c2y-extensions\0"
  "c99-compat\0"
  "c99-designator\0"
  "c99-extensions\0"
  "call-to-pure-virtual-from-ctor-dtor\0"
  "called-once-parameter\0"
  "cast-align\0"
  "cast-calling-convention\0"
  "cast-function-type\0"
  "cast-function-type-mismatch\0"
  "cast-function-type-strict\0"
  "cast-of-sel-type\0"
  "cast-qual\0"
  "cast-qual-unrelated\0"
  "char-align\0"
  "char-subscripts\0"
  "clang-cl-pch\0"
  "class-conversion\0"
  "class-varargs\0"
  "cmse-union-leak\0"
  "comma\0"
  "comment\0"
  "comments\0"
  "compare-distinct-pointer-types\0"
  "completion-handler\0"
  "complex-component-init\0"
  "compound-token-split\0"
  "compound-token-split-by-macro\0"
  "compound-token-split-by-space\0"
  "conditional-type-mismatch\0"
  "conditional-uninitialized\0"
  "config-macros\0"
  "constant-conversion\0"
  "constant-evaluated\0"
  "constant-logical-operand\0"
  "constexpr-not-const\0"
  "consumed\0"
  "conversion\0"
  "conversion-null\0"
  "coro-non-aligned-allocation-function\0"
  "coro-type-aware-allocation-function\0"
  "coroutine\0"
  "coroutine-missing-unhandled-exception\0"
  "covered-switch-default\0"
  "cpp\0"
  "cstring-format-directive\0"
  "ctad-maybe-unsupported\0"
  "ctor-dtor-privacy\0"
  "ctu\0"
  "cuda-compat\0"
  "custom-atomic-properties\0"
  "cxx-attribute-extension\0"
  "dangling\0"
  "dangling-assignment\0"
  "dangling-assignment-gsl\0"
  "dangling-capture\0"
  "dangling-else\0"
  "dangling-field\0"
  "dangling-gsl\0"
  "dangling-initializer-list\0"
  "darwin-sdk-settings\0"
  "date-time\0"
  "dealloc-in-category\0"
  "debug-compression-unavailable\0"
  "declaration-after-statement\0"
  "decls-in-multiple-modules\0"
  "defaulted-function-deleted\0"
  "delayed-template-parsing-in-cxx20\0"
  "delegating-ctor-cycles\0"
  "delete-abstract-non-virtual-dtor\0"
  "delete-incomplete\0"
  "delete-non-abstract-non-virtual-dtor\0"
  "delete-non-virtual-dtor\0"
  "delimited-escape-sequence-extension\0"
  "deprecate-lax-vec-conv-all\0"
  "deprecated\0"
  "deprecated-altivec-src-compat\0"
  "deprecated-anon-enum-enum-conversion\0"
  "deprecated-array-compare\0"
  "deprecated-attributes\0"
  "deprecated-builtins\0"
  "deprecated-comma-subscript\0"
  "deprecated-copy\0"
  "deprecated-copy-dtor\0"
  "deprecated-copy-with-dtor\0"
  "deprecated-copy-with-user-provided-copy\0"
  "deprecated-copy-with-user-provided-dtor\0"
  "deprecated-coroutine\0"
  "deprecated-declarations\0"
  "deprecated-dynamic-exception-spec\0"
  "deprecated-enum-compare\0"
  "deprecated-enum-compare-conditional\0"
  "deprecated-enum-enum-conversion\0"
  "deprecated-enum-float-conversion\0"
  "deprecated-implementations\0"
  "deprecated-increment-bool\0"
  "deprecated-literal-operator\0"
  "deprecated-missing-comma-variadic-parameter\0"
  "deprecated-module-dot-map\0"
  "deprecated-non-prototype\0"
  "deprecated-objc-isa-usage\0"
  "deprecated-objc-pointer-introspection\0"
  "deprecated-objc-pointer-introspection-performSelector\0"
  "deprecated-octal-literals\0"
  "deprecated-ofast\0"
  "deprecated-pragma\0"
  "deprecated-redundant-constexpr-static-def\0"
  "deprecated-register\0"
  "deprecated-this-capture\0"
  "deprecated-type\0"
  "deprecated-volatile\0"
  "deprecated-writable-strings\0"
  "direct-ivar-access\0"
  "disabled-macro-expansion\0"
  "disabled-optimization\0"
  "discard-qual\0"
  "distributed-object-modifiers\0"
  "div-by-zero\0"
  "division-by-zero\0"
  "dll-attribute-on-redeclaration\0"
  "dllexport-explicit-instantiation-decl\0"
  "dllimport-static-field-def\0"
  "documentation\0"
  "documentation-deprecated-sync\0"
  "documentation-html\0"
  "documentation-pedantic\0"
  "documentation-unknown-command\0"
  "dollar-in-identifier-extension\0"
  "double-promotion\0"
  "dtor-name\0"
  "dtor-typedef\0"
  "duplicate-decl-specifier\0"
  "duplicate-enum\0"
  "duplicate-method-arg\0"
  "duplicate-method-match\0"
  "duplicate-protocol\0"
  "dxil-validation\0"
  "dynamic-class-memaccess\0"
  "dynamic-exception-spec\0"
  "eager-load-cxx-named-modules\0"
  "effc++\0"
  "elaborated-enum-base\0"
  "elaborated-enum-class\0"
  "embedded-directive\0"
  "empty-body\0"
  "empty-decomposition\0"
  "empty-init-stmt\0"
  "empty-translation-unit\0"
  "encode-type\0"
  "endif-labels\0"
  "enum-compare\0"
  "enum-compare-conditional\0"
  "enum-compare-switch\0"
  "enum-conversion\0"
  "enum-enum-conversion\0"
  "enum-float-conversion\0"
  "enum-too-large\0"
  "exceptions\0"
  "excess-initializers\0"
  "excessive-regsave\0"
  "exit-time-destructors\0"
  "expansion-to-defined\0"
  "experimental-header-units\0"
  "experimental-option\0"
  "explicit-initialize-call\0"
  "explicit-ownership-type\0"
  "explicit-specialization-storage-class\0"
  "export-unnamed\0"
  "extern-c-compat\0"
  "extern-initializer\0"
  "extra\0"
  "extra-qualification\0"
  "extra-semi\0"
  "extra-semi-stmt\0"
  "extra-tokens\0"
  "extractapi-misuse\0"
  "extraneous-template-head\0"
  "final-dtor-non-final-class\0"
  "final-macro\0"
  "fixed-point-overflow\0"
  "flag-enum\0"
  "flexible-array-extensions\0"
  "float-conversion\0"
  "float-equal\0"
  "float-overflow-conversion\0"
  "float-zero-conversion\0"
  "for-loop-analysis\0"
  "format\0"
  "format-extra-args\0"
  "format-insufficient-args\0"
  "format-invalid-specifier\0"
  "format-non-iso\0"
  "format-nonliteral\0"
  "format-overflow\0"
  "format-overflow-non-kprintf\0"
  "format-pedantic\0"
  "format-security\0"
  "format-signedness\0"
  "format-truncation\0"
  "format-truncation-non-kprintf\0"
  "format-type-confusion\0"
  "format-y2k\0"
  "format-zero-length\0"
  "format=2\0"
  "fortify-source\0"
  "four-char-constants\0"
  "frame-address\0"
  "frame-larger-than\0"
  "frame-larger-than=\0"
  "framework-include-private-from-public\0"
  "free-nonheap-object\0"
  "friend-enum\0"
  "function-def-in-objc-container\0"
  "function-effects\0"
  "function-multiversion\0"
  "fuse-ld-path\0"
  "future-attribute-extensions\0"
  "future-compat\0"
  "gcc-compat\0"
  "global-constructors\0"
  "global-isel\0"
  "gnu\0"
  "gnu-alignof-expression\0"
  "gnu-anonymous-struct\0"
  "gnu-array-member-paren-init\0"
  "gnu-auto-type\0"
  "gnu-binary-literal\0"
  "gnu-case-range\0"
  "gnu-complex-integer\0"
  "gnu-compound-literal-initializer\0"
  "gnu-conditional-omitted-operand\0"
  "gnu-designator\0"
  "gnu-empty-initializer\0"
  "gnu-empty-struct\0"
  "gnu-flexible-array-initializer\0"
  "gnu-flexible-array-union-member\0"
  "gnu-folding-constant\0"
  "gnu-imaginary-constant\0"
  "gnu-include-next\0"
  "gnu-inline-cpp-without-extern\0"
  "gnu-label-as-value\0"
  "gnu-line-marker\0"
  "gnu-null-pointer-arithmetic\0"
  "gnu-offsetof-extensions\0"
  "gnu-pointer-arith\0"
  "gnu-redeclared-enum\0"
  "gnu-statement-expression\0"
  "gnu-statement-expression-from-macro-expansion\0"
  "gnu-static-float-init\0"
  "gnu-string-literal-operator-template\0"
  "gnu-union-cast\0"
  "gnu-variable-sized-type-not-at-end\0"
  "gnu-zero-line-directive\0"
  "gnu-zero-variadic-macro-arguments\0"
  "gpu-maybe-wrong-side\0"
  "header-guard\0"
  "header-hygiene\0"
  "higher-precision-for-complex-division\0"
  "hip-omp-target-directives\0"
  "hip-only\0"
  "hlsl-202y-extensions\0"
  "hlsl-availability\0"
  "hlsl-dxc-compatability\0"
  "hlsl-extensions\0"
  "hlsl-implicit-binding\0"
  "idiomatic-parentheses\0"
  "ignored-attributes\0"
  "ignored-availability-without-sdk-settings\0"
  "ignored-base-class-qualifiers\0"
  "ignored-gch\0"
  "ignored-optimization-argument\0"
  "ignored-pragma-intrinsic\0"
  "ignored-pragma-optimize\0"
  "ignored-pragmas\0"
  "ignored-qualifiers\0"
  "ignored-reference-qualifiers\0"
  "implicit\0"
  "implicit-atomic-properties\0"
  "implicit-const-int-float-conversion\0"
  "implicit-conversion-floating-point-to-bool\0"
  "implicit-exception-spec-mismatch\0"
  "implicit-fallthrough\0"
  "implicit-fallthrough-per-function\0"
  "implicit-fixed-point-conversion\0"
  "implicit-float-conversion\0"
  "implicit-function-declaration\0"
  "implicit-int\0"
  "implicit-int-conversion\0"
  "implicit-int-float-conversion\0"
  "implicit-retain-self\0"
  "implicitly-unsigned-literal\0"
  "import\0"
  "import-implementation-partition-unit-in-interface-unit\0"
  "import-preprocessor-directive-pedantic\0"
  "inaccessible-base\0"
  "include-angled-in-module-purview\0"
  "include-next-absolute-path\0"
  "include-next-outside-header\0"
  "incompatible-exception-spec\0"
  "incompatible-function-pointer-types\0"
  "incompatible-function-pointer-types-strict\0"
  "incompatible-library-redeclaration\0"
  "incompatible-ms-pragma-section\0"
  "incompatible-ms-struct\0"
  "incompatible-pointer-types\0"
  "incompatible-pointer-types-discards-qualifiers\0"
  "incompatible-property-type\0"
  "incompatible-sysroot\0"
  "incomplete-framework-module-declaration\0"
  "incomplete-implementation\0"
  "incomplete-module\0"
  "incomplete-setjmp-declaration\0"
  "incomplete-umbrella\0"
  "inconsistent-dllimport\0"
  "inconsistent-missing-destructor-override\0"
  "inconsistent-missing-override\0"
  "increment-bool\0"
  "infinite-recursion\0"
  "init-priority-reserved\0"
  "init-self\0"
  "initializer-overrides\0"
  "injected-class-name\0"
  "inline\0"
  "inline-asm\0"
  "inline-namespace-reopened-noninline\0"
  "inline-new-delete\0"
  "installapi-violation\0"
  "instantiation-after-specialization\0"
  "int-conversion\0"
  "int-conversions\0"
  "int-in-bool-context\0"
  "int-to-pointer-cast\0"
  "int-to-void-pointer-cast\0"
  "integer-overflow\0"
  "invalid-command-line-argument\0"
  "invalid-constexpr\0"
  "invalid-feature-combination\0"
  "invalid-gnu-asm-cast\0"
  "invalid-iboutlet\0"
  "invalid-initializer-from-system-header\0"
  "invalid-ios-deployment-target\0"
  "invalid-no-builtin-names\0"
  "invalid-noreturn\0"
  "invalid-offsetof\0"
  "invalid-or-nonexistent-directory\0"
  "invalid-partial-specialization\0"
  "invalid-pch\0"
  "invalid-pp-token\0"
  "invalid-source-encoding\0"
  "invalid-specialization\0"
  "invalid-static-assert-message\0"
  "invalid-token-paste\0"
  "invalid-unevaluated-string\0"
  "invalid-utf8\0"
  "jump-seh-finally\0"
  "keyword-compat\0"
  "keyword-macro\0"
  "knr-promoted-parameter\0"
  "language-extension-token\0"
  "large-by-value-copy\0"
  "legacy-constant-register-binding\0"
  "liblto\0"
  "linker-warnings\0"
  "literal-conversion\0"
  "literal-range\0"
  "local-type-template-args\0"
  "logical-not-parentheses\0"
  "logical-op-parentheses\0"
  "long-long\0"
  "loop-analysis\0"
  "macro-redefined\0"
  "main\0"
  "main-return-type\0"
  "malformed-warning-check\0"
  "many-braces-around-scalar-init\0"
  "math-errno-enabled-with-veclib\0"
  "mathematical-notation-identifier-extension\0"
  "max-tokens\0"
  "max-unsigned-zero\0"
  "memset-transposed-args\0"
  "memsize-comparison\0"
  "method-signatures\0"
  "microsoft\0"
  "microsoft-abstract\0"
  "microsoft-anon-tag\0"
  "microsoft-cast\0"
  "microsoft-charize\0"
  "microsoft-comment-paste\0"
  "microsoft-const-init\0"
  "microsoft-cpp-macro\0"
  "microsoft-default-arg-redefinition\0"
  "microsoft-drectve-section\0"
  "microsoft-end-of-file\0"
  "microsoft-enum-forward-reference\0"
  "microsoft-enum-value\0"
  "microsoft-exception-spec\0"
  "microsoft-exists\0"
  "microsoft-explicit-constructor-call\0"
  "microsoft-extra-qualification\0"
  "microsoft-fixed-enum\0"
  "microsoft-flexible-array\0"
  "microsoft-goto\0"
  "microsoft-inaccessible-base\0"
  "microsoft-include\0"
  "microsoft-init-from-predefined\0"
  "microsoft-inline-on-non-function\0"
  "microsoft-mutable-reference\0"
  "microsoft-pure-definition\0"
  "microsoft-redeclare-static\0"
  "microsoft-sealed\0"
  "microsoft-static-assert\0"
  "microsoft-string-literal-from-predefined\0"
  "microsoft-template\0"
  "microsoft-template-shadow\0"
  "microsoft-union-member-reference\0"
  "microsoft-unqualified-friend\0"
  "microsoft-using-decl\0"
  "microsoft-void-pseudo-dtor\0"
  "misexpect\0"
  "misleading-indentation\0"
  "mismatched-new-delete\0"
  "mismatched-parameter-types\0"
  "mismatched-return-types\0"
  "mismatched-tags\0"
  "missing-braces\0"
  "missing-constinit\0"
  "missing-declarations\0"
  "missing-designated-field-initializers\0"
  "missing-exception-spec\0"
  "missing-field-initializers\0"
  "missing-format-attribute\0"
  "missing-include-dirs\0"
  "missing-method-return-type\0"
  "missing-multilib\0"
  "missing-noescape\0"
  "missing-noreturn\0"
  "missing-prototype-for-cc\0"
  "missing-prototypes\0"
  "missing-selector-name\0"
  "missing-sysroot\0"
  "missing-template-arg-list-after-template-kw\0"
  "missing-variable-declarations\0"
  "misspelled-assumption\0"
  "mix-packoffset\0"
  "module-build\0"
  "module-conflict\0"
  "module-file-config-mismatch\0"
  "module-file-extension\0"
  "module-import\0"
  "module-import-in-extern-c\0"
  "module-include-translation\0"
  "module-lock\0"
  "modules-ambiguous-internal-linkage\0"
  "modules-import-nested-redundant\0"
  "most\0"
  "move\0"
  "msvc-include\0"
  "msvc-not-found\0"
  "multi-gpu\0"
  "multichar\0"
  "multiple-move-vbase\0"
  "nan-infinity-disabled\0"
  "narrowing\0"
  "nested-anon-types\0"
  "nested-externs\0"
  "new-returns-null\0"
  "newline-eof\0"
  "noderef\0"
  "noexcept-type\0"
  "non-c-typedef-for-linkage\0"
  "non-gcc\0"
  "non-literal-null-conversion\0"
  "non-modular-include-in-framework-module\0"
  "non-modular-include-in-module\0"
  "non-pod-varargs\0"
  "non-power-of-two-alignment\0"
  "non-virtual-dtor\0"
  "nonnull\0"
  "nonportable-cfstrings\0"
  "nonportable-include-path\0"
  "nonportable-private-apinotes-path\0"
  "nonportable-private-system-apinotes-path\0"
  "nonportable-sycl\0"
  "nonportable-system-include-path\0"
  "nonportable-vector-initialization\0"
  "nontrivial-memaccess\0"
  "nontrivial-memcall\0"
  "nsconsumed-mismatch\0"
  "nsreturns-mismatch\0"
  "null-arithmetic\0"
  "null-character\0"
  "null-conversion\0"
  "null-dereference\0"
  "null-pointer-arithmetic\0"
  "null-pointer-subtraction\0"
  "nullability\0"
  "nullability-completeness\0"
  "nullability-completeness-on-arrays\0"
  "nullability-declspec\0"
  "nullability-extension\0"
  "nullability-inferred-on-nested-type\0"
  "nullable-to-nonnull-conversion\0"
  "nvcc-compat\0"
  "objc-autosynthesis-property-ivar-name-match\0"
  "objc-bool-constant-conversion\0"
  "objc-boxing\0"
  "objc-circular-container\0"
  "objc-cocoa-api\0"
  "objc-designated-initializers\0"
  "objc-dictionary-duplicate-keys\0"
  "objc-duplicate-category-definition\0"
  "objc-flexible-array\0"
  "objc-forward-class-redefinition\0"
  "objc-interface-ivars\0"
  "objc-literal-compare\0"
  "objc-literal-conversion\0"
  "objc-macro-redefinition\0"
  "objc-messaging-id\0"
  "objc-method-access\0"
  "objc-missing-property-synthesis\0"
  "objc-missing-super-calls\0"
  "objc-multiple-method-names\0"
  "objc-noncopy-retain-block-property\0"
  "objc-nonunified-exceptions\0"
  "objc-property-assign-on-object-type\0"
  "objc-property-implementation\0"
  "objc-property-implicit-mismatch\0"
  "objc-property-matches-cocoa-ownership-rule\0"
  "objc-property-no-attribute\0"
  "objc-property-synthesis\0"
  "objc-protocol-method-implementation\0"
  "objc-protocol-property-synthesis\0"
  "objc-protocol-qualifiers\0"
  "objc-readonly-with-setter-property\0"
  "objc-redundant-api-use\0"
  "objc-redundant-literal-use\0"
  "objc-root-class\0"
  "objc-signed-char-bool\0"
  "objc-signed-char-bool-implicit-float-conversion\0"
  "objc-signed-char-bool-implicit-int-conversion\0"
  "objc-string-compare\0"
  "objc-string-concatenation\0"
  "objc-unsafe-perform-selector\0"
  "octal-prefix-extension\0"
  "odr\0"
  "old-style-cast\0"
  "old-style-definition\0"
  "openacc\0"
  "openacc-confusing-routine-name\0"
  "openacc-deprecated-clause-alias\0"
  "openacc-self-if-potential-conflict\0"
  "opencl-unsupported-rgba\0"
  "openmp\0"
  "openmp-51-extensions\0"
  "openmp-clauses\0"
  "openmp-extensions\0"
  "openmp-loop-form\0"
  "openmp-mapping\0"
  "openmp-target\0"
  "openmp-target-exception\0"
  "option-ignored\0"
  "ordered-compare-function-pointers\0"
  "out-of-line-declaration\0"
  "out-of-scope-function\0"
  "over-aligned\0"
  "overflow\0"
  "overlength-strings\0"
  "overloaded-shift-op-parentheses\0"
  "overloaded-virtual\0"
  "override-init\0"
  "override-module\0"
  "overriding-method-mismatch\0"
  "overriding-option\0"
  "packed\0"
  "packed-non-pod\0"
  "padded\0"
  "padded-bitfield\0"
  "parentheses\0"
  "parentheses-equality\0"
  "partial-availability\0"
  "pass\0"
  "pass-analysis\0"
  "pass-failed\0"
  "pass-missed\0"
  "pch-date-time\0"
  "pch-vfs-diff\0"
  "pedantic\0"
  "pedantic-core-features\0"
  "pedantic-macros\0"
  "perf-constraint-implies-noexcept\0"
  "pessimizing-move\0"
  "pointer-arith\0"
  "pointer-bool-conversion\0"
  "pointer-compare\0"
  "pointer-integer-compare\0"
  "pointer-sign\0"
  "pointer-to-enum-cast\0"
  "pointer-to-int-cast\0"
  "pointer-type-mismatch\0"
  "poison-system-directories\0"
  "potentially-direct-selector\0"
  "potentially-evaluated-expression\0"
  "pragma-clang-attribute\0"
  "pragma-once-outside-header\0"
  "pragma-pack\0"
  "pragma-pack-suspicious-include\0"
  "pragma-system-header-outside-header\0"
  "pragmas\0"
  "pre-c++14-compat\0"
  "pre-c++14-compat-pedantic\0"
  "pre-c++17-compat\0"
  "pre-c++17-compat-pedantic\0"
  "pre-c++20-compat\0"
  "pre-c++20-compat-pedantic\0"
  "pre-c++23-compat\0"
  "pre-c++23-compat-pedantic\0"
  "pre-c++26-compat\0"
  "pre-c++26-compat-pedantic\0"
  "pre-c++2c-compat\0"
  "pre-c++2c-compat-pedantic\0"
  "pre-c11-compat\0"
  "pre-c11-compat-pedantic\0"
  "pre-c23-compat\0"
  "pre-c23-compat-pedantic\0"
  "pre-c2x-compat\0"
  "pre-c2x-compat-pedantic\0"
  "pre-c2y-compat\0"
  "pre-c2y-compat-pedantic\0"
  "pre-openmp-51-compat\0"
  "predefined-identifier-outside-function\0"
  "preferred-type-bitfield-enum-conversion\0"
  "preferred-type-bitfield-width\0"
  "private-extern\0"
  "private-header\0"
  "private-module\0"
  "profile-instr-missing\0"
  "profile-instr-out-of-date\0"
  "profile-instr-unprofiled\0"
  "property-access-dot-syntax\0"
  "property-attribute-mismatch\0"
  "protocol\0"
  "protocol-property-synthesis-ambiguity\0"
  "psabi\0"
  "ptrauth-null-pointers\0"
  "qualified-void-return-type\0"
  "quoted-include-in-framework-header\0"
  "range-loop-analysis\0"
  "range-loop-bind-reference\0"
  "range-loop-construct\0"
  "read-only-types\0"
  "readonly-iboutlet-property\0"
  "receiver-expr\0"
  "receiver-forward-class\0"
  "redeclared-class-member\0"
  "reduced-bmi-output-overrided\0"
  "redundant-attribute\0"
  "redundant-consteval-if\0"
  "redundant-decls\0"
  "redundant-move\0"
  "redundant-parens\0"
  "register\0"
  "reinterpret-base-class\0"
  "remark-backend-plugin\0"
  "reorder\0"
  "reorder-ctor\0"
  "reorder-init-list\0"
  "requires-super-attribute\0"
  "reserved-attribute-identifier\0"
  "reserved-id-macro\0"
  "reserved-identifier\0"
  "reserved-macro-identifier\0"
  "reserved-module-identifier\0"
  "reserved-user-defined-literal\0"
  "restrict-expansion\0"
  "retained-language-linkage\0"
  "return-local-addr\0"
  "return-mismatch\0"
  "return-stack-address\0"
  "return-std-move\0"
  "return-type\0"
  "return-type-c-linkage\0"
  "rewrite-not-bool\0"
  "round-trip-cc1-args\0"
  "rtti\0"
  "sanitize-address\0"
  "sarif-format-unstable\0"
  "search-path-usage\0"
  "section\0"
  "selector\0"
  "selector-type-mismatch\0"
  "self-assign\0"
  "self-assign-field\0"
  "self-assign-overloaded\0"
  "self-move\0"
  "semicolon-before-method-body\0"
  "sentinel\0"
  "sequence-point\0"
  "serialized-diagnostics\0"
  "shadow\0"
  "shadow-all\0"
  "shadow-field\0"
  "shadow-field-in-constructor\0"
  "shadow-field-in-constructor-modified\0"
  "shadow-ivar\0"
  "shadow-uncaptured-local\0"
  "shift-bool\0"
  "shift-count-negative\0"
  "shift-count-overflow\0"
  "shift-negative-value\0"
  "shift-op-parentheses\0"
  "shift-overflow\0"
  "shift-sign-overflow\0"
  "shorten-64-to-32\0"
  "sign-compare\0"
  "sign-conversion\0"
  "sign-promo\0"
  "signed-enum-bitfield\0"
  "signed-unsigned-wchar\0"
  "single-bit-bitfield-constant-conversion\0"
  "sizeof-array-argument\0"
  "sizeof-array-decay\0"
  "sizeof-array-div\0"
  "sizeof-pointer-div\0"
  "sizeof-pointer-memaccess\0"
  "slash-u-filename\0"
  "slh-asm-goto\0"
  "sloc-usage\0"
  "sometimes-uninitialized\0"
  "source-mgr\0"
  "source-uses-openacc\0"
  "source-uses-openmp\0"
  "spir-compat\0"
  "spirv-compat\0"
  "stack-exhausted\0"
  "stack-protector\0"
  "static-float-init\0"
  "static-in-inline\0"
  "static-inline-explicit-instantiation\0"
  "static-local-in-inline\0"
  "static-self-init\0"
  "stdlibcxx-not-found\0"
  "strict-aliasing\0"
  "strict-aliasing=0\0"
  "strict-aliasing=1\0"
  "strict-aliasing=2\0"
  "strict-overflow\0"
  "strict-overflow=0\0"
  "strict-overflow=1\0"
  "strict-overflow=2\0"
  "strict-overflow=3\0"
  "strict-overflow=4\0"
  "strict-overflow=5\0"
  "strict-potentially-direct-selector\0"
  "strict-primary-template-shadow\0"
  "strict-prototypes\0"
  "strict-selector-match\0"
  "string-compare\0"
  "string-concatenation\0"
  "string-conversion\0"
  "string-plus-char\0"
  "string-plus-int\0"
  "strlcpy-strlcat-size\0"
  "strncat-size\0"
  "suggest-destructor-override\0"
  "suggest-override\0"
  "super-class-method-mismatch\0"
  "suspicious-bzero\0"
  "suspicious-memaccess\0"
  "swift-name-attribute\0"
  "switch\0"
  "switch-bool\0"
  "switch-default\0"
  "switch-enum\0"
  "sync-alignment\0"
  "sync-fetch-and-nand-semantics-changed\0"
  "synth\0"
  "target-clones-mixed-specifiers\0"
  "tautological-bitwise-compare\0"
  "tautological-compare\0"
  "tautological-constant-compare\0"
  "tautological-constant-in-range-compare\0"
  "tautological-constant-out-of-range-compare\0"
  "tautological-negation-compare\0"
  "tautological-objc-bool-compare\0"
  "tautological-overlap-compare\0"
  "tautological-pointer-compare\0"
  "tautological-type-limit-compare\0"
  "tautological-undefined-compare\0"
  "tautological-unsigned-char-zero-compare\0"
  "tautological-unsigned-enum-zero-compare\0"
  "tautological-unsigned-zero-compare\0"
  "tautological-value-range-compare\0"
  "tcb-enforcement\0"
  "template-in-declaration-name\0"
  "tentative-definition-array\0"
  "tentative-definition-incomplete-type\0"
  "thread-safety\0"
  "thread-safety-analysis\0"
  "thread-safety-attributes\0"
  "thread-safety-beta\0"
  "thread-safety-negative\0"
  "thread-safety-pointer\0"
  "thread-safety-precise\0"
  "thread-safety-reference\0"
  "thread-safety-reference-return\0"
  "thread-safety-verbose\0"
  "trigraphs\0"
  "type-limits\0"
  "type-safety\0"
  "typedef-redefinition\0"
  "typename-missing\0"
  "unable-to-open-stats-file\0"
  "unaligned-access\0"
  "unaligned-qualifier-implicit-cast\0"
  "unavailable-declarations\0"
  "undeclared-selector\0"
  "undef\0"
  "undef-prefix\0"
  "undef-true\0"
  "undefined-arm-za\0"
  "undefined-arm-zt0\0"
  "undefined-bool-conversion\0"
  "undefined-func-template\0"
  "undefined-inline\0"
  "undefined-internal\0"
  "undefined-internal-type\0"
  "undefined-reinterpret-cast\0"
  "undefined-var-template\0"
  "underaligned-exception-object\0"
  "unevaluated-expression\0"
  "unguarded-availability\0"
  "unguarded-availability-new\0"
  "unicode\0"
  "unicode-homoglyph\0"
  "unicode-whitespace\0"
  "unicode-zero-width\0"
  "uninitialized\0"
  "uninitialized-const-reference\0"
  "uninitialized-explicit-init\0"
  "unique-object-duplication\0"
  "unknown-argument\0"
  "unknown-assumption\0"
  "unknown-attributes\0"
  "unknown-cuda-version\0"
  "unknown-directives\0"
  "unknown-escape-sequence\0"
  "unknown-pragmas\0"
  "unknown-sanitizers\0"
  "unknown-warning-option\0"
  "unnamed-type-template-args\0"
  "unnecessary-virtual-specifier\0"
  "unneeded-internal-declaration\0"
  "unneeded-member-function\0"
  "unqualified-std-cast-call\0"
  "unreachable-code\0"
  "unreachable-code-aggressive\0"
  "unreachable-code-break\0"
  "unreachable-code-fallthrough\0"
  "unreachable-code-generic-assoc\0"
  "unreachable-code-loop-increment\0"
  "unreachable-code-return\0"
  "unsafe-buffer-usage\0"
  "unsafe-buffer-usage-in-container\0"
  "unsafe-buffer-usage-in-libc-call\0"
  "unsequenced\0"
  "unsupported-abi\0"
  "unsupported-abs\0"
  "unsupported-availability-guard\0"
  "unsupported-cb\0"
  "unsupported-dll-base-class-template\0"
  "unsupported-floating-point-opt\0"
  "unsupported-friend\0"
  "unsupported-gpopt\0"
  "unsupported-nan\0"
  "unsupported-target-opt\0"
  "unsupported-visibility\0"
  "unusable-partial-specialization\0"
  "unused\0"
  "unused-argument\0"
  "unused-but-set-parameter\0"
  "unused-but-set-variable\0"
  "unused-command-line-argument\0"
  "unused-comparison\0"
  "unused-const-variable\0"
  "unused-exception-parameter\0"
  "unused-function\0"
  "unused-getter-return-value\0"
  "unused-label\0"
  "unused-lambda-capture\0"
  "unused-local-typedef\0"
  "unused-local-typedefs\0"
  "unused-macros\0"
  "unused-member-function\0"
  "unused-parameter\0"
  "unused-private-field\0"
  "unused-property-ivar\0"
  "unused-result\0"
  "unused-template\0"
  "unused-value\0"
  "unused-variable\0"
  "unused-volatile-lvalue\0"
  "used-but-marked-unused\0"
  "user-defined-literals\0"
  "user-defined-warnings\0"
  "varargs\0"
  "variadic-macro-arguments-omitted\0"
  "variadic-macros\0"
  "vec-elem-size\0"
  "vector-conversion\0"
  "vector-conversions\0"
  "vexing-parse\0"
  "visibility\0"
  "vla\0"
  "vla-cxx-extension\0"
  "vla-extension\0"
  "vla-extension-static-assert\0"
  "void-pointer-to-enum-cast\0"
  "void-pointer-to-int-cast\0"
  "void-ptr-dereference\0"
  "volatile-register-var\0"
  "wasm-exception-spec\0"
  "weak-template-vtables\0"
  "weak-vtables\0"
  "writable-strings\0"
  "write-strings\0"
  "xor-used-as-pow\0"
  "zero-as-null-pointer-constant\0"
  "zero-length-array\0"
  ;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static constexpr llvm::StringTable DiagGroupNames =
    DiagGroupNamesStorage;

#endif // GET_DIAG_ARRAYS


#ifdef DIAG_ENTRY
DIAG_ENTRY(anonymous_59 /*  */, 0, 0, /* DiagSubGroup0 */ 1, R"()")
DIAG_ENTRY(PoundPragmaMessage /* #pragma-messages */, 1, /* DiagArray1 */ 1, 0, R"()")
DIAG_ENTRY(PoundWarning /* #warnings */, 18, /* DiagArray2 */ 3, 0, R"()")
DIAG_ENTRY(anonymous_320 /* CFString-literal */, 28, /* DiagArray3 */ 5, 0, R"()")
DIAG_ENTRY(anonymous_58 /* CL4 */, 45, 0, /* DiagSubGroup4 */ 3, R"()")
DIAG_ENTRY(IndependentClassAttribute /* IndependentClass-attribute */, 49, /* DiagArray5 */ 7, 0, R"()")
DIAG_ENTRY(NSobjectAttribute /* NSObject-attribute */, 76, /* DiagArray6 */ 10, 0, R"()")
DIAG_ENTRY(AArch64SMEAttributes /* aarch64-sme-attributes */, 95, /* DiagArray7 */ 12, 0, R"()")
DIAG_ENTRY(anonymous_0 /* abi */, 118, 0, 0, R"()")
DIAG_ENTRY(AbsoluteValue /* absolute-value */, 122, /* DiagArray9 */ 16, 0, R"()")
DIAG_ENTRY(AbstractFinalClass /* abstract-final-class */, 137, /* DiagArray10 */ 21, 0, R"()")
DIAG_ENTRY(anonymous_312 /* abstract-vbase-init */, 158, /* DiagArray11 */ 23, 0, R"()")
DIAG_ENTRY(anonymous_57 /* address */, 178, 0, /* DiagSubGroup12 */ 6, R"()")
DIAG_ENTRY(anonymous_286 /* address-of-packed-member */, 186, /* DiagArray13 */ 25, 0, R"()")
DIAG_ENTRY(AddressOfTemporary /* address-of-temporary */, 211, /* DiagArray14 */ 27, 0, R"()")
DIAG_ENTRY(anonymous_2 /* aggregate-return */, 232, 0, 0, R"()")
DIAG_ENTRY(AIXCompat /* aix-compat */, 249, /* DiagArray16 */ 29, 0, R"()")
DIAG_ENTRY(anonymous_298 /* alias-template-in-declaration-name */, 260, /* DiagArray17 */ 32, 0, R"()")
DIAG_ENTRY(anonymous_287 /* align-mismatch */, 295, /* DiagArray18 */ 34, 0, R"()")
DIAG_ENTRY(All /* all */, 310, 0, /* DiagSubGroup19 */ 10, R"()")
DIAG_ENTRY(anonymous_226 /* alloca */, 314, /* DiagArray20 */ 36, 0, R"()")
DIAG_ENTRY(anonymous_227 /* alloca-with-align-alignof */, 321, /* DiagArray21 */ 38, 0, R"()")
DIAG_ENTRY(AlwaysInlineCoroutine /* always-inline-coroutine */, 347, /* DiagArray22 */ 40, 0, R"()")
DIAG_ENTRY(anonymous_293 /* ambiguous-delete */, 371, /* DiagArray23 */ 42, 0, R"()")
DIAG_ENTRY(anonymous_151 /* ambiguous-ellipsis */, 388, /* DiagArray24 */ 44, 0, R"()")
DIAG_ENTRY(AmbiguousMacro /* ambiguous-macro */, 407, /* DiagArray25 */ 46, 0, R"()")
DIAG_ENTRY(AmbigMemberTemplate /* ambiguous-member-template */, 423, /* DiagArray26 */ 48, 0, R"()")
DIAG_ENTRY(anonymous_247 /* ambiguous-reversed-operator */, 449, /* DiagArray27 */ 50, 0, R"()")
DIAG_ENTRY(anonymous_108 /* analyzer-incompatible-plugin */, 477, /* DiagArray28 */ 52, 0, R"()")
DIAG_ENTRY(anonymous_105 /* android-unversioned-fallback */, 506, /* DiagArray29 */ 54, 0, R"()")
DIAG_ENTRY(AnonEnumEnumConversion /* anon-enum-enum-conversion */, 535, /* DiagArray30 */ 56, /* DiagSubGroup30 */ 19, R"()")
DIAG_ENTRY(anonymous_152 /* anonymous-pack-parens */, 561, /* DiagArray31 */ 58, 0, R"()")
DIAG_ENTRY(anonymous_89 /* apinotes */, 583, /* DiagArray32 */ 60, 0, R"()")
DIAG_ENTRY(AutomaticReferenceCounting /* arc */, 592, 0, /* DiagSubGroup33 */ 21, R"()")
DIAG_ENTRY(anonymous_150 /* arc-bridge-casts-disallowed-in-nonarc */, 596, /* DiagArray34 */ 62, 0, R"()")
DIAG_ENTRY(ARCRepeatedUseOfWeakMaybe /* arc-maybe-repeated-use-of-weak */, 634, /* DiagArray35 */ 64, 0, R"()")
DIAG_ENTRY(ARCNonPodMemAccess /* arc-non-pod-memaccess */, 665, /* DiagArray36 */ 66, 0, R"()")
DIAG_ENTRY(anonymous_206 /* arc-performSelector-leaks */, 687, /* DiagArray37 */ 68, 0, R"()")
DIAG_ENTRY(ARCRepeatedUseOfWeak /* arc-repeated-use-of-weak */, 713, /* DiagArray38 */ 70, /* DiagSubGroup38 */ 25, R"()")
DIAG_ENTRY(ARCRetainCycles /* arc-retain-cycles */, 738, /* DiagArray39 */ 72, 0, R"()")
DIAG_ENTRY(ARCUnsafeRetainedAssign /* arc-unsafe-retained-assign */, 756, /* DiagArray40 */ 74, 0, R"()")
DIAG_ENTRY(anonymous_322 /* argument-outside-range */, 783, /* DiagArray41 */ 78, 0, R"()")
DIAG_ENTRY(anonymous_323 /* argument-undefined-behaviour */, 806, /* DiagArray42 */ 80, 0, R"()")
DIAG_ENTRY(anonymous_185 /* arm-interrupt-save-fp-no-vfp-unit */, 835, /* DiagArray43 */ 82, 0, R"()")
DIAG_ENTRY(anonymous_184 /* arm-interrupt-vfp-clobber */, 869, /* DiagArray44 */ 84, 0, R"()")
DIAG_ENTRY(ArrayBounds /* array-bounds */, 895, /* DiagArray45 */ 86, 0, R"()")
DIAG_ENTRY(ArrayBoundsPointerArithmetic /* array-bounds-pointer-arithmetic */, 908, /* DiagArray46 */ 92, 0, R"()")
DIAG_ENTRY(anonymous_318 /* array-compare */, 940, /* DiagArray47 */ 95, 0, R"()")
DIAG_ENTRY(anonymous_319 /* array-compare-cxx26 */, 954, /* DiagArray48 */ 97, 0, R"()")
DIAG_ENTRY(ArrayParameter /* array-parameter */, 974, /* DiagArray49 */ 99, 0, R"()")
DIAG_ENTRY(ASM /* asm */, 990, 0, /* DiagSubGroup50 */ 27, R"()")
DIAG_ENTRY(ASMOperandWidths /* asm-operand-widths */, 994, /* DiagArray51 */ 101, 0, R"()")
DIAG_ENTRY(anonymous_321 /* assign-enum */, 1013, /* DiagArray52 */ 103, 0, R"()")
DIAG_ENTRY(anonymous_193 /* assume */, 1025, /* DiagArray53 */ 105, 0, R"()")
DIAG_ENTRY(anonymous_56 /* at-protocol */, 1032, 0, 0, R"()")
DIAG_ENTRY(FrameworkHdrAtImport /* atimport-in-framework-header */, 1044, /* DiagArray55 */ 107, 0, R"()")
DIAG_ENTRY(anonymous_273 /* atomic-access */, 1073, /* DiagArray56 */ 109, 0, R"()")
DIAG_ENTRY(AtomicAlignment /* atomic-alignment */, 1087, /* DiagArray57 */ 111, 0, R"()")
DIAG_ENTRY(anonymous_303 /* atomic-implicit-seq-cst */, 1104, /* DiagArray58 */ 114, 0, R"()")
DIAG_ENTRY(anonymous_302 /* atomic-memory-ordering */, 1128, /* DiagArray59 */ 116, 0, R"()")
DIAG_ENTRY(AtomicProperties /* atomic-properties */, 1151, 0, /* DiagSubGroup60 */ 29, R"()")
DIAG_ENTRY(anonymous_201 /* atomic-property-with-user-defined-accessor */, 1169, /* DiagArray61 */ 118, 0, R"()")
DIAG_ENTRY(anonymous_243 /* attribute-packed-for-bitfield */, 1212, /* DiagArray62 */ 120, 0, R"()")
DIAG_ENTRY(HLSLAttributeStatement /* attribute-statement */, 1242, /* DiagArray63 */ 122, 0, R"()")
DIAG_ENTRY(BackendWarningAttributes /* attribute-warning */, 1262, /* DiagArray64 */ 124, 0, R"()")
DIAG_ENTRY(Attributes /* attributes */, 1280, 0, /* DiagSubGroup65 */ 32, R"()")
DIAG_ENTRY(anonymous_182 /* auto-decl-extensions */, 1291, /* DiagArray66 */ 127, 0, R"()")
DIAG_ENTRY(AutoDisableVptrSanitizer /* auto-disable-vptr-sanitizer */, 1312, /* DiagArray67 */ 129, 0, R"()")
DIAG_ENTRY(anonymous_3 /* auto-import */, 1340, 0, 0, R"()")
DIAG_ENTRY(anonymous_148 /* auto-storage-class */, 1352, /* DiagArray69 */ 131, 0, R"()")
DIAG_ENTRY(anonymous_220 /* auto-var-id */, 1371, /* DiagArray70 */ 133, 0, R"()")
DIAG_ENTRY(Availability /* availability */, 1383, /* DiagArray71 */ 135, 0, R"()")
DIAG_ENTRY(AVRRtlibLinkingQuirks /* avr-rtlib-linking-quirks */, 1396, /* DiagArray72 */ 147, 0, R"()")
DIAG_ENTRY(BackendPlugin /* backend-plugin */, 1421, /* DiagArray73 */ 153, 0, R"()")
DIAG_ENTRY(anonymous_112 /* backslash-newline-escape */, 1436, /* DiagArray74 */ 156, 0, R"()")
DIAG_ENTRY(BadFunctionCast /* bad-function-cast */, 1461, /* DiagArray75 */ 158, 0, R"()")
DIAG_ENTRY(BinaryLiteral /* binary-literal */, 1479, 0, /* DiagSubGroup76 */ 35, R"()")
DIAG_ENTRY(BindToTemporaryCopy /* bind-to-temporary-copy */, 1494, /* DiagArray77 */ 160, /* DiagSubGroup77 */ 38, R"()")
DIAG_ENTRY(BitIntExtension /* bit-int-extension */, 1517, /* DiagArray78 */ 163, 0, R"()")
DIAG_ENTRY(BitFieldConstantConversion /* bitfield-constant-conversion */, 1535, /* DiagArray79 */ 166, /* DiagSubGroup79 */ 40, R"()")
DIAG_ENTRY(BitFieldEnumConversion /* bitfield-enum-conversion */, 1564, /* DiagArray80 */ 168, 0, R"()")
DIAG_ENTRY(BitFieldWidth /* bitfield-width */, 1589, /* DiagArray81 */ 172, 0, R"()")
DIAG_ENTRY(BitwiseConditionalParentheses /* bitwise-conditional-parentheses */, 1604, /* DiagArray82 */ 174, 0, R"()")
DIAG_ENTRY(BitwiseInsteadOfLogical /* bitwise-instead-of-logical */, 1636, /* DiagArray83 */ 176, 0, R"()")
DIAG_ENTRY(BitwiseOpParentheses /* bitwise-op-parentheses */, 1663, /* DiagArray84 */ 178, 0, R"()")
DIAG_ENTRY(BlockCaptureAutoReleasing /* block-capture-autoreleasing */, 1686, /* DiagArray85 */ 180, 0, R"()")
DIAG_ENTRY(BoolConversion /* bool-conversion */, 1714, /* DiagArray86 */ 182, /* DiagSubGroup86 */ 42, R"()")
DIAG_ENTRY(anonymous_64 /* bool-conversions */, 1730, 0, /* DiagSubGroup87 */ 45, R"()")
DIAG_ENTRY(BoolOperation /* bool-operation */, 1747, /* DiagArray88 */ 184, /* DiagSubGroup88 */ 47, R"()")
DIAG_ENTRY(BoundsSafetyCountedByEltTyUnknownSize /* bounds-safety-counted-by-elt-type-unknown-size */, 1762, /* DiagArray89 */ 186, 0, R"()")
DIAG_ENTRY(anonymous_267 /* braced-scalar-init */, 1809, /* DiagArray90 */ 188, 0, R"()")
DIAG_ENTRY(BranchProtection /* branch-protection */, 1828, /* DiagArray91 */ 190, 0, R"()")
DIAG_ENTRY(ObjCBridge /* bridge-cast */, 1846, /* DiagArray92 */ 195, 0, R"()")
DIAG_ENTRY(anonymous_229 /* builtin-assume-aligned-alignment */, 1858, /* DiagArray93 */ 198, 0, R"()")
DIAG_ENTRY(BuiltinMacroRedefined /* builtin-macro-redefined */, 1891, /* DiagArray94 */ 200, 0, R"()")
DIAG_ENTRY(anonymous_194 /* builtin-memcpy-chk-size */, 1915, /* DiagArray95 */ 203, 0, R"()")
DIAG_ENTRY(BuiltinRequiresHeader /* builtin-requires-header */, 1939, /* DiagArray96 */ 205, 0, R"()")
DIAG_ENTRY(CXXCompat /* c++-compat */, 1963, /* DiagArray97 */ 207, 0, R"()")
DIAG_ENTRY(anonymous_26 /* c++0x-compat */, 1974, 0, /* DiagSubGroup98 */ 49, R"()")
DIAG_ENTRY(anonymous_68 /* c++0x-extensions */, 1987, 0, /* DiagSubGroup99 */ 51, R"()")
DIAG_ENTRY(anonymous_24 /* c++0x-narrowing */, 2004, 0, /* DiagSubGroup100 */ 53, R"()")
DIAG_ENTRY(CXX11Compat /* c++11-compat */, 2020, /* DiagArray101 */ 209, /* DiagSubGroup101 */ 55, R"()")
DIAG_ENTRY(CXX11CompatDeprecatedWritableStr /* c++11-compat-deprecated-writable-strings */, 2033, /* DiagArray102 */ 220, 0, R"()")
DIAG_ENTRY(CXX11CompatPedantic /* c++11-compat-pedantic */, 2074, 0, /* DiagSubGroup103 */ 64, R"()")
DIAG_ENTRY(CXX11CompatReservedUserDefinedLiteral /* c++11-compat-reserved-user-defined-literal */, 2096, /* DiagArray104 */ 222, 0, R"()")
DIAG_ENTRY(CXX11 /* c++11-extensions */, 2139, /* DiagArray105 */ 224, /* DiagSubGroup105 */ 71, R"()")
DIAG_ENTRY(CXX11ExtraSemi /* c++11-extra-semi */, 2156, /* DiagArray106 */ 252, 0, R"()")
DIAG_ENTRY(CXX11InlineNamespace /* c++11-inline-namespace */, 2173, /* DiagArray107 */ 254, 0, R"()")
DIAG_ENTRY(CXX11LongLong /* c++11-long-long */, 2196, /* DiagArray108 */ 256, 0, R"()")
DIAG_ENTRY(CXX11Narrowing /* c++11-narrowing */, 2212, /* DiagArray109 */ 258, /* DiagSubGroup109 */ 75, R"()")
DIAG_ENTRY(CXX11NarrowingConstReference /* c++11-narrowing-const-reference */, 2228, /* DiagArray110 */ 266, 0, R"()")
DIAG_ENTRY(CXX14Attrs /* c++14-attribute-extensions */, 2260, /* DiagArray111 */ 270, 0, R"()")
DIAG_ENTRY(CXX14BinaryLiteral /* c++14-binary-literal */, 2287, /* DiagArray112 */ 272, 0, R"()")
DIAG_ENTRY(CXX14Compat /* c++14-compat */, 2308, 0, /* DiagSubGroup113 */ 77, R"()")
DIAG_ENTRY(CXX14CompatPedantic /* c++14-compat-pedantic */, 2321, 0, /* DiagSubGroup114 */ 82, R"()")
DIAG_ENTRY(CXX14 /* c++14-extensions */, 2343, /* DiagArray115 */ 274, /* DiagSubGroup115 */ 88, R"()")
DIAG_ENTRY(CXX17Attrs /* c++17-attribute-extensions */, 2360, /* DiagArray116 */ 282, 0, R"()")
DIAG_ENTRY(CXX17Compat /* c++17-compat */, 2387, 0, /* DiagSubGroup117 */ 91, R"()")
DIAG_ENTRY(CXX17CompatMangling /* c++17-compat-mangling */, 2400, /* DiagArray118 */ 284, 0, R"()")
DIAG_ENTRY(CXX17CompatPedantic /* c++17-compat-pedantic */, 2422, 0, /* DiagSubGroup119 */ 98, R"()")
DIAG_ENTRY(CXX17 /* c++17-extensions */, 2444, /* DiagArray120 */ 286, /* DiagSubGroup120 */ 103, R"()")
DIAG_ENTRY(anonymous_69 /* c++1y-extensions */, 2461, 0, /* DiagSubGroup121 */ 105, R"()")
DIAG_ENTRY(anonymous_27 /* c++1z-compat */, 2478, 0, /* DiagSubGroup122 */ 107, R"()")
DIAG_ENTRY(anonymous_13 /* c++1z-compat-mangling */, 2491, 0, /* DiagSubGroup123 */ 109, R"()")
DIAG_ENTRY(anonymous_70 /* c++1z-extensions */, 2513, 0, /* DiagSubGroup124 */ 111, R"()")
DIAG_ENTRY(CXX20Attrs /* c++20-attribute-extensions */, 2530, /* DiagArray125 */ 304, 0, R"()")
DIAG_ENTRY(CXX20Compat /* c++20-compat */, 2557, /* DiagArray126 */ 306, /* DiagSubGroup126 */ 113, R"()")
DIAG_ENTRY(CXX20CompatPedantic /* c++20-compat-pedantic */, 2570, 0, /* DiagSubGroup127 */ 116, R"()")
DIAG_ENTRY(CXX20Designator /* c++20-designator */, 2592, /* DiagArray128 */ 317, 0, R"()")
DIAG_ENTRY(CXX20 /* c++20-extensions */, 2609, /* DiagArray129 */ 319, /* DiagSubGroup129 */ 120, R"()")
DIAG_ENTRY(CXX23Attrs /* c++23-attribute-extensions */, 2626, /* DiagArray130 */ 342, 0, R"()")
DIAG_ENTRY(CXX23Compat /* c++23-compat */, 2653, 0, /* DiagSubGroup131 */ 124, R"()")
DIAG_ENTRY(CXX23 /* c++23-extensions */, 2666, /* DiagArray132 */ 344, /* DiagSubGroup132 */ 126, R"()")
DIAG_ENTRY(CXX23AttrsOnLambda /* c++23-lambda-attributes */, 2683, /* DiagArray133 */ 355, 0, R"()")
DIAG_ENTRY(CXX26 /* c++26-extensions */, 2707, /* DiagArray134 */ 357, 0, R"()")
DIAG_ENTRY(anonymous_28 /* c++2a-compat */, 2724, 0, /* DiagSubGroup135 */ 128, R"()")
DIAG_ENTRY(anonymous_29 /* c++2a-compat-pedantic */, 2737, 0, /* DiagSubGroup136 */ 130, R"()")
DIAG_ENTRY(anonymous_71 /* c++2a-extensions */, 2759, 0, /* DiagSubGroup137 */ 132, R"()")
DIAG_ENTRY(anonymous_72 /* c++2b-extensions */, 2776, 0, /* DiagSubGroup138 */ 134, R"()")
DIAG_ENTRY(CXX26Compat /* c++2c-compat */, 2793, 0, /* DiagSubGroup139 */ 136, R"()")
DIAG_ENTRY(anonymous_73 /* c++2c-extensions */, 2806, 0, /* DiagSubGroup140 */ 138, R"()")
DIAG_ENTRY(anonymous_20 /* c++98-c++11-c++14-c++17-compat */, 2823, 0, /* DiagSubGroup141 */ 140, R"()")
DIAG_ENTRY(anonymous_21 /* c++98-c++11-c++14-c++17-compat-pedantic */, 2854, 0, /* DiagSubGroup142 */ 142, R"()")
DIAG_ENTRY(anonymous_18 /* c++98-c++11-c++14-compat */, 2894, 0, /* DiagSubGroup143 */ 144, R"()")
DIAG_ENTRY(anonymous_19 /* c++98-c++11-c++14-compat-pedantic */, 2919, 0, /* DiagSubGroup144 */ 146, R"()")
DIAG_ENTRY(anonymous_16 /* c++98-c++11-compat */, 2953, 0, /* DiagSubGroup145 */ 148, R"()")
DIAG_ENTRY(CXXPre14CompatBinaryLiteral /* c++98-c++11-compat-binary-literal */, 2972, /* DiagArray146 */ 368, 0, R"()")
DIAG_ENTRY(anonymous_17 /* c++98-c++11-compat-pedantic */, 3006, 0, /* DiagSubGroup147 */ 150, R"()")
DIAG_ENTRY(CXX98Compat /* c++98-compat */, 3034, /* DiagArray148 */ 370, /* DiagSubGroup148 */ 152, R"()")
DIAG_ENTRY(CXX98CompatBindToTemporaryCopy /* c++98-compat-bind-to-temporary-copy */, 3047, /* DiagArray149 */ 429, 0, R"()")
DIAG_ENTRY(CXX98CompatExtraSemi /* c++98-compat-extra-semi */, 3083, /* DiagArray150 */ 431, 0, R"()")
DIAG_ENTRY(CXX98CompatLocalTypeTemplateArgs /* c++98-compat-local-type-template-args */, 3107, /* DiagArray151 */ 433, 0, R"()")
DIAG_ENTRY(CXX98CompatPedantic /* c++98-compat-pedantic */, 3145, /* DiagArray152 */ 435, /* DiagSubGroup152 */ 160, R"()")
DIAG_ENTRY(CXX98CompatUnnamedTypeTemplateArgs /* c++98-compat-unnamed-type-template-args */, 3167, /* DiagArray153 */ 444, 0, R"()")
DIAG_ENTRY(C11 /* c11-extensions */, 3207, /* DiagArray154 */ 446, 0, R"()")
DIAG_ENTRY(C23Compat /* c23-compat */, 3222, /* DiagArray155 */ 451, 0, R"()")
DIAG_ENTRY(C23 /* c23-extensions */, 3233, /* DiagArray156 */ 454, /* DiagSubGroup156 */ 169, R"()")
DIAG_ENTRY(anonymous_4 /* c2x-compat */, 3248, 0, /* DiagSubGroup157 */ 171, R"()")
DIAG_ENTRY(anonymous_74 /* c2x-extensions */, 3259, 0, /* DiagSubGroup158 */ 173, R"()")
DIAG_ENTRY(C2y /* c2y-extensions */, 3274, /* DiagArray159 */ 472, 0, R"()")
DIAG_ENTRY(C99Compat /* c99-compat */, 3289, /* DiagArray160 */ 481, 0, R"()")
DIAG_ENTRY(C99Designator /* c99-designator */, 3300, /* DiagArray161 */ 486, /* DiagSubGroup161 */ 175, R"()")
DIAG_ENTRY(C99 /* c99-extensions */, 3315, /* DiagArray162 */ 492, /* DiagSubGroup162 */ 179, R"()")
DIAG_ENTRY(PureVirtualCallFromCtorDtor /* call-to-pure-virtual-from-ctor-dtor */, 3330, /* DiagArray163 */ 503, 0, R"()")
DIAG_ENTRY(CalledOnceParameter /* called-once-parameter */, 3366, /* DiagArray164 */ 505, /* DiagSubGroup164 */ 181, R"()")
DIAG_ENTRY(CastAlign /* cast-align */, 3388, /* DiagArray165 */ 509, 0, R"()")
DIAG_ENTRY(anonymous_306 /* cast-calling-convention */, 3399, /* DiagArray166 */ 511, 0, R"()")
DIAG_ENTRY(CastFunctionType /* cast-function-type */, 3423, 0, /* DiagSubGroup167 */ 183, R"()")
DIAG_ENTRY(CastFunctionTypeMismatch /* cast-function-type-mismatch */, 3442, /* DiagArray168 */ 513, 0, R"()")
DIAG_ENTRY(CastFunctionTypeStrict /* cast-function-type-strict */, 3470, /* DiagArray169 */ 515, 0, R"()")
DIAG_ENTRY(SelTypeCast /* cast-of-sel-type */, 3496, /* DiagArray170 */ 517, 0, R"()")
DIAG_ENTRY(CastQual /* cast-qual */, 3513, /* DiagArray171 */ 519, 0, R"()")
DIAG_ENTRY(anonymous_291 /* cast-qual-unrelated */, 3523, /* DiagArray172 */ 522, 0, R"()")
DIAG_ENTRY(anonymous_5 /* char-align */, 3543, 0, 0, R"()")
DIAG_ENTRY(CharSubscript /* char-subscripts */, 3554, /* DiagArray174 */ 524, 0, R"()")
DIAG_ENTRY(ClangClPch /* clang-cl-pch */, 3570, /* DiagArray175 */ 527, 0, R"()")
DIAG_ENTRY(ClassConversion /* class-conversion */, 3583, /* DiagArray176 */ 532, 0, R"()")
DIAG_ENTRY(ClassVarargs /* class-varargs */, 3600, /* DiagArray177 */ 536, /* DiagSubGroup177 */ 186, R"()")
DIAG_ENTRY(anonymous_230 /* cmse-union-leak */, 3614, /* DiagArray178 */ 538, 0, R"()")
DIAG_ENTRY(anonymous_180 /* comma */, 3630, /* DiagArray179 */ 540, 0, R"()")
DIAG_ENTRY(Comment /* comment */, 3636, /* DiagArray180 */ 542, 0, R"()")
DIAG_ENTRY(anonymous_62 /* comments */, 3644, 0, /* DiagSubGroup181 */ 188, R"()")
DIAG_ENTRY(CompareDistinctPointerType /* compare-distinct-pointer-types */, 3653, /* DiagArray182 */ 547, 0, R"()")
DIAG_ENTRY(CompletionHandler /* completion-handler */, 3684, /* DiagArray183 */ 549, 0, R"()")
DIAG_ENTRY(anonymous_269 /* complex-component-init */, 3703, /* DiagArray184 */ 553, 0, R"()")
DIAG_ENTRY(CompoundTokenSplit /* compound-token-split */, 3726, 0, /* DiagSubGroup185 */ 190, R"()")
DIAG_ENTRY(CompoundTokenSplitByMacro /* compound-token-split-by-macro */, 3747, /* DiagArray186 */ 555, 0, R"()")
DIAG_ENTRY(CompoundTokenSplitBySpace /* compound-token-split-by-space */, 3777, /* DiagArray187 */ 557, 0, R"()")
DIAG_ENTRY(anonymous_308 /* conditional-type-mismatch */, 3807, /* DiagArray188 */ 559, 0, R"()")
DIAG_ENTRY(UninitializedMaybe /* conditional-uninitialized */, 3833, /* DiagArray189 */ 561, 0, R"()")
DIAG_ENTRY(ConfigMacros /* config-macros */, 3859, /* DiagArray190 */ 563, 0, R"()")
DIAG_ENTRY(ConstantConversion /* constant-conversion */, 3873, /* DiagArray191 */ 565, /* DiagSubGroup191 */ 193, R"()")
DIAG_ENTRY(anonymous_81 /* constant-evaluated */, 3893, /* DiagArray192 */ 567, 0, R"()")
DIAG_ENTRY(anonymous_281 /* constant-logical-operand */, 3912, /* DiagArray193 */ 569, 0, R"()")
DIAG_ENTRY(anonymous_218 /* constexpr-not-const */, 3937, /* DiagArray194 */ 571, 0, R"()")
DIAG_ENTRY(Consumed /* consumed */, 3957, /* DiagArray195 */ 573, 0, R"()")
DIAG_ENTRY(Conversion /* conversion */, 3966, /* DiagArray196 */ 582, /* DiagSubGroup196 */ 196, R"()")
DIAG_ENTRY(anonymous_63 /* conversion-null */, 3977, 0, /* DiagSubGroup197 */ 211, R"()")
DIAG_ENTRY(CoroNonAlignedAllocationFunction /* coro-non-aligned-allocation-function */, 3993, /* DiagArray198 */ 589, 0, R"()")
DIAG_ENTRY(CoroTypeAwareAllocationFunction /* coro-type-aware-allocation-function */, 4030, /* DiagArray199 */ 591, 0, R"()")
DIAG_ENTRY(Coroutine /* coroutine */, 4066, /* DiagArray200 */ 593, /* DiagSubGroup200 */ 213, R"()")
DIAG_ENTRY(CoroutineMissingUnhandledException /* coroutine-missing-unhandled-exception */, 4076, /* DiagArray201 */ 595, 0, R"()")
DIAG_ENTRY(CoveredSwitchDefault /* covered-switch-default */, 4114, /* DiagArray202 */ 597, 0, R"()")
DIAG_ENTRY(anonymous_61 /* cpp */, 4137, 0, /* DiagSubGroup203 */ 219, R"()")
DIAG_ENTRY(ObjCCStringFormat /* cstring-format-directive */, 4141, /* DiagArray204 */ 599, 0, R"()")
DIAG_ENTRY(CTADMaybeUnsupported /* ctad-maybe-unsupported */, 4166, /* DiagArray205 */ 601, 0, R"()")
DIAG_ENTRY(anonymous_6 /* ctor-dtor-privacy */, 4189, 0, 0, R"()")
DIAG_ENTRY(CrossTU /* ctu */, 4207, /* DiagArray207 */ 603, 0, R"()")
DIAG_ENTRY(CudaCompat /* cuda-compat */, 4211, /* DiagArray208 */ 605, 0, R"()")
DIAG_ENTRY(CustomAtomic /* custom-atomic-properties */, 4223, /* DiagArray209 */ 611, 0, R"()")
DIAG_ENTRY(anonymous_153 /* cxx-attribute-extension */, 4248, /* DiagArray210 */ 613, 0, R"()")
DIAG_ENTRY(Dangling /* dangling */, 4272, /* DiagArray211 */ 615, /* DiagSubGroup211 */ 221, R"()")
DIAG_ENTRY(DanglingAssignment /* dangling-assignment */, 4281, /* DiagArray212 */ 617, 0, R"()")
DIAG_ENTRY(DanglingAssignmentGsl /* dangling-assignment-gsl */, 4301, /* DiagArray213 */ 619, 0, R"()")
DIAG_ENTRY(DanglingCapture /* dangling-capture */, 4325, /* DiagArray214 */ 621, 0, R"()")
DIAG_ENTRY(DanglingElse /* dangling-else */, 4342, /* DiagArray215 */ 624, 0, R"()")
DIAG_ENTRY(DanglingField /* dangling-field */, 4356, /* DiagArray216 */ 626, 0, R"()")
DIAG_ENTRY(DanglingGsl /* dangling-gsl */, 4371, /* DiagArray217 */ 631, 0, R"()")
DIAG_ENTRY(DanglingInitializerList /* dangling-initializer-list */, 4384, /* DiagArray218 */ 634, 0, R"()")
DIAG_ENTRY(anonymous_101 /* darwin-sdk-settings */, 4410, /* DiagArray219 */ 636, 0, R"()")
DIAG_ENTRY(anonymous_138 /* date-time */, 4430, /* DiagArray220 */ 638, 0, R"()")
DIAG_ENTRY(DeallocInCategory /* dealloc-in-category */, 4440, /* DiagArray221 */ 640, 0, R"()")
DIAG_ENTRY(anonymous_96 /* debug-compression-unavailable */, 4460, /* DiagArray222 */ 642, 0, R"()")
DIAG_ENTRY(DeclarationAfterStatement /* declaration-after-statement */, 4490, /* DiagArray223 */ 644, 0, R"()")
DIAG_ENTRY(anonymous_340 /* decls-in-multiple-modules */, 4518, /* DiagArray224 */ 647, 0, R"()")
DIAG_ENTRY(DefaultedFunctionDeleted /* defaulted-function-deleted */, 4544, /* DiagArray225 */ 649, 0, R"()")
DIAG_ENTRY(anonymous_98 /* delayed-template-parsing-in-cxx20 */, 4571, /* DiagArray226 */ 652, 0, R"()")
DIAG_ENTRY(DelegatingCtorCycles /* delegating-ctor-cycles */, 4605, /* DiagArray227 */ 654, 0, R"()")
DIAG_ENTRY(DeleteAbstractNonVirtualDtor /* delete-abstract-non-virtual-dtor */, 4628, /* DiagArray228 */ 656, 0, R"()")
DIAG_ENTRY(DeleteIncomplete /* delete-incomplete */, 4661, /* DiagArray229 */ 658, 0, R"()")
DIAG_ENTRY(DeleteNonAbstractNonVirtualDtor /* delete-non-abstract-non-virtual-dtor */, 4679, /* DiagArray230 */ 661, 0, R"()")
DIAG_ENTRY(DeleteNonVirtualDtor /* delete-non-virtual-dtor */, 4716, 0, /* DiagSubGroup231 */ 229, R"()")
DIAG_ENTRY(anonymous_120 /* delimited-escape-sequence-extension */, 4740, /* DiagArray232 */ 663, 0, R"()")
DIAG_ENTRY(anonymous_295 /* deprecate-lax-vec-conv-all */, 4776, /* DiagArray233 */ 665, 0, R"()")
DIAG_ENTRY(Deprecated /* deprecated */, 4803, /* DiagArray234 */ 667, /* DiagSubGroup234 */ 232, R"()")
DIAG_ENTRY(anonymous_294 /* deprecated-altivec-src-compat */, 4814, /* DiagArray235 */ 677, 0, R"()")
DIAG_ENTRY(DeprecatedAnonEnumEnumConversion /* deprecated-anon-enum-enum-conversion */, 4844, /* DiagArray236 */ 679, 0, R"()")
DIAG_ENTRY(DeprecatedArrayCompare /* deprecated-array-compare */, 4881, /* DiagArray237 */ 681, 0, R"()")
DIAG_ENTRY(DeprecatedAttributes /* deprecated-attributes */, 4906, /* DiagArray238 */ 683, 0, R"()")
DIAG_ENTRY(DeprecatedBuiltins /* deprecated-builtins */, 4928, /* DiagArray239 */ 687, 0, R"()")
DIAG_ENTRY(DeprecatedCommaSubscript /* deprecated-comma-subscript */, 4948, /* DiagArray240 */ 689, 0, R"()")
DIAG_ENTRY(DeprecatedCopy /* deprecated-copy */, 4975, /* DiagArray241 */ 691, /* DiagSubGroup241 */ 258, R"()")
DIAG_ENTRY(anonymous_8 /* deprecated-copy-dtor */, 4991, 0, /* DiagSubGroup242 */ 260, R"()")
DIAG_ENTRY(DeprecatedCopyWithDtor /* deprecated-copy-with-dtor */, 5012, /* DiagArray243 */ 693, /* DiagSubGroup243 */ 262, R"()")
DIAG_ENTRY(DeprecatedCopyWithUserProvidedCopy /* deprecated-copy-with-user-provided-copy */, 5038, /* DiagArray244 */ 695, 0, R"()")
DIAG_ENTRY(DeprecatedCopyWithUserProvidedDtor /* deprecated-copy-with-user-provided-dtor */, 5078, /* DiagArray245 */ 697, 0, R"()")
DIAG_ENTRY(DeprecatedCoroutine /* deprecated-coroutine */, 5118, /* DiagArray246 */ 699, 0, R"()")
DIAG_ENTRY(DeprecatedDeclarations /* deprecated-declarations */, 5139, /* DiagArray247 */ 701, 0, R"()")
DIAG_ENTRY(DeprecatedDynamicExceptionSpec /* deprecated-dynamic-exception-spec */, 5163, /* DiagArray248 */ 708, 0, R"()")
DIAG_ENTRY(DeprecatedEnumCompare /* deprecated-enum-compare */, 5197, /* DiagArray249 */ 710, 0, R"()")
DIAG_ENTRY(DeprecatedEnumCompareConditional /* deprecated-enum-compare-conditional */, 5221, /* DiagArray250 */ 712, 0, R"()")
DIAG_ENTRY(DeprecatedEnumEnumConversion /* deprecated-enum-enum-conversion */, 5257, /* DiagArray251 */ 714, 0, R"()")
DIAG_ENTRY(DeprecatedEnumFloatConversion /* deprecated-enum-float-conversion */, 5289, /* DiagArray252 */ 716, 0, R"()")
DIAG_ENTRY(DeprecatedImplementations /* deprecated-implementations */, 5322, /* DiagArray253 */ 718, 0, R"()")
DIAG_ENTRY(DeprecatedIncrementBool /* deprecated-increment-bool */, 5349, /* DiagArray254 */ 721, 0, R"()")
DIAG_ENTRY(DeprecatedLiteralOperator /* deprecated-literal-operator */, 5375, /* DiagArray255 */ 723, 0, R"()")
DIAG_ENTRY(DeprecatedMissingCommaVariadicParam /* deprecated-missing-comma-variadic-parameter */, 5403, /* DiagArray256 */ 725, 0, R"()")
DIAG_ENTRY(DeprecatedModuleDotMap /* deprecated-module-dot-map */, 5447, /* DiagArray257 */ 727, 0, R"()")
DIAG_ENTRY(DeprecatedNonPrototype /* deprecated-non-prototype */, 5473, /* DiagArray258 */ 729, 0, R"()")
DIAG_ENTRY(DeprecatedObjCIsaUsage /* deprecated-objc-isa-usage */, 5498, /* DiagArray259 */ 732, 0, R"()")
DIAG_ENTRY(ObjCPointerIntrospect /* deprecated-objc-pointer-introspection */, 5524, /* DiagArray260 */ 735, /* DiagSubGroup260 */ 264, R"()")
DIAG_ENTRY(ObjCPointerIntrospectPerformSelector /* deprecated-objc-pointer-introspection-performSelector */, 5562, /* DiagArray261 */ 737, 0, R"()")
DIAG_ENTRY(DeprecatedOctalLiterals /* deprecated-octal-literals */, 5616, /* DiagArray262 */ 739, 0, R"()")
DIAG_ENTRY(DeprecatedOFast /* deprecated-ofast */, 5642, /* DiagArray263 */ 741, 0, R"()")
DIAG_ENTRY(DeprecatedPragma /* deprecated-pragma */, 5659, /* DiagArray264 */ 744, 0, R"()")
DIAG_ENTRY(DeprecatedRedundantConstexprStaticDef /* deprecated-redundant-constexpr-static-def */, 5677, /* DiagArray265 */ 746, 0, R"()")
DIAG_ENTRY(DeprecatedRegister /* deprecated-register */, 5719, /* DiagArray266 */ 748, 0, R"()")
DIAG_ENTRY(DeprecatedThisCapture /* deprecated-this-capture */, 5739, /* DiagArray267 */ 750, 0, R"()")
DIAG_ENTRY(DeprecatedType /* deprecated-type */, 5763, /* DiagArray268 */ 752, 0, R"()")
DIAG_ENTRY(DeprecatedVolatile /* deprecated-volatile */, 5779, /* DiagArray269 */ 754, 0, R"()")
DIAG_ENTRY(DeprecatedWritableStr /* deprecated-writable-strings */, 5799, 0, /* DiagSubGroup270 */ 266, R"()")
DIAG_ENTRY(anonymous_327 /* direct-ivar-access */, 5827, /* DiagArray271 */ 760, 0, R"()")
DIAG_ENTRY(anonymous_130 /* disabled-macro-expansion */, 5846, /* DiagArray272 */ 762, 0, R"()")
DIAG_ENTRY(anonymous_10 /* disabled-optimization */, 5871, 0, 0, R"()")
DIAG_ENTRY(anonymous_11 /* discard-qual */, 5893, 0, 0, R"()")
DIAG_ENTRY(DistributedObjectModifiers /* distributed-object-modifiers */, 5906, /* DiagArray275 */ 764, 0, R"()")
DIAG_ENTRY(anonymous_12 /* div-by-zero */, 5935, 0, /* DiagSubGroup276 */ 268, R"()")
DIAG_ENTRY(DivZero /* division-by-zero */, 5947, /* DiagArray277 */ 767, 0, R"()")
DIAG_ENTRY(anonymous_231 /* dll-attribute-on-redeclaration */, 5964, /* DiagArray278 */ 769, 0, R"()")
DIAG_ENTRY(DllexportExplicitInstantiationDecl /* dllexport-explicit-instantiation-decl */, 5995, /* DiagArray279 */ 771, 0, R"()")
DIAG_ENTRY(anonymous_232 /* dllimport-static-field-def */, 6033, /* DiagArray280 */ 773, 0, R"()")
DIAG_ENTRY(Documentation /* documentation */, 6060, /* DiagArray281 */ 775, /* DiagSubGroup281 */ 270, R"()")
DIAG_ENTRY(DocumentationDeprecatedSync /* documentation-deprecated-sync */, 6074, /* DiagArray282 */ 796, 0, R"()")
DIAG_ENTRY(DocumentationHTML /* documentation-html */, 6104, /* DiagArray283 */ 798, 0, R"()")
DIAG_ENTRY(DocumentationPedantic /* documentation-pedantic */, 6123, /* DiagArray284 */ 803, /* DiagSubGroup284 */ 273, R"()")
DIAG_ENTRY(DocumentationUnknownCommand /* documentation-unknown-command */, 6146, /* DiagArray285 */ 805, 0, R"()")
DIAG_ENTRY(anonymous_113 /* dollar-in-identifier-extension */, 6176, /* DiagArray286 */ 808, 0, R"()")
DIAG_ENTRY(DoublePromotion /* double-promotion */, 6207, /* DiagArray287 */ 810, 0, R"()")
DIAG_ENTRY(DtorName /* dtor-name */, 6224, /* DiagArray288 */ 812, 0, R"()")
DIAG_ENTRY(anonymous_217 /* dtor-typedef */, 6234, /* DiagArray289 */ 816, 0, R"()")
DIAG_ENTRY(DuplicateDeclSpecifier /* duplicate-decl-specifier */, 6247, /* DiagArray290 */ 818, 0, R"()")
DIAG_ENTRY(anonymous_179 /* duplicate-enum */, 6272, /* DiagArray291 */ 823, 0, R"()")
DIAG_ENTRY(DuplicateArgDecl /* duplicate-method-arg */, 6287, /* DiagArray292 */ 825, 0, R"()")
DIAG_ENTRY(MethodDuplicate /* duplicate-method-match */, 6308, /* DiagArray293 */ 827, 0, R"()")
DIAG_ENTRY(anonymous_196 /* duplicate-protocol */, 6331, /* DiagArray294 */ 829, 0, R"()")
DIAG_ENTRY(DXILValidation /* dxil-validation */, 6350, /* DiagArray295 */ 831, 0, R"()")
DIAG_ENTRY(DynamicClassMemaccess /* dynamic-class-memaccess */, 6366, /* DiagArray296 */ 833, 0, R"()")
DIAG_ENTRY(DynamicExceptionSpec /* dynamic-exception-spec */, 6390, /* DiagArray297 */ 835, /* DiagSubGroup297 */ 275, R"()")
DIAG_ENTRY(anonymous_110 /* eager-load-cxx-named-modules */, 6413, /* DiagArray298 */ 837, 0, R"()")
DIAG_ENTRY(anonymous_38 /* effc++ */, 6442, 0, /* DiagSubGroup299 */ 277, R"()")
DIAG_ENTRY(anonymous_144 /* elaborated-enum-base */, 6449, /* DiagArray300 */ 839, 0, R"()")
DIAG_ENTRY(anonymous_145 /* elaborated-enum-class */, 6470, /* DiagArray301 */ 841, 0, R"()")
DIAG_ENTRY(anonymous_135 /* embedded-directive */, 6492, /* DiagArray302 */ 843, 0, R"()")
DIAG_ENTRY(EmptyBody /* empty-body */, 6511, /* DiagArray303 */ 845, 0, R"()")
DIAG_ENTRY(anonymous_149 /* empty-decomposition */, 6522, /* DiagArray304 */ 851, 0, R"()")
DIAG_ENTRY(EmptyInitStatement /* empty-init-stmt */, 6542, /* DiagArray305 */ 853, 0, R"()")
DIAG_ENTRY(anonymous_142 /* empty-translation-unit */, 6558, /* DiagArray306 */ 855, 0, R"()")
DIAG_ENTRY(anonymous_235 /* encode-type */, 6581, /* DiagArray307 */ 857, 0, R"()")
DIAG_ENTRY(anonymous_60 /* endif-labels */, 6593, 0, /* DiagSubGroup308 */ 279, R"()")
DIAG_ENTRY(EnumCompare /* enum-compare */, 6606, /* DiagArray309 */ 859, /* DiagSubGroup309 */ 281, R"()")
DIAG_ENTRY(EnumCompareConditional /* enum-compare-conditional */, 6619, /* DiagArray310 */ 861, /* DiagSubGroup310 */ 284, R"()")
DIAG_ENTRY(EnumCompareSwitch /* enum-compare-switch */, 6644, /* DiagArray311 */ 863, 0, R"()")
DIAG_ENTRY(EnumConversion /* enum-conversion */, 6664, /* DiagArray312 */ 865, /* DiagSubGroup312 */ 286, R"()")
DIAG_ENTRY(EnumEnumConversion /* enum-enum-conversion */, 6680, /* DiagArray313 */ 867, /* DiagSubGroup313 */ 290, R"()")
DIAG_ENTRY(EnumFloatConversion /* enum-float-conversion */, 6701, /* DiagArray314 */ 870, /* DiagSubGroup314 */ 292, R"()")
DIAG_ENTRY(EnumTooLarge /* enum-too-large */, 6723, /* DiagArray315 */ 872, 0, R"()")
DIAG_ENTRY(Exceptions /* exceptions */, 6738, /* DiagArray316 */ 875, 0, R"()")
DIAG_ENTRY(ExcessInitializers /* excess-initializers */, 6749, /* DiagArray317 */ 879, 0, R"()")
DIAG_ENTRY(anonymous_183 /* excessive-regsave */, 6769, /* DiagArray318 */ 884, 0, R"()")
DIAG_ENTRY(ExitTimeDestructors /* exit-time-destructors */, 6787, /* DiagArray319 */ 886, 0, R"()")
DIAG_ENTRY(ExpansionToDefined /* expansion-to-defined */, 6809, /* DiagArray320 */ 888, 0, R"()")
DIAG_ENTRY(anonymous_331 /* experimental-header-units */, 6830, /* DiagArray321 */ 891, 0, R"()")
DIAG_ENTRY(ExperimentalOption /* experimental-option */, 6856, /* DiagArray322 */ 893, 0, R"()")
DIAG_ENTRY(ExplicitInitializeCall /* explicit-initialize-call */, 6876, /* DiagArray323 */ 895, 0, R"()")
DIAG_ENTRY(anonymous_271 /* explicit-ownership-type */, 6901, /* DiagArray324 */ 898, 0, R"()")
DIAG_ENTRY(ExplicitSpecializationStorageClass /* explicit-specialization-storage-class */, 6925, /* DiagArray325 */ 900, 0, R"()")
DIAG_ENTRY(ExportUnnamed /* export-unnamed */, 6963, 0, 0, R"()")
DIAG_ENTRY(ExternCCompat /* extern-c-compat */, 6978, /* DiagArray327 */ 902, 0, R"()")
DIAG_ENTRY(anonymous_266 /* extern-initializer */, 6994, /* DiagArray328 */ 904, 0, R"()")
DIAG_ENTRY(Extra /* extra */, 7013, 0, /* DiagSubGroup329 */ 294, R"()")
DIAG_ENTRY(anonymous_282 /* extra-qualification */, 7019, /* DiagArray330 */ 906, 0, R"()")
DIAG_ENTRY(ExtraSemi /* extra-semi */, 7039, /* DiagArray331 */ 908, /* DiagSubGroup331 */ 310, R"()")
DIAG_ENTRY(ExtraSemiStmt /* extra-semi-stmt */, 7050, /* DiagArray332 */ 911, /* DiagSubGroup332 */ 313, R"()")
DIAG_ENTRY(ExtraTokens /* extra-tokens */, 7066, /* DiagArray333 */ 913, 0, R"()")
DIAG_ENTRY(ExtractAPIMisuse /* extractapi-misuse */, 7079, /* DiagArray334 */ 916, 0, R"()")
DIAG_ENTRY(anonymous_252 /* extraneous-template-head */, 7097, /* DiagArray335 */ 918, 0, R"()")
DIAG_ENTRY(FinalDtorNonFinalClass /* final-dtor-non-final-class */, 7122, /* DiagArray336 */ 920, 0, R"()")
DIAG_ENTRY(FinalMacro /* final-macro */, 7149, /* DiagArray337 */ 922, 0, R"()")
DIAG_ENTRY(anonymous_80 /* fixed-point-overflow */, 7161, /* DiagArray338 */ 924, 0, R"()")
DIAG_ENTRY(FlagEnum /* flag-enum */, 7182, /* DiagArray339 */ 926, 0, R"()")
DIAG_ENTRY(FlexibleArrayExtensions /* flexible-array-extensions */, 7192, /* DiagArray340 */ 928, 0, R"()")
DIAG_ENTRY(FloatConversion /* float-conversion */, 7218, /* DiagArray341 */ 931, /* DiagSubGroup341 */ 315, R"()")
DIAG_ENTRY(anonymous_274 /* float-equal */, 7235, /* DiagArray342 */ 933, 0, R"()")
DIAG_ENTRY(FloatOverflowConversion /* float-overflow-conversion */, 7247, /* DiagArray343 */ 935, 0, R"()")
DIAG_ENTRY(FloatZeroConversion /* float-zero-conversion */, 7273, /* DiagArray344 */ 938, 0, R"()")
DIAG_ENTRY(ForLoopAnalysis /* for-loop-analysis */, 7295, /* DiagArray345 */ 940, 0, R"()")
DIAG_ENTRY(Format /* format */, 7313, /* DiagArray346 */ 943, /* DiagSubGroup346 */ 318, R"()")
DIAG_ENTRY(FormatExtraArgs /* format-extra-args */, 7320, /* DiagArray347 */ 977, 0, R"()")
DIAG_ENTRY(FormatInsufficientArgs /* format-insufficient-args */, 7338, /* DiagArray348 */ 979, 0, R"()")
DIAG_ENTRY(FormatInvalidSpecifier /* format-invalid-specifier */, 7363, /* DiagArray349 */ 982, 0, R"()")
DIAG_ENTRY(FormatNonStandard /* format-non-iso */, 7388, /* DiagArray350 */ 984, 0, R"()")
DIAG_ENTRY(FormatNonLiteral /* format-nonliteral */, 7403, /* DiagArray351 */ 988, 0, R"()")
DIAG_ENTRY(FormatOverflow /* format-overflow */, 7421, /* DiagArray352 */ 990, /* DiagSubGroup352 */ 328, R"()")
DIAG_ENTRY(FormatOverflowNonKprintf /* format-overflow-non-kprintf */, 7437, /* DiagArray353 */ 992, 0, R"()")
DIAG_ENTRY(FormatPedantic /* format-pedantic */, 7465, /* DiagArray354 */ 994, 0, R"()")
DIAG_ENTRY(FormatSecurity /* format-security */, 7481, /* DiagArray355 */ 998, 0, R"()")
DIAG_ENTRY(FormatSignedness /* format-signedness */, 7497, /* DiagArray356 */ 1000, 0, R"()")
DIAG_ENTRY(FormatTruncation /* format-truncation */, 7515, /* DiagArray357 */ 1002, /* DiagSubGroup357 */ 330, R"()")
DIAG_ENTRY(FormatTruncationNonKprintf /* format-truncation-non-kprintf */, 7533, /* DiagArray358 */ 1004, 0, R"()")
DIAG_ENTRY(FormatTypeConfusion /* format-type-confusion */, 7563, /* DiagArray359 */ 1006, 0, R"()")
DIAG_ENTRY(FormatY2K /* format-y2k */, 7585, 0, 0, R"()")
DIAG_ENTRY(FormatZeroLength /* format-zero-length */, 7596, /* DiagArray361 */ 1008, 0, R"()")
DIAG_ENTRY(Format2 /* format=2 */, 7615, 0, /* DiagSubGroup362 */ 332, R"()")
DIAG_ENTRY(FortifySource /* fortify-source */, 7624, /* DiagArray363 */ 1010, /* DiagSubGroup363 */ 336, R"()")
DIAG_ENTRY(FourByteMultiChar /* four-char-constants */, 7639, /* DiagArray364 */ 1015, 0, R"()")
DIAG_ENTRY(FrameAddress /* frame-address */, 7659, /* DiagArray365 */ 1017, 0, R"()")
DIAG_ENTRY(BackendFrameLargerThan /* frame-larger-than */, 7673, /* DiagArray366 */ 1019, 0, R"(More fine grained information about the stack layout is available by adding the
`-Rpass-analysis=stack-frame-layout` command-line flag to the compiler
invocation.

The diagnostic information can be saved to a file in a machine readable format,
like YAML by adding the `-foptimization-record-file=<file>` command-line flag.

Results can be filtered by function name by passing
`-mllvm -filter-print-funcs=foo`, where `foo` is the target function's name.

   .. code-block:: console

      clang -c a.cpp -Rpass-analysis=stack-frame-layout -mllvm -filter-print-funcs=foo

   .. code-block:: console

      clang -c a.cpp -Rpass-analysis=stack-frame-layout -foptimization-record-file=<file>)")
DIAG_ENTRY(anonymous_76 /* frame-larger-than= */, 7691, 0, /* DiagSubGroup367 */ 339, R"()")
DIAG_ENTRY(FrameworkIncludePrivateFromPublic /* framework-include-private-from-public */, 7710, /* DiagArray368 */ 1022, 0, R"()")
DIAG_ENTRY(FreeNonHeapObject /* free-nonheap-object */, 7748, /* DiagArray369 */ 1024, 0, R"()")
DIAG_ENTRY(anonymous_212 /* friend-enum */, 7768, /* DiagArray370 */ 1026, 0, R"()")
DIAG_ENTRY(FunctionDefInObjCContainer /* function-def-in-objc-container */, 7780, /* DiagArray371 */ 1028, 0, R"()")
DIAG_ENTRY(FunctionEffects /* function-effects */, 7811, /* DiagArray372 */ 1030, 0, R"()")
DIAG_ENTRY(FunctionMultiVersioning /* function-multiversion */, 7828, /* DiagArray373 */ 1038, /* DiagSubGroup373 */ 341, R"()")
DIAG_ENTRY(FUseLdPath /* fuse-ld-path */, 7850, /* DiagArray374 */ 1043, 0, R"()")
DIAG_ENTRY(FutureAttrs /* future-attribute-extensions */, 7863, 0, /* DiagSubGroup375 */ 343, R"()")
DIAG_ENTRY(FutureCompat /* future-compat */, 7891, 0, 0, R"()")
DIAG_ENTRY(GccCompat /* gcc-compat */, 7905, /* DiagArray377 */ 1045, 0, R"()")
DIAG_ENTRY(GlobalConstructors /* global-constructors */, 7916, /* DiagArray378 */ 1057, 0, R"(Emit a warning for each variable declaration that generates code run at startup.)")
DIAG_ENTRY(GlobalISel /* global-isel */, 7936, /* DiagArray379 */ 1060, 0, R"()")
DIAG_ENTRY(GNU /* gnu */, 7948, 0, /* DiagSubGroup380 */ 348, R"()")
DIAG_ENTRY(GNUAlignofExpression /* gnu-alignof-expression */, 7952, /* DiagArray381 */ 1063, 0, R"()")
DIAG_ENTRY(GNUAnonymousStruct /* gnu-anonymous-struct */, 7975, /* DiagArray382 */ 1065, 0, R"()")
DIAG_ENTRY(anonymous_285 /* gnu-array-member-paren-init */, 7996, /* DiagArray383 */ 1067, 0, R"()")
DIAG_ENTRY(GNUAutoType /* gnu-auto-type */, 8024, /* DiagArray384 */ 1069, 0, R"()")
DIAG_ENTRY(GNUBinaryLiteral /* gnu-binary-literal */, 8038, 0, 0, R"()")
DIAG_ENTRY(GNUCaseRange /* gnu-case-range */, 8057, /* DiagArray386 */ 1071, 0, R"()")
DIAG_ENTRY(GNUComplexInteger /* gnu-complex-integer */, 8072, /* DiagArray387 */ 1073, 0, R"()")
DIAG_ENTRY(GNUCompoundLiteralInitializer /* gnu-compound-literal-initializer */, 8092, /* DiagArray388 */ 1075, 0, R"()")
DIAG_ENTRY(GNUConditionalOmittedOperand /* gnu-conditional-omitted-operand */, 8125, /* DiagArray389 */ 1077, 0, R"()")
DIAG_ENTRY(GNUDesignator /* gnu-designator */, 8157, /* DiagArray390 */ 1079, 0, R"()")
DIAG_ENTRY(anonymous_1 /* gnu-empty-initializer */, 8172, 0, 0, R"()")
DIAG_ENTRY(GNUEmptyStruct /* gnu-empty-struct */, 8194, /* DiagArray392 */ 1083, 0, R"()")
DIAG_ENTRY(GNUFlexibleArrayInitializer /* gnu-flexible-array-initializer */, 8211, /* DiagArray393 */ 1087, 0, R"()")
DIAG_ENTRY(GNUFlexibleArrayUnionMember /* gnu-flexible-array-union-member */, 8242, /* DiagArray394 */ 1089, 0, R"()")
DIAG_ENTRY(GNUFoldingConstant /* gnu-folding-constant */, 8274, /* DiagArray395 */ 1091, 0, R"()")
DIAG_ENTRY(GNUImaginaryConstant /* gnu-imaginary-constant */, 8295, /* DiagArray396 */ 1095, 0, R"()")
DIAG_ENTRY(GNUIncludeNext /* gnu-include-next */, 8318, /* DiagArray397 */ 1097, 0, R"()")
DIAG_ENTRY(anonymous_236 /* gnu-inline-cpp-without-extern */, 8335, /* DiagArray398 */ 1099, 0, R"()")
DIAG_ENTRY(GNULabelsAsValue /* gnu-label-as-value */, 8365, /* DiagArray399 */ 1101, 0, R"()")
DIAG_ENTRY(GNULineMarker /* gnu-line-marker */, 8384, /* DiagArray400 */ 1104, 0, R"()")
DIAG_ENTRY(GNUNullPointerArithmetic /* gnu-null-pointer-arithmetic */, 8400, /* DiagArray401 */ 1106, 0, R"()")
DIAG_ENTRY(GNUOffsetofExtensions /* gnu-offsetof-extensions */, 8428, 0, 0, R"()")
DIAG_ENTRY(GNUPointerArith /* gnu-pointer-arith */, 8452, /* DiagArray403 */ 1108, 0, R"()")
DIAG_ENTRY(GNURedeclaredEnum /* gnu-redeclared-enum */, 8470, /* DiagArray404 */ 1112, 0, R"()")
DIAG_ENTRY(GNUStatementExpression /* gnu-statement-expression */, 8490, /* DiagArray405 */ 1114, /* DiagSubGroup405 */ 380, R"()")
DIAG_ENTRY(GNUStatementExpressionFromMacroExpansion /* gnu-statement-expression-from-macro-expansion */, 8515, /* DiagArray406 */ 1116, 0, R"()")
DIAG_ENTRY(GNUStaticFloatInit /* gnu-static-float-init */, 8561, /* DiagArray407 */ 1118, 0, R"()")
DIAG_ENTRY(GNUStringLiteralOperatorTemplate /* gnu-string-literal-operator-template */, 8583, /* DiagArray408 */ 1120, 0, R"()")
DIAG_ENTRY(GNUUnionCast /* gnu-union-cast */, 8620, /* DiagArray409 */ 1122, 0, R"()")
DIAG_ENTRY(GNUVariableSizedTypeNotAtEnd /* gnu-variable-sized-type-not-at-end */, 8635, /* DiagArray410 */ 1124, 0, R"()")
DIAG_ENTRY(GNUZeroLineDirective /* gnu-zero-line-directive */, 8670, /* DiagArray411 */ 1126, 0, R"()")
DIAG_ENTRY(GNUZeroVariadicMacroArguments /* gnu-zero-variadic-macro-arguments */, 8694, /* DiagArray412 */ 1128, /* DiagSubGroup412 */ 382, R"()")
DIAG_ENTRY(anonymous_304 /* gpu-maybe-wrong-side */, 8728, /* DiagArray413 */ 1130, 0, R"()")
DIAG_ENTRY(anonymous_141 /* header-guard */, 8749, /* DiagArray414 */ 1132, 0, R"()")
DIAG_ENTRY(HeaderHygiene /* header-hygiene */, 8762, /* DiagArray415 */ 1134, 0, R"()")
DIAG_ENTRY(anonymous_272 /* higher-precision-for-complex-division */, 8777, /* DiagArray416 */ 1136, 0, R"()")
DIAG_ENTRY(HIPOpenMPOffloading /* hip-omp-target-directives */, 8815, /* DiagArray417 */ 1138, 0, R"()")
DIAG_ENTRY(HIPOnly /* hip-only */, 8841, /* DiagArray418 */ 1140, 0, R"()")
DIAG_ENTRY(HLSL202y /* hlsl-202y-extensions */, 8850, /* DiagArray419 */ 1142, 0, R"()")
DIAG_ENTRY(HLSLAvailability /* hlsl-availability */, 8871, /* DiagArray420 */ 1144, 0, R"()")
DIAG_ENTRY(HLSLDXCCompat /* hlsl-dxc-compatability */, 8889, /* DiagArray421 */ 1147, 0, R"()")
DIAG_ENTRY(HLSLExtension /* hlsl-extensions */, 8912, /* DiagArray422 */ 1149, /* DiagSubGroup422 */ 384, R"()")
DIAG_ENTRY(HLSLImplicitBinding /* hlsl-implicit-binding */, 8928, /* DiagArray423 */ 1152, 0, R"()")
DIAG_ENTRY(anonymous_299 /* idiomatic-parentheses */, 8950, /* DiagArray424 */ 1154, 0, R"()")
DIAG_ENTRY(IgnoredAttributes /* ignored-attributes */, 8972, /* DiagArray425 */ 1156, 0, R"()")
DIAG_ENTRY(anonymous_239 /* ignored-availability-without-sdk-settings */, 8991, /* DiagArray426 */ 1249, 0, R"()")
DIAG_ENTRY(IgnoredBaseClassQualifiers /* ignored-base-class-qualifiers */, 9033, /* DiagArray427 */ 1251, /* DiagSubGroup427 */ 386, R"()")
DIAG_ENTRY(IgnoredGCH /* ignored-gch */, 9063, /* DiagArray428 */ 1253, 0, R"()")
DIAG_ENTRY(IgnoredOptimizationArgument /* ignored-optimization-argument */, 9075, /* DiagArray429 */ 1256, 0, R"()")
DIAG_ENTRY(IgnoredPragmaIntrinsic /* ignored-pragma-intrinsic */, 9105, /* DiagArray430 */ 1259, 0, R"()")
DIAG_ENTRY(IgnoredPragmaOptimize /* ignored-pragma-optimize */, 9130, 0, 0, R"()")
DIAG_ENTRY(IgnoredPragmas /* ignored-pragmas */, 9154, /* DiagArray432 */ 1261, /* DiagSubGroup432 */ 388, R"()")
DIAG_ENTRY(IgnoredQualifiers /* ignored-qualifiers */, 9170, /* DiagArray433 */ 1312, /* DiagSubGroup433 */ 391, R"()")
DIAG_ENTRY(IgnoredReferenceQualifiers /* ignored-reference-qualifiers */, 9189, /* DiagArray434 */ 1318, 0, R"()")
DIAG_ENTRY(Implicit /* implicit */, 9218, 0, /* DiagSubGroup435 */ 393, R"()")
DIAG_ENTRY(ImplicitAtomic /* implicit-atomic-properties */, 9227, /* DiagArray436 */ 1320, 0, R"()")
DIAG_ENTRY(ImplicitConstIntFloatConversion /* implicit-const-int-float-conversion */, 9254, /* DiagArray437 */ 1323, 0, R"()")
DIAG_ENTRY(ImplicitConversionFloatingPointToBool /* implicit-conversion-floating-point-to-bool */, 9290, /* DiagArray438 */ 1325, 0, R"()")
DIAG_ENTRY(anonymous_317 /* implicit-exception-spec-mismatch */, 9333, /* DiagArray439 */ 1327, 0, R"()")
DIAG_ENTRY(ImplicitFallthrough /* implicit-fallthrough */, 9366, /* DiagArray440 */ 1329, /* DiagSubGroup440 */ 396, R"()")
DIAG_ENTRY(ImplicitFallthroughPerFunction /* implicit-fallthrough-per-function */, 9387, /* DiagArray441 */ 1331, 0, R"()")
DIAG_ENTRY(ImplicitFixedPointConversion /* implicit-fixed-point-conversion */, 9421, /* DiagArray442 */ 1333, 0, R"()")
DIAG_ENTRY(ImplicitFloatConversion /* implicit-float-conversion */, 9453, /* DiagArray443 */ 1335, /* DiagSubGroup443 */ 398, R"()")
DIAG_ENTRY(ImplicitFunctionDeclare /* implicit-function-declaration */, 9479, /* DiagArray444 */ 1338, 0, R"()")
DIAG_ENTRY(ImplicitInt /* implicit-int */, 9509, /* DiagArray445 */ 1344, 0, R"()")
DIAG_ENTRY(ImplicitIntConversion /* implicit-int-conversion */, 9522, /* DiagArray446 */ 1348, /* DiagSubGroup446 */ 401, R"()")
DIAG_ENTRY(ImplicitIntFloatConversion /* implicit-int-float-conversion */, 9546, /* DiagArray447 */ 1351, /* DiagSubGroup447 */ 404, R"()")
DIAG_ENTRY(anonymous_207 /* implicit-retain-self */, 9576, /* DiagArray448 */ 1353, 0, R"()")
DIAG_ENTRY(ImplicitlyUnsignedLiteral /* implicitly-unsigned-literal */, 9597, /* DiagArray449 */ 1355, 0, R"()")
DIAG_ENTRY(anonymous_31 /* import */, 9625, 0, 0, R"()")
DIAG_ENTRY(anonymous_187 /* import-implementation-partition-unit-in-interface-unit */, 9632, /* DiagArray451 */ 1357, 0, R"()")
DIAG_ENTRY(anonymous_134 /* import-preprocessor-directive-pedantic */, 9687, /* DiagArray452 */ 1359, 0, R"()")
DIAG_ENTRY(anonymous_314 /* inaccessible-base */, 9726, /* DiagArray453 */ 1361, 0, R"()")
DIAG_ENTRY(anonymous_140 /* include-angled-in-module-purview */, 9744, /* DiagArray454 */ 1363, 0, R"()")
DIAG_ENTRY(anonymous_125 /* include-next-absolute-path */, 9777, /* DiagArray455 */ 1365, 0, R"()")
DIAG_ENTRY(anonymous_124 /* include-next-outside-header */, 9804, /* DiagArray456 */ 1367, 0, R"()")
DIAG_ENTRY(IncompatibleExceptionSpec /* incompatible-exception-spec */, 9832, /* DiagArray457 */ 1369, 0, R"()")
DIAG_ENTRY(IncompatibleFunctionPointerTypes /* incompatible-function-pointer-types */, 9860, /* DiagArray458 */ 1372, 0, R"()")
DIAG_ENTRY(anonymous_301 /* incompatible-function-pointer-types-strict */, 9896, /* DiagArray459 */ 1374, 0, R"()")
DIAG_ENTRY(anonymous_190 /* incompatible-library-redeclaration */, 9939, /* DiagArray460 */ 1376, 0, R"()")
DIAG_ENTRY(IncompatibleMSPragmaSection /* incompatible-ms-pragma-section */, 9974, /* DiagArray461 */ 1378, 0, R"()")
DIAG_ENTRY(IncompatibleMSStruct /* incompatible-ms-struct */, 10005, /* DiagArray462 */ 1380, 0, R"()")
DIAG_ENTRY(IncompatiblePointerTypes /* incompatible-pointer-types */, 10028, /* DiagArray463 */ 1383, /* DiagSubGroup463 */ 406, R"()")
DIAG_ENTRY(IncompatiblePointerTypesDiscardsQualifiers /* incompatible-pointer-types-discards-qualifiers */, 10055, /* DiagArray464 */ 1385, 0, R"()")
DIAG_ENTRY(anonymous_197 /* incompatible-property-type */, 10102, /* DiagArray465 */ 1389, 0, R"()")
DIAG_ENTRY(anonymous_95 /* incompatible-sysroot */, 10129, /* DiagArray466 */ 1391, 0, R"()")
DIAG_ENTRY(IncompleteFrameworkModuleDeclaration /* incomplete-framework-module-declaration */, 10150, /* DiagArray467 */ 1393, 0, R"()")
DIAG_ENTRY(anonymous_199 /* incomplete-implementation */, 10190, /* DiagArray468 */ 1395, 0, R"()")
DIAG_ENTRY(IncompleteModule /* incomplete-module */, 10216, 0, /* DiagSubGroup469 */ 409, R"()")
DIAG_ENTRY(anonymous_189 /* incomplete-setjmp-declaration */, 10234, /* DiagArray470 */ 1397, 0, R"()")
DIAG_ENTRY(IncompleteUmbrella /* incomplete-umbrella */, 10264, /* DiagArray471 */ 1399, 0, R"()")
DIAG_ENTRY(MicrosoftInconsistentDllImport /* inconsistent-dllimport */, 10284, /* DiagArray472 */ 1403, 0, R"()")
DIAG_ENTRY(CXX11WarnInconsistentOverrideDestructor /* inconsistent-missing-destructor-override */, 10307, /* DiagArray473 */ 1406, 0, R"()")
DIAG_ENTRY(CXX11WarnInconsistentOverrideMethod /* inconsistent-missing-override */, 10348, /* DiagArray474 */ 1408, 0, R"()")
DIAG_ENTRY(IncrementBool /* increment-bool */, 10378, /* DiagArray475 */ 1410, /* DiagSubGroup475 */ 412, R"()")
DIAG_ENTRY(InfiniteRecursion /* infinite-recursion */, 10393, /* DiagArray476 */ 1412, 0, R"()")
DIAG_ENTRY(anonymous_223 /* init-priority-reserved */, 10412, /* DiagArray477 */ 1414, 0, R"()")
DIAG_ENTRY(anonymous_32 /* init-self */, 10435, 0, 0, R"()")
DIAG_ENTRY(InitializerOverrides /* initializer-overrides */, 10445, /* DiagArray479 */ 1416, 0, R"()")
DIAG_ENTRY(anonymous_216 /* injected-class-name */, 10467, /* DiagArray480 */ 1420, 0, R"()")
DIAG_ENTRY(anonymous_33 /* inline */, 10487, 0, 0, R"()")
DIAG_ENTRY(BackendInlineAsm /* inline-asm */, 10494, /* DiagArray482 */ 1422, 0, R"()")
DIAG_ENTRY(InlineNamespaceReopenedNoninline /* inline-namespace-reopened-noninline */, 10505, /* DiagArray483 */ 1424, 0, R"()")
DIAG_ENTRY(anonymous_315 /* inline-new-delete */, 10541, /* DiagArray484 */ 1426, 0, R"()")
DIAG_ENTRY(InstallAPIViolation /* installapi-violation */, 10559, /* DiagArray485 */ 1428, 0, R"()")
DIAG_ENTRY(anonymous_255 /* instantiation-after-specialization */, 10580, /* DiagArray486 */ 1442, 0, R"()")
DIAG_ENTRY(IntConversion /* int-conversion */, 10615, /* DiagArray487 */ 1444, 0, R"()")
DIAG_ENTRY(anonymous_65 /* int-conversions */, 10630, 0, /* DiagSubGroup488 */ 414, R"()")
DIAG_ENTRY(IntInBoolContext /* int-in-bool-context */, 10646, /* DiagArray489 */ 1447, 0, R"()")
DIAG_ENTRY(IntToPointerCast /* int-to-pointer-cast */, 10666, /* DiagArray490 */ 1450, /* DiagSubGroup490 */ 416, R"()")
DIAG_ENTRY(IntToVoidPointerCast /* int-to-void-pointer-cast */, 10686, /* DiagArray491 */ 1452, 0, R"()")
DIAG_ENTRY(anonymous_79 /* integer-overflow */, 10711, /* DiagArray492 */ 1454, 0, R"()")
DIAG_ENTRY(InvalidCommandLineArgument /* invalid-command-line-argument */, 10728, /* DiagArray493 */ 1456, /* DiagSubGroup493 */ 418, R"()")
DIAG_ENTRY(anonymous_219 /* invalid-constexpr */, 10758, /* DiagArray494 */ 1468, 0, R"()")
DIAG_ENTRY(anonymous_87 /* invalid-feature-combination */, 10776, /* DiagArray495 */ 1470, 0, R"()")
DIAG_ENTRY(anonymous_311 /* invalid-gnu-asm-cast */, 10804, /* DiagArray496 */ 1472, 0, R"()")
DIAG_ENTRY(ObjCInvalidIBOutletProperty /* invalid-iboutlet */, 10825, /* DiagArray497 */ 1474, 0, R"()")
DIAG_ENTRY(anonymous_233 /* invalid-initializer-from-system-header */, 10842, /* DiagArray498 */ 1477, 0, R"()")
DIAG_ENTRY(InvalidIOSDeploymentTarget /* invalid-ios-deployment-target */, 10881, /* DiagArray499 */ 1479, 0, R"()")
DIAG_ENTRY(anonymous_245 /* invalid-no-builtin-names */, 10911, /* DiagArray500 */ 1481, 0, R"()")
DIAG_ENTRY(InvalidNoreturn /* invalid-noreturn */, 10936, /* DiagArray501 */ 1483, 0, R"()")
DIAG_ENTRY(InvalidOffsetof /* invalid-offsetof */, 10953, /* DiagArray502 */ 1487, 0, R"()")
DIAG_ENTRY(InvalidOrNonExistentDirectory /* invalid-or-nonexistent-directory */, 10970, /* DiagArray503 */ 1490, 0, R"()")
DIAG_ENTRY(anonymous_253 /* invalid-partial-specialization */, 11003, /* DiagArray504 */ 1492, 0, R"()")
DIAG_ENTRY(anonymous_34 /* invalid-pch */, 11034, 0, 0, R"()")
DIAG_ENTRY(InvalidPPToken /* invalid-pp-token */, 11046, /* DiagArray506 */ 1494, 0, R"()")
DIAG_ENTRY(InvalidSourceEncoding /* invalid-source-encoding */, 11063, /* DiagArray507 */ 1497, 0, R"()")
DIAG_ENTRY(anonymous_251 /* invalid-specialization */, 11087, /* DiagArray508 */ 1500, 0, R"()")
DIAG_ENTRY(anonymous_210 /* invalid-static-assert-message */, 11110, /* DiagArray509 */ 1502, 0, R"()")
DIAG_ENTRY(anonymous_137 /* invalid-token-paste */, 11140, /* DiagArray510 */ 1504, 0, R"()")
DIAG_ENTRY(anonymous_123 /* invalid-unevaluated-string */, 11160, /* DiagArray511 */ 1506, 0, R"()")
DIAG_ENTRY(anonymous_115 /* invalid-utf8 */, 11187, /* DiagArray512 */ 1508, 0, R"()")
DIAG_ENTRY(anonymous_296 /* jump-seh-finally */, 11200, /* DiagArray513 */ 1510, 0, R"()")
DIAG_ENTRY(KeywordCompat /* keyword-compat */, 11217, /* DiagArray514 */ 1512, 0, R"()")
DIAG_ENTRY(KeywordAsMacro /* keyword-macro */, 11232, /* DiagArray515 */ 1514, 0, R"()")
DIAG_ENTRY(KNRPromotedParameter /* knr-promoted-parameter */, 11246, /* DiagArray516 */ 1516, 0, R"()")
DIAG_ENTRY(anonymous_114 /* language-extension-token */, 11269, /* DiagArray517 */ 1518, 0, R"()")
DIAG_ENTRY(LargeByValueCopy /* large-by-value-copy */, 11294, /* DiagArray518 */ 1520, 0, R"()")
DIAG_ENTRY(LegacyConstantRegisterBinding /* legacy-constant-register-binding */, 11314, /* DiagArray519 */ 1523, 0, R"()")
DIAG_ENTRY(LibLTO /* liblto */, 11347, 0, 0, R"()")
DIAG_ENTRY(LinkerWarnings /* linker-warnings */, 11354, /* DiagArray521 */ 1528, 0, R"()")
DIAG_ENTRY(LiteralConversion /* literal-conversion */, 11370, /* DiagArray522 */ 1530, 0, R"()")
DIAG_ENTRY(LiteralRange /* literal-range */, 11389, /* DiagArray523 */ 1533, 0, R"()")
DIAG_ENTRY(LocalTypeTemplateArgs /* local-type-template-args */, 11403, /* DiagArray524 */ 1537, /* DiagSubGroup524 */ 420, R"()")
DIAG_ENTRY(LogicalNotParentheses /* logical-not-parentheses */, 11428, /* DiagArray525 */ 1539, 0, R"()")
DIAG_ENTRY(LogicalOpParentheses /* logical-op-parentheses */, 11452, /* DiagArray526 */ 1541, 0, R"()")
DIAG_ENTRY(LongLong /* long-long */, 11475, /* DiagArray527 */ 1543, /* DiagSubGroup527 */ 422, R"()")
DIAG_ENTRY(LoopAnalysis /* loop-analysis */, 11485, 0, /* DiagSubGroup528 */ 424, R"()")
DIAG_ENTRY(MacroRedefined /* macro-redefined */, 11499, /* DiagArray529 */ 1545, 0, R"()")
DIAG_ENTRY(Main /* main */, 11515, /* DiagArray530 */ 1547, 0, R"()")
DIAG_ENTRY(MainReturnType /* main-return-type */, 11520, /* DiagArray531 */ 1556, 0, R"()")
DIAG_ENTRY(MalformedWarningCheck /* malformed-warning-check */, 11537, /* DiagArray532 */ 1559, 0, R"()")
DIAG_ENTRY(anonymous_268 /* many-braces-around-scalar-init */, 11561, /* DiagArray533 */ 1561, 0, R"()")
DIAG_ENTRY(MathErrnoEnabledWithVecLib /* math-errno-enabled-with-veclib */, 11592, /* DiagArray534 */ 1563, 0, R"()")
DIAG_ENTRY(anonymous_119 /* mathematical-notation-identifier-extension */, 11623, /* DiagArray535 */ 1565, 0, R"()")
DIAG_ENTRY(MaxTokens /* max-tokens */, 11666, /* DiagArray536 */ 1567, 0, R"(The warning is issued if the number of pre-processor tokens exceeds
the token limit, which can be set in three ways:

1. As a limit at a specific point in a file, using the ``clang max_tokens_here``
   pragma:

   .. code-block:: c++

      #pragma clang max_tokens_here 1234

2. As a per-translation unit limit, using the ``-fmax-tokens=`` command-line
   flag:

   .. code-block:: console

      clang -c a.cpp -fmax-tokens=1234

3. As a per-translation unit limit using the ``clang max_tokens_total`` pragma,
   which works like and overrides the ``-fmax-tokens=`` flag:

   .. code-block:: c++

      #pragma clang max_tokens_total 1234

These limits can be helpful in limiting code growth through included files.

Setting a token limit of zero means no limit.

Note that the warning is disabled by default, so -Wmax-tokens must be used
in addition with the pragmas or -fmax-tokens flag to get any warnings.)")
DIAG_ENTRY(MaxUnsignedZero /* max-unsigned-zero */, 11677, /* DiagArray537 */ 1570, 0, R"()")
DIAG_ENTRY(MemsetTransposedArgs /* memset-transposed-args */, 11695, /* DiagArray538 */ 1572, 0, R"()")
DIAG_ENTRY(anonymous_192 /* memsize-comparison */, 11718, /* DiagArray539 */ 1574, 0, R"()")
DIAG_ENTRY(MethodSignatures /* method-signatures */, 11737, /* DiagArray540 */ 1576, 0, R"()")
DIAG_ENTRY(Microsoft /* microsoft */, 11755, 0, /* DiagSubGroup541 */ 427, R"()")
DIAG_ENTRY(MicrosoftAbstract /* microsoft-abstract */, 11765, /* DiagArray542 */ 1579, 0, R"()")
DIAG_ENTRY(MicrosoftAnonTag /* microsoft-anon-tag */, 11784, /* DiagArray543 */ 1581, 0, R"()")
DIAG_ENTRY(MicrosoftCast /* microsoft-cast */, 11803, /* DiagArray544 */ 1584, 0, R"()")
DIAG_ENTRY(MicrosoftCharize /* microsoft-charize */, 11818, /* DiagArray545 */ 1587, 0, R"()")
DIAG_ENTRY(MicrosoftCommentPaste /* microsoft-comment-paste */, 11836, /* DiagArray546 */ 1589, 0, R"()")
DIAG_ENTRY(MicrosoftConstInit /* microsoft-const-init */, 11860, /* DiagArray547 */ 1591, 0, R"()")
DIAG_ENTRY(MicrosoftCppMacro /* microsoft-cpp-macro */, 11881, /* DiagArray548 */ 1593, 0, R"()")
DIAG_ENTRY(MicrosoftDefaultArgRedefinition /* microsoft-default-arg-redefinition */, 11901, /* DiagArray549 */ 1595, 0, R"()")
DIAG_ENTRY(MicrosoftDrectveSection /* microsoft-drectve-section */, 11936, /* DiagArray550 */ 1597, 0, R"()")
DIAG_ENTRY(MicrosoftEndOfFile /* microsoft-end-of-file */, 11962, /* DiagArray551 */ 1599, 0, R"()")
DIAG_ENTRY(MicrosoftEnumForwardReference /* microsoft-enum-forward-reference */, 11984, /* DiagArray552 */ 1601, 0, R"()")
DIAG_ENTRY(MicrosoftEnumValue /* microsoft-enum-value */, 12017, /* DiagArray553 */ 1603, 0, R"()")
DIAG_ENTRY(MicrosoftExceptionSpec /* microsoft-exception-spec */, 12038, /* DiagArray554 */ 1605, 0, R"()")
DIAG_ENTRY(anonymous_146 /* microsoft-exists */, 12063, /* DiagArray555 */ 1611, 0, R"()")
DIAG_ENTRY(MicrosoftExplicitConstructorCall /* microsoft-explicit-constructor-call */, 12080, /* DiagArray556 */ 1613, 0, R"()")
DIAG_ENTRY(MicrosoftExtraQualification /* microsoft-extra-qualification */, 12116, /* DiagArray557 */ 1615, 0, R"()")
DIAG_ENTRY(MicrosoftFixedEnum /* microsoft-fixed-enum */, 12146, /* DiagArray558 */ 1617, 0, R"()")
DIAG_ENTRY(MicrosoftFlexibleArray /* microsoft-flexible-array */, 12167, /* DiagArray559 */ 1619, 0, R"()")
DIAG_ENTRY(MicrosoftGoto /* microsoft-goto */, 12192, /* DiagArray560 */ 1622, 0, R"()")
DIAG_ENTRY(MicrosoftInaccessibleBase /* microsoft-inaccessible-base */, 12207, /* DiagArray561 */ 1624, 0, R"()")
DIAG_ENTRY(MicrosoftInclude /* microsoft-include */, 12235, /* DiagArray562 */ 1626, 0, R"()")
DIAG_ENTRY(MicrosoftInitFromPredefined /* microsoft-init-from-predefined */, 12253, /* DiagArray563 */ 1628, 0, R"()")
DIAG_ENTRY(MicrosoftInlineOnNonFunction /* microsoft-inline-on-non-function */, 12284, /* DiagArray564 */ 1630, 0, R"()")
DIAG_ENTRY(MicrosoftMutableReference /* microsoft-mutable-reference */, 12317, /* DiagArray565 */ 1632, 0, R"()")
DIAG_ENTRY(MicrosoftPureDefinition /* microsoft-pure-definition */, 12345, /* DiagArray566 */ 1634, 0, R"()")
DIAG_ENTRY(MicrosoftRedeclareStatic /* microsoft-redeclare-static */, 12371, /* DiagArray567 */ 1636, 0, R"()")
DIAG_ENTRY(MicrosoftSealed /* microsoft-sealed */, 12398, /* DiagArray568 */ 1638, 0, R"()")
DIAG_ENTRY(MicrosoftStaticAssert /* microsoft-static-assert */, 12415, /* DiagArray569 */ 1640, 0, R"()")
DIAG_ENTRY(MicrosoftStringLiteralFromPredefined /* microsoft-string-literal-from-predefined */, 12439, /* DiagArray570 */ 1642, 0, R"()")
DIAG_ENTRY(MicrosoftTemplate /* microsoft-template */, 12480, /* DiagArray571 */ 1644, /* DiagSubGroup571 */ 461, R"()")
DIAG_ENTRY(MicrosoftTemplateShadow /* microsoft-template-shadow */, 12499, /* DiagArray572 */ 1655, 0, R"()")
DIAG_ENTRY(MicrosoftUnionMemberReference /* microsoft-union-member-reference */, 12525, /* DiagArray573 */ 1657, 0, R"()")
DIAG_ENTRY(MicrosoftUnqualifiedFriend /* microsoft-unqualified-friend */, 12558, /* DiagArray574 */ 1659, 0, R"()")
DIAG_ENTRY(MicrosoftUsingDecl /* microsoft-using-decl */, 12587, /* DiagArray575 */ 1661, 0, R"()")
DIAG_ENTRY(MicrosoftVoidPseudoDtor /* microsoft-void-pseudo-dtor */, 12608, /* DiagArray576 */ 1663, 0, R"()")
DIAG_ENTRY(MisExpect /* misexpect */, 12635, /* DiagArray577 */ 1665, 0, R"()")
DIAG_ENTRY(MisleadingIndentation /* misleading-indentation */, 12645, /* DiagArray578 */ 1667, 0, R"()")
DIAG_ENTRY(anonymous_292 /* mismatched-new-delete */, 12668, /* DiagArray579 */ 1669, 0, R"()")
DIAG_ENTRY(MismatchedParameterTypes /* mismatched-parameter-types */, 12690, /* DiagArray580 */ 1671, 0, R"()")
DIAG_ENTRY(MismatchedReturnTypes /* mismatched-return-types */, 12717, /* DiagArray581 */ 1673, 0, R"()")
DIAG_ENTRY(MismatchedTags /* mismatched-tags */, 12741, /* DiagArray582 */ 1675, 0, R"()")
DIAG_ENTRY(MissingBraces /* missing-braces */, 12757, /* DiagArray583 */ 1678, 0, R"()")
DIAG_ENTRY(anonymous_310 /* missing-constinit */, 12772, /* DiagArray584 */ 1680, 0, R"()")
DIAG_ENTRY(MissingDeclarations /* missing-declarations */, 12790, /* DiagArray585 */ 1682, 0, R"()")
DIAG_ENTRY(MissingDesignatedFieldInitializers /* missing-designated-field-initializers */, 12811, /* DiagArray586 */ 1687, 0, R"(Warn about designated initializers with some fields missing (only in C++).)")
DIAG_ENTRY(anonymous_215 /* missing-exception-spec */, 12849, /* DiagArray587 */ 1689, 0, R"()")
DIAG_ENTRY(MissingFieldInitializers /* missing-field-initializers */, 12872, /* DiagArray588 */ 1691, /* DiagSubGroup588 */ 463, R"()")
DIAG_ENTRY(anonymous_35 /* missing-format-attribute */, 12899, 0, 0, R"()")
DIAG_ENTRY(MissingIncludeDirs /* missing-include-dirs */, 12924, /* DiagArray590 */ 1693, 0, R"()")
DIAG_ENTRY(MissingMethodReturnType /* missing-method-return-type */, 12945, /* DiagArray591 */ 1695, 0, R"()")
DIAG_ENTRY(anonymous_104 /* missing-multilib */, 12972, /* DiagArray592 */ 1697, 0, R"()")
DIAG_ENTRY(MissingNoEscape /* missing-noescape */, 12989, /* DiagArray593 */ 1699, 0, R"()")
DIAG_ENTRY(MissingNoreturn /* missing-noreturn */, 13006, /* DiagArray594 */ 1701, 0, R"()")
DIAG_ENTRY(anonymous_237 /* missing-prototype-for-cc */, 13023, /* DiagArray595 */ 1704, 0, R"()")
DIAG_ENTRY(anonymous_257 /* missing-prototypes */, 13048, /* DiagArray596 */ 1706, 0, R"()")
DIAG_ENTRY(anonymous_147 /* missing-selector-name */, 13067, /* DiagArray597 */ 1708, 0, R"()")
DIAG_ENTRY(anonymous_94 /* missing-sysroot */, 13089, /* DiagArray598 */ 1710, 0, R"()")
DIAG_ENTRY(anonymous_154 /* missing-template-arg-list-after-template-kw */, 13105, /* DiagArray599 */ 1712, 0, R"()")
DIAG_ENTRY(anonymous_258 /* missing-variable-declarations */, 13149, /* DiagArray600 */ 1714, 0, R"()")
DIAG_ENTRY(MisspelledAssumption /* misspelled-assumption */, 13179, /* DiagArray601 */ 1716, 0, R"()")
DIAG_ENTRY(HLSLMixPackOffset /* mix-packoffset */, 13201, /* DiagArray602 */ 1718, 0, R"()")
DIAG_ENTRY(ModuleBuild /* module-build */, 13216, /* DiagArray603 */ 1720, 0, R"()")
DIAG_ENTRY(ModuleConflict /* module-conflict */, 13229, /* DiagArray604 */ 1725, 0, R"()")
DIAG_ENTRY(anonymous_109 /* module-file-config-mismatch */, 13245, /* DiagArray605 */ 1728, 0, R"()")
DIAG_ENTRY(ModuleFileExtension /* module-file-extension */, 13273, /* DiagArray606 */ 1730, 0, R"()")
DIAG_ENTRY(ModuleImport /* module-import */, 13295, /* DiagArray607 */ 1732, 0, R"()")
DIAG_ENTRY(anonymous_329 /* module-import-in-extern-c */, 13309, /* DiagArray608 */ 1734, 0, R"()")
DIAG_ENTRY(ModuleIncludeDirectiveTranslation /* module-include-translation */, 13335, /* DiagArray609 */ 1736, 0, R"()")
DIAG_ENTRY(ModuleLock /* module-lock */, 13362, /* DiagArray610 */ 1738, 0, R"()")
DIAG_ENTRY(anonymous_332 /* modules-ambiguous-internal-linkage */, 13374, /* DiagArray611 */ 1740, 0, R"()")
DIAG_ENTRY(anonymous_330 /* modules-import-nested-redundant */, 13409, /* DiagArray612 */ 1742, 0, R"()")
DIAG_ENTRY(Most /* most */, 13441, 0, /* DiagSubGroup613 */ 465, R"()")
DIAG_ENTRY(Move /* move */, 13446, 0, /* DiagSubGroup614 */ 503, R"()")
DIAG_ENTRY(anonymous_75 /* msvc-include */, 13451, 0, /* DiagSubGroup615 */ 508, R"()")
DIAG_ENTRY(anonymous_100 /* msvc-not-found */, 13464, /* DiagArray616 */ 1744, 0, R"()")
DIAG_ENTRY(MultiGPU /* multi-gpu */, 13479, /* DiagArray617 */ 1746, 0, R"()")
DIAG_ENTRY(MultiChar /* multichar */, 13489, /* DiagArray618 */ 1748, 0, R"()")
DIAG_ENTRY(anonymous_316 /* multiple-move-vbase */, 13499, /* DiagArray619 */ 1750, 0, R"()")
DIAG_ENTRY(anonymous_83 /* nan-infinity-disabled */, 13519, /* DiagArray620 */ 1752, 0, R"(This warning is enabled when source code using the macros ``INFINITY`` or ``NAN``
is compiled with floating-point options preventing these two values. This can
lead to undefined behavior. Check the order of command line arguments that modify
this behavior, such as ``-ffast-math``, ``-fhonor-infinities``, and
``-fhonor-nans`` (etc), as well as ``#pragma`` directives if this diagnostic is
generated unexpectedly.)")
DIAG_ENTRY(anonymous_25 /* narrowing */, 13541, 0, /* DiagSubGroup621 */ 510, R"()")
DIAG_ENTRY(anonymous_313 /* nested-anon-types */, 13551, /* DiagArray622 */ 1754, 0, R"()")
DIAG_ENTRY(anonymous_36 /* nested-externs */, 13569, 0, 0, R"()")
DIAG_ENTRY(OperatorNewReturnsNull /* new-returns-null */, 13584, /* DiagArray624 */ 1756, 0, R"()")
DIAG_ENTRY(NewlineEOF /* newline-eof */, 13601, /* DiagArray625 */ 1758, 0, R"()")
DIAG_ENTRY(NoDeref /* noderef */, 13613, /* DiagArray626 */ 1760, 0, R"()")
DIAG_ENTRY(NoexceptType /* noexcept-type */, 13621, 0, /* DiagSubGroup627 */ 512, R"()")
DIAG_ENTRY(anonymous_195 /* non-c-typedef-for-linkage */, 13635, /* DiagArray628 */ 1764, 0, R"()")
DIAG_ENTRY(NonGCC /* non-gcc */, 13661, 0, /* DiagSubGroup629 */ 514, R"()")
DIAG_ENTRY(NonLiteralNullConversion /* non-literal-null-conversion */, 13669, /* DiagArray630 */ 1766, 0, R"()")
DIAG_ENTRY(NonModularIncludeInFrameworkModule /* non-modular-include-in-framework-module */, 13697, /* DiagArray631 */ 1768, 0, R"()")
DIAG_ENTRY(NonModularIncludeInModule /* non-modular-include-in-module */, 13737, /* DiagArray632 */ 1770, /* DiagSubGroup632 */ 518, R"()")
DIAG_ENTRY(NonPODVarargs /* non-pod-varargs */, 13767, /* DiagArray633 */ 1772, 0, R"()")
DIAG_ENTRY(anonymous_228 /* non-power-of-two-alignment */, 13783, /* DiagArray634 */ 1777, 0, R"()")
DIAG_ENTRY(NonVirtualDtor /* non-virtual-dtor */, 13810, /* DiagArray635 */ 1779, 0, R"()")
DIAG_ENTRY(NonNull /* nonnull */, 13827, /* DiagArray636 */ 1781, 0, R"()")
DIAG_ENTRY(anonymous_37 /* nonportable-cfstrings */, 13835, 0, 0, R"()")
DIAG_ENTRY(anonymous_126 /* nonportable-include-path */, 13857, /* DiagArray638 */ 1784, 0, R"()")
DIAG_ENTRY(anonymous_90 /* nonportable-private-apinotes-path */, 13882, /* DiagArray639 */ 1786, 0, R"()")
DIAG_ENTRY(anonymous_91 /* nonportable-private-system-apinotes-path */, 13916, /* DiagArray640 */ 1788, 0, R"()")
DIAG_ENTRY(anonymous_334 /* nonportable-sycl */, 13957, /* DiagArray641 */ 1790, 0, R"()")
DIAG_ENTRY(anonymous_127 /* nonportable-system-include-path */, 13974, /* DiagArray642 */ 1792, 0, R"()")
DIAG_ENTRY(anonymous_324 /* nonportable-vector-initialization */, 14006, /* DiagArray643 */ 1794, 0, R"()")
DIAG_ENTRY(NonTrivialMemaccess /* nontrivial-memaccess */, 14040, /* DiagArray644 */ 1796, /* DiagSubGroup644 */ 520, R"()")
DIAG_ENTRY(NonTrivialMemcall /* nontrivial-memcall */, 14061, /* DiagArray645 */ 1798, 0, R"()")
DIAG_ENTRY(NSConsumedMismatch /* nsconsumed-mismatch */, 14080, /* DiagArray646 */ 1800, 0, R"()")
DIAG_ENTRY(NSReturnsMismatch /* nsreturns-mismatch */, 14100, /* DiagArray647 */ 1802, 0, R"()")
DIAG_ENTRY(NullArithmetic /* null-arithmetic */, 14119, /* DiagArray648 */ 1804, 0, R"()")
DIAG_ENTRY(NullCharacter /* null-character */, 14135, /* DiagArray649 */ 1807, 0, R"()")
DIAG_ENTRY(NullConversion /* null-conversion */, 14150, /* DiagArray650 */ 1810, 0, R"()")
DIAG_ENTRY(NullDereference /* null-dereference */, 14166, /* DiagArray651 */ 1812, 0, R"()")
DIAG_ENTRY(NullPointerArithmetic /* null-pointer-arithmetic */, 14183, /* DiagArray652 */ 1815, /* DiagSubGroup652 */ 522, R"()")
DIAG_ENTRY(NullPointerSubtraction /* null-pointer-subtraction */, 14207, /* DiagArray653 */ 1817, 0, R"()")
DIAG_ENTRY(Nullability /* nullability */, 14232, /* DiagArray654 */ 1819, 0, R"()")
DIAG_ENTRY(NullabilityCompleteness /* nullability-completeness */, 14244, /* DiagArray655 */ 1825, /* DiagSubGroup655 */ 524, R"()")
DIAG_ENTRY(NullabilityCompletenessOnArrays /* nullability-completeness-on-arrays */, 14269, /* DiagArray656 */ 1827, 0, R"()")
DIAG_ENTRY(NullabilityDeclSpec /* nullability-declspec */, 14304, /* DiagArray657 */ 1829, 0, R"()")
DIAG_ENTRY(anonymous_143 /* nullability-extension */, 14325, /* DiagArray658 */ 1831, 0, R"()")
DIAG_ENTRY(NullabilityInferredOnNestedType /* nullability-inferred-on-nested-type */, 14347, /* DiagArray659 */ 1833, 0, R"()")
DIAG_ENTRY(NullableToNonNullConversion /* nullable-to-nonnull-conversion */, 14383, /* DiagArray660 */ 1835, 0, R"()")
DIAG_ENTRY(anonymous_305 /* nvcc-compat */, 14414, /* DiagArray661 */ 1837, 0, R"()")
DIAG_ENTRY(anonymous_204 /* objc-autosynthesis-property-ivar-name-match */, 14426, /* DiagArray662 */ 1839, 0, R"()")
DIAG_ENTRY(ObjCBoolConstantConversion /* objc-bool-constant-conversion */, 14470, /* DiagArray663 */ 1841, 0, R"()")
DIAG_ENTRY(ObjCBoxing /* objc-boxing */, 14500, /* DiagArray664 */ 1843, 0, R"()")
DIAG_ENTRY(anonymous_289 /* objc-circular-container */, 14512, /* DiagArray665 */ 1845, 0, R"()")
DIAG_ENTRY(ObjCCocoaAPI /* objc-cocoa-api */, 14536, 0, /* DiagSubGroup666 */ 526, R"()")
DIAG_ENTRY(ObjCDesignatedInit /* objc-designated-initializers */, 14551, /* DiagArray667 */ 1847, 0, R"()")
DIAG_ENTRY(anonymous_225 /* objc-dictionary-duplicate-keys */, 14580, /* DiagArray668 */ 1854, 0, R"()")
DIAG_ENTRY(anonymous_86 /* objc-duplicate-category-definition */, 14611, /* DiagArray669 */ 1856, 0, R"()")
DIAG_ENTRY(ObjCFlexibleArray /* objc-flexible-array */, 14646, /* DiagArray670 */ 1858, 0, R"()")
DIAG_ENTRY(anonymous_264 /* objc-forward-class-redefinition */, 14666, /* DiagArray671 */ 1861, 0, R"()")
DIAG_ENTRY(anonymous_265 /* objc-interface-ivars */, 14698, /* DiagArray672 */ 1863, 0, R"()")
DIAG_ENTRY(ObjCLiteralComparison /* objc-literal-compare */, 14719, /* DiagArray673 */ 1865, /* DiagSubGroup673 */ 528, R"()")
DIAG_ENTRY(ObjCLiteralConversion /* objc-literal-conversion */, 14740, /* DiagArray674 */ 1867, 0, R"()")
DIAG_ENTRY(anonymous_133 /* objc-macro-redefinition */, 14764, /* DiagArray675 */ 1870, 0, R"()")
DIAG_ENTRY(anonymous_209 /* objc-messaging-id */, 14788, /* DiagArray676 */ 1872, 0, R"()")
DIAG_ENTRY(MethodAccess /* objc-method-access */, 14806, /* DiagArray677 */ 1874, 0, R"()")
DIAG_ENTRY(anonymous_205 /* objc-missing-property-synthesis */, 14825, /* DiagArray678 */ 1881, 0, R"()")
DIAG_ENTRY(ObjCMissingSuperCalls /* objc-missing-super-calls */, 14857, /* DiagArray679 */ 1883, 0, R"()")
DIAG_ENTRY(ObjCMultipleMethodNames /* objc-multiple-method-names */, 14882, /* DiagArray680 */ 1885, 0, R"()")
DIAG_ENTRY(ObjCRetainBlockProperty /* objc-noncopy-retain-block-property */, 14909, /* DiagArray681 */ 1887, 0, R"()")
DIAG_ENTRY(ObjCNonUnifiedException /* objc-nonunified-exceptions */, 14944, /* DiagArray682 */ 1889, 0, R"()")
DIAG_ENTRY(ObjCPropertyAssignOnObjectType /* objc-property-assign-on-object-type */, 14971, /* DiagArray683 */ 1891, 0, R"()")
DIAG_ENTRY(ObjCPropertyImpl /* objc-property-implementation */, 15007, /* DiagArray684 */ 1893, 0, R"()")
DIAG_ENTRY(anonymous_200 /* objc-property-implicit-mismatch */, 15036, /* DiagArray685 */ 1898, 0, R"()")
DIAG_ENTRY(anonymous_202 /* objc-property-matches-cocoa-ownership-rule */, 15068, /* DiagArray686 */ 1900, 0, R"()")
DIAG_ENTRY(ObjCPropertyNoAttribute /* objc-property-no-attribute */, 15111, /* DiagArray687 */ 1902, 0, R"()")
DIAG_ENTRY(ObjCNoPropertyAutoSynthesis /* objc-property-synthesis */, 15138, /* DiagArray688 */ 1905, 0, R"()")
DIAG_ENTRY(ObjCProtocolMethodImpl /* objc-protocol-method-implementation */, 15162, /* DiagArray689 */ 1909, 0, R"()")
DIAG_ENTRY(anonymous_203 /* objc-protocol-property-synthesis */, 15198, /* DiagArray690 */ 1911, 0, R"()")
DIAG_ENTRY(ObjCProtocolQualifiers /* objc-protocol-qualifiers */, 15231, /* DiagArray691 */ 1913, 0, R"()")
DIAG_ENTRY(ObjCReadonlyPropertyHasSetter /* objc-readonly-with-setter-property */, 15256, /* DiagArray692 */ 1915, 0, R"()")
DIAG_ENTRY(ObjCRedundantAPIUse /* objc-redundant-api-use */, 15291, 0, /* DiagSubGroup693 */ 530, R"()")
DIAG_ENTRY(ObjCRedundantLiteralUse /* objc-redundant-literal-use */, 15314, /* DiagArray694 */ 1917, 0, R"()")
DIAG_ENTRY(ObjCRootClass /* objc-root-class */, 15341, /* DiagArray695 */ 1919, 0, R"()")
DIAG_ENTRY(ObjCSignedCharBool /* objc-signed-char-bool */, 15357, 0, /* DiagSubGroup696 */ 532, R"()")
DIAG_ENTRY(ObjCSignedCharBoolImplicitFloatConversion /* objc-signed-char-bool-implicit-float-conversion */, 15379, /* DiagArray697 */ 1921, 0, R"()")
DIAG_ENTRY(ObjCSignedCharBoolImplicitIntConversion /* objc-signed-char-bool-implicit-int-conversion */, 15427, /* DiagArray698 */ 1923, 0, R"()")
DIAG_ENTRY(ObjCStringComparison /* objc-string-compare */, 15473, /* DiagArray699 */ 1925, 0, R"()")
DIAG_ENTRY(ObjCStringConcatenation /* objc-string-concatenation */, 15493, /* DiagArray700 */ 1927, 0, R"()")
DIAG_ENTRY(anonymous_290 /* objc-unsafe-perform-selector */, 15519, /* DiagArray701 */ 1929, 0, R"()")
DIAG_ENTRY(anonymous_122 /* octal-prefix-extension */, 15548, /* DiagArray702 */ 1931, 0, R"()")
DIAG_ENTRY(ODR /* odr */, 15571, /* DiagArray703 */ 1933, 0, R"()")
DIAG_ENTRY(OldStyleCast /* old-style-cast */, 15575, /* DiagArray704 */ 1952, 0, R"()")
DIAG_ENTRY(anonymous_39 /* old-style-definition */, 15590, 0, 0, R"()")
DIAG_ENTRY(OpenACC /* openacc */, 15611, 0, /* DiagSubGroup706 */ 537, R"()")
DIAG_ENTRY(anonymous_338 /* openacc-confusing-routine-name */, 15619, /* DiagArray707 */ 1954, 0, R"()")
DIAG_ENTRY(anonymous_337 /* openacc-deprecated-clause-alias */, 15650, /* DiagArray708 */ 1956, 0, R"()")
DIAG_ENTRY(anonymous_336 /* openacc-self-if-potential-conflict */, 15682, /* DiagArray709 */ 1958, 0, R"()")
DIAG_ENTRY(OpenCLUnsupportedRGBA /* opencl-unsupported-rgba */, 15717, /* DiagArray710 */ 1960, 0, R"()")
DIAG_ENTRY(OpenMP /* openmp */, 15741, 0, /* DiagSubGroup711 */ 539, R"()")
DIAG_ENTRY(OpenMP51Ext /* openmp-51-extensions */, 15748, /* DiagArray712 */ 1962, 0, R"()")
DIAG_ENTRY(OpenMPClauses /* openmp-clauses */, 15769, /* DiagArray713 */ 1964, 0, R"()")
DIAG_ENTRY(OpenMPExtensions /* openmp-extensions */, 15784, /* DiagArray714 */ 1984, 0, R"()")
DIAG_ENTRY(OpenMPLoopForm /* openmp-loop-form */, 15802, /* DiagArray715 */ 1986, 0, R"()")
DIAG_ENTRY(OpenMPMapping /* openmp-mapping */, 15819, /* DiagArray716 */ 1989, 0, R"()")
DIAG_ENTRY(OpenMPTarget /* openmp-target */, 15834, /* DiagArray717 */ 1991, /* DiagSubGroup717 */ 548, R"()")
DIAG_ENTRY(OpenMPTargetException /* openmp-target-exception */, 15848, /* DiagArray718 */ 1995, 0, R"()")
DIAG_ENTRY(OptionIgnored /* option-ignored */, 15872, /* DiagArray719 */ 1998, 0, R"()")
DIAG_ENTRY(OrderedCompareFunctionPointers /* ordered-compare-function-pointers */, 15887, /* DiagArray720 */ 2016, 0, R"()")
DIAG_ENTRY(OutOfLineDeclaration /* out-of-line-declaration */, 15921, /* DiagArray721 */ 2019, 0, R"()")
DIAG_ENTRY(anonymous_188 /* out-of-scope-function */, 15945, /* DiagArray722 */ 2021, 0, R"()")
DIAG_ENTRY(OveralignedType /* over-aligned */, 15967, /* DiagArray723 */ 2023, 0, R"()")
DIAG_ENTRY(anonymous_40 /* overflow */, 15980, 0, 0, R"()")
DIAG_ENTRY(OverlengthStrings /* overlength-strings */, 15989, /* DiagArray725 */ 2025, 0, R"()")
DIAG_ENTRY(OverloadedShiftOpParentheses /* overloaded-shift-op-parentheses */, 16008, /* DiagArray726 */ 2027, 0, R"()")
DIAG_ENTRY(OverloadedVirtual /* overloaded-virtual */, 16040, /* DiagArray727 */ 2029, 0, R"()")
DIAG_ENTRY(anonymous_7 /* override-init */, 16059, 0, /* DiagSubGroup728 */ 550, R"()")
DIAG_ENTRY(anonymous_106 /* override-module */, 16073, /* DiagArray729 */ 2031, 0, R"()")
DIAG_ENTRY(OverridingMethodMismatch /* overriding-method-mismatch */, 16089, /* DiagArray730 */ 2033, 0, R"()")
DIAG_ENTRY(anonymous_93 /* overriding-option */, 16116, /* DiagArray731 */ 2041, 0, R"()")
DIAG_ENTRY(Packed /* packed */, 16134, /* DiagArray732 */ 2043, /* DiagSubGroup732 */ 552, R"()")
DIAG_ENTRY(PackedNonPod /* packed-non-pod */, 16141, /* DiagArray733 */ 2045, 0, R"()")
DIAG_ENTRY(Padded /* padded */, 16156, /* DiagArray734 */ 2047, /* DiagSubGroup734 */ 554, R"()")
DIAG_ENTRY(PaddedBitField /* padded-bitfield */, 16163, /* DiagArray735 */ 2051, 0, R"()")
DIAG_ENTRY(Parentheses /* parentheses */, 16179, /* DiagArray736 */ 2054, /* DiagSubGroup736 */ 556, R"()")
DIAG_ENTRY(ParenthesesOnEquality /* parentheses-equality */, 16191, /* DiagArray737 */ 2059, 0, R"()")
DIAG_ENTRY(anonymous_9 /* partial-availability */, 16212, 0, /* DiagSubGroup738 */ 565, R"()")
DIAG_ENTRY(BackendOptimizationRemark /* pass */, 16233, /* DiagArray739 */ 2061, 0, R"()")
DIAG_ENTRY(BackendOptimizationRemarkAnalysis /* pass-analysis */, 16238, /* DiagArray740 */ 2063, 0, R"()")
DIAG_ENTRY(BackendOptimizationFailure /* pass-failed */, 16252, /* DiagArray741 */ 2067, 0, R"()")
DIAG_ENTRY(BackendOptimizationRemarkMissed /* pass-missed */, 16264, /* DiagArray742 */ 2069, 0, R"()")
DIAG_ENTRY(anonymous_341 /* pch-date-time */, 16276, /* DiagArray743 */ 2071, 0, R"()")
DIAG_ENTRY(anonymous_339 /* pch-vfs-diff */, 16290, /* DiagArray744 */ 2073, 0, R"()")
DIAG_ENTRY(Pedantic /* pedantic */, 16303, /* DiagArray745 */ 2075, /* DiagSubGroup745 */ 567, R"()")
DIAG_ENTRY(OpenCLCoreFeaturesDiagGroup /* pedantic-core-features */, 16312, /* DiagArray746 */ 2165, 0, R"()")
DIAG_ENTRY(PedanticMacros /* pedantic-macros */, 16335, 0, /* DiagSubGroup747 */ 626, R"()")
DIAG_ENTRY(PerfConstraintImpliesNoexcept /* perf-constraint-implies-noexcept */, 16351, /* DiagArray748 */ 2168, 0, R"()")
DIAG_ENTRY(PessimizingMove /* pessimizing-move */, 16384, /* DiagArray749 */ 2170, 0, R"()")
DIAG_ENTRY(PointerArith /* pointer-arith */, 16401, /* DiagArray750 */ 2173, /* DiagSubGroup750 */ 632, R"()")
DIAG_ENTRY(PointerBoolConversion /* pointer-bool-conversion */, 16415, /* DiagArray751 */ 2177, 0, R"()")
DIAG_ENTRY(anonymous_240 /* pointer-compare */, 16439, /* DiagArray752 */ 2180, 0, R"()")
DIAG_ENTRY(anonymous_288 /* pointer-integer-compare */, 16455, /* DiagArray753 */ 2182, 0, R"()")
DIAG_ENTRY(anonymous_300 /* pointer-sign */, 16479, /* DiagArray754 */ 2184, 0, R"()")
DIAG_ENTRY(PointerToEnumCast /* pointer-to-enum-cast */, 16492, /* DiagArray755 */ 2186, /* DiagSubGroup755 */ 634, R"()")
DIAG_ENTRY(PointerToIntCast /* pointer-to-int-cast */, 16513, /* DiagArray756 */ 2188, /* DiagSubGroup756 */ 636, R"()")
DIAG_ENTRY(anonymous_307 /* pointer-type-mismatch */, 16533, /* DiagArray757 */ 2190, 0, R"()")
DIAG_ENTRY(anonymous_92 /* poison-system-directories */, 16555, /* DiagArray758 */ 2192, 0, R"()")
DIAG_ENTRY(ObjCPotentiallyDirectSelector /* potentially-direct-selector */, 16581, /* DiagArray759 */ 2194, 0, R"()")
DIAG_ENTRY(PotentiallyEvaluatedExpression /* potentially-evaluated-expression */, 16609, /* DiagArray760 */ 2196, 0, R"()")
DIAG_ENTRY(PragmaClangAttribute /* pragma-clang-attribute */, 16642, /* DiagArray761 */ 2198, 0, R"()")
DIAG_ENTRY(anonymous_128 /* pragma-once-outside-header */, 16665, /* DiagArray762 */ 2200, 0, R"()")
DIAG_ENTRY(PragmaPack /* pragma-pack */, 16692, /* DiagArray763 */ 2202, /* DiagSubGroup763 */ 639, R"()")
DIAG_ENTRY(PragmaPackSuspiciousInclude /* pragma-pack-suspicious-include */, 16704, /* DiagArray764 */ 2205, 0, R"()")
DIAG_ENTRY(anonymous_129 /* pragma-system-header-outside-header */, 16735, /* DiagArray765 */ 2207, 0, R"()")
DIAG_ENTRY(Pragmas /* pragmas */, 16771, /* DiagArray766 */ 2209, /* DiagSubGroup766 */ 641, R"()")
DIAG_ENTRY(CXXPre14Compat /* pre-c++14-compat */, 16779, /* DiagArray767 */ 2212, 0, R"()")
DIAG_ENTRY(CXXPre14CompatPedantic /* pre-c++14-compat-pedantic */, 16796, 0, /* DiagSubGroup768 */ 646, R"()")
DIAG_ENTRY(CXXPre17Compat /* pre-c++17-compat */, 16822, /* DiagArray769 */ 2224, 0, R"()")
DIAG_ENTRY(CXXPre17CompatPedantic /* pre-c++17-compat-pedantic */, 16839, /* DiagArray770 */ 2242, /* DiagSubGroup770 */ 649, R"()")
DIAG_ENTRY(CXXPre20Compat /* pre-c++20-compat */, 16865, /* DiagArray771 */ 2245, 0, R"()")
DIAG_ENTRY(CXXPre20CompatPedantic /* pre-c++20-compat-pedantic */, 16882, /* DiagArray772 */ 2274, /* DiagSubGroup772 */ 651, R"()")
DIAG_ENTRY(CXXPre23Compat /* pre-c++23-compat */, 16908, /* DiagArray773 */ 2277, 0, R"()")
DIAG_ENTRY(CXXPre23CompatPedantic /* pre-c++23-compat-pedantic */, 16925, 0, /* DiagSubGroup774 */ 653, R"()")
DIAG_ENTRY(CXXPre26Compat /* pre-c++26-compat */, 16951, /* DiagArray775 */ 2293, 0, R"()")
DIAG_ENTRY(CXXPre26CompatPedantic /* pre-c++26-compat-pedantic */, 16968, 0, /* DiagSubGroup776 */ 655, R"()")
DIAG_ENTRY(anonymous_22 /* pre-c++2c-compat */, 16994, 0, /* DiagSubGroup777 */ 657, R"()")
DIAG_ENTRY(anonymous_23 /* pre-c++2c-compat-pedantic */, 17011, 0, /* DiagSubGroup778 */ 659, R"()")
DIAG_ENTRY(CPre11Compat /* pre-c11-compat */, 17037, /* DiagArray779 */ 2305, 0, R"()")
DIAG_ENTRY(CPre11CompatPedantic /* pre-c11-compat-pedantic */, 17052, 0, /* DiagSubGroup780 */ 661, R"()")
DIAG_ENTRY(CPre23Compat /* pre-c23-compat */, 17076, /* DiagArray781 */ 2307, /* DiagSubGroup781 */ 663, R"()")
DIAG_ENTRY(CPre23CompatPedantic /* pre-c23-compat-pedantic */, 17091, 0, /* DiagSubGroup782 */ 665, R"()")
DIAG_ENTRY(anonymous_14 /* pre-c2x-compat */, 17115, 0, /* DiagSubGroup783 */ 667, R"()")
DIAG_ENTRY(anonymous_15 /* pre-c2x-compat-pedantic */, 17130, 0, /* DiagSubGroup784 */ 669, R"()")
DIAG_ENTRY(CPre2yCompat /* pre-c2y-compat */, 17154, /* DiagArray785 */ 2330, 0, R"()")
DIAG_ENTRY(CPre2yCompatPedantic /* pre-c2y-compat-pedantic */, 17169, 0, /* DiagSubGroup786 */ 671, R"()")
DIAG_ENTRY(OpenMPPre51Compat /* pre-openmp-51-compat */, 17193, /* DiagArray787 */ 2340, 0, R"()")
DIAG_ENTRY(anonymous_181 /* predefined-identifier-outside-function */, 17214, /* DiagArray788 */ 2342, 0, R"()")
DIAG_ENTRY(PreferredTypeBitFieldEnumConversion /* preferred-type-bitfield-enum-conversion */, 17253, /* DiagArray789 */ 2344, 0, R"()")
DIAG_ENTRY(PreferredTypeBitFieldWidth /* preferred-type-bitfield-width */, 17293, 0, 0, R"()")
DIAG_ENTRY(PrivateExtern /* private-extern */, 17323, /* DiagArray791 */ 2348, 0, R"()")
DIAG_ENTRY(anonymous_139 /* private-header */, 17338, /* DiagArray792 */ 2350, 0, R"()")
DIAG_ENTRY(PrivateModule /* private-module */, 17353, /* DiagArray793 */ 2352, 0, R"()")
DIAG_ENTRY(ProfileInstrMissing /* profile-instr-missing */, 17368, /* DiagArray794 */ 2357, 0, R"()")
DIAG_ENTRY(ProfileInstrOutOfDate /* profile-instr-out-of-date */, 17390, /* DiagArray795 */ 2359, 0, R"()")
DIAG_ENTRY(ProfileInstrUnprofiled /* profile-instr-unprofiled */, 17416, /* DiagArray796 */ 2361, 0, R"()")
DIAG_ENTRY(PropertyAccessDotSyntax /* property-access-dot-syntax */, 17441, /* DiagArray797 */ 2363, 0, R"()")
DIAG_ENTRY(PropertyAttr /* property-attribute-mismatch */, 17468, /* DiagArray798 */ 2365, 0, R"()")
DIAG_ENTRY(Protocol /* protocol */, 17496, /* DiagArray799 */ 2370, 0, R"()")
DIAG_ENTRY(anonymous_198 /* protocol-property-synthesis-ambiguity */, 17505, /* DiagArray800 */ 2372, 0, R"()")
DIAG_ENTRY(anonymous_111 /* psabi */, 17543, /* DiagArray801 */ 2374, 0, R"()")
DIAG_ENTRY(PtrAuthNullPointers /* ptrauth-null-pointers */, 17549, /* DiagArray802 */ 2376, 0, R"()")
DIAG_ENTRY(anonymous_270 /* qualified-void-return-type */, 17571, /* DiagArray803 */ 2379, 0, R"()")
DIAG_ENTRY(FrameworkHdrQuotedInclude /* quoted-include-in-framework-header */, 17598, /* DiagArray804 */ 2381, 0, R"()")
DIAG_ENTRY(RangeLoopAnalysis /* range-loop-analysis */, 17633, 0, /* DiagSubGroup805 */ 673, R"()")
DIAG_ENTRY(RangeLoopBindReference /* range-loop-bind-reference */, 17653, /* DiagArray806 */ 2383, 0, R"()")
DIAG_ENTRY(RangeLoopConstruct /* range-loop-construct */, 17679, /* DiagArray807 */ 2385, 0, R"()")
DIAG_ENTRY(ReadOnlyPlacementChecks /* read-only-types */, 17700, /* DiagArray808 */ 2388, 0, R"()")
DIAG_ENTRY(anonymous_208 /* readonly-iboutlet-property */, 17716, /* DiagArray809 */ 2390, 0, R"()")
DIAG_ENTRY(ObjCReceiver /* receiver-expr */, 17743, /* DiagArray810 */ 2392, 0, R"()")
DIAG_ENTRY(ForwardClassReceiver /* receiver-forward-class */, 17757, /* DiagArray811 */ 2394, 0, R"()")
DIAG_ENTRY(RedeclaredClassMember /* redeclared-class-member */, 17780, /* DiagArray812 */ 2397, 0, R"()")
DIAG_ENTRY(anonymous_97 /* reduced-bmi-output-overrided */, 17804, /* DiagArray813 */ 2399, 0, R"()")
DIAG_ENTRY(RedundantAttribute /* redundant-attribute */, 17833, /* DiagArray814 */ 2401, 0, R"()")
DIAG_ENTRY(anonymous_211 /* redundant-consteval-if */, 17853, /* DiagArray815 */ 2403, 0, R"()")
DIAG_ENTRY(anonymous_41 /* redundant-decls */, 17876, 0, 0, R"()")
DIAG_ENTRY(RedundantMove /* redundant-move */, 17892, /* DiagArray817 */ 2405, 0, R"()")
DIAG_ENTRY(anonymous_186 /* redundant-parens */, 17907, /* DiagArray818 */ 2407, 0, R"()")
DIAG_ENTRY(Register /* register */, 17924, /* DiagArray819 */ 2409, /* DiagSubGroup819 */ 676, R"()")
DIAG_ENTRY(ReinterpretBaseClass /* reinterpret-base-class */, 17933, /* DiagArray820 */ 2411, 0, R"()")
DIAG_ENTRY(RemarkBackendPlugin /* remark-backend-plugin */, 17956, /* DiagArray821 */ 2413, 0, R"()")
DIAG_ENTRY(Reorder /* reorder */, 17978, 0, /* DiagSubGroup822 */ 678, R"()")
DIAG_ENTRY(ReorderCtor /* reorder-ctor */, 17986, /* DiagArray823 */ 2415, 0, R"()")
DIAG_ENTRY(ReorderInitList /* reorder-init-list */, 17999, /* DiagArray824 */ 2418, 0, R"()")
DIAG_ENTRY(anonymous_246 /* requires-super-attribute */, 18017, /* DiagArray825 */ 2420, 0, R"()")
DIAG_ENTRY(ReservedAttributeIdentifier /* reserved-attribute-identifier */, 18042, /* DiagArray826 */ 2422, 0, R"()")
DIAG_ENTRY(ReservedIdAsMacroAlias /* reserved-id-macro */, 18072, 0, /* DiagSubGroup827 */ 681, R"()")
DIAG_ENTRY(ReservedIdentifier /* reserved-identifier */, 18090, /* DiagArray828 */ 2424, /* DiagSubGroup828 */ 683, R"()")
DIAG_ENTRY(ReservedIdAsMacro /* reserved-macro-identifier */, 18110, /* DiagArray829 */ 2426, 0, R"()")
DIAG_ENTRY(ReservedModuleIdentifier /* reserved-module-identifier */, 18136, /* DiagArray830 */ 2428, 0, R"()")
DIAG_ENTRY(ReservedUserDefinedLiteral /* reserved-user-defined-literal */, 18163, /* DiagArray831 */ 2430, /* DiagSubGroup831 */ 688, R"()")
DIAG_ENTRY(RestrictExpansionMacro /* restrict-expansion */, 18193, /* DiagArray832 */ 2433, 0, R"()")
DIAG_ENTRY(anonymous_263 /* retained-language-linkage */, 18212, /* DiagArray833 */ 2435, 0, R"()")
DIAG_ENTRY(anonymous_30 /* return-local-addr */, 18238, 0, /* DiagSubGroup834 */ 690, R"()")
DIAG_ENTRY(ReturnMismatch /* return-mismatch */, 18256, /* DiagArray835 */ 2437, 0, R"()")
DIAG_ENTRY(ReturnStackAddress /* return-stack-address */, 18272, /* DiagArray836 */ 2441, 0, R"()")
DIAG_ENTRY(ReturnStdMove /* return-std-move */, 18293, 0, 0, R"()")
DIAG_ENTRY(ReturnType /* return-type */, 18309, /* DiagArray838 */ 2446, /* DiagSubGroup838 */ 692, R"()")
DIAG_ENTRY(ReturnTypeCLinkage /* return-type-c-linkage */, 18321, /* DiagArray839 */ 2448, 0, R"()")
DIAG_ENTRY(anonymous_248 /* rewrite-not-bool */, 18343, /* DiagArray840 */ 2451, 0, R"()")
DIAG_ENTRY(RoundTripCC1Args /* round-trip-cc1-args */, 18360, /* DiagArray841 */ 2453, 0, R"()")
DIAG_ENTRY(RTTI /* rtti */, 18380, /* DiagArray842 */ 2455, 0, R"()")
DIAG_ENTRY(SanitizeAddressRemarks /* sanitize-address */, 18385, /* DiagArray843 */ 2458, 0, R"()")
DIAG_ENTRY(anonymous_103 /* sarif-format-unstable */, 18402, /* DiagArray844 */ 2461, 0, R"()")
DIAG_ENTRY(UsedSearchPath /* search-path-usage */, 18424, /* DiagArray845 */ 2463, 0, R"()")
DIAG_ENTRY(Section /* section */, 18442, /* DiagArray846 */ 2465, 0, R"()")
DIAG_ENTRY(Selector /* selector */, 18450, /* DiagArray847 */ 2469, /* DiagSubGroup847 */ 695, R"()")
DIAG_ENTRY(SelectorTypeMismatch /* selector-type-mismatch */, 18459, /* DiagArray848 */ 2471, 0, R"()")
DIAG_ENTRY(SelfAssignment /* self-assign */, 18482, /* DiagArray849 */ 2473, /* DiagSubGroup849 */ 697, R"()")
DIAG_ENTRY(SelfAssignmentField /* self-assign-field */, 18494, /* DiagArray850 */ 2475, 0, R"()")
DIAG_ENTRY(SelfAssignmentOverloaded /* self-assign-overloaded */, 18512, /* DiagArray851 */ 2477, 0, R"()")
DIAG_ENTRY(SelfMove /* self-move */, 18535, /* DiagArray852 */ 2479, 0, R"()")
DIAG_ENTRY(SemiBeforeMethodBody /* semicolon-before-method-body */, 18545, /* DiagArray853 */ 2481, 0, R"()")
DIAG_ENTRY(Sentinel /* sentinel */, 18574, /* DiagArray854 */ 2483, 0, R"()")
DIAG_ENTRY(anonymous_44 /* sequence-point */, 18583, 0, /* DiagSubGroup855 */ 700, R"()")
DIAG_ENTRY(SerializedDiagnostics /* serialized-diagnostics */, 18598, /* DiagArray856 */ 2486, 0, R"()")
DIAG_ENTRY(Shadow /* shadow */, 18621, /* DiagArray857 */ 2490, /* DiagSubGroup857 */ 702, R"()")
DIAG_ENTRY(ShadowAll /* shadow-all */, 18628, 0, /* DiagSubGroup858 */ 705, R"()")
DIAG_ENTRY(ShadowField /* shadow-field */, 18639, /* DiagArray859 */ 2492, 0, R"()")
DIAG_ENTRY(ShadowFieldInConstructor /* shadow-field-in-constructor */, 18652, /* DiagArray860 */ 2494, /* DiagSubGroup860 */ 710, R"()")
DIAG_ENTRY(ShadowFieldInConstructorModified /* shadow-field-in-constructor-modified */, 18680, /* DiagArray861 */ 2496, 0, R"()")
DIAG_ENTRY(ShadowIvar /* shadow-ivar */, 18717, /* DiagArray862 */ 2498, 0, R"()")
DIAG_ENTRY(ShadowUncapturedLocal /* shadow-uncaptured-local */, 18729, /* DiagArray863 */ 2500, 0, R"()")
DIAG_ENTRY(anonymous_280 /* shift-bool */, 18753, /* DiagArray864 */ 2502, 0, R"()")
DIAG_ENTRY(anonymous_276 /* shift-count-negative */, 18764, /* DiagArray865 */ 2504, 0, R"()")
DIAG_ENTRY(anonymous_277 /* shift-count-overflow */, 18785, /* DiagArray866 */ 2506, 0, R"()")
DIAG_ENTRY(anonymous_275 /* shift-negative-value */, 18806, /* DiagArray867 */ 2508, 0, R"()")
DIAG_ENTRY(ShiftOpParentheses /* shift-op-parentheses */, 18827, /* DiagArray868 */ 2510, 0, R"()")
DIAG_ENTRY(anonymous_278 /* shift-overflow */, 18848, /* DiagArray869 */ 2512, 0, R"()")
DIAG_ENTRY(anonymous_279 /* shift-sign-overflow */, 18863, /* DiagArray870 */ 2514, 0, R"()")
DIAG_ENTRY(Shorten64To32 /* shorten-64-to-32 */, 18883, /* DiagArray871 */ 2516, 0, R"()")
DIAG_ENTRY(SignCompare /* sign-compare */, 18900, /* DiagArray872 */ 2518, 0, R"()")
DIAG_ENTRY(SignConversion /* sign-conversion */, 18913, /* DiagArray873 */ 2520, 0, R"()")
DIAG_ENTRY(anonymous_42 /* sign-promo */, 18929, 0, 0, R"()")
DIAG_ENTRY(SignedEnumBitfield /* signed-enum-bitfield */, 18940, /* DiagArray875 */ 2524, 0, R"()")
DIAG_ENTRY(anonymous_326 /* signed-unsigned-wchar */, 18961, /* DiagArray876 */ 2526, 0, R"()")
DIAG_ENTRY(SingleBitBitFieldConstantConversion /* single-bit-bitfield-constant-conversion */, 18983, /* DiagArray877 */ 2528, 0, R"()")
DIAG_ENTRY(SizeofArrayArgument /* sizeof-array-argument */, 19023, /* DiagArray878 */ 2530, 0, R"()")
DIAG_ENTRY(SizeofArrayDecay /* sizeof-array-decay */, 19045, /* DiagArray879 */ 2532, 0, R"()")
DIAG_ENTRY(anonymous_242 /* sizeof-array-div */, 19064, /* DiagArray880 */ 2534, 0, R"()")
DIAG_ENTRY(anonymous_241 /* sizeof-pointer-div */, 19081, /* DiagArray881 */ 2536, 0, R"()")
DIAG_ENTRY(SizeofPointerMemaccess /* sizeof-pointer-memaccess */, 19100, /* DiagArray882 */ 2538, 0, R"()")
DIAG_ENTRY(anonymous_99 /* slash-u-filename */, 19125, /* DiagArray883 */ 2541, 0, R"()")
DIAG_ENTRY(anonymous_85 /* slh-asm-goto */, 19142, /* DiagArray884 */ 2543, 0, R"()")
DIAG_ENTRY(anonymous_88 /* sloc-usage */, 19155, /* DiagArray885 */ 2545, 0, R"()")
DIAG_ENTRY(UninitializedSometimes /* sometimes-uninitialized */, 19166, /* DiagArray886 */ 2547, 0, R"()")
DIAG_ENTRY(BackendSourceMgr /* source-mgr */, 19190, /* DiagArray887 */ 2549, 0, R"()")
DIAG_ENTRY(SourceUsesOpenACC /* source-uses-openacc */, 19201, /* DiagArray888 */ 2551, 0, R"()")
DIAG_ENTRY(SourceUsesOpenMP /* source-uses-openmp */, 19221, /* DiagArray889 */ 2555, 0, R"()")
DIAG_ENTRY(SpirCompat /* spir-compat */, 19240, /* DiagArray890 */ 2564, 0, R"()")
DIAG_ENTRY(anonymous_77 /* spirv-compat */, 19252, 0, /* DiagSubGroup891 */ 712, R"()")
DIAG_ENTRY(anonymous_82 /* stack-exhausted */, 19265, /* DiagArray892 */ 2566, 0, R"()")
DIAG_ENTRY(anonymous_84 /* stack-protector */, 19281, /* DiagArray893 */ 2568, 0, R"()")
DIAG_ENTRY(StaticFloatInit /* static-float-init */, 19297, /* DiagArray894 */ 2570, /* DiagSubGroup894 */ 714, R"()")
DIAG_ENTRY(StaticInInline /* static-in-inline */, 19315, /* DiagArray895 */ 2572, 0, R"()")
DIAG_ENTRY(anonymous_155 /* static-inline-explicit-instantiation */, 19332, /* DiagArray896 */ 2575, 0, R"()")
DIAG_ENTRY(StaticLocalInInline /* static-local-in-inline */, 19369, /* DiagArray897 */ 2577, 0, R"()")
DIAG_ENTRY(UninitializedStaticSelfInit /* static-self-init */, 19392, /* DiagArray898 */ 2579, 0, R"()")
DIAG_ENTRY(anonymous_102 /* stdlibcxx-not-found */, 19409, /* DiagArray899 */ 2581, 0, R"()")
DIAG_ENTRY(anonymous_48 /* strict-aliasing */, 19429, 0, 0, R"()")
DIAG_ENTRY(anonymous_45 /* strict-aliasing=0 */, 19445, 0, 0, R"()")
DIAG_ENTRY(anonymous_46 /* strict-aliasing=1 */, 19463, 0, 0, R"()")
DIAG_ENTRY(anonymous_47 /* strict-aliasing=2 */, 19481, 0, 0, R"()")
DIAG_ENTRY(anonymous_55 /* strict-overflow */, 19499, 0, 0, R"()")
DIAG_ENTRY(anonymous_49 /* strict-overflow=0 */, 19515, 0, 0, R"()")
DIAG_ENTRY(anonymous_50 /* strict-overflow=1 */, 19533, 0, 0, R"()")
DIAG_ENTRY(anonymous_51 /* strict-overflow=2 */, 19551, 0, 0, R"()")
DIAG_ENTRY(anonymous_52 /* strict-overflow=3 */, 19569, 0, 0, R"()")
DIAG_ENTRY(anonymous_53 /* strict-overflow=4 */, 19587, 0, 0, R"()")
DIAG_ENTRY(anonymous_54 /* strict-overflow=5 */, 19605, 0, 0, R"()")
DIAG_ENTRY(ObjCStrictPotentiallyDirectSelector /* strict-potentially-direct-selector */, 19623, /* DiagArray911 */ 2583, /* DiagSubGroup911 */ 716, R"()")
DIAG_ENTRY(anonymous_249 /* strict-primary-template-shadow */, 19658, /* DiagArray912 */ 2585, 0, R"()")
DIAG_ENTRY(StrictPrototypes /* strict-prototypes */, 19689, /* DiagArray913 */ 2587, /* DiagSubGroup913 */ 718, R"()")
DIAG_ENTRY(StrictSelector /* strict-selector-match */, 19707, /* DiagArray914 */ 2589, 0, R"()")
DIAG_ENTRY(StringCompare /* string-compare */, 19729, /* DiagArray915 */ 2591, 0, R"()")
DIAG_ENTRY(StringConcatation /* string-concatenation */, 19744, /* DiagArray916 */ 2593, 0, R"()")
DIAG_ENTRY(StringConversion /* string-conversion */, 19765, /* DiagArray917 */ 2595, 0, R"()")
DIAG_ENTRY(StringPlusChar /* string-plus-char */, 19783, /* DiagArray918 */ 2597, 0, R"()")
DIAG_ENTRY(StringPlusInt /* string-plus-int */, 19800, /* DiagArray919 */ 2599, 0, R"()")
DIAG_ENTRY(anonymous_191 /* strlcpy-strlcat-size */, 19816, /* DiagArray920 */ 2601, 0, R"()")
DIAG_ENTRY(StrncatSize /* strncat-size */, 19837, /* DiagArray921 */ 2603, 0, R"()")
DIAG_ENTRY(CXX11WarnSuggestOverrideDestructor /* suggest-destructor-override */, 19850, /* DiagArray922 */ 2607, 0, R"()")
DIAG_ENTRY(CXX11WarnSuggestOverride /* suggest-override */, 19878, /* DiagArray923 */ 2609, 0, R"()")
DIAG_ENTRY(SuperSubClassMismatch /* super-class-method-mismatch */, 19895, /* DiagArray924 */ 2611, 0, R"()")
DIAG_ENTRY(SuspiciousBzero /* suspicious-bzero */, 19923, /* DiagArray925 */ 2613, 0, R"()")
DIAG_ENTRY(SuspiciousMemaccess /* suspicious-memaccess */, 19940, 0, /* DiagSubGroup926 */ 720, R"()")
DIAG_ENTRY(SwiftNameAttribute /* swift-name-attribute */, 19961, /* DiagArray927 */ 2615, 0, R"()")
DIAG_ENTRY(Switch /* switch */, 19982, /* DiagArray928 */ 2629, 0, R"()")
DIAG_ENTRY(SwitchBool /* switch-bool */, 19989, /* DiagArray929 */ 2633, 0, R"()")
DIAG_ENTRY(SwitchDefault /* switch-default */, 20001, /* DiagArray930 */ 2635, 0, R"()")
DIAG_ENTRY(SwitchEnum /* switch-enum */, 20016, /* DiagArray931 */ 2637, 0, R"()")
DIAG_ENTRY(SyncAlignment /* sync-alignment */, 20028, /* DiagArray932 */ 2639, 0, R"()")
DIAG_ENTRY(anonymous_325 /* sync-fetch-and-nand-semantics-changed */, 20043, /* DiagArray933 */ 2641, 0, R"()")
DIAG_ENTRY(anonymous_43 /* synth */, 20081, 0, 0, R"()")
DIAG_ENTRY(TargetClonesMixedSpecifiers /* target-clones-mixed-specifiers */, 20087, /* DiagArray935 */ 2643, 0, R"()")
DIAG_ENTRY(TautologicalBitwiseCompare /* tautological-bitwise-compare */, 20118, /* DiagArray936 */ 2645, 0, R"()")
DIAG_ENTRY(TautologicalCompare /* tautological-compare */, 20147, /* DiagArray937 */ 2648, /* DiagSubGroup937 */ 726, R"()")
DIAG_ENTRY(TautologicalConstantCompare /* tautological-constant-compare */, 20168, /* DiagArray938 */ 2651, /* DiagSubGroup938 */ 734, R"()")
DIAG_ENTRY(TautologicalInRangeCompare /* tautological-constant-in-range-compare */, 20198, 0, /* DiagSubGroup939 */ 736, R"()")
DIAG_ENTRY(TautologicalOutOfRangeCompare /* tautological-constant-out-of-range-compare */, 20237, /* DiagArray940 */ 2655, 0, R"()")
DIAG_ENTRY(TautologicalNegationCompare /* tautological-negation-compare */, 20280, /* DiagArray941 */ 2657, 0, R"()")
DIAG_ENTRY(TautologicalObjCBoolCompare /* tautological-objc-bool-compare */, 20310, /* DiagArray942 */ 2660, 0, R"()")
DIAG_ENTRY(TautologicalOverlapCompare /* tautological-overlap-compare */, 20341, /* DiagArray943 */ 2662, 0, R"()")
DIAG_ENTRY(TautologicalPointerCompare /* tautological-pointer-compare */, 20370, /* DiagArray944 */ 2664, 0, R"()")
DIAG_ENTRY(TautologicalTypeLimitCompare /* tautological-type-limit-compare */, 20399, /* DiagArray945 */ 2667, 0, R"()")
DIAG_ENTRY(TautologicalUndefinedCompare /* tautological-undefined-compare */, 20431, /* DiagArray946 */ 2669, 0, R"()")
DIAG_ENTRY(TautologicalUnsignedCharZeroCompare /* tautological-unsigned-char-zero-compare */, 20462, /* DiagArray947 */ 2672, 0, R"()")
DIAG_ENTRY(TautologicalUnsignedEnumZeroCompare /* tautological-unsigned-enum-zero-compare */, 20502, /* DiagArray948 */ 2674, 0, R"()")
DIAG_ENTRY(TautologicalUnsignedZeroCompare /* tautological-unsigned-zero-compare */, 20542, /* DiagArray949 */ 2676, 0, R"()")
DIAG_ENTRY(TautologicalValueRangeCompare /* tautological-value-range-compare */, 20577, /* DiagArray950 */ 2678, 0, R"()")
DIAG_ENTRY(anonymous_335 /* tcb-enforcement */, 20610, /* DiagArray951 */ 2680, 0, R"()")
DIAG_ENTRY(anonymous_297 /* template-in-declaration-name */, 20626, /* DiagArray952 */ 2682, 0, R"()")
DIAG_ENTRY(anonymous_284 /* tentative-definition-array */, 20655, /* DiagArray953 */ 2684, 0, R"()")
DIAG_ENTRY(anonymous_283 /* tentative-definition-incomplete-type */, 20682, /* DiagArray954 */ 2686, 0, R"()")
DIAG_ENTRY(ThreadSafety /* thread-safety */, 20719, 0, /* DiagSubGroup955 */ 739, R"()")
DIAG_ENTRY(ThreadSafetyAnalysis /* thread-safety-analysis */, 20733, /* DiagArray956 */ 2688, 0, R"()")
DIAG_ENTRY(ThreadSafetyAttributes /* thread-safety-attributes */, 20756, /* DiagArray957 */ 2710, 0, R"()")
DIAG_ENTRY(ThreadSafetyBeta /* thread-safety-beta */, 20781, /* DiagArray958 */ 2718, 0, R"()")
DIAG_ENTRY(ThreadSafetyNegative /* thread-safety-negative */, 20800, /* DiagArray959 */ 2720, 0, R"()")
DIAG_ENTRY(ThreadSafetyPointer /* thread-safety-pointer */, 20823, /* DiagArray960 */ 2722, 0, R"()")
DIAG_ENTRY(ThreadSafetyPrecise /* thread-safety-precise */, 20845, /* DiagArray961 */ 2727, 0, R"()")
DIAG_ENTRY(ThreadSafetyReference /* thread-safety-reference */, 20867, /* DiagArray962 */ 2731, /* DiagSubGroup962 */ 744, R"()")
DIAG_ENTRY(ThreadSafetyReferenceReturn /* thread-safety-reference-return */, 20891, /* DiagArray963 */ 2734, 0, R"()")
DIAG_ENTRY(ThreadSafetyVerbose /* thread-safety-verbose */, 20922, /* DiagArray964 */ 2737, 0, R"()")
DIAG_ENTRY(Trigraphs /* trigraphs */, 20944, /* DiagArray965 */ 2739, 0, R"()")
DIAG_ENTRY(TypeLimits /* type-limits */, 20954, 0, /* DiagSubGroup966 */ 746, R"()")
DIAG_ENTRY(TypeSafety /* type-safety */, 20966, /* DiagArray967 */ 2744, 0, R"()")
DIAG_ENTRY(anonymous_262 /* typedef-redefinition */, 20978, /* DiagArray968 */ 2748, 0, R"()")
DIAG_ENTRY(anonymous_256 /* typename-missing */, 20999, /* DiagArray969 */ 2750, 0, R"()")
DIAG_ENTRY(anonymous_107 /* unable-to-open-stats-file */, 21016, /* DiagArray970 */ 2752, 0, R"()")
DIAG_ENTRY(UnalignedAccess /* unaligned-access */, 21042, /* DiagArray971 */ 2754, 0, R"()")
DIAG_ENTRY(anonymous_328 /* unaligned-qualifier-implicit-cast */, 21059, /* DiagArray972 */ 2756, 0, R"()")
DIAG_ENTRY(UnavailableDeclarations /* unavailable-declarations */, 21093, /* DiagArray973 */ 2758, 0, R"()")
DIAG_ENTRY(UndeclaredSelector /* undeclared-selector */, 21118, /* DiagArray974 */ 2760, 0, R"()")
DIAG_ENTRY(UndefinedIdentifier /* undef */, 21138, /* DiagArray975 */ 2763, /* DiagSubGroup975 */ 751, R"()")
DIAG_ENTRY(anonymous_132 /* undef-prefix */, 21144, /* DiagArray976 */ 2765, 0, R"()")
DIAG_ENTRY(UndefinedTrueIdentifier /* undef-true */, 21157, /* DiagArray977 */ 2767, 0, R"()")
DIAG_ENTRY(anonymous_221 /* undefined-arm-za */, 21168, /* DiagArray978 */ 2769, 0, R"()")
DIAG_ENTRY(anonymous_222 /* undefined-arm-zt0 */, 21185, /* DiagArray979 */ 2771, 0, R"()")
DIAG_ENTRY(UndefinedBoolConversion /* undefined-bool-conversion */, 21203, /* DiagArray980 */ 2773, 0, R"()")
DIAG_ENTRY(UndefinedFuncTemplate /* undefined-func-template */, 21229, /* DiagArray981 */ 2776, 0, R"()")
DIAG_ENTRY(anonymous_261 /* undefined-inline */, 21253, /* DiagArray982 */ 2778, 0, R"()")
DIAG_ENTRY(anonymous_259 /* undefined-internal */, 21270, /* DiagArray983 */ 2780, 0, R"()")
DIAG_ENTRY(anonymous_260 /* undefined-internal-type */, 21289, /* DiagArray984 */ 2782, 0, R"()")
DIAG_ENTRY(UndefinedReinterpretCast /* undefined-reinterpret-cast */, 21313, /* DiagArray985 */ 2784, 0, R"()")
DIAG_ENTRY(UndefinedVarTemplate /* undefined-var-template */, 21340, /* DiagArray986 */ 2787, 0, R"()")
DIAG_ENTRY(UnderalignedExceptionObject /* underaligned-exception-object */, 21363, /* DiagArray987 */ 2789, 0, R"()")
DIAG_ENTRY(UnevaluatedExpression /* unevaluated-expression */, 21393, /* DiagArray988 */ 2791, /* DiagSubGroup988 */ 753, R"()")
DIAG_ENTRY(UnguardedAvailability /* unguarded-availability */, 21416, /* DiagArray989 */ 2793, /* DiagSubGroup989 */ 755, R"()")
DIAG_ENTRY(UnguardedAvailabilityNew /* unguarded-availability-new */, 21439, /* DiagArray990 */ 2796, 0, R"()")
DIAG_ENTRY(Unicode /* unicode */, 21466, /* DiagArray991 */ 2799, 0, R"()")
DIAG_ENTRY(anonymous_117 /* unicode-homoglyph */, 21474, /* DiagArray992 */ 2807, 0, R"()")
DIAG_ENTRY(anonymous_116 /* unicode-whitespace */, 21492, /* DiagArray993 */ 2809, 0, R"()")
DIAG_ENTRY(anonymous_118 /* unicode-zero-width */, 21511, /* DiagArray994 */ 2811, 0, R"()")
DIAG_ENTRY(Uninitialized /* uninitialized */, 21530, /* DiagArray995 */ 2813, /* DiagSubGroup995 */ 757, R"()")
DIAG_ENTRY(UninitializedConstReference /* uninitialized-const-reference */, 21544, /* DiagArray996 */ 2822, 0, R"()")
DIAG_ENTRY(UninitializedExplicitInit /* uninitialized-explicit-init */, 21574, /* DiagArray997 */ 2824, 0, R"()")
DIAG_ENTRY(UniqueObjectDuplication /* unique-object-duplication */, 21602, /* DiagArray998 */ 2826, 0, R"(Warns when objects which are supposed to be globally unique might get duplicated
when built into a shared library.

If an object with hidden visibility is built into a shared library, each instance
of the library will get its own copy. This can cause very subtle bugs if there was
only supposed to be one copy of the object in question: singletons aren't single,
changes to one object won't affect the others, the object's initializer will run
once per copy, etc.

Specifically, this warning fires when it detects an object which:
  1. Appears in a header file (so it might get compiled into multiple libaries), and
  2. Has external linkage (otherwise it's supposed to be duplicated), and
  3. Has hidden visibility.

As well as one of the following:
  1. The object is mutable, or
  2. The object's initializer definitely has side effects.

The warning is best resolved by making the object ``const`` (if possible), or by explicitly
giving the object non-hidden visibility, e.g. using ``__attribute((visibility("default")))``.
Note that all levels of a pointer variable must be constant; ``const int*`` will
trigger the warning because the pointer itself is mutable.

This warning is currently disabled on Windows since it uses import/export rules
instead of visibility.)")
DIAG_ENTRY(UnknownArgument /* unknown-argument */, 21628, /* DiagArray999 */ 2829, 0, R"()")
DIAG_ENTRY(UnknownAssumption /* unknown-assumption */, 21645, /* DiagArray1000 */ 2833, 0, R"()")
DIAG_ENTRY(UnknownAttributes /* unknown-attributes */, 21664, /* DiagArray1001 */ 2835, 0, R"()")
DIAG_ENTRY(CudaUnknownVersion /* unknown-cuda-version */, 21683, /* DiagArray1002 */ 2837, 0, R"()")
DIAG_ENTRY(anonymous_136 /* unknown-directives */, 21704, /* DiagArray1003 */ 2840, 0, R"()")
DIAG_ENTRY(anonymous_121 /* unknown-escape-sequence */, 21723, /* DiagArray1004 */ 2842, 0, R"()")
DIAG_ENTRY(UnknownPragmas /* unknown-pragmas */, 21747, /* DiagArray1005 */ 2844, 0, R"()")
DIAG_ENTRY(UnknownSanitizers /* unknown-sanitizers */, 21763, /* DiagArray1006 */ 2865, 0, R"()")
DIAG_ENTRY(UnknownWarningOption /* unknown-warning-option */, 21782, /* DiagArray1007 */ 2867, 0, R"()")
DIAG_ENTRY(UnnamedTypeTemplateArgs /* unnamed-type-template-args */, 21805, /* DiagArray1008 */ 2871, /* DiagSubGroup1008 */ 761, R"()")
DIAG_ENTRY(WarnUnnecessaryVirtualSpecifier /* unnecessary-virtual-specifier */, 21832, /* DiagArray1009 */ 2873, 0, R"(Warns when a ``final`` class contains a virtual method (including virtual
destructors). Since ``final`` classes cannot be subclassed, their methods
cannot be overridden, and hence the ``virtual`` specifier is useless.

The warning also detects virtual methods in classes whose destructor is
``final``, for the same reason.

The warning does not fire on virtual methods which are also marked ``override``.)")
DIAG_ENTRY(UnneededInternalDecl /* unneeded-internal-declaration */, 21862, /* DiagArray1010 */ 2875, 0, R"()")
DIAG_ENTRY(UnneededMemberFunction /* unneeded-member-function */, 21892, /* DiagArray1011 */ 2878, 0, R"()")
DIAG_ENTRY(anonymous_250 /* unqualified-std-cast-call */, 21917, /* DiagArray1012 */ 2880, 0, R"()")
DIAG_ENTRY(UnreachableCode /* unreachable-code */, 21943, /* DiagArray1013 */ 2882, /* DiagSubGroup1013 */ 763, R"()")
DIAG_ENTRY(UnreachableCodeAggressive /* unreachable-code-aggressive */, 21960, 0, /* DiagSubGroup1014 */ 767, R"()")
DIAG_ENTRY(UnreachableCodeBreak /* unreachable-code-break */, 21988, /* DiagArray1015 */ 2884, 0, R"()")
DIAG_ENTRY(UnreachableCodeFallthrough /* unreachable-code-fallthrough */, 22011, /* DiagArray1016 */ 2886, 0, R"()")
DIAG_ENTRY(UnreachableCodeGenericAssoc /* unreachable-code-generic-assoc */, 22040, /* DiagArray1017 */ 2888, 0, R"()")
DIAG_ENTRY(UnreachableCodeLoopIncrement /* unreachable-code-loop-increment */, 22071, /* DiagArray1018 */ 2890, 0, R"()")
DIAG_ENTRY(UnreachableCodeReturn /* unreachable-code-return */, 22103, /* DiagArray1019 */ 2892, 0, R"()")
DIAG_ENTRY(UnsafeBufferUsage /* unsafe-buffer-usage */, 22127, /* DiagArray1020 */ 2894, /* DiagSubGroup1020 */ 771, R"()")
DIAG_ENTRY(UnsafeBufferUsageInContainer /* unsafe-buffer-usage-in-container */, 22147, /* DiagArray1021 */ 2897, 0, R"()")
DIAG_ENTRY(UnsafeBufferUsageInLibcCall /* unsafe-buffer-usage-in-libc-call */, 22180, /* DiagArray1022 */ 2899, 0, R"()")
DIAG_ENTRY(Unsequenced /* unsequenced */, 22213, /* DiagArray1023 */ 2901, 0, R"()")
DIAG_ENTRY(UnsupportedABI /* unsupported-abi */, 22225, /* DiagArray1024 */ 2904, 0, R"()")
DIAG_ENTRY(UnsupportedAbs /* unsupported-abs */, 22241, /* DiagArray1025 */ 2907, 0, R"()")
DIAG_ENTRY(anonymous_238 /* unsupported-availability-guard */, 22257, /* DiagArray1026 */ 2910, 0, R"()")
DIAG_ENTRY(UnsupportedCB /* unsupported-cb */, 22288, /* DiagArray1027 */ 2912, 0, R"()")
DIAG_ENTRY(anonymous_234 /* unsupported-dll-base-class-template */, 22303, /* DiagArray1028 */ 2914, 0, R"()")
DIAG_ENTRY(UnsupportedFPOpt /* unsupported-floating-point-opt */, 22339, /* DiagArray1029 */ 2916, 0, R"()")
DIAG_ENTRY(UnsupportedFriend /* unsupported-friend */, 22370, /* DiagArray1030 */ 2919, 0, R"()")
DIAG_ENTRY(UnsupportedGPOpt /* unsupported-gpopt */, 22389, /* DiagArray1031 */ 2922, 0, R"()")
DIAG_ENTRY(UnsupportedNan /* unsupported-nan */, 22407, /* DiagArray1032 */ 2924, 0, R"()")
DIAG_ENTRY(UnsupportedTargetOpt /* unsupported-target-opt */, 22423, /* DiagArray1033 */ 2927, 0, R"()")
DIAG_ENTRY(anonymous_244 /* unsupported-visibility */, 22446, /* DiagArray1034 */ 2930, 0, R"()")
DIAG_ENTRY(anonymous_254 /* unusable-partial-specialization */, 22469, /* DiagArray1035 */ 2932, 0, R"()")
DIAG_ENTRY(Unused /* unused */, 22501, 0, /* DiagSubGroup1036 */ 774, R"()")
DIAG_ENTRY(UnusedArgument /* unused-argument */, 22508, 0, 0, R"()")
DIAG_ENTRY(UnusedButSetParameter /* unused-but-set-parameter */, 22524, /* DiagArray1038 */ 2934, 0, R"()")
DIAG_ENTRY(UnusedButSetVariable /* unused-but-set-variable */, 22549, /* DiagArray1039 */ 2936, 0, R"()")
DIAG_ENTRY(UnusedCommandLineArgument /* unused-command-line-argument */, 22573, /* DiagArray1040 */ 2938, 0, R"()")
DIAG_ENTRY(UnusedComparison /* unused-comparison */, 22602, /* DiagArray1041 */ 2953, 0, R"()")
DIAG_ENTRY(UnusedConstVariable /* unused-const-variable */, 22620, /* DiagArray1042 */ 2955, 0, R"()")
DIAG_ENTRY(UnusedExceptionParameter /* unused-exception-parameter */, 22642, /* DiagArray1043 */ 2957, 0, R"()")
DIAG_ENTRY(UnusedFunction /* unused-function */, 22669, /* DiagArray1044 */ 2959, /* DiagSubGroup1044 */ 785, R"()")
DIAG_ENTRY(UnusedGetterReturnValue /* unused-getter-return-value */, 22685, /* DiagArray1045 */ 2961, 0, R"()")
DIAG_ENTRY(UnusedLabel /* unused-label */, 22712, /* DiagArray1046 */ 2963, 0, R"()")
DIAG_ENTRY(UnusedLambdaCapture /* unused-lambda-capture */, 22725, /* DiagArray1047 */ 2965, 0, R"()")
DIAG_ENTRY(UnusedLocalTypedef /* unused-local-typedef */, 22747, /* DiagArray1048 */ 2967, 0, R"()")
DIAG_ENTRY(anonymous_67 /* unused-local-typedefs */, 22768, 0, /* DiagSubGroup1049 */ 787, R"()")
DIAG_ENTRY(anonymous_131 /* unused-macros */, 22790, /* DiagArray1050 */ 2969, 0, R"()")
DIAG_ENTRY(UnusedMemberFunction /* unused-member-function */, 22804, /* DiagArray1051 */ 2971, /* DiagSubGroup1051 */ 789, R"()")
DIAG_ENTRY(UnusedParameter /* unused-parameter */, 22827, /* DiagArray1052 */ 2973, 0, R"()")
DIAG_ENTRY(UnusedPrivateField /* unused-private-field */, 22844, /* DiagArray1053 */ 2975, 0, R"()")
DIAG_ENTRY(UnusedPropertyIvar /* unused-property-ivar */, 22865, /* DiagArray1054 */ 2977, 0, R"()")
DIAG_ENTRY(UnusedResult /* unused-result */, 22886, /* DiagArray1055 */ 2979, 0, R"()")
DIAG_ENTRY(UnusedTemplate /* unused-template */, 22900, /* DiagArray1056 */ 2981, /* DiagSubGroup1056 */ 791, R"()")
DIAG_ENTRY(UnusedValue /* unused-value */, 22916, /* DiagArray1057 */ 2983, /* DiagSubGroup1057 */ 793, R"()")
DIAG_ENTRY(UnusedVariable /* unused-variable */, 22929, /* DiagArray1058 */ 2991, /* DiagSubGroup1058 */ 797, R"()")
DIAG_ENTRY(anonymous_309 /* unused-volatile-lvalue */, 22945, /* DiagArray1059 */ 2993, 0, R"()")
DIAG_ENTRY(UsedButMarkedUnused /* used-but-marked-unused */, 22968, /* DiagArray1060 */ 2995, 0, R"()")
DIAG_ENTRY(UserDefinedLiterals /* user-defined-literals */, 22991, /* DiagArray1061 */ 2997, 0, R"()")
DIAG_ENTRY(UserDefinedWarnings /* user-defined-warnings */, 23013, /* DiagArray1062 */ 2999, 0, R"()")
DIAG_ENTRY(Varargs /* varargs */, 23035, /* DiagArray1063 */ 3001, 0, R"()")
DIAG_ENTRY(VariadicMacroArgumentsOmitted /* variadic-macro-arguments-omitted */, 23043, /* DiagArray1064 */ 3006, 0, R"()")
DIAG_ENTRY(VariadicMacros /* variadic-macros */, 23076, /* DiagArray1065 */ 3010, 0, R"()")
DIAG_ENTRY(anonymous_224 /* vec-elem-size */, 23092, /* DiagArray1066 */ 3014, 0, R"()")
DIAG_ENTRY(VectorConversion /* vector-conversion */, 23106, /* DiagArray1067 */ 3016, 0, R"()")
DIAG_ENTRY(anonymous_66 /* vector-conversions */, 23124, 0, /* DiagSubGroup1068 */ 799, R"()")
DIAG_ENTRY(VexingParse /* vexing-parse */, 23143, /* DiagArray1069 */ 3018, 0, R"()")
DIAG_ENTRY(Visibility /* visibility */, 23156, /* DiagArray1070 */ 3022, 0, R"()")
DIAG_ENTRY(VLA /* vla */, 23167, /* DiagArray1071 */ 3025, /* DiagSubGroup1071 */ 801, R"()")
DIAG_ENTRY(VLACxxExtension /* vla-cxx-extension */, 23171, /* DiagArray1072 */ 3027, /* DiagSubGroup1072 */ 803, R"()")
DIAG_ENTRY(VLAExtension /* vla-extension */, 23189, /* DiagArray1073 */ 3030, /* DiagSubGroup1073 */ 805, R"()")
DIAG_ENTRY(VLAUseStaticAssert /* vla-extension-static-assert */, 23203, /* DiagArray1074 */ 3032, 0, R"()")
DIAG_ENTRY(VoidPointerToEnumCast /* void-pointer-to-enum-cast */, 23231, /* DiagArray1075 */ 3035, 0, R"()")
DIAG_ENTRY(VoidPointerToIntCast /* void-pointer-to-int-cast */, 23257, /* DiagArray1076 */ 3037, /* DiagSubGroup1076 */ 807, R"()")
DIAG_ENTRY(VoidPointerDeref /* void-ptr-dereference */, 23282, /* DiagArray1077 */ 3039, 0, R"()")
DIAG_ENTRY(VolatileRegisterVar /* volatile-register-var */, 23303, 0, 0, R"()")
DIAG_ENTRY(WebAssemblyExceptionSpec /* wasm-exception-spec */, 23325, /* DiagArray1079 */ 3041, 0, R"()")
DIAG_ENTRY(anonymous_214 /* weak-template-vtables */, 23345, /* DiagArray1080 */ 3043, 0, R"()")
DIAG_ENTRY(anonymous_213 /* weak-vtables */, 23367, /* DiagArray1081 */ 3045, 0, R"()")
DIAG_ENTRY(WritableStrings /* writable-strings */, 23380, /* DiagArray1082 */ 3047, /* DiagSubGroup1082 */ 809, R"()")
DIAG_ENTRY(GCCWriteStrings /* write-strings */, 23397, 0, /* DiagSubGroup1083 */ 811, R"(**Note:** enabling this warning in C will change the semantic behavior of the
program by treating all string literals as having type ``const char *``
instead of ``char *``. This can cause unexpected behaviors with type-sensitive
constructs like ``_Generic``.)")
DIAG_ENTRY(XorUsedAsPow /* xor-used-as-pow */, 23411, /* DiagArray1084 */ 3049, 0, R"()")
DIAG_ENTRY(anonymous_333 /* zero-as-null-pointer-constant */, 23427, /* DiagArray1085 */ 3053, 0, R"()")
DIAG_ENTRY(ZeroLengthArray /* zero-length-array */, 23457, /* DiagArray1086 */ 3055, 0, R"()")
#endif // DIAG_ENTRY


#ifdef GET_CATEGORY_TABLE
CATEGORY("", DiagCat_None)
CATEGORY("Lexical or Preprocessor Issue", DiagCat_Lexical_or_Preprocessor_Issue)
CATEGORY("Parse Issue", DiagCat_Parse_Issue)
CATEGORY("Semantic Issue", DiagCat_Semantic_Issue)
CATEGORY("Lambda Issue", DiagCat_Lambda_Issue)
CATEGORY("Verification", DiagCat_Verification)
CATEGORY("ARC Semantic Issue", DiagCat_ARC_Semantic_Issue)
CATEGORY("ARC and @properties", DiagCat_ARC_and__properties)
CATEGORY("ARC Casting Rules", DiagCat_ARC_Casting_Rules)
CATEGORY("ARC Parse Issue", DiagCat_ARC_Parse_Issue)
CATEGORY("ARC Weak References", DiagCat_ARC_Weak_References)
CATEGORY("ARC Restrictions", DiagCat_ARC_Restrictions)
CATEGORY("OpenMP Issue", DiagCat_OpenMP_Issue)
CATEGORY("Inline Assembly Issue", DiagCat_Inline_Assembly_Issue)
CATEGORY("AST Deserialization Issue", DiagCat_AST_Deserialization_Issue)
CATEGORY("Modules Issue", DiagCat_Modules_Issue)
CATEGORY("Codegen ABI Check", DiagCat_Codegen_ABI_Check)
CATEGORY("Coroutines Issue", DiagCat_Coroutines_Issue)
CATEGORY("Command line", DiagCat_Command_line)
CATEGORY("Concepts Issue", DiagCat_Concepts_Issue)
CATEGORY("Dependency Directive Source Scanner Issue", DiagCat_Dependency_Directive_Source_Scanner_Issue)
CATEGORY("Backend Issue", DiagCat_Backend_Issue)
CATEGORY("SourceMgr Reported Issue", DiagCat_SourceMgr_Reported_Issue)
CATEGORY("API Notes Issue", DiagCat_API_Notes_Issue)
CATEGORY("Related Result Type Issue", DiagCat_Related_Result_Type_Issue)
CATEGORY("AST Serialization Issue", DiagCat_AST_Serialization_Issue)
CATEGORY("Nullability Issue", DiagCat_Nullability_Issue)
CATEGORY("Generics Issue", DiagCat_Generics_Issue)
CATEGORY("User-Defined Issue", DiagCat_User_Defined_Issue)
CATEGORY("Refactoring Invocation Issue", DiagCat_Refactoring_Invocation_Issue)
CATEGORY("VTable ABI Issue", DiagCat_VTable_ABI_Issue)
CATEGORY("Value Conversion Issue", DiagCat_Value_Conversion_Issue)
CATEGORY("Documentation Issue", DiagCat_Documentation_Issue)
CATEGORY("ARC Retain Cycle", DiagCat_ARC_Retain_Cycle)
CATEGORY("Deprecations", DiagCat_Deprecations)
CATEGORY("Format String Issue", DiagCat_Format_String_Issue)
CATEGORY("Cocoa API Issue", DiagCat_Cocoa_API_Issue)
CATEGORY("#pragma message Directive", DiagCat__pragma_message_Directive)
CATEGORY("Instrumentation Issue", DiagCat_Instrumentation_Issue)
CATEGORY("Unused Entity Issue", DiagCat_Unused_Entity_Issue)
#endif // GET_CATEGORY_TABLE


RUN: diagtool list-warnings > %t 2>&1
RUN: FileCheck --input-file=%t %s

This test serves two purposes:

(1) It documents all existing warnings that currently have no associated -W flag,
    and ensures that the list never grows.

    If you take an existing warning and add a flag, this test will fail.
    To fix this test, simply remove that warning from the list below.

(2) It prevents us adding new warnings to Clang that have no -W flag.  All
    new warnings should have -W flags.

    If you add a new warning without a flag, this test will fail.  To fix
    this test, simply add a warning group to that warning.


The list of warnings below should NEVER grow.  It should gradually shrink to 0.

CHECK: Warnings without flags (67):

CHECK-NEXT:   ext_expected_semi_decl_list
CHECK-NEXT:   ext_explicit_specialization_storage_class
CHECK-NEXT:   ext_missing_whitespace_after_macro_name
CHECK-NEXT:   ext_new_paren_array_nonconst
CHECK-NEXT:   ext_plain_complex
CHECK-NEXT:   ext_template_arg_extra_parens
CHECK-NEXT:   ext_typecheck_cond_incompatible_operands
CHECK-NEXT:   ext_typecheck_ordered_comparison_of_pointer_integer
CHECK-NEXT:   ext_using_undefined_std
CHECK-NEXT:   pp_invalid_string_literal
CHECK-NEXT:   pp_out_of_date_dependency
CHECK-NEXT:   pp_poisoning_existing_macro
CHECK-NEXT:   warn_accessor_property_type_mismatch
CHECK-NEXT:   warn_analyzer_deprecated_option
CHECK-NEXT:   warn_arcmt_nsalloc_realloc
CHECK-NEXT:   warn_asm_label_on_auto_decl
CHECK-NEXT:   warn_c_kext
CHECK-NEXT:   warn_call_wrong_number_of_arguments
CHECK-NEXT:   warn_case_empty_range
CHECK-NEXT:   warn_char_constant_too_large
CHECK-NEXT:   warn_collection_expr_type
CHECK-NEXT:   warn_conflicting_variadic
CHECK-NEXT:   warn_delete_array_type
CHECK-NEXT:   warn_double_const_requires_fp64
CHECK-NEXT:   warn_drv_assuming_mfloat_abi_is
CHECK-NEXT:   warn_drv_clang_unsupported
CHECK-NEXT:   warn_drv_pch_not_first_include
CHECK-NEXT:   warn_dup_category_def
CHECK-NEXT:   warn_enum_value_overflow
CHECK-NEXT:   warn_expected_qualified_after_typename
CHECK-NEXT:   warn_fe_backend_unsupported
CHECK-NEXT:   warn_fe_cc_log_diagnostics_failure
CHECK-NEXT:   warn_fe_cc_print_header_failure
CHECK-NEXT:   warn_fe_macro_contains_embedded_newline
CHECK-NEXT:   warn_ignoring_ftabstop_value
CHECK-NEXT:   warn_implements_nscopying
CHECK-NEXT:   warn_incompatible_qualified_id
CHECK-NEXT:   warn_invalid_asm_cast_lvalue
CHECK-NEXT:   warn_maynot_respond
CHECK-NEXT:   warn_method_param_redefinition
CHECK-NEXT:   warn_missing_case_for_condition
CHECK-NEXT:   warn_missing_dependent_template_keyword
CHECK-NEXT:   warn_missing_whitespace_after_macro_name
CHECK-NEXT:   warn_mt_message
CHECK-NEXT:   warn_no_constructor_for_refconst
CHECK-NEXT:   warn_not_compound_assign
CHECK-NEXT:   warn_objc_property_copy_missing_on_block
CHECK-NEXT:   warn_objc_protocol_qualifier_missing_id
CHECK-NEXT:   warn_on_superclass_use
CHECK-NEXT:   warn_pp_convert_to_positive
CHECK-NEXT:   warn_pp_expr_overflow
CHECK-NEXT:   warn_pp_line_decimal
CHECK-NEXT:   warn_pragma_pack_pop_identifier_and_alignment
CHECK-NEXT:   warn_pragma_pack_show
CHECK-NEXT:   warn_property_getter_owning_mismatch
CHECK-NEXT:   warn_register_objc_catch_parm
CHECK-NEXT:   warn_related_result_type_compatibility_class
CHECK-NEXT:   warn_related_result_type_compatibility_protocol
CHECK-NEXT:   warn_template_export_unsupported
CHECK-NEXT:   warn_template_spec_extra_headers
CHECK-NEXT:   warn_tentative_incomplete_array
CHECK-NEXT:   warn_typecheck_function_qualifiers
CHECK-NEXT:   warn_undef_interface
CHECK-NEXT:   warn_undef_interface_suggest
CHECK-NEXT:   warn_undef_protocolref
CHECK-NEXT:   warn_weak_identifier_undeclared
CHECK-NEXT:   warn_weak_import

The list of warnings in -Wpedantic should NEVER grow.

CHECK: Number in -Wpedantic (not covered by other -W flags): 26

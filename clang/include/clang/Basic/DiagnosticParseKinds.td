//==--- DiagnosticParseKinds.td - libparse diagnostics --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parser Diagnostics
//===----------------------------------------------------------------------===//

let Component = "Parse" in {

def w_asm_qualifier_ignored : Warning<"ignored %0 qualifier on asm">,
                              CatInlineAsm;
def warn_file_asm_volatile : Warning<
  "meaningless 'volatile' on asm outside function">, CatInlineAsm;

let CategoryName = "Parse Issue" in {

def ext_empty_source_file : Extension<"ISO C forbids an empty source file">;
def ext_top_level_semi : Extension<
  "extra ';' outside of a function">;
def warn_cxx98_compat_top_level_semi : Warning<
  "extra ';' outside of a function is incompatible with C++98">,
  InGroup<CXX98CompatPedantic>, DefaultIgnore;
def ext_extra_struct_semi : Extension<
  "extra ';' inside a %0">;
def ext_extra_ivar_semi : Extension<
  "extra ';' inside instance variable list">;

def ext_duplicate_declspec : Extension<"duplicate '%0' declaration specifier">;
def ext_plain_complex : ExtWarn<
  "plain '_Complex' requires a type specifier; assuming '_Complex double'">;
def ext_integer_complex : Extension<
  "complex integer types are an extension">;
def ext_thread_before : Extension<"'__thread' before 'static'">;

def ext_empty_struct_union : Extension<
  "empty %select{struct|union}0 is a GNU extension">, InGroup<GNU>;
def warn_empty_struct_union_compat : Warning<"empty %select{struct|union}0 "
  "has size 0 in C, size 1 in C++">, InGroup<CXXCompat>, DefaultIgnore;
def error_empty_enum : Error<"use of empty enum">;
def err_invalid_sign_spec : Error<"'%0' cannot be signed or unsigned">;
def err_invalid_short_spec : Error<"'short %0' is invalid">;
def err_invalid_long_spec : Error<"'long %0' is invalid">;
def err_invalid_longlong_spec : Error<"'long long %0' is invalid">;
def err_invalid_complex_spec : Error<"'_Complex %0' is invalid">;
def err_friend_storage_spec : Error<"'%0' is invalid in friend declarations">;

def ext_ident_list_in_param : Extension<
  "type-less parameter names in function declaration">;
def ext_c99_variable_decl_in_for_loop : Extension<
  "variable declaration in for loop is a C99-specific feature">, InGroup<C99>;
def ext_c99_compound_literal : Extension<
  "compound literals are a C99-specific feature">, InGroup<C99>;
def ext_c99_flexible_array_member : Extension<
  "Flexible array members are a C99-specific feature">, InGroup<C99>;
def ext_enumerator_list_comma : Extension<
  "commas at the end of enumerator lists are a %select{C99|C++11}0-specific "
  "feature">;
def warn_cxx98_compat_enumerator_list_comma : Warning<
  "commas at the end of enumerator lists are incompatible with C++98">,
  InGroup<CXX98CompatPedantic>, DefaultIgnore;
def err_enumerator_list_missing_comma : Error<
  "missing ',' between enumerators">;
def err_enumerator_unnamed_no_def : Error<
  "unnamed enumeration must be a definition">;
def ext_ms_enum_fixed_underlying_type : Extension<
  "enumeration types with a fixed underlying type are a Microsoft extension">, 
  InGroup<Microsoft>;
def warn_cxx98_compat_enum_fixed_underlying_type : Warning<
  "enumeration types with a fixed underlying type are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def warn_cxx98_compat_alignof : Warning<
  "alignof expressions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

def warn_microsoft_dependent_exists : Warning<
  "dependent %select{__if_not_exists|__if_exists}0 declarations are ignored">, 
  InGroup<DiagGroup<"microsoft-exists">>;

def ext_c11_generic_selection : Extension<
  "generic selections are a C11-specific feature">, InGroup<C11>;
def err_duplicate_default_assoc : Error<
  "duplicate default generic association">;
def note_previous_default_assoc : Note<
  "previous default generic association is here">;

def ext_c11_alignas : Extension<
  "_Alignas is a C11-specific feature">, InGroup<C11>;

def ext_gnu_indirect_goto : Extension<
  "use of GNU indirect-goto extension">, InGroup<GNU>;
def ext_gnu_address_of_label : Extension<
  "use of GNU address-of-label extension">, InGroup<GNU>;
def ext_gnu_local_label : Extension<
  "use of GNU locally declared label extension">, InGroup<GNU>;
def ext_gnu_statement_expr : Extension<
  "use of GNU statement expression extension">, InGroup<GNU>;
def ext_gnu_conditional_expr : Extension<
  "use of GNU ?: expression extension, eliding middle term">, InGroup<GNU>;
def ext_gnu_empty_initializer : Extension<
  "use of GNU empty initializer extension">, InGroup<GNU>;
def ext_gnu_array_range : Extension<"use of GNU array range extension">, 
  InGroup<GNUDesignator>;
def ext_gnu_missing_equal_designator : ExtWarn<
  "use of GNU 'missing =' extension in designator">, 
  InGroup<GNUDesignator>;
def err_expected_equal_designator : Error<"expected '=' or another designator">;
def ext_gnu_old_style_field_designator : ExtWarn<
  "use of GNU old-style field designator extension">, 
  InGroup<GNUDesignator>;
def ext_gnu_case_range : Extension<"use of GNU case range extension">,
  InGroup<GNU>;

// Generic errors.
def err_expected_expression : Error<"expected expression">;
def err_expected_type : Error<"expected a type">;
def err_expected_external_declaration : Error<"expected external declaration">;
def err_extraneous_closing_brace : Error<"extraneous closing brace ('}')">;
def err_expected_ident : Error<"expected identifier">;
def err_expected_ident_lparen : Error<"expected identifier or '('">;
def err_expected_ident_lbrace : Error<"expected identifier or '{'">;
def err_expected_lbrace : Error<"expected '{'">;
def err_expected_lparen : Error<"expected '('">;
def err_expected_lparen_or_lbrace : Error<"expected '('or '{'">;
def err_expected_rparen : Error<"expected ')'">;
def err_expected_lsquare : Error<"expected '['">;
def err_expected_rsquare : Error<"expected ']'">;
def err_expected_rbrace : Error<"expected '}'">;
def err_expected_greater : Error<"expected '>'">;
def err_expected_ggg : Error<"expected '>>>'">;
def err_expected_semi_declaration : Error<
  "expected ';' at end of declaration">;
def err_expected_semi_decl_list : Error<
  "expected ';' at end of declaration list">;
def ext_expected_semi_decl_list : ExtWarn<
  "expected ';' at end of declaration list">;
def err_expected_member_name_or_semi : Error<
  "expected member name or ';' after declaration specifiers">;
def err_function_declared_typedef : Error<
  "function definition declared 'typedef'">;
def err_iboutletcollection_builtintype : Error<
  "type argument of iboutletcollection attribute cannot be a builtin type">;
def err_iboutletcollection_with_protocol : Error<
  "invalid argument of iboutletcollection attribute">;
def err_at_defs_cxx : Error<"@defs is not supported in Objective-C++">;
def err_at_in_class : Error<"unexpected '@' in member specification">;

def err_expected_fn_body : Error<
  "expected function body after function declarator">;
def warn_attribute_on_function_definition : Warning<
  "GCC does not allow %0 attribute in this position on a function definition">, 
  InGroup<GccCompat>;
def warn_attribute_no_decl : Warning<
  "attribute %0 ignored, because it is not attached to a declaration">, 
  InGroup<IgnoredAttributes>;
def err_expected_method_body : Error<"expected method body">;
def err_invalid_token_after_toplevel_declarator : Error<
  "expected ';' after top level declarator">;
def err_invalid_token_after_declarator_suggest_equal : Error<
  "invalid '%0' at end of declaration; did you mean '='?">;
def err_expected_statement : Error<"expected statement">;
def err_expected_lparen_after : Error<"expected '(' after '%0'">;
def err_expected_lparen_after_id : Error<"expected '(' after %0">;
def err_expected_less_after : Error<"expected '<' after '%0'">;
def err_expected_equal_after : Error<"expected '=' after %0">;
def err_expected_comma : Error<"expected ','">;
def err_expected_lbrace_in_compound_literal : Error<
  "expected '{' in compound literal">;
def err_expected_while : Error<"expected 'while' in do/while loop">;

def err_expected_semi_after : Error<"expected ';' after %0">;
def err_expected_semi_after_stmt : Error<"expected ';' after %0 statement">;
def err_expected_semi_after_expr : Error<"expected ';' after expression">;
def err_extraneous_token_before_semi : Error<"extraneous '%0' before ';'">;

def err_expected_semi_after_method_proto : Error<
  "expected ';' after method prototype">;
def err_expected_semi_after_namespace_name : Error<
  "expected ';' after namespace name">;
def err_unexpected_namespace_attributes_alias : Error<
  "attributes can not be specified on namespace alias">;
def err_inline_namespace_alias : Error<"namespace alias cannot be inline">;
def err_namespace_nonnamespace_scope : Error<
  "namespaces can only be defined in global or namespace scope">;
def err_nested_namespaces_with_double_colon : Error<
  "nested namespace definition must define each namespace separately">;
def err_expected_semi_after_attribute_list : Error<
  "expected ';' after attribute list">;
def err_expected_semi_after_static_assert : Error<
  "expected ';' after static_assert">;
def err_expected_semi_for : Error<"expected ';' in 'for' statement specifier">;
def err_expected_colon_after : Error<"expected ':' after %0">;
def err_label_end_of_compound_statement : Error<
  "label at end of compound statement: expected statement">;
def err_address_of_label_outside_fn : Error<
  "use of address-of-label extension outside of a function body">;
def err_expected_string_literal : Error<"expected string literal">;
def err_asm_operand_wide_string_literal : Error<
  "cannot use %select{unicode|wide}0 string literal in 'asm'">;
def err_expected_selector_for_method : Error<
  "expected selector for Objective-C method">;
def err_expected_property_name : Error<"expected property name">;

def err_unexpected_at : Error<"unexpected '@' in program">;

def err_invalid_reference_qualifier_application : Error<
  "'%0' qualifier may not be applied to a reference">;
def err_illegal_decl_reference_to_reference : Error<
  "%0 declared as a reference to a reference">;
def ext_rvalue_reference : ExtWarn<
  "rvalue references are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_rvalue_reference : Warning<
  "rvalue references are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def ext_ref_qualifier : ExtWarn<
  "reference qualifiers on functions are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_ref_qualifier : Warning<
  "reference qualifiers on functions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def ext_inline_namespace : ExtWarn<
  "inline namespaces are a C++11 feature">, InGroup<CXX11>;
def warn_cxx98_compat_inline_namespace : Warning<
  "inline namespaces are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def ext_generalized_initializer_lists : ExtWarn<
  "generalized initializer lists are a C++11 extension">,
  InGroup<CXX11>;
def warn_cxx98_compat_generalized_initializer_lists : Warning<
  "generalized initializer lists are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_init_list_bin_op : Error<"initializer list cannot be used on the "
  "%select{left|right}0 hand side of operator '%1'">;
def warn_cxx98_compat_trailing_return_type : Warning<
  "trailing return types are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def ext_auto_type_specifier : ExtWarn<
  "'auto' type specifier is a C++11 extension">, InGroup<CXX11>;
def warn_auto_storage_class : Warning<
  "'auto' storage class specifier is redundant and incompatible with C++11">,
  InGroup<CXX11Compat>, DefaultIgnore;
def ext_auto_storage_class : ExtWarn<
  "'auto' storage class specifier is not permitted in C++11, and will not "
  "be supported in future releases">, InGroup<DiagGroup<"auto-storage-class">>;
def ext_for_range : ExtWarn<
  "range-based for loop is a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_for_range : Warning<
  "range-based for loop is incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_for_range_expected_decl : Error<
  "for range declaration must declare a variable">;
def err_argument_required_after_attribute : Error<
  "argument required after attribute">;
def err_missing_param : Error<"expected parameter declarator">;
def err_missing_comma_before_ellipsis : Error<
  "C requires a comma prior to the ellipsis in a variadic function type">;
def err_unexpected_typedef_ident : Error<
  "unexpected type name %0: expected identifier">;
def warn_cxx98_compat_decltype : Warning<
  "'decltype' type specifier is incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_unexpected_scope_on_base_decltype : Error<
  "unexpected namespace scope prior to decltype">;
def err_expected_class_name : Error<"expected class name">;
def err_expected_class_name_not_template : 
  Error<"'typename' is redundant; base classes are implicitly types">;
def err_unspecified_vla_size_with_static : Error<
  "'static' may not be used with an unspecified variable length array size">;

def err_expected_case_before_expression: Error<
  "expected 'case' keyword before expression">;

// Declarations.
def err_typename_requires_specqual : Error<
  "type name requires a specifier or qualifier">;
def err_typename_invalid_storageclass : Error<
  "type name does not allow storage class to be specified">;
def err_typename_invalid_functionspec : Error<
  "type name does not allow function specifier to be specified">;
def err_typename_invalid_constexpr : Error<
  "type name does not allow constexpr specifier to be specified">;
def err_typename_identifiers_only : Error<
  "typename is allowed for identifiers only">;

def err_invalid_decl_spec_combination : Error<
  "cannot combine with previous '%0' declaration specifier">;
def err_invalid_vector_decl_spec_combination : Error<
  "cannot combine with previous '%0' declaration specifier. "
  "'__vector' must be first">;
def err_invalid_pixel_decl_spec_combination : Error<
  "'__pixel' must be preceded by '__vector'.  "
  "'%0' declaration specifier not allowed here">;
def err_invalid_vector_decl_spec : Error<
  "cannot use '%0' with '__vector'">;
def err_invalid_vector_bool_decl_spec : Error<
  "cannot use '%0' with '__vector bool'">;
def warn_vector_long_decl_spec_combination : Warning<
  "Use of 'long' with '__vector' is deprecated">, InGroup<Deprecated>;
def err_friend_invalid_in_context : Error<
  "'friend' used outside of class">;
def err_unknown_typename : Error<
  "unknown type name %0">;
def err_use_of_tag_name_without_tag : Error<
  "must use '%1' tag to refer to type %0%select{| in this scope}2">;
def err_templated_using_directive : Error<
  "cannot template a using directive">;
def err_templated_using_declaration : Error<
  "cannot template a using declaration">;
def err_unexected_colon_in_nested_name_spec : Error<
  "unexpected ':' in nested name specifier">;
def err_bool_redeclaration : Error<
  "redeclaration of C++ built-in type 'bool'">;
def ext_c11_static_assert : Extension<
  "_Static_assert is a C11-specific feature">, InGroup<C11>;
def warn_cxx98_compat_static_assert : Warning<
  "static_assert declarations are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

/// Objective-C parser diagnostics
def err_expected_minus_or_plus : Error<
  "method type specifier must start with '-' or '+'">;
def err_objc_no_attributes_on_category : Error<
  "attributes may not be specified on a category">;
def err_objc_missing_end : Error<"missing '@end'">;
def note_objc_container_start : Note<
  "%select{class|protocol|category|class extension|implementation"
  "|category implementation}0 started here">;
def warn_objc_protocol_qualifier_missing_id : Warning<
  "protocol qualifiers without 'id' is archaic">;
def err_objc_unknown_at : Error<"expected an Objective-C directive after '@'">;
def err_illegal_super_cast : Error<
  "cannot cast 'super' (it isn't an expression)">;
def err_nsnumber_nonliteral_unary : Error<
  "@%0 must be followed by a number to form an NSNumber object">;

let CategoryName = "ARC Parse Issue" in {
def err_arc_bridge_retain : Error<
  "unknown cast annotation __bridge_retain; did you mean __bridge_retained?">;
// To be default mapped to an error later.
def warn_arc_bridge_cast_nonarc : Warning<
  "'%0' casts have no effect when not using ARC">,
  InGroup<DiagGroup<"arc-bridge-casts-disallowed-in-nonarc">>;
}
  
def err_objc_illegal_visibility_spec : Error<
  "illegal visibility specification">;
def err_objc_illegal_interface_qual : Error<"illegal interface qualifier">;
def err_objc_expected_equal_for_getter : Error<
  "expected '=' for Objective-C getter">;
def err_objc_expected_equal_for_setter : Error<
  "expected '=' for Objective-C setter">;
def err_objc_expected_selector_for_getter_setter : Error<
  "expected selector for Objective-C %select{setter|getter}0">;
def err_objc_property_requires_field_name : Error<
  "property requires fields to be named">;
def err_objc_property_bitfield : Error<"property name cannot be a bitfield">;
def err_objc_expected_property_attr : Error<"unknown property attribute %0">;
def err_objc_properties_require_objc2 : Error<
  "properties are an Objective-C 2 feature">;
def err_objc_unexpected_attr : Error<
  "prefix attribute must be followed by an interface or protocol">;
def err_objc_directive_only_in_protocol : Error<
  "directive may only be specified in protocols only">;
def err_missing_catch_finally : Error<
  "@try statement without a @catch and @finally clause">;
def err_objc_concat_string : Error<"unexpected token after Objective-C string">;
def err_expected_objc_container : Error<
  "'@end' must appear in an Objective-C context">;
def error_property_ivar_decl : Error<
  "property synthesize requires specification of an ivar">;
def err_synthesized_property_name : Error<
  "expected a property name in @synthesize">;
def warn_semicolon_before_method_body : Warning<
  "semicolon before method body is ignored">,
  InGroup<DiagGroup<"semicolon-before-method-body">>, DefaultIgnore;

def err_expected_field_designator : Error<
  "expected a field designator, such as '.field = 4'">;

def err_declaration_does_not_declare_param : Error<
  "declaration does not declare a parameter">;
def err_no_matching_param : Error<"parameter named %0 is missing">;

/// C++ parser diagnostics
def err_expected_unqualified_id : Error<
  "expected %select{identifier|unqualified-id}0">;
def err_func_def_no_params : Error<
  "function definition does not declare parameters">;
def err_expected_lparen_after_type : Error<
  "expected '(' for function-style cast or type construction">;
def err_expected_init_in_condition : Error<
  "variable declaration in condition must have an initializer">;
def err_expected_init_in_condition_lparen : Error<
  "variable declaration in condition cannot have a parenthesized initializer">;
def warn_parens_disambiguated_as_function_decl : Warning<
  "parentheses were disambiguated as a function declarator">,
  InGroup<VexingParse>;
def warn_dangling_else : Warning<
  "add explicit braces to avoid dangling else">,
  InGroup<DanglingElse>;
def err_expected_member_or_base_name : Error<
  "expected class member or base class name">;
def err_expected_lbrace_after_base_specifiers : Error<
  "expected '{' after base class list">;
def ext_ellipsis_exception_spec : Extension<
  "exception specification of '...' is a Microsoft extension">;
def err_dynamic_and_noexcept_specification : Error<
  "cannot have both throw() and noexcept() clause on the same function">;
def warn_cxx98_compat_noexcept_decl : Warning<
  "noexcept specifications are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_expected_catch : Error<"expected catch">;
def err_expected_lbrace_or_comma : Error<"expected '{' or ','">;
def err_expected_rbrace_or_comma : Error<"expected '}' or ','">;
def err_expected_rsquare_or_comma : Error<"expected ']' or ','">;
def err_using_namespace_in_class : Error<
  "'using namespace' is not allowed in classes">;
def err_destructor_tilde_identifier : Error<
  "expected a class name after '~' to name a destructor">;
def err_destructor_template_id : Error<
  "destructor name %0 does not refer to a template">;
def err_default_arg_unparsed : Error<
  "unexpected end of default argument expression">;
def err_parser_impl_limit_overflow : Error<
  "parser recursion limit reached, program too complex">, DefaultFatal;
def err_misplaced_ellipsis_in_declaration : Error<
  "'...' must %select{immediately precede declared identifier|"
  "be innermost component of anonymous pack declaration}0">;

// C++ derived classes
def err_dup_virtual : Error<"duplicate 'virtual' in base specifier">;

// C++ operator overloading
def err_literal_operator_string_prefix : Error<
  "string literal after 'operator' cannot have an encoding prefix">;
def err_literal_operator_string_not_empty : Error<
  "string literal after 'operator' must be '\"\"'">;
def err_literal_operator_missing_space : Error<
  "C++11 requires a space between the \"\" and the user-defined suffix in a "
  "literal operator">;
def warn_cxx98_compat_literal_operator : Warning<
  "literal operators are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

// Classes.
def err_anon_type_definition : Error<
  "declaration of anonymous %0 must be a definition">;
def err_default_delete_in_multiple_declaration : Error<
  "'= %select{default|delete}0' is a function definition and must occur in a "
  "standalone declaration">;

def warn_cxx98_compat_noexcept_expr : Warning<
  "noexcept expressions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def warn_cxx98_compat_nullptr : Warning<
  "'nullptr' is incompatible with C++98">, InGroup<CXX98Compat>, DefaultIgnore;

def warn_cxx98_compat_alignas : Warning<"'alignas' is incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def warn_cxx98_compat_attribute : Warning<
  "attributes are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_cxx11_attribute_forbids_arguments : Error<
  "attribute '%0' cannot have an argument list">;
def err_cxx11_attribute_forbids_ellipsis : Error<
  "attribute '%0' cannot be used as an attribute pack">;
def err_attributes_not_allowed : Error<"an attribute list cannot appear here">;
def err_l_square_l_square_not_attribute : Error<
  "C++11 only allows consecutive left square brackets when "
  "introducing an attribute">;
def err_alignas_pack_exp_unsupported : Error<
  "pack expansions in alignment specifiers are not supported yet">;

/// C++ Templates
def err_expected_template : Error<"expected template">;
def err_unknown_template_name : Error<
  "unknown template name %0">;
def err_expected_comma_greater : Error<
  "expected ',' or '>' in template-parameter-list">;
def err_class_on_template_template_param : Error<
  "template template parameter requires 'class' after the parameter list">;
def err_template_spec_syntax_non_template : Error<
  "identifier followed by '<' indicates a class template specialization but "
  "%0 %select{does not refer to a template|refers to a function "
  "template|<unused>|refers to a template template parameter}1">;
def err_id_after_template_in_nested_name_spec : Error<
  "expected template name after 'template' keyword in nested name specifier">;
def err_two_right_angle_brackets_need_space : Error<
  "a space is required between consecutive right angle brackets (use '> >')">;
def warn_cxx0x_right_shift_in_template_arg : Warning<
  "use of right-shift operator ('>>') in template argument will require "
  "parentheses in C++11">;
def warn_cxx98_compat_two_right_angle_brackets : Warning<
  "consecutive right angle brackets are incompatible with C++98 (use '> >')">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_multiple_template_declarators : Error<
    "%select{|a template declaration|an explicit template specialization|"
    "an explicit template instantiation}0 can "
    "only %select{|declare|declare|instantiate}0 a single entity">;
def err_explicit_instantiation_with_definition : Error<
    "explicit template instantiation cannot have a definition; if this "
    "definition is meant to be an explicit specialization, add '<>' after the "
    "'template' keyword">;
def err_enum_template : Error<"enumeration cannot be a template">;
def err_explicit_instantiation_enum : Error<
    "enumerations cannot be explicitly instantiated">;
def err_expected_template_parameter : Error<"expected template parameter">;

def err_missing_dependent_template_keyword : Error<
  "use 'template' keyword to treat '%0' as a dependent template name">;
def warn_missing_dependent_template_keyword : ExtWarn<
  "use 'template' keyword to treat '%0' as a dependent template name">;

def ext_extern_template : Extension<
  "extern templates are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_extern_template : Warning<
  "extern templates are incompatible with C++98">,
  InGroup<CXX98CompatPedantic>, DefaultIgnore;
def warn_static_inline_explicit_inst_ignored : Warning<
  "ignoring '%select{static|inline}0' keyword on explicit template "
  "instantiation">;
  
// Constructor template diagnostics.
def err_out_of_line_constructor_template_id : Error<
  "out-of-line constructor for %0 cannot have template arguments">;
def err_out_of_line_template_id_names_constructor : Error<
  "qualified reference to %0 is a constructor name rather than a "
  "template name wherever a constructor can be declared">;
def err_out_of_line_type_names_constructor : Error<
  "qualified reference to %0 is a constructor name rather than a "
  "type wherever a constructor can be declared">;

def err_expected_qualified_after_typename : Error<
  "expected a qualified name after 'typename'">;
def warn_expected_qualified_after_typename : ExtWarn<
  "expected a qualified name after 'typename'">;
def err_expected_semi_after_tagdecl : Error<
  "expected ';' after %0">;

def err_typename_refers_to_non_type_template : Error<
  "typename specifier refers to a non-template">;
def err_expected_type_name_after_typename : Error<
  "expected an identifier or template-id after '::'">;
def err_explicit_spec_non_template : Error<
  "explicit %select{specialization|instantiation}0 of non-template "
  "%select{class|struct|union}1 %2">;
  
def err_default_template_template_parameter_not_template : Error<
  "default template argument for a template template parameter must be a class "
  "template">;
  
def err_ctor_init_missing_comma : Error<
  "missing ',' between base or member initializers">;

// C++ declarations
def err_friend_decl_defines_type : Error<
  "cannot define a type in a friend declaration">;
def err_missing_whitespace_digraph : Error<
  "found '<::' after a "
  "%select{template name|const_cast|dynamic_cast|reinterpret_cast|static_cast}0"
  " which forms the digraph '<:' (aka '[') and a ':', did you mean '< ::'?">;

def ext_deleted_function : ExtWarn<
  "deleted function definitions are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_deleted_function : Warning<
  "deleted function definitions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def ext_defaulted_function : ExtWarn<
  "defaulted function definitions are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_defaulted_function : Warning<
  "defaulted function definitions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

// C++11 in-class member initialization
def ext_nonstatic_member_init : ExtWarn<
  "in-class initialization of non-static data member is a C++11 extension">,
  InGroup<CXX11>;
def warn_cxx98_compat_nonstatic_member_init : Warning<
  "in-class initialization of non-static data members is incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_bitfield_member_init: Error<
  "bitfield member cannot have an in-class initializer">;
def err_incomplete_array_member_init: Error<
  "array bound cannot be deduced from an in-class initializer">;

// C++11 alias-declaration
def ext_alias_declaration : ExtWarn<
  "alias declarations are a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_alias_declaration : Warning<
  "alias declarations are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_alias_declaration_not_identifier : Error<
  "name defined in alias declaration must be an identifier">;
def err_alias_declaration_specialization : Error<
  "%select{partial specialization|explicit specialization|explicit instantiation}0 of alias templates is not permitted">;
    
// C++11 override control
def ext_override_control_keyword : ExtWarn<
  "'%0' keyword is a C++11 extension">, InGroup<CXX11>;
def warn_cxx98_compat_override_control_keyword : Warning<
  "'%0' keyword is incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

def err_duplicate_virt_specifier : Error<
  "class member already marked '%0'">;

def err_scoped_enum_missing_identifier : Error<
  "scoped enumeration requires a name">;
def warn_cxx98_compat_scoped_enum : Warning<
  "scoped enumerations are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;

def err_expected_parameter_pack : Error<
  "expected the name of a parameter pack">;
def err_paren_sizeof_parameter_pack : Error<
  "missing parentheses around the size of parameter pack %0">;
def err_sizeof_parameter_pack : Error<
  "expected parenthesized parameter pack name in 'sizeof...' expression">;

// C++11 lambda expressions
def err_expected_comma_or_rsquare : Error<
  "expected ',' or ']' in lambda capture list">;
def err_this_captured_by_reference : Error<
  "'this' cannot be captured by reference">;
def err_expected_capture : Error<
  "expected variable name or 'this' in lambda capture list">;
def err_expected_lambda_body : Error<"expected body of lambda expression">;
def warn_cxx98_compat_lambda : Warning<
  "lambda expressions are incompatible with C++98">,
  InGroup<CXX98Compat>, DefaultIgnore;
def err_lambda_missing_parens : Error<
  "lambda requires '()' before %select{'mutable'|return type}0">;

// Availability attribute
def err_expected_version : Error<
  "expected a version of the form 'major[.minor[.subminor]]'">;
def err_zero_version : Error<
  "version number must have non-zero major, minor, or sub-minor version">;
def err_availability_expected_platform : Error<
  "expected a platform name, e.g., 'macosx'">;
def err_availability_expected_change : Error<
  "expected 'introduced', 'deprecated', or 'obsoleted'">;
def err_availability_unknown_change : Error<
  "%0 is not an availability stage; use 'introduced', 'deprecated', or "
  "'obsoleted'">;
def err_availability_redundant : Error<
  "redundant %0 availability change; only the last specified change will "        "be used">;
def warn_availability_and_unavailable : Warning<
  "'unavailable' availability overrides all other availability information">;

// Language specific pragmas
// - Generic warnings
def warn_pragma_expected_lparen : Warning<
  "missing '(' after '#pragma %0' - ignoring">;
def warn_pragma_expected_rparen : Warning<
  "missing ')' after '#pragma %0' - ignoring">;
def warn_pragma_expected_identifier : Warning<
  "expected identifier in '#pragma %0' - ignored">;  
def warn_pragma_ms_struct : Warning<
  "incorrect use of '#pragma ms_struct on|off' - ignored">;  
def warn_pragma_extra_tokens_at_eol : Warning<
  "extra tokens at end of '#pragma %0' - ignored">; 
// - #pragma options
def warn_pragma_options_expected_align : Warning<
  "expected 'align' following '#pragma options' - ignored">;
def warn_pragma_align_expected_equal : Warning<
  "expected '=' following '#pragma %select{align|options align}0' - ignored">;
def warn_pragma_align_invalid_option : Warning<
  "invalid alignment option in '#pragma %select{align|options align}0' - ignored">;
// - #pragma pack
def warn_pragma_pack_invalid_action : Warning<
  "unknown action for '#pragma pack' - ignored">;
def warn_pragma_pack_malformed : Warning<
  "expected integer or identifier in '#pragma pack' - ignored">;
// - #pragma unused
def warn_pragma_unused_expected_var : Warning<
  "expected '#pragma unused' argument to be a variable name">;
def warn_pragma_unused_expected_punc : Warning<
  "expected ')' or ',' in '#pragma unused'">;

// OpenCL Section 6.8.g
def err_not_opencl_storage_class_specifier : Error<
  "OpenCL does not support the '%0' storage class specifier">;

// OpenCL EXTENSION pragma (OpenCL 1.1 [9.1])
def warn_pragma_expected_colon : Warning<
  "missing ':' after %0 - ignoring">;
def warn_pragma_expected_enable_disable : Warning<
  "expected 'enable' or 'disable' - ignoring">;
def warn_pragma_unknown_extension : Warning<
  "unknown OpenCL extension %0 - ignoring">;

def err_seh_expected_handler : Error<
  "expected '__except' or '__finally' block">;

def err_seh___except_block : Error<
  "%0 only allowed in __except block">;

def err_seh___except_filter : Error<
  "%0 only allowed in __except filter expression">;

def err_seh___finally_block : Error<
  "%0 only allowed in __finally block">;
  
} // end of Parse Issue category.

let CategoryName = "Modules Issue" in {
def err_module_expected_ident : Error<
  "expected a module name after module import">;
def err_module_expected_semi : Error<
  "expected a semicolon name after module name">;
}

} // end of Parser diagnostics

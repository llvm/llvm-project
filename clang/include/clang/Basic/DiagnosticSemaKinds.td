//==--- DiagnosticSemaKinds.td - libsema diagnostics ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Semantic Analysis
//===----------------------------------------------------------------------===//

let Component = "Sema" in {
let CategoryName = "Semantic Issue" in {

// Constant expressions
def err_expr_not_ice : Error<
  "expression is not an integer constant expression">;
def ext_expr_not_ice : Extension<
  "expression is not integer constant expression "
  "(but is allowed as an extension)">;

// Semantic analysis of constant literals.
def ext_predef_outside_function : Warning<
  "predefined identifier is only valid inside function">;
def warn_float_overflow : Warning<
  "magnitude of floating-point constant too large for type %0; maximum is %1">,
   InGroup<LiteralRange>;
def warn_float_underflow : Warning<
  "magnitude of floating-point constant too small for type %0; minimum is %1">,
  InGroup<LiteralRange>;

// C99 variable-length arrays
def ext_vla : Extension<
  "variable length arrays are a C99 feature, accepted as an extension">,
  InGroup<VLA>;
def err_vla_non_pod : Error<"variable length array of non-POD element type %0">;
def err_vla_in_sfinae : Error<
  "variable length array cannot be formed during template argument deduction">;
def err_array_star_in_function_definition : Error<
  "variable length array must be bound in function definition">;
def err_vla_decl_in_file_scope : Error<
  "variable length array declaration not allowed at file scope">;
def err_vla_decl_has_static_storage : Error<
  "variable length array declaration can not have 'static' storage duration">;
def err_vla_decl_has_extern_linkage : Error<
  "variable length array declaration can not have 'extern' linkage">;

// C99 variably modified types
def err_variably_modified_template_arg : Error<
  "variably modified type %0 cannot be used as a template argument">;
def err_variably_modified_nontype_template_param : Error<
  "non-type template parameter of variably modified type %0">;
def err_variably_modified_new_type : Error<
  "'new' cannot allocate object of variably modified type %0">;

// C99 Designated Initializers
def ext_designated_init : Extension<
  "designated initializers are a C99 feature, accepted in C++ as an extension">;
def err_array_designator_negative : Error<
  "array designator value '%0' is negative">;
def err_array_designator_empty_range : Error<
  "array designator range [%0, %1] is empty">;
def err_array_designator_non_array : Error<
  "array designator cannot initialize non-array type %0">;
def err_array_designator_too_large : Error<
  "array designator index (%0) exceeds array bounds (%1)">;
def err_field_designator_non_aggr : Error<
  "field designator cannot initialize a "
  "%select{non-struct, non-union|non-class}0 type %1">;
def err_field_designator_unknown : Error<
  "field designator %0 does not refer to any field in type %1">;
def err_field_designator_nonfield : Error<
  "field designator %0 does not refer to a non-static data member">;
def note_field_designator_found : Note<"field designator refers here">;
def err_designator_for_scalar_init : Error<
  "designator in initializer for scalar type %0">;
def warn_subobject_initializer_overrides : Warning<
  "subobject initialization overrides initialization of other fields "
  "within its enclosing subobject">, InGroup<InitializerOverrides>;
def warn_initializer_overrides : Warning<
  "initializer overrides prior initialization of this subobject">,
  InGroup<InitializerOverrides>;
def note_previous_initializer : Note<
  "previous initialization %select{|with side effects }0is here"
  "%select{| (side effects may not occur at run time)}0">;
def err_designator_into_flexible_array_member : Error<
  "designator into flexible array member subobject">;
def note_flexible_array_member : Note<
  "initialized flexible array member %0 is here">;
def ext_flexible_array_init : Extension<
  "flexible array initialization is a GNU extension">, InGroup<GNU>;

// Declarations.
def ext_anon_param_requires_type_specifier : Extension<
  "type specifier required for unnamed parameter, defaults to int">;
def err_bad_variable_name : Error<
  "'%0' cannot be the name of a variable or data member">;
def err_bad_parameter_name : Error<
  "'%0' cannot be the name of a parameter">;
def err_parameter_name_omitted : Error<"parameter name omitted">;
def warn_unused_parameter : Warning<"unused parameter %0">,
  InGroup<UnusedParameter>, DefaultIgnore;
def warn_unused_variable : Warning<"unused variable %0">,
  InGroup<UnusedVariable>, DefaultIgnore;
def warn_unused_exception_param : Warning<"unused exception parameter %0">,
  InGroup<UnusedExceptionParameter>, DefaultIgnore;
def warn_decl_in_param_list : Warning<
  "declaration of %0 will not be visible outside of this function">;
def warn_unused_function : Warning<"unused function %0">,
  InGroup<UnusedFunction>, DefaultIgnore;
def warn_unused_member_function : Warning<"unused member function %0">,
  InGroup<UnusedMemberFunction>, DefaultIgnore;
def warn_used_but_marked_unused: Warning<"%0 was marked unused but was used">,
  InGroup<UsedButMarkedUnused>, DefaultIgnore;

def warn_parameter_size: Warning<
  "%0 is a large (%1 bytes) pass-by-value argument; "
  "pass it by reference instead ?">, InGroup<LargeByValueCopy>;
def warn_return_value_size: Warning<
  "return value of %0 is a large (%1 bytes) pass-by-value object; "
  "pass it by reference instead ?">, InGroup<LargeByValueCopy>;

def warn_implicit_function_decl : Warning<
  "implicit declaration of function %0">,
  InGroup<ImplicitFunctionDeclare>, DefaultIgnore;
def ext_implicit_function_decl : ExtWarn<
  "implicit declaration of function %0 is invalid in C99">,
  InGroup<ImplicitFunctionDeclare>;

def err_ellipsis_first_arg : Error<
  "ISO C requires a named argument before '...'">;
def err_declarator_need_ident : Error<"declarator requires an identifier">;
def err_bad_language : Error<"unknown linkage language">;
def warn_use_out_of_scope_declaration : Warning<
  "use of out-of-scope declaration of %0">;
def err_inline_non_function : Error<
  "'inline' can only appear on functions">;
def warn_qual_return_type : Warning< 
  "'%0' type qualifier%s1 on return type %plural{1:has|:have}1 no effect">,
  InGroup<IgnoredQualifiers>, DefaultIgnore;

def warn_decl_shadow :
  Warning<"declaration shadows a %select{"
          "local variable|"
          "variable in %2|"
          "static data member of %2|"
          "field of %2}1">,
  InGroup<Shadow>, DefaultIgnore;

// C++ using declarations
def err_using_requires_qualname : Error<
  "using declaration requires a qualified name">;
def err_using_typename_non_type : Error<
  "'typename' keyword used on a non-type">;
def err_using_dependent_value_is_type : Error<
  "dependent using declaration resolved to type without 'typename'">;
def err_using_decl_nested_name_specifier_is_not_class : Error<
  "using declaration in class refers into '%0', which is not a class">;
def err_using_decl_nested_name_specifier_is_current_class : Error<
  "using declaration refers to its own class">;
def err_using_decl_nested_name_specifier_is_not_base_class : Error<
  "using declaration refers into '%0', which is not a base class of %1">;
def err_using_decl_inherited_constructor_unsupported : Error<
  "inherited constructors are not supported">;
def err_using_decl_constructor_not_in_direct_base : Error<
  "%0 is not a direct base of %1, can not inherit constructors">;
def err_using_decl_constructor_conflict : Error<
  "can not inherit constructor, already inherited constructor with "
  "the same signature">;
def note_using_decl_constructor_conflict_current_ctor : Note<
  "conflicting constructor">;
def note_using_decl_constructor_conflict_previous_ctor : Note<
  "previous constructor">;
def note_using_decl_constructor_conflict_previous_using : Note<
  "previously inherited here">;
def err_using_decl_can_not_refer_to_class_member : Error<
  "using declaration can not refer to class member">;
def err_using_decl_can_not_refer_to_namespace : Error<
  "using declaration can not refer to namespace">;
def err_using_decl_constructor : Error<
  "using declaration can not refer to a constructor">;
def err_using_decl_destructor : Error<
  "using declaration can not refer to a destructor">;
def err_using_decl_template_id : Error<
  "using declaration can not refer to a template specialization">;
def note_using_decl_target : Note<"target of using declaration">;
def note_using_decl_conflict : Note<"conflicting declaration">;
def err_using_decl_redeclaration : Error<"redeclaration of using decl">;
def err_using_decl_conflict : Error<
  "target of using declaration conflicts with declaration already in scope">;
def err_using_decl_conflict_reverse : Error<
  "declaration conflicts with target of using declaration already in scope">;
def note_using_decl : Note<"%select{|previous }0using declaration">;

def warn_access_decl_deprecated : Warning<
  "access declarations are deprecated; use using declarations instead">,
  InGroup<Deprecated>;

def warn_global_constructor : Warning<
  "declaration requires a global constructor">,
  InGroup<GlobalConstructors>, DefaultIgnore;
def warn_global_destructor : Warning<
  "declaration requires a global destructor">,
   InGroup<GlobalConstructors>, DefaultIgnore;

def err_invalid_thread : Error<
  "'__thread' is only allowed on variable declarations">;
def err_thread_non_global : Error<
  "'__thread' variables must have global storage">;
def err_thread_unsupported : Error<
  "thread-local storage is unsupported for the current target">;

def warn_maybe_falloff_nonvoid_function : Warning<
  "control may reach end of non-void function">,
  InGroup<ReturnType>;
def warn_falloff_nonvoid_function : Warning<
  "control reaches end of non-void function">,
  InGroup<ReturnType>;
def err_maybe_falloff_nonvoid_block : Error<
  "control may reach end of non-void block">;
def err_falloff_nonvoid_block : Error<
  "control reaches end of non-void block">;
def warn_suggest_noreturn_function : Warning<
  "function could be attribute 'noreturn'">,
  InGroup<DiagGroup<"missing-noreturn">>, DefaultIgnore;
def warn_suggest_noreturn_block : Warning<
  "block could be attribute 'noreturn'">,
  InGroup<DiagGroup<"missing-noreturn">>, DefaultIgnore;
def warn_unreachable : Warning<"will never be executed">,
  InGroup<DiagGroup<"unreachable-code">>, DefaultIgnore;

/// Built-in functions.
def ext_implicit_lib_function_decl : ExtWarn<
  "implicitly declaring C library function '%0' with type %1">;
def note_please_include_header : Note<
  "please include the header <%0> or explicitly provide a "
  "declaration for '%1'">;
def note_previous_builtin_declaration : Note<"%0 is a builtin with type %1">;
def warn_implicit_decl_requires_stdio : Warning<
  "declaration of built-in function '%0' requires inclusion of the header "
  "<stdio.h>">;
def warn_implicit_decl_requires_setjmp : Warning<
  "declaration of built-in function '%0' requires inclusion of the header "
  "<setjmp.h>">;
def warn_redecl_library_builtin : Warning<
  "incompatible redeclaration of library function %0">;
def err_builtin_definition : Error<"definition of builtin function %0">;
def err_types_compatible_p_in_cplusplus : Error<
  "__builtin_types_compatible_p is not valid in C++">;
def warn_builtin_unknown : Warning<"use of unknown builtin %0">, DefaultError;

/// main()
// static/inline main() are not errors in C, just in C++.
def warn_unusual_main_decl : Warning<"'main' should not be declared "
    "%select{static|inline|static or inline}0">;
def err_unusual_main_decl : Error<"'main' is not allowed to be declared "
    "%select{static|inline|static or inline}0">;
def err_main_template_decl : Error<"'main' cannot be a template">;
def err_main_returns_nonint : Error<"'main' must return 'int'">;
def err_main_surplus_args : Error<"too many parameters (%0) for 'main': "
    "must be 0, 2, or 3">;
def warn_main_one_arg : Warning<"only one parameter on 'main' declaration">;
def err_main_arg_wrong : Error<"%select{first|second|third|fourth}0 "
    "parameter of 'main' (%select{argument count|argument array|environment|"
    "platform-specific data}0) must be of type %1">;

/// parser diagnostics
def ext_typedef_without_a_name : ExtWarn<"typedef requires a name">,
  InGroup<MissingDeclarations>;
def err_typedef_not_identifier : Error<"typedef name must be an identifier">;
def err_statically_allocated_object : Error<
  "interface type cannot be statically allocated">;
def err_object_cannot_be_passed_returned_by_value : Error<
  "interface type %1 cannot be %select{returned|passed}0 by value"
  "; did you forget * in %1">;
def warn_enum_value_overflow : Warning<"overflow in enumeration value">;
def warn_pragma_options_align_unsupported_option : Warning<
  "unsupported alignment option in '#pragma options align'">;
def warn_pragma_options_align_reset_failed : Warning<
  "#pragma options align=reset failed: %0">;
def err_pragma_options_align_mac68k_target_unsupported : Error<
  "mac68k alignment pragma is not supported on this target">;
def warn_pragma_pack_invalid_alignment : Warning<
  "expected #pragma pack parameter to be '1', '2', '4', '8', or '16'">;
// Follow the MSVC implementation.
def warn_pragma_pack_show : Warning<"value of #pragma pack(show) == %0">;
def warn_pragma_pack_pop_identifer_and_alignment : Warning<
  "specifying both a name and alignment to 'pop' is undefined">;
def warn_pragma_pack_pop_failed : Warning<"#pragma pack(pop, ...) failed: %0">;

def warn_pragma_unused_undeclared_var : Warning<
  "undeclared variable %0 used as an argument for '#pragma unused'">;
def warn_pragma_unused_expected_var_arg : Warning<
  "only variables can be arguments to '#pragma unused'">;
def err_unsupported_pragma_weak : Error<
  "using '#pragma weak' to refer to an undeclared identifier is not yet supported">;

/// Objective-C parser diagnostics
def err_duplicate_class_def : Error<
  "duplicate interface definition for class %0">;
def err_undef_superclass : Error<
  "cannot find interface declaration for %0, superclass of %1">;
def err_no_nsconstant_string_class : Error<
  "cannot find interface declaration for %0">;
def err_recursive_superclass : Error<
  "trying to recursively use %0 as superclass of %1">;
def warn_previous_alias_decl : Warning<"previously declared alias is ignored">;
def err_conflicting_aliasing_type : Error<"conflicting types for alias %0">;
def warn_undef_interface : Warning<"cannot find interface declaration for %0">;
def warn_duplicate_protocol_def : Warning<"duplicate protocol definition of %0 is ignored">;
def err_protocol_has_circular_dependency : Error<
  "protocol has circular dependency">;
def err_undeclared_protocol : Error<"cannot find protocol declaration for %0">;
def warn_undef_protocolref : Warning<"cannot find protocol definition for %0">;
def warn_readonly_property : Warning<
  "attribute 'readonly' of property %0 restricts attribute "
  "'readwrite' of property inherited from %1">;

def warn_property_attribute : Warning<
  "property %0 '%1' attribute does not match the property inherited from %2">;
def warn_property_types_are_incompatible : Warning<
  "property type %0 is incompatible with type %1 inherited from %2">;
def err_undef_interface : Error<"cannot find interface declaration for %0">;
def err_class_extension_after_impl : Error<
  "cannot declare class extension for %0 after class implementation">;
def note_implementation_declared : Note<
  "class implementation is declared here">;
def warn_dup_category_def : Warning<
  "duplicate definition of category %1 on interface %0">;
def err_conflicting_super_class : Error<"conflicting super class name %0">;
def err_dup_implementation_class : Error<"reimplementation of class %0">;
def err_dup_implementation_category : Error<
  "reimplementation of category %1 for class %0">;
def err_conflicting_ivar_type : Error<
  "instance variable %0 has conflicting type: %1 vs %2">;
def err_duplicate_ivar_declaration : Error<
  "instance variable is already declared">;
def warn_on_superclass_use : Warning<
  "class implementation may not have super class">;
def err_conflicting_ivar_bitwidth : Error<
  "instance variable %0 has conflicting bit-field width">;
def err_conflicting_ivar_name : Error<
  "conflicting instance variable names: %0 vs %1">;
def err_inconsistant_ivar_count : Error<
  "inconsistent number of instance variables specified">;
def warn_incomplete_impl : Warning<"incomplete implementation">,
  InGroup<DiagGroup<"incomplete-implementation">>;
def note_undef_method_impl : Note<"method definition for %0 not found">;
def note_required_for_protocol_at : 
  Note<"required for direct or indirect protocol %0">;

def warn_conflicting_ret_types : Warning<
  "conflicting return type in implementation of %0: %1 vs %2">;
def warn_conflicting_ret_type_modifiers : Warning<
  "conflicting distributed object modifiers on return type "
  "in implementation of %0">;
def warn_non_covariant_ret_types : Warning<
  "conflicting return type in implementation of %0: %1 vs %2">,
  InGroup<DiagGroup<"method-signatures">>, DefaultIgnore;

def warn_conflicting_param_types : Warning<
  "conflicting parameter types in implementation of %0: %1 vs %2">;
def warn_conflicting_param_modifiers : Warning<
  "conflicting distributed object modifiers on parameter type "
  "in implementation of %0">;
def warn_non_contravariant_param_types : Warning<
  "conflicting parameter types in implementation of %0: %1 vs %2">,
  InGroup<DiagGroup<"method-signatures">>, DefaultIgnore;
def warn_conflicting_variadic :Warning<
  "conflicting variadic declaration of method and its implementation">;

def warn_implements_nscopying : Warning<
"default assign attribute on property %0 which implements "
"NSCopying protocol is not appropriate with -fobjc-gc[-only]">;

def warn_multiple_method_decl : Warning<"multiple methods named %0 found">;
def warn_strict_multiple_method_decl : Warning<
  "multiple methods named %0 found">, InGroup<StrictSelector>, DefaultIgnore;
def warn_accessor_property_type_mismatch : Warning<
  "type of property %0 does not match type of accessor %1">;
def note_declared_at : Note<"declared here">;
def note_method_declared_at : Note<"method declared here">;
def err_setter_type_void : Error<"type of setter must be void">;
def err_duplicate_method_decl : Error<"duplicate declaration of method %0">;
def warn_missing_atend : Warning<"'@end' is missing in implementation context">;
def err_objc_var_decl_inclass : 
    Error<"cannot declare variable inside @interface or @protocol">;
def error_missing_method_context : Error<
  "missing context for method declaration">;
def err_objc_property_attr_mutually_exclusive : Error<
  "property attributes '%0' and '%1' are mutually exclusive">;
def err_objc_property_requires_object : Error<
  "property with '%0' attribute must be of object type">;
def warn_objc_property_no_assignment_attribute : Warning<
  "no 'assign', 'retain', or 'copy' attribute is specified - "
  "'assign' is assumed">;
def warn_objc_property_default_assign_on_object : Warning<
  "default property attribute 'assign' not appropriate for non-gc object">;
def warn_property_attr_mismatch : Warning<
  "property attribute in continuation class does not match the primary class">;
def warn_objc_property_copy_missing_on_block : Warning<
    "'copy' attribute must be specified for the block property "
    "when -fobjc-gc-only is specified">;
def warn_atomic_property_rule : Warning<
  "writable atomic property %0 cannot pair a synthesized setter/getter "
  "with a user defined setter/getter">;
def warn_default_atomic_custom_getter_setter : Warning<
  "atomic by default property %0 has a user defined %select{getter|setter}1 "
  "(property should be marked 'atomic' if this is intended)">,
  InGroup<CustomAtomic>, DefaultIgnore;
def err_use_continuation_class : Error<
  "illegal redeclaration of property in continuation class %0"
  " (attribute must be 'readwrite', while its primary must be 'readonly')">;
def err_use_continuation_class_redeclaration_readwrite : Error<
  "illegal redeclaration of 'readwrite' property in continuation class %0"
  " (perhaps you intended this to be a 'readwrite' redeclaration of a "
  "'readonly' public property?)">;
def err_continuation_class : Error<"continuation class has no primary class">;
def err_property_type : Error<"property cannot have array or function type %0">;
def error_missing_property_context : Error<
  "missing context for property implementation declaration">;
def error_bad_property_decl : Error<
  "property implementation must have its declaration in interface %0">;
def error_category_property : Error<
  "property declared in category %0 cannot be implemented in "
  "class implementation">;
def note_property_declare : Note<
  "property declared here">;
def error_synthesize_category_decl : Error<
  "@synthesize not allowed in a category's implementation">;
def error_reference_property : Error<
  "property of reference type is not supported">;
def error_missing_property_interface : Error<
  "property implementation in a category with no category declaration">;
def error_bad_category_property_decl : Error<
  "property implementation must have its declaration in the category %0">;
def error_bad_property_context : Error<
  "property implementation must be in a class or category implementation">;
def error_missing_property_ivar_decl : Error<
  "synthesized property %0 must either be named the same as a compatible"
  " ivar or must explicitly name an ivar">;

def error_synthesized_ivar_yet_not_supported : Error<
  "instance variable synthesis not yet supported"
  " (need to declare %0 explicitly)">;

def error_property_ivar_type : Error<
  "type of property %0 (%1) does not match type of ivar %2 (%3)">;
def error_ivar_in_superclass_use : Error<
  "property %0 attempting to use ivar %1 declared in super class %2">;
def error_weak_property : Error<
  "existing ivar %1 for __weak property %0 must be __weak">;
def error_strong_property : Error<
  "property %0 must be declared __weak to match existing ivar %1 with __weak attribute">;
def error_dynamic_property_ivar_decl : Error<
  "dynamic property can not have ivar specification">;
def error_duplicate_ivar_use : Error<
  "synthesized properties %0 and %1 both claim ivar %2">;
def error_property_implemented : Error<"property %0 is already implemented">;
def warn_objc_property_attr_mutually_exclusive : Warning<
  "property attributes '%0' and '%1' are mutually exclusive">,
  InGroup<ReadOnlySetterAttrs>, DefaultIgnore;
def warn_undeclared_selector : Warning<
  "undeclared selector %0">, InGroup<UndeclaredSelector>, DefaultIgnore;
def warn_implicit_atomic_property : Warning<
  "property is assumed atomic by default">, InGroup<ImplicitAtomic>, DefaultIgnore;
def warn_auto_implicit_atomic_property : Warning<
  "property is assumed atomic when auto-synthesizing the property">, 
  InGroup<ImplicitAtomic>, DefaultIgnore;
def warn_unimplemented_selector:  Warning<
  "unimplemented selector %0">, InGroup<Selector>, DefaultIgnore;
def warn_unimplemented_protocol_method : Warning<
  "method in protocol not implemented">, InGroup<Protocol>;

// C++ declarations
def err_static_assert_expression_is_not_constant : Error<
  "static_assert expression is not an integral constant expression">;
def err_static_assert_failed : Error<"static_assert failed \"%0\"">;

def err_inline_namespace_mismatch : Error<
  "%select{|non-}0inline namespace "
  "cannot be reopened as %select{non-|}0inline">;

def err_unexpected_friend : Error<
  "friends can only be classes or functions">;
def ext_enum_friend : ExtWarn<
  "enumeration type %0 cannot be a friend">;
def ext_nonclass_type_friend : ExtWarn<
  "non-class type %0 cannot be a friend">;
def err_friend_is_member : Error<
  "friends cannot be members of the declaring class">;
def ext_unelaborated_friend_type : ExtWarn<
  "must specify '%select{struct|union|class|enum}0' to befriend %1">;
def err_qualified_friend_not_found : Error<
  "no function named %0 with type %1 was found in the specified scope">;
def err_introducing_special_friend : Error<
  "must use a qualified name when declaring a %select{constructor|"
  "destructor|conversion operator}0 as a friend">;
def err_tagless_friend_type_template : Error<
  "friend type templates must use an elaborated type">;
def err_no_matching_local_friend : Error<
  "no matching function found in local scope">;
def err_partial_specialization_friend : Error<
  "partial specialization cannot be declared as a friend">;

def err_abstract_type_in_decl : Error<
  "%select{return|parameter|variable|field}0 type %1 is an abstract class">;
def err_allocation_of_abstract_type : Error<
  "allocating an object of abstract class type %0">;
def err_throw_abstract_type : Error<
  "cannot throw an object of abstract type %0">;
def err_array_of_abstract_type : Error<"array of abstract class type %0">;

def err_multiple_final_overriders : Error<
  "virtual function %q0 has more than one final overrider in %1">; 
def note_final_overrider : Note<"final overrider of %q0 in %1">;

def err_type_defined_in_type_specifier : Error<
  "%0 can not be defined in a type specifier">;
def err_type_defined_in_result_type : Error<
  "%0 can not be defined in the result type of a function">;
def err_type_defined_in_param_type : Error<
  "%0 can not be defined in a parameter type">;

def note_pure_virtual_function : Note<
  "unimplemented pure virtual method %0 in %1">;

def err_deleted_non_function : Error<
  "only functions can have deleted definitions">;
def err_deleted_decl_not_first : Error<
  "deleted definition must be first declaration">;

def warn_weak_vtable : Warning<
  "%0 has no out-of-line virtual method definitions; its vtable will be "
  "emitted in every translation unit">,
  InGroup<DiagGroup<"weak-vtables">>, DefaultIgnore;

def ext_using_undefined_std : ExtWarn<
  "using directive refers to implicitly-defined namespace 'std'">;
  
// C++ exception specifications
def err_exception_spec_in_typedef : Error<
  "exception specifications are not allowed in typedefs">;
def err_distant_exception_spec : Error<
  "exception specifications are not allowed beyond a single level "
  "of indirection">;
def err_incomplete_in_exception_spec : Error<
  "%select{|pointer to |reference to }0incomplete type %1 is not allowed "
  "in exception specification">;
def err_mismatched_exception_spec : Error<
  "exception specification in declaration does not match previous declaration">;
def err_override_exception_spec : Error<
  "exception specification of overriding function is more lax than "
  "base version">;
def err_incompatible_exception_specs : Error<
  "target exception specification is not superset of source">;
def err_deep_exception_specs_differ : Error<
  "exception specifications of %select{return|argument}0 types differ">;
def warn_missing_exception_specification : Warning<
  "%0 is missing exception specification '%1'">;

// C++ access checking
def err_class_redeclared_with_different_access : Error<
  "%0 redeclared with '%1' access">;
def err_access : Error<
  "%1 is a %select{private|protected}0 member of %3">, AccessControl;
def err_access_ctor : Error<
  "calling a %select{private|protected}0 constructor of class %2">, 
  AccessControl;
def ext_rvalue_to_reference_access_ctor : ExtWarn<
  "C++98 requires an accessible copy constructor for class %2 when binding "
  "a reference to a temporary; was %select{private|protected}0">,
  AccessControl, InGroup<BindToTemporaryCopy>;
def err_access_base : Error<
  "%select{base class|inherited virtual base class}0 %1 has %select{private|"
  "protected}3 %select{constructor|copy constructor|copy assignment operator|"
  "destructor}2">, AccessControl;
def err_access_field: Error<
  "field of type %0 has %select{private|protected}2 %select{constructor|copy "
  "constructor|copy assignment operator|destructor}1">, AccessControl;

def err_access_ctor_field :
    Error<"field of type %1 has %select{private|protected}2 constructor">,
    AccessControl;
def err_access_dtor : Error<
  "calling a %select{private|protected}1 destructor of class %0">, 
  AccessControl;
def err_access_dtor_base :
    Error<"base class %0 has %select{private|protected}1 destructor">,
    AccessControl;
def err_access_dtor_vbase :
    Error<"inherited virtual base class %0 has "
    "%select{private|protected}1 destructor">,
    AccessControl;
def err_access_dtor_temp :
    Error<"temporary of type %0 has %select{private|protected}1 destructor">,
    AccessControl;
def err_access_dtor_exception :
    Error<"exception object of type %0 has %select{private|protected}1 "
          "destructor">, AccessControl;
def err_access_dtor_field :
    Error<"field of type %1 has %select{private|protected}2 destructor">,
    AccessControl;
def err_access_dtor_var :
    Error<"variable of type %1 has %select{private|protected}2 destructor">,
    AccessControl;
def err_access_assign_field :
    Error<"field of type %1 has %select{private|protected}2 copy assignment"
          " operator">,
    AccessControl;
def err_access_assign_base :
    Error<"base class %0 has %select{private|protected}1 copy assignment"
          " operator">,
    AccessControl;
def err_access_copy_field :
    Error<"field of type %1 has %select{private|protected}2 copy constructor">,
    AccessControl;
def err_access_copy_base :
    Error<"base class %0 has %select{private|protected}1 copy constructor">,
    AccessControl;
def err_access_dtor_ivar :
    Error<"instance variable of type %0 has %select{private|protected}1 "
          "destructor">,
    AccessControl;
def note_previous_access_declaration : Note<
  "previously declared '%1' here">;
def err_access_outside_class : Error<
  "access to %select{private|protected}0 member outside any class context">,
  AccessControl;
def note_access_natural : Note<
  "%select{|implicitly }1declared %select{private|protected}0 here">;
def note_access_constrained_by_path : Note<
  "constrained by %select{|implicitly }1%select{private|protected}0"
  " inheritance here">;
def note_access_protected_restricted : Note<
  "object type %select{|%1 }0must derive from context type %2">;
  
// C++ name lookup
def err_incomplete_nested_name_spec : Error<
  "incomplete type %0 named in nested name specifier">;
def err_dependent_nested_name_spec : Error<
  "nested name specifier for a declaration cannot depend on a template "
  "parameter">;
def err_nested_name_member_ref_lookup_ambiguous : Error<
  "lookup of %0 in member access expression is ambiguous">;
def ext_nested_name_member_ref_lookup_ambiguous : ExtWarn<
  "lookup of %0 in member access expression is ambiguous; using member of %1">,
  InGroup<AmbigMemberTemplate>;
def note_ambig_member_ref_object_type : Note<
  "lookup in the object type %0 refers here">;
def note_ambig_member_ref_scope : Note<
  "lookup from the current scope refers here">;
def err_qualified_member_nonclass : Error<
  "qualified member access refers to a member in %0">;
def err_incomplete_member_access : Error<
  "member access into incomplete type %0">;
def err_incomplete_type : Error<
  "incomplete type %0 where a complete type is required">;
  
// C++ class members
def err_storageclass_invalid_for_member : Error<
  "storage class specified for a member declaration">;
def err_mutable_function : Error<"'mutable' cannot be applied to functions">;
def err_mutable_reference : Error<"'mutable' cannot be applied to references">;
def err_mutable_const : Error<"'mutable' and 'const' cannot be mixed">;
def err_mutable_nonmember : Error<
  "'mutable' can only be applied to member variables">;
def err_virtual_non_function : Error<
  "'virtual' can only appear on non-static member functions">;
def err_virtual_out_of_class : Error<
  "'virtual' can only be specified inside the class definition">;
def err_virtual_member_function_template : Error<
  "'virtual' can not be specified on member function templates">;
def err_static_overrides_virtual : Error<
  "'static' member function %0 overrides a virtual function in a base class">;
def err_explicit_non_function : Error<
  "'explicit' can only appear on non-static member functions">;
def err_explicit_out_of_class : Error<
  "'explicit' can only be specified inside the class definition">;
def err_explicit_non_ctor_or_conv_function : Error<
  "'explicit' can only be applied to a constructor or conversion function">;
def err_static_not_bitfield : Error<"static member %0 cannot be a bit-field">;
def err_static_out_of_line : Error<
  "'static' can only be specified inside the class definition">;
def err_typedef_not_bitfield : Error<"typedef member %0 cannot be a bit-field">;
def err_not_integral_type_bitfield : Error<
  "bit-field %0 has non-integral type %1">;
def err_not_integral_type_anon_bitfield : Error<
  "anonymous bit-field has non-integral type %0">;
def err_member_initialization : Error<
  "fields can only be initialized in constructors">;
def err_member_function_initialization : Error<
  "initializer on function does not look like a pure-specifier">;
def err_non_virtual_pure : Error<
  "%0 is not virtual and cannot be declared pure">;
def err_implicit_object_parameter_init : Error<
  "cannot initialize object parameter of type %0 with an expression "
  "of type %1">;
def err_qualified_member_of_unrelated : Error<
  "%q0 is not a member of class %1">;

def note_field_decl : Note<"member is declared here">;
def note_ivar_decl : Note<"ivar is declared here">;
def note_bitfield_decl : Note<"bit-field is declared here">;
def note_previous_decl : Note<"%0 declared here">;
def note_member_synthesized_at : Note<
  "implicit default %select{constructor|copy constructor|"
  "copy assignment operator|destructor}0 for %1 first required here">;
def err_missing_default_ctor : Error<
  "%select{|implicit default }0constructor for %1 must explicitly initialize "
  "the %select{base class|member}2 %3 which does not have a default "
  "constructor">;
def err_illegal_union_or_anon_struct_member : Error<
  "%select{anonymous struct|union}0 member %1 has a non-trivial "
  "%select{constructor|copy constructor|copy assignment operator|destructor}2">;
def note_nontrivial_has_virtual : Note<
  "because type %0 has a virtual %select{member function|base class}1">;
def note_nontrivial_has_nontrivial : Note<
  "because type %0 has a %select{member|base class}1 with a non-trivial "
  "%select{constructor|copy constructor|copy assignment operator|destructor}2">;
def note_nontrivial_user_defined : Note<
  "because type %0 has a user-declared %select{constructor|copy constructor|"
  "copy assignment operator|destructor}1">;
def err_static_data_member_not_allowed_in_union_or_anon_struct : Error<
  "static data member %0 not allowed in %select{anonymous struct|union}1">; 
def err_union_member_of_reference_type : Error<
  "union member %0 has reference type %1">;

def err_different_return_type_for_overriding_virtual_function : Error<
  "virtual function %0 has a different return type (%1) than the "
  "function it overrides (which has return type %2)">;
def note_overridden_virtual_function : Note<
  "overridden virtual function is here">;

def err_covariant_return_inaccessible_base : Error<
  "invalid covariant return for virtual function: %1 is a "
  "%select{private|protected}2 base class of %0">, AccessControl;
def err_covariant_return_ambiguous_derived_to_base_conv : Error<
  "return type of virtual function %3 is not covariant with the return type of "
  "the function it overrides (ambiguous conversion from derived class "
  "%0 to base class %1:%2)">;
def err_covariant_return_not_derived : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (%1 is not derived from %2)">;
def err_covariant_return_incomplete : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (%1 is incomplete)">;
def err_covariant_return_type_different_qualifications : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (%1 has different qualifiers than %2)">;
def err_covariant_return_type_class_type_more_qualified : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (class type %1 is more qualified than class "
  "type %2">;
  
// C++ constructors
def err_constructor_cannot_be : Error<"constructor cannot be declared '%0'">;
def err_invalid_qualified_constructor : Error<
  "'%0' qualifier is not allowed on a constructor">;
def err_ref_qualifier_constructor : Error<
  "ref-qualifier '%select{&&|&}0' is not allowed on a constructor">;

def err_constructor_return_type : Error<
  "constructor cannot have a return type">;
def err_constructor_redeclared : Error<"constructor cannot be redeclared">;
def err_constructor_byvalue_arg : Error<
  "copy constructor must pass its first argument by reference">;
def warn_no_constructor_for_refconst : Warning<
  "%select{struct|union|class|enum}0 %1 does not declare any constructor to "
  "initialize its non-modifiable members">;
def note_refconst_member_not_initialized : Note<
  "%select{const|reference}0 member %1 will never be initialized">;
def ext_ms_explicit_constructor_call : ExtWarn<
  "explicit constructor calls are a Microsoft extension">, InGroup<Microsoft>;

// C++ destructors
def err_destructor_not_member : Error<
  "destructor must be a non-static member function">;
def err_destructor_cannot_be : Error<"destructor cannot be declared '%0'">;
def err_invalid_qualified_destructor : Error<
  "'%0' qualifier is not allowed on a destructor">;
def err_ref_qualifier_destructor : Error<
  "ref-qualifier '%select{&&|&}0' is not allowed on a destructor">;
def err_destructor_return_type : Error<"destructor cannot have a return type">;
def err_destructor_redeclared : Error<"destructor cannot be redeclared">;
def err_destructor_with_params : Error<"destructor cannot have any parameters">;
def err_destructor_variadic : Error<"destructor cannot be variadic">;
def err_destructor_typedef_name : Error<
  "destructor cannot be declared using a typedef %0 of the class name">;
def err_destructor_name : Error<
  "expected the class name after '~' to name the enclosing class">;
def err_destructor_class_name : Error<
  "expected the class name after '~' to name a destructor">;
def err_ident_in_dtor_not_a_type : Error<
  "identifier %0 in object destruction expression does not name a type">;
def err_destructor_expr_type_mismatch : Error<
  "destructor type %0 in object destruction expression does not match the "
  "type %1 of the object being destroyed">;
def note_destructor_type_here : Note<
  "type %0 is declared here">;

def err_destructor_template : Error<
  "destructor cannot be declared as a template">;

// C++ initialization
def err_init_conversion_failed : Error<
  "cannot initialize %select{a variable|a parameter|return object|an "
  "exception object|a member subobject|an array element|a new value|a value|a "
  "base class|a constructor delegation|a vector element}0 of type %1 with an "
  "%select{rvalue|lvalue}2 of type %3">;

def err_lvalue_to_rvalue_ref : Error<"rvalue reference to type %0 cannot bind "
  "to lvalue of type %1">;
def err_invalid_initialization : Error<
"invalid initialization of reference of type %0 from expression of type %1">;
def err_lvalue_to_rvalue_ambig_ref : Error<"rvalue reference cannot bind to lvalue "
                                           "due to multiple conversion functions">;
def err_not_reference_to_const_init : Error<
  "%select{non-const|volatile}0 lvalue reference to type %1 cannot be "
  "initialized with a %select{value|temporary}2 of type %3">;
def err_lvalue_reference_bind_to_temporary : Error<
  "%select{non-const|volatile}0 lvalue reference to type %1 cannot bind to a "
  "temporary of type %2">;
def err_lvalue_reference_bind_to_unrelated : Error<
  "%select{non-const|volatile}0 lvalue reference to type %1 cannot bind to a "
  "value of unrelated type %2">;
def err_reference_bind_drops_quals : Error<
  "binding of reference to type %0 to a value of type %1 drops qualifiers">;
def err_reference_bind_failed : Error<
  "reference to type %0 could not bind to an %select{rvalue|lvalue}1 of type "
  "%2">;
def err_reference_bind_init_list : Error<
  "reference to type %0 cannot bind to an initializer list">;
def err_init_list_bad_dest_type : Error<
  "%select{|non-aggregate }0type %1 cannot be initialized with an initializer "
  "list">;
def err_member_function_call_bad_cvr : Error<"member function %0 not viable: "
    "'this' argument has type %1, but function is not marked "
    "%select{const|restrict|const or restrict|volatile|const or volatile|"
    "volatile or restrict|const, volatile, or restrict}2">;

def err_reference_init_drops_quals : Error<
  "initialization of reference to type %0 with a %select{value|temporary}1 of type %2 drops "
  "qualifiers">;
def err_reference_bind_to_bitfield : Error<
  "%select{non-const|volatile}0 reference cannot bind to bit-field %1">;
def err_reference_bind_to_vector_element : Error<
  "%select{non-const|volatile}0 reference cannot bind to vector element">;
def err_reference_var_requires_init : Error<
  "declaration of reference variable %0 requires an initializer">;
def err_const_var_requires_init : Error<
  "declaration of const variable '%0' requires an initializer">;
def err_reference_without_init : Error<
  "reference to type %0 requires an initializer">;
def err_reference_has_multiple_inits : Error<
  "reference cannot be initialized with multiple values">;
def err_init_non_aggr_init_list : Error<
  "initialization of non-aggregate type %0 with an initializer list">;
def err_init_reference_member_uninitialized : Error<
  "reference member of type %0 uninitialized">;
def note_uninit_reference_member : Note<
  "uninitialized reference member is here">;
def warn_field_is_uninit : Warning<"field is uninitialized when used here">,
  InGroup<Uninitialized>;
def warn_uninit_var : Warning<"variable %0 is possibly uninitialized when used here">,
  InGroup<DiagGroup<"uninitialized-experimental">>, DefaultIgnore;
def note_uninit_var_def : Note<
  "variable %0 is declared here">;
def warn_uninit_var_captured_by_block : Warning<
  "variable %0 is possibly uninitialized when captured by block">,
  InGroup<DiagGroup<"uninitialized-experimental">>, DefaultIgnore;
def note_var_fixit_add_initialization : Note<
  "add initialization to silence this warning">;
def err_init_incomplete_type : Error<"initialization of incomplete type %0">;

def err_temp_copy_no_viable : Error<
  "no viable constructor %select{copying variable|copying parameter|"
  "returning object|throwing object|copying member subobject|copying array "
  "element|allocating object|copying temporary|initializing base subobject|"
  "initializing vector element}0 of type %1">;
def ext_rvalue_to_reference_temp_copy_no_viable : ExtWarn<
  "no viable constructor %select{copying variable|copying parameter|"
  "returning object|throwing object|copying member subobject|copying array "
  "element|allocating object|copying temporary|initializing base subobject|"
  "initializing vector element}0 of type %1; C++98 requires a copy "
  "constructor when binding a reference to a temporary">,
  InGroup<BindToTemporaryCopy>;
def err_temp_copy_ambiguous : Error<
  "ambiguous constructor call when %select{copying variable|copying "
  "parameter|returning object|throwing object|copying member subobject|copying "
  "array element|allocating object|copying temporary|initializing base subobject|"
  "initializing vector element}0 of type %1">;
def err_temp_copy_deleted : Error<
  "%select{copying variable|copying parameter|returning object|throwing "
  "object|copying member subobject|copying array element|allocating object|"
  "copying temporary|initializing base subobject|initializing vector element}0 "
  "of type %1 invokes deleted constructor">;
def err_temp_copy_incomplete : Error<
  "copying a temporary object of incomplete type %0">;

// C++0x decltype
def err_cannot_determine_declared_type_of_overloaded_function : Error<
    "cannot determine the type of an overloaded function">;
    
// C++0x auto
def err_auto_variable_cannot_appear_in_own_initializer : Error<
  "variable %0 declared with 'auto' type cannot appear in its own initializer">;
def err_illegal_decl_array_of_auto : Error<
  "'%0' declared as array of %1">;
def err_new_array_of_auto : Error<
  "cannot allocate array of 'auto'">;
def err_auto_not_allowed : Error<
  "'auto' not allowed %select{in function prototype|in struct member"
  "|in union member|in class member|in exception declaration"
  "|in template parameter|in block literal|in template argument"
  "|in typedef|in function return type|here}0">;
def err_auto_var_requires_init : Error<
  "declaration of variable %0 with type %1 requires an initializer">;
def err_auto_new_requires_ctor_arg : Error<
  "new expression for type %0 requires a constructor argument">;
def err_auto_var_init_multiple_expressions : Error<
  "initializer for variable %0 with type %1 contains multiple expressions">;
def err_auto_new_ctor_multiple_expressions : Error<
  "new expression for type %0 contains multiple constructor arguments">;
def err_auto_missing_trailing_return : Error<
  "'auto' return without trailing return type">;
def err_trailing_return_without_auto : Error<
  "function with trailing return type must specify return type 'auto', not %0">;
def err_trailing_return_in_parens : Error<
  "trailing return type may not be nested within parentheses">;
def err_auto_var_deduction_failure : Error<
  "variable %0 with type %1 has incompatible initializer of type %2">;
def err_auto_new_deduction_failure : Error<
  "new expression for type %0 has incompatible constructor argument of type %1">;
def err_auto_different_deductions : Error<
  "'auto' deduced as %0 in declaration of %1 and deduced as %2 in declaration of %3">;

// C++0x override control
def override_keyword_only_allowed_on_virtual_member_functions : Error<
  "only virtual member functions can be marked '%0'">;
def err_function_marked_override_not_overriding : Error<
  "%0 marked 'override' but does not override any member functions">;
def err_class_marked_final_used_as_base : Error<
  "base %0 is marked 'final'">;

// C++0x attributes
def err_repeat_attribute : Error<"'%0' attribute cannot be repeated">;

// C++0x [[final]]
def err_final_function_overridden : Error<
  "declaration of %0 overrides a 'final' function">;
def err_final_base : Error<
  "derivation from 'final' %0">;

def err_function_overriding_without_override : Error<
  "%0 overrides function%s1 without being marked 'override'">;

// C++0x scoped enumerations
def err_enum_invalid_underlying : Error<
  "non-integral type %0 is an invalid underlying type">;
def err_enumerator_too_large : Error<
  "enumerator value is not representable in the underlying type %0">;
def ext_enumerator_too_large : ExtWarn<
  "enumerator value is not representable in the underlying type %0">,
  InGroup<Microsoft>;
def err_enumerator_wrapped : Error<
  "enumerator value %0 is not representable in the underlying type %1">;
def err_enum_redeclare_type_mismatch : Error<
  "enumeration redeclared with different underlying type %0 (was %1)">;
def err_enum_redeclare_fixed_mismatch : Error<
  "enumeration previously declared with %select{non|}0fixed underlying type">;
def err_enum_redeclare_scoped_mismatch : Error<
  "enumeration previously declared as %select{un|}0scoped">;

// C++0x delegating constructors
def err_delegation_0x_only : Error<
  "delegating constructors are permitted only in C++0x">;
def err_delegation_unimplemented : Error<
  "delegating constructors are not fully implemented">;
def err_delegating_initializer_alone : Error<
  "an initializer for a delegating constructor must appear alone">;
  
// Objective-C++
def err_objc_decls_may_only_appear_in_global_scope : Error<
  "Objective-C declarations may only appear in global scope">;
def err_nsobject_attribute : Error<
  "__attribute ((NSObject)) is for pointer types only">;

// Attributes
def err_attribute_can_be_applied_only_to_symbol_declaration : Error<
  "%0 attribute can be applied only to symbol declaration">;
def err_attributes_are_not_compatible : Error<
  "%0 and %1 attributes are not compatible">;
def err_attribute_wrong_number_arguments : Error<
  "attribute %plural{0:takes no arguments|1:takes one argument|"
  ":requires exactly %0 arguments}0">;
def err_attribute_too_many_arguments : Error<
  "attribute takes no more than %0 argument%s0">;
def err_iboutletcollection_type : Error<
  "invalid type %0 as argument of iboutletcollection attribute">;
def err_iboutletcollection_object_type : Error<
  "%select{ivar|property}1 with iboutletcollection attribute must "
  "have object type (invalid %0)">;
def err_attribute_missing_parameter_name : Error<
  "attribute requires unquoted parameter">;
def err_attribute_invalid_vector_type : Error<"invalid vector element type %0">;
def err_attribute_bad_neon_vector_size : Error<
  "Neon vector size must be 64 or 128 bits">;
def err_attribute_argument_not_int : Error<
  "'%0' attribute requires integer constant">;
def err_attribute_argument_outof_range : Error<
  "init_priority attribute requires integer constant between "
  "101 and 65535 inclusive">;
def err_init_priority_object_attr : Error<
  "can only use 'init_priority' attribute on file-scope definitions "
  "of objects of class type">;
def err_attribute_argument_n_not_int : Error<
  "'%0' attribute requires parameter %1 to be an integer constant">;
def err_attribute_argument_n_not_string : Error<
  "'%0' attribute requires parameter %1 to be a string">;
def err_attribute_argument_out_of_bounds : Error<
  "'%0' attribute parameter %1 is out of bounds">;
def err_attribute_requires_objc_interface : Error<
  "attribute may only be applied to an Objective-C interface">;
def err_attribute_uuid_malformed_guid : Error<
  "uuid attribute contains a malformed GUID">;
def warn_nonnull_pointers_only : Warning<
  "nonnull attribute only applies to pointer arguments">;
def err_attribute_invalid_implicit_this_argument : Error<
  "'%0' attribute is invalid for the implicit this argument">;
def err_ownership_type : Error<
  "%0 attribute only applies to %1 arguments">;
def err_format_strftime_third_parameter : Error<
  "strftime format attribute requires 3rd parameter to be 0">;
def err_format_attribute_requires_variadic : Error<
  "format attribute requires variadic function">;
def err_format_attribute_not : Error<"format argument not %0">;
def err_format_attribute_result_not : Error<"function does not return %0">;
def err_format_attribute_implicit_this_format_string : Error<
  "format attribute cannot specify the implicit this argument as the format "
  "string">;
def warn_unknown_method_family : Warning<"unrecognized method family">;
def err_attribute_invalid_size : Error<
  "vector size not an integral multiple of component size">;
def err_attribute_zero_size : Error<"zero vector size">;
def err_typecheck_vector_not_convertable : Error<
  "can't convert between vector values of different size (%0 and %1)">;
def err_typecheck_ext_vector_not_typedef : Error<
  "ext_vector_type only applies to types, not variables">;
def err_unsupported_vector_size : Error<
  "unsupported type %0 for vector_size attribute, please use on typedef">;
def err_ext_vector_component_exceeds_length : Error<
  "vector component access exceeds type %0">;
def err_ext_vector_component_name_illegal : Error<
  "illegal vector component name '%0'">;
def err_attribute_address_space_not_int : Error<
  "address space attribute requires an integer constant">;
def err_attribute_address_space_negative : Error<
  "address space is negative">;
def err_attribute_address_space_too_high : Error<
  "address space is larger than the maximum supported (%0)">;
def err_attribute_address_multiple_qualifiers : Error<
  "multiple address spaces specified for type">;
def err_as_qualified_auto_decl : Error<
  "automatic variable qualified with an address space">;
def err_arg_with_address_space : Error<
  "parameter may not be qualified with an address space">;
def err_attribute_not_string : Error<
  "argument to %0 attribute was not a string literal">;
def err_attribute_section_invalid_for_target : Error<
  "argument to 'section' attribute is not valid for this target: %0">;
def err_attribute_section_local_variable : Error<
  "'section' attribute is not valid on local variables">;
def err_attribute_aligned_not_power_of_two : Error<
  "requested alignment is not a power of 2">;
def warn_redeclaration_without_attribute_prev_attribute_ignored : Warning<
  "'%0' redeclared without %1 attribute: previous %1 ignored">;
def warn_attribute_ignored : Warning<"%0 attribute ignored">;
def warn_unknown_attribute_ignored : Warning<
  "unknown attribute %0 ignored">, InGroup<UnknownAttributes>;
def warn_attribute_precede_definition : Warning<
  "attribute declaration must precede definition">;
def warn_attribute_void_function_method : Warning<
  "attribute %0 cannot be applied to "
  "%select{functions|Objective-C method}1 without return value">;
def warn_attribute_weak_on_field : Warning<
  "__weak attribute cannot be specified on a field declaration">;
def warn_attribute_weak_on_local : Warning<
  "__weak attribute cannot be specified on an automatic variable">;
def warn_weak_identifier_undeclared : Warning<
  "weak identifier %0 never declared">;
def err_attribute_weak_static : Error<
  "weak declaration cannot have internal linkage">;
def warn_attribute_weak_import_invalid_on_definition : Warning<
  "'weak_import' attribute cannot be specified on a definition">;
def err_attribute_weakref_not_static : Error<
  "weakref declaration must have internal linkage">;
def err_attribute_weakref_not_global_context : Error<
  "weakref declaration of '%0' must be in a global context">;
def err_attribute_weakref_without_alias : Error<
  "weakref declaration of '%0' must also have an alias attribute">;
def err_alias_not_supported_on_darwin : Error <
  "only weak aliases are supported on darwin">;
def warn_attribute_wrong_decl_type : Warning<
  "%0 attribute only applies to %select{functions|unions|"
  "variables and functions|functions and methods|parameters|"
  "parameters and methods|functions, methods and blocks|"
  "classes and virtual methods|functions, methods, and parameters|"
  "classes|virtual methods|class members|variables|methods|"
  "variables, functions and labels}1">;
def err_attribute_wrong_decl_type : Error<
  "%0 attribute only applies to %select{functions|unions|"
  "variables and functions|functions and methods|parameters|"
  "parameters and methods|functions, methods and blocks|"
  "classes and virtual methods|functions, methods, and parameters|"
  "classes|virtual methods|class members|variables|methods}1">;
def warn_function_attribute_wrong_type : Warning<
  "%0 only applies to function types; type here is %1">;
def warn_gnu_inline_attribute_requires_inline : Warning<
  "'gnu_inline' attribute requires function to be marked 'inline',"
  " attribute ignored">;
def err_attribute_vecreturn_only_vector_member : Error<
  "the vecreturn attribute can only be used on a class or structure with one member, which must be a vector">;
def err_attribute_vecreturn_only_pod_record : Error<
  "the vecreturn attribute can only be used on a POD (plain old data) class or structure (i.e. no virtual functions)">;
def err_cconv_change : Error<
  "function declared '%0' here was previously declared "
  "%select{'%2'|without calling convention}1">;
def err_cconv_knr : Error<
  "function with no prototype cannot use %0 calling convention">;
def err_cconv_varargs : Error<
  "variadic function cannot use %0 calling convention">;
def err_regparm_mismatch : Error<"function declared with with regparm(%0) "
  "attribute was previously declared "
  "%plural{0:without the regparm|:with the regparm(%1)}1 attribute">;

def warn_impcast_vector_scalar : Warning<
  "implicit conversion turns vector to scalar: %0 to %1">,
  InGroup<DiagGroup<"conversion">>, DefaultIgnore;
def warn_impcast_complex_scalar : Warning<
  "implicit conversion discards imaginary component: %0 to %1">,
  InGroup<DiagGroup<"conversion">>, DefaultIgnore;
def warn_impcast_float_precision : Warning<
  "implicit conversion loses floating-point precision: %0 to %1">,
  InGroup<DiagGroup<"conversion">>, DefaultIgnore;
def warn_impcast_float_integer : Warning<
  "implicit conversion turns floating-point number into integer: %0 to %1">,
  InGroup<DiagGroup<"conversion">>, DefaultIgnore;
def warn_impcast_integer_sign : Warning<
  "implicit conversion changes signedness: %0 to %1">,
  InGroup<DiagGroup<"sign-conversion">>, DefaultIgnore;
def warn_impcast_integer_sign_conditional : Warning<
  "operand of ? changes signedness: %0 to %1">,
  InGroup<DiagGroup<"sign-conversion">>, DefaultIgnore;
def warn_impcast_integer_precision : Warning<
  "implicit conversion loses integer precision: %0 to %1">,
  InGroup<DiagGroup<"conversion">>, DefaultIgnore;
def warn_impcast_integer_64_32 : Warning<
  "implicit conversion loses integer precision: %0 to %1">,
  InGroup<DiagGroup<"shorten-64-to-32">>, DefaultIgnore;
def warn_impcast_integer_precision_constant : Warning<
  "implicit conversion from %2 to %3 changes value from %0 to %1">,
  InGroup<DiagGroup<"constant-conversion">>;
def warn_impcast_bitfield_precision_constant : Warning<
  "implicit truncation from %2 to bitfield changes value from %0 to %1">,
  InGroup<DiagGroup<"constant-conversion">>;
def warn_impcast_literal_float_to_integer : Warning<
  "implicit conversion turns literal floating-point number into integer: "
  "%0 to %1">,
  InGroup<DiagGroup<"literal-conversion">>, DefaultIgnore;
def warn_impcast_different_enum_types : Warning<
  "implicit conversion from enumeration type %0 to different enumeration type "
  "%1">, InGroup<DiagGroup<"conversion">>;

def warn_cast_align : Warning<
  "cast from %0 to %1 increases required alignment from %2 to %3">,
  InGroup<CastAlign>, DefaultIgnore;

def warn_attribute_ignored_for_field_of_type : Warning<
  "%0 attribute ignored for field of type %1">;
def warn_transparent_union_attribute_field_size_align : Warning<
  "%select{alignment|size}0 of field %1 (%2 bits) does not match the "
  "%select{alignment|size}0 of the first field in transparent union; "
  "transparent_union attribute ignored">;
def note_transparent_union_first_field_size_align : Note<
  "%select{alignment|size}0 of first field is %1 bits">;
def warn_transparent_union_attribute_not_definition : Warning<
  "transparent_union attribute can only be applied to a union definition; "
  "attribute ignored">;
def warn_transparent_union_attribute_floating : Warning<
  "first field of a transparent union cannot have %select{floating point|"
  "vector}0 type %1; transparent_union attribute ignored">;
def warn_transparent_union_attribute_zero_fields : Warning<
  "transparent union definition must contain at least one field; "
  "transparent_union attribute ignored">;
def warn_attribute_type_not_supported : Warning<
  "'%0' attribute argument not supported: %1">;
def warn_attribute_unknown_visibility : Warning<"unknown visibility '%0'">;
def err_unknown_machine_mode : Error<"unknown machine mode %0">;
def err_unsupported_machine_mode : Error<"unsupported machine mode %0">;
def err_mode_not_primitive : Error<
  "mode attribute only supported for integer and floating-point types">;
def err_mode_wrong_type : Error<
  "type of machine mode does not match type of base type">;
def err_attr_wrong_decl : Error<
  "'%0' attribute invalid on this declaration, requires typedef or value">;
def warn_attribute_nonnull_no_pointers : Warning<
  "'nonnull' attribute applied to function with no pointer arguments">;
def warn_attribute_malloc_pointer_only : Warning<
  "'malloc' attribute only applies to functions returning a pointer type">;
def warn_transparent_union_nonpointer : Warning<
  "'transparent_union' attribute support incomplete; only supported for "
  "pointer unions">;
def warn_attribute_sentinel_named_arguments : Warning<
  "'sentinel' attribute requires named arguments">;
def warn_attribute_sentinel_not_variadic : Warning<
  "'sentinel' attribute only supported for variadic %select{functions|blocks}0">;
def err_attribute_sentinel_less_than_zero : Error<
  "'sentinel' parameter 1 less than zero">;
def err_attribute_sentinel_not_zero_or_one : Error<
  "'sentinel' parameter 2 not 0 or 1">;
def err_attribute_cleanup_arg_not_found : Error<
  "'cleanup' argument %0 not found">;
def err_attribute_cleanup_arg_not_function : Error<
  "'cleanup' argument %0 is not a function">;
def err_attribute_cleanup_func_must_take_one_arg : Error<
  "'cleanup' function %0 must take 1 parameter">;
def err_attribute_cleanup_func_arg_incompatible_type : Error<
  "'cleanup' function %0 parameter has type %1 which is incompatible with "
  "type %2">;
def err_attribute_regparm_wrong_platform : Error<
  "'regparm' is not valid on this platform">;
def err_attribute_regparm_invalid_number : Error<
  "'regparm' parameter must be between 0 and %0 inclusive">;


// Clang-Specific Attributes
def warn_attribute_iboutlet : Warning<
  "%0 attribute can only be applied to instance variables or properties">;
def warn_attribute_ibaction: Warning<
  "ibaction attribute can only be applied to Objective-C instance methods">;
def err_attribute_overloadable_not_function : Error<
  "'overloadable' attribute can only be applied to a function">;
def err_attribute_overloadable_missing : Error<
  "%select{overloaded function|redeclaration of}0 %1 must have the "
  "'overloadable' attribute">;
def note_attribute_overloadable_prev_overload : Note<
  "previous overload of function is here">;
def err_attribute_overloadable_no_prototype : Error<
  "'overloadable' function %0 must have a prototype">;
def warn_ns_attribute_wrong_return_type : Warning<
  "%0 attribute only applies to %select{functions|methods}1 that "
  "return %select{an Objective-C object|a pointer}2">;
def warn_ns_attribute_wrong_parameter_type : Warning<
  "%0 attribute only applies to %select{Objective-C object|pointer}1 "
  "parameters">;

// Function Parameter Semantic Analysis.
def err_param_with_void_type : Error<"argument may not have 'void' type">;
def err_void_only_param : Error<
  "'void' must be the first and only parameter if specified">;
def err_void_param_qualified : Error<
  "'void' as parameter must not have type qualifiers">;
def err_ident_list_in_fn_declaration : Error<
  "a parameter list without types is only allowed in a function definition">;
def ext_param_not_declared : Extension<
  "parameter %0 was not declared, defaulting to type 'int'">;
def err_param_typedef_of_void : Error<
  "empty parameter list defined with a typedef of 'void' not allowed in C++">;
def err_param_default_argument : Error<
  "C does not support default arguments">;
def err_param_default_argument_redefinition : Error<
  "redefinition of default argument">;
def err_param_default_argument_missing : Error<
  "missing default argument on parameter">;
def err_param_default_argument_missing_name : Error<
  "missing default argument on parameter %0">;
def err_param_default_argument_references_param : Error<
  "default argument references parameter %0">;
def err_param_default_argument_references_local : Error<
  "default argument references local variable %0 of enclosing function">;
def err_param_default_argument_references_this : Error<
  "default argument references 'this'">;
def err_param_default_argument_nonfunc : Error<
  "default arguments can only be specified for parameters in a function "
  "declaration">;
def err_param_default_argument_template_redecl : Error<
  "default arguments cannot be added to a function template that has already "
  "been declared">;
def err_param_default_argument_member_template_redecl : Error<
  "default arguments cannot be added to an out-of-line definition of a member "
  "of a %select{class template|class template partial specialization|nested "
  "class in a template}0">;
def err_uninitialized_member_for_assign : Error<
  "cannot define the implicit default assignment operator for %0, because "
  "non-static %select{reference|const}1 member %2 can't use default "
  "assignment operator">;
def note_first_required_here : Note<
  "synthesized method is first required here">;
def err_uninitialized_member_in_ctor : Error<
  "%select{|implicit default }0constructor for %1 must explicitly initialize "
  "the %select{reference|const}2 member %3">;

def err_use_of_default_argument_to_function_declared_later : Error<
  "use of default argument to function %0 that is declared later in class %1">;
def note_default_argument_declared_here : Note<
  "default argument declared here">;

def ext_param_promoted_not_compatible_with_prototype : ExtWarn<
  "promoted type %0 of K&R function parameter is not compatible with the "
  "parameter type %1 declared in a previous prototype">;


// C++ Overloading Semantic Analysis.
def err_ovl_diff_return_type : Error<
  "functions that differ only in their return type cannot be overloaded">;
def err_ovl_static_nonstatic_member : Error<
  "static and non-static member functions with the same parameter types "
  "cannot be overloaded">;

def err_ovl_no_viable_function_in_call : Error<
  "no matching function for call to %0">;
def err_ovl_no_viable_member_function_in_call : Error<
  "no matching member function for call to %0">;
def err_ovl_ambiguous_call : Error<
  "call to %0 is ambiguous">;
def err_ovl_deleted_call : Error<
  "call to %select{unavailable|deleted}0 function %1 %2">;
def err_ovl_ambiguous_member_call : Error<
  "call to member function %0 is ambiguous">;
def err_ovl_deleted_member_call : Error<
  "call to %select{unavailable|deleted}0 member function %1 %2">;
def note_ovl_too_many_candidates : Note<
    "remaining %0 candidate%s0 omitted; "
    "pass -fshow-overloads=all to show them">;
def note_ovl_candidate : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "is the implicit default constructor|"
    "is the implicit copy constructor|"
    "is the implicit copy assignment operator|"
    "is an inherited constructor}0%1">;

def warn_init_pointer_from_false : Warning<
    "initialization of pointer of type %0 from literal 'false'">,
    InGroup<BoolConversions>;

def note_ovl_candidate_inherited_constructor : Note<"inherited from here">;
def note_ovl_candidate_bad_deduction : Note<
    "candidate template ignored: failed template argument deduction">;
def note_ovl_candidate_incomplete_deduction : Note<"candidate template ignored: "
    "couldn't infer template argument %0">;
def note_ovl_candidate_inconsistent_deduction : Note<
    "candidate template ignored: deduced conflicting %select{types|values|"
    "templates}0 for parameter %1 (%2 vs. %3)">;
def note_ovl_candidate_explicit_arg_mismatch_named : Note<
    "candidate template ignored: invalid explicitly-specified argument "
    "for template parameter %0">;
def note_ovl_candidate_explicit_arg_mismatch_unnamed : Note<
    "candidate template ignored: invalid explicitly-specified argument "
    "for %ordinal0 template parameter">;
def note_ovl_candidate_instantiation_depth : Note<
    "candidate template ignored: substitution exceeded maximum template "
    "instantiation depth">;
def note_ovl_candidate_underqualified : Note<
    "candidate template ignored: can't deduce a type for %0 which would "
    "make %2 equal %1">;
def note_ovl_candidate_substitution_failure : Note<
    "candidate template ignored: substitution failure %0">;
    
// Note that we don't treat templates differently for this diagnostic.
def note_ovl_candidate_arity : Note<"candidate "
    "%select{function|function|constructor|function|function|constructor|"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0 %select{|template }1"
    "not viable: requires%select{ at least| at most|}2 %3 argument%s3, but %4 "
    "%plural{1:was|:were}4 provided">;

def note_ovl_candidate_deleted : Note<
    "candidate %select{function|function|constructor|"
    "function |function |constructor ||||constructor (inherited)}0%1 "
    "has been explicitly %select{made unavailable|deleted}2">;

// Giving the index of the bad argument really clutters this message, and
// it's relatively unimportant because 1) it's generally obvious which
// argument(s) are of the given object type and 2) the fix is usually
// to complete the type, which doesn't involve changes to the call line
// anyway.  If people complain, we can change it.
def note_ovl_candidate_bad_conv_incomplete : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1 "
    "not viable: cannot convert argument of incomplete type %2 to %3">;
def note_ovl_candidate_bad_overload : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1"
    " not viable: no overload of %3 matching %2 for %ordinal4 argument">;
def note_ovl_candidate_bad_conv : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1"
    " not viable: no known conversion from %2 to %3 for "
    "%select{%ordinal5 argument|object argument}4">;
def note_ovl_candidate_bad_addrspace : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1 not viable: "
    "%select{%ordinal6|'this'}5 argument (%2) is in "
    "address space %3, but parameter must be in address space %4">;
def note_ovl_candidate_bad_cvr_this : Note<"candidate "
    "%select{|function|||function||||"
    "function (the implicit copy assignment operator)|}0 not viable: "
    "'this' argument has type %2, but method is not marked "
    "%select{const|restrict|const or restrict|volatile|const or volatile|"
    "volatile or restrict|const, volatile, or restrict}3">;
def note_ovl_candidate_bad_cvr : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1 not viable: "
    "%ordinal4 argument (%2) would lose "
    "%select{const|restrict|const and restrict|volatile|const and volatile|"
    "volatile and restrict|const, volatile, and restrict}3 qualifier"
    "%select{||s||s|s|s}3">;
def note_ovl_candidate_bad_base_to_derived_conv : Note<"candidate "
    "%select{function|function|constructor|"
    "function |function |constructor |"
    "constructor (the implicit default constructor)|"
    "constructor (the implicit copy constructor)|"
    "function (the implicit copy assignment operator)|"
    "constructor (inherited)}0%1"
    " not viable: cannot %select{convert from|convert from|bind}2 "
    "%select{base class pointer|superclass|base class object of type}2 %3 to "
    "%select{derived class pointer|subclass|derived class reference}2 %4 for "
    "%ordinal5 argument">;

def note_ambiguous_type_conversion: Note<
    "because of ambiguity in conversion of %0 to %1">;
def note_ovl_builtin_binary_candidate : Note<
    "built-in candidate %0">;
def note_ovl_builtin_unary_candidate : Note<
    "built-in candidate %0">;
def err_ovl_no_viable_function_in_init : Error<
  "no matching constructor for initialization of %0">;
def err_ovl_no_conversion_in_cast : Error<
  "cannot convert %1 to %2 without a conversion operator">;
def err_ovl_no_viable_conversion_in_cast : Error<
  "no matching conversion for %select{|static_cast|reinterpret_cast|"
  "dynamic_cast|C-style cast|functional-style cast}0 from %1 to %2">;
def err_ovl_ambiguous_conversion_in_cast : Error<
  "ambiguous conversion for %select{|static_cast|reinterpret_cast|"
  "dynamic_cast|C-style cast|functional-style cast}0 from %1 to %2">;
def err_ovl_deleted_conversion_in_cast : Error<
  "%select{|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from %1 to %2 uses deleted function">;
def err_ovl_ambiguous_init : Error<"call to constructor of %0 is ambiguous">;
def err_ref_init_ambiguous : Error<
  "reference initialization of type %0 with initializer of type %1 is ambiguous">;
def err_ovl_deleted_init : Error<
  "call to %select{unavailable|deleted}0 constructor of %1">;
def err_ovl_ambiguous_oper_unary : Error<
  "use of overloaded operator '%0' is ambiguous (operand type %1)">;
def err_ovl_ambiguous_oper_binary : Error<
  "use of overloaded operator '%0' is ambiguous (with operand types %1 and %2)">;
def err_ovl_no_viable_oper : Error<"no viable overloaded '%0'">;
def err_ovl_deleted_oper : Error<
  "overload resolution selected %select{unavailable|deleted}0 operator '%1' %2">;
def err_ovl_no_viable_subscript :
    Error<"no viable overloaded operator[] for type %0">;
def err_ovl_no_oper :
    Error<"type %0 does not provide a %select{subscript|call}1 operator">;
def err_ovl_unresolvable :
    Error<"cannot resolve overloaded function from context">;
  

def err_ovl_no_viable_object_call : Error<
  "no matching function for call to object of type %0">;
def err_ovl_ambiguous_object_call : Error<
  "call to object of type %0 is ambiguous">;
def err_ovl_deleted_object_call : Error<
  "call to %select{unavailable|deleted}0 function call operator in type %1 %2">;
def note_ovl_surrogate_cand : Note<"conversion candidate of type %0">;
def err_member_call_without_object : Error<
  "call to non-static member function without an object argument">;

// C++ Address of Overloaded Function
def err_addr_ovl_no_viable : Error<
  "address of overloaded function %0 does not match required type %1">;
def err_addr_ovl_ambiguous : Error<
  "address of overloaded function %0 is ambiguous">;
def err_addr_ovl_not_func_ptrref : Error<
  "address of overloaded function %0 cannot be converted to type %1">;
def err_addr_ovl_no_qualifier : Error<
  "can't form member pointer of type %0 without '&' and class name">;
  
// C++ Template Declarations
def err_template_param_shadow : Error<
  "declaration of %0 shadows template parameter">;
def note_template_param_here : Note<"template parameter is declared here">;
def warn_template_export_unsupported : Warning<
  "exported templates are unsupported">;
def err_template_outside_namespace_or_class_scope : Error<
  "templates can only be declared in namespace or class scope">;
def err_template_linkage : Error<"templates must have C++ linkage">;
def err_template_typedef : Error<"a typedef cannot be a template">;
def err_template_unnamed_class : Error<
  "cannot declare a class template with no name">;
def err_template_param_list_different_arity : Error<
  "%select{too few|too many}0 template parameters in template "
  "%select{|template parameter }1redeclaration">;
def note_template_param_list_different_arity : Note<
  "%select{too few|too many}0 template parameters in template template "
  "argument">;
def note_template_prev_declaration : Note<
  "previous template %select{declaration|template parameter}0 is here">;
def err_template_param_different_kind : Error<
  "template parameter has a different kind in template "
  "%select{|template parameter }0redeclaration">;
def note_template_param_different_kind : Note<
  "template parameter has a different kind in template argument">;
  
def err_template_nontype_parm_different_type : Error<
  "template non-type parameter has a different type %0 in template "
  "%select{|template parameter }1redeclaration">;

def note_template_nontype_parm_different_type : Note<
  "template non-type parameter has a different type %0 in template argument">;
def note_template_nontype_parm_prev_declaration : Note<
  "previous non-type template parameter with type %0 is here">;
def err_template_nontype_parm_bad_type : Error<
  "a non-type template parameter cannot have type %0">;
def err_template_param_default_arg_redefinition : Error<
  "template parameter redefines default argument">;
def note_template_param_prev_default_arg : Note<
  "previous default template argument defined here">;
def err_template_param_default_arg_missing : Error<
  "template parameter missing a default argument">;
def ext_template_parameter_default_in_function_template : ExtWarn<
  "default template arguments for a function template are a C++0x extension">,
  InGroup<CXX0x>;
def err_template_parameter_default_template_member : Error<
  "cannot add a default template argument to the definition of a member of a "
  "class template">;
def err_template_parameter_default_friend_template : Error<
  "default template argument not permitted on a friend template">;
def err_template_template_parm_no_parms : Error<
  "template template parameter must have its own template parameters">;

def err_template_variable : Error<"variable %0 declared as a template">;
def err_template_variable_noparams : Error<
  "extraneous 'template<>' in declaration of variable %0">;
def err_template_tag_noparams : Error<
  "extraneous 'template<>' in declaration of %0 %1">;
def err_template_decl_ref : Error<
  "cannot refer to class template %0 without a template argument list">;

// C++ Template Argument Lists
def err_template_missing_args : Error<
  "use of class template %0 requires template arguments">;
def err_template_arg_list_different_arity : Error<
  "%select{too few|too many}0 template arguments for "
  "%select{class template|function template|template template parameter"
  "|template}1 %2">;
def note_template_decl_here : Note<"template is declared here">;
def note_member_of_template_here : Note<"member is declared here">;
def err_template_arg_must_be_type : Error<
  "template argument for template type parameter must be a type">;
def err_template_arg_must_be_expr : Error<
  "template argument for non-type template parameter must be an expression">;
def err_template_arg_nontype_ambig : Error<
  "template argument for non-type template parameter is treated as type %0">;
def err_template_arg_must_be_template : Error<
  "template argument for template template parameter must be a class template">;
def ext_template_arg_local_type : ExtWarn<
  "template argument uses local type %0">, InGroup<LocalTypeTemplateArgs>;
def ext_template_arg_unnamed_type : ExtWarn<
  "template argument uses unnamed type">, InGroup<UnnamedTypeTemplateArgs>;
def note_template_unnamed_type_here : Note<
  "unnamed type used in template argument was declared here">;
def err_template_arg_overload_type : Error<
  "template argument is the type of an unresolved overloaded function">;
def err_template_arg_not_class_template : Error<
  "template argument does not refer to a class template or template "
  "template parameter">;
def note_template_arg_refers_here_func : Note<
  "template argument refers to function template %0, here">;
def err_template_arg_template_params_mismatch : Error<
  "template template argument has different template parameters than its "
  "corresponding template template parameter">;
def err_template_arg_not_integral_or_enumeral : Error<
  "non-type template argument of type %0 must have an integral or enumeration"
  " type">;
def err_template_arg_not_ice : Error<
  "non-type template argument of type %0 is not an integral constant "
  "expression">;
def err_deduced_non_type_template_arg_type_mismatch : Error<
  "deduced non-type template argument does not have the same type as the "
  "its corresponding template parameter (%0 vs %1)">;
def err_template_arg_not_convertible : Error<
  "non-type template argument of type %0 cannot be converted to a value "
  "of type %1">;
def warn_template_arg_negative : Warning<
  "non-type template argument with value '%0' converted to '%1' for unsigned "
  "template parameter of type %2">, InGroup<Conversion>, DefaultIgnore;
def warn_template_arg_too_large : Warning<
  "non-type template argument value '%0' truncated to '%1' for "
  "template parameter of type %2">, InGroup<Conversion>, DefaultIgnore;
def err_template_arg_no_ref_bind : Error<
  "non-type template parameter of reference type %0 cannot bind to template "
  "argument of type %1">;
def err_template_arg_ref_bind_ignores_quals : Error<
  "reference binding of non-type template parameter of type %0 to template "
  "argument of type %1 ignores qualifiers">;
def err_template_arg_not_decl_ref : Error<
  "non-type template argument does not refer to any declaration">;
def err_template_arg_not_object_or_func_form : Error<
  "non-type template argument does not directly refer to an object or "
  "function">;
def err_template_arg_not_address_of : Error<
  "non-type template argument for template parameter of pointer type %0 must "
  "have its address taken">;
def err_template_arg_address_of_non_pointer : Error<
  "address taken in non-type template argument for template parameter of "
  "reference type %0">;
def err_template_arg_reference_var : Error<
  "non-type template argument of reference type %0 is not an object">;
def err_template_arg_field : Error<
  "non-type template argument refers to non-static data member %0">;
def err_template_arg_method : Error<
  "non-type template argument refers to non-static member function %0">;
def err_template_arg_function_not_extern : Error<
  "non-template argument refers to function %0 with internal linkage">;
def err_template_arg_object_not_extern : Error<
  "non-template argument refers to object %0 that does not have external "
  "linkage">;
def note_template_arg_internal_object : Note<
  "non-template argument refers to %select{function|object}0 here">;
def note_template_arg_refers_here : Note<"non-template argument refers here">;
def err_template_arg_not_object_or_func : Error<
  "non-type template argument does not refer to an object or function">;
def err_template_arg_not_pointer_to_member_form : Error<
  "non-type template argument is not a pointer to member constant">;
def ext_template_arg_extra_parens : ExtWarn<
  "address non-type template argument cannot be surrounded by parentheses">;
def err_pointer_to_member_type : Error<
  "invalid use of pointer to member type after %select{.*|->*}0">;
def err_pointer_to_member_call_drops_quals : Error<
  "call to pointer to member function of type %0 drops '%1' qualifier%s2">;
def err_pointer_to_member_oper_value_classify: Error<
  "pointer-to-member function type %0 can only be called on an "
  "%select{rvalue|lvalue}1">;

// C++ template specialization
def err_template_spec_unknown_kind : Error<
  "can only provide an explicit specialization for a class template, function "
  "template, or a member function, static data member, or member class of a "
  "class template">;
def note_specialized_entity : Note<
  "explicitly specialized declaration is here">;
def err_template_spec_decl_function_scope : Error<
  "explicit specialization of %0 in function scope">;
def err_template_spec_decl_class_scope : Error<
  "explicit specialization of %0 in class scope">;
def err_template_spec_decl_friend : Error<
  "cannot declare an explicit specialization in a friend">;
def err_template_spec_decl_out_of_scope_global : Error<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 must "
  "originally be declared in the global scope">;
def ext_template_spec_decl_out_of_scope_global : ExtWarn<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 must "
  "originally be declared in the global scope; accepted as a C++0x extension">,
  InGroup<CXX0x>;
def err_template_spec_decl_out_of_scope : Error<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 must "
  "originally be declared in namespace %2">;
def ext_template_spec_decl_out_of_scope : ExtWarn<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 must "
  "originally be declared in namespace %2; accepted as a C++0x extension">,
  InGroup<CXX0x>;
def err_template_spec_redecl_out_of_scope : Error<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 not in a "
  "namespace enclosing %2">;
def err_template_spec_redecl_global_scope : Error<
  "%select{class template|class template partial|function template|member "
  "function|static data member|member class}0 specialization of %1 must occur "
  "at global scope">;
def err_spec_member_not_instantiated : Error<
  "specialization of member %q0 does not specialize an instantiated member">;
def note_specialized_decl : Note<"attempt to specialize declaration here">;
def err_specialization_after_instantiation : Error<
  "explicit specialization of %0 after instantiation">;
def note_instantiation_required_here : Note<
  "%select{implicit|explicit}0 instantiation first required here">;
def err_template_spec_friend : Error<
  "template specialization declaration cannot be a friend">;
def err_template_spec_default_arg : Error<
  "default argument not permitted on an explicit "
  "%select{instantiation|specialization}0 of function %1">;
def err_not_class_template_specialization : Error<
  "cannot specialize a %select{dependent template|template template "
  "parameter}0">;

// C++ class template specializations and out-of-line definitions
def err_template_spec_needs_header : Error<
  "template specialization requires 'template<>'">;
def err_template_spec_needs_template_parameters : Error<
  "template specialization or definition requires a template parameter list "
  "corresponding to the nested type %0">;
def err_template_param_list_matches_nontemplate : Error<
  "template parameter list matching the non-templated nested type %0 should "
  "be empty ('template<>')">;
def err_template_spec_extra_headers : Error<
  "extraneous template parameter list in template specialization or "
  "out-of-line template definition">;
def warn_template_spec_extra_headers : Warning<
  "extraneous template parameter list in template specialization">;
def note_explicit_template_spec_does_not_need_header : Note<
  "'template<>' header not required for explicitly-specialized class %0 "
  "declared here">;
def err_template_qualified_declarator_no_match : Error<
  "nested name specifier '%0' for declaration does not refer into a class, "
  "class template or class template partial specialization">;

// C++ Class Template Partial Specialization
def err_default_arg_in_partial_spec : Error<
    "default template argument in a class template partial specialization">;
def err_dependent_non_type_arg_in_partial_spec : Error<
    "non-type template argument depends on a template parameter of the "
    "partial specialization">;
def err_dependent_typed_non_type_arg_in_partial_spec : Error<
    "non-type template argument specializes a template parameter with "
    "dependent type %0">;
def err_partial_spec_args_match_primary_template : Error<
    "class template partial specialization does not specialize any template "
    "argument; to %select{declare|define}0 the primary template, remove the "
    "template argument list">; 
def warn_partial_specs_not_deducible : Warning<
    "class template partial specialization contains "
    "%select{a template parameter|template parameters}0 that can not be "
    "deduced; this partial specialization will never be used">;
def note_partial_spec_unused_parameter : Note<
    "non-deducible template parameter %0">;
def err_partial_spec_ordering_ambiguous : Error<
    "ambiguous partial specializations of %0">;
def note_partial_spec_match : Note<"partial specialization matches %0">;
def err_partial_spec_redeclared : Error<
  "class template partial specialization %0 cannot be redeclared">;
def note_prev_partial_spec_here : Note<
  "previous declaration of class template partial specialization %0 is here">;
def err_partial_spec_fully_specialized : Error<
  "partial specialization of %0 does not use any of its template parameters">;
  
// C++ Function template specializations
def err_function_template_spec_no_match : Error<
    "no function template matches function template specialization %0">;
def err_function_template_spec_ambiguous : Error<
    "function template specialization %0 ambiguously refers to more than one "
    "function template; explicitly specify%select{|additional }1 template "
    "arguments to identify a particular function template">;
def note_function_template_spec_matched : Note<
    "function template matches specialization %0">;
def err_function_template_partial_spec : Error<
    "function template partial specialization is not allowed">;

// C++ Template Instantiation
def err_template_recursion_depth_exceeded : Error<
  "recursive template instantiation exceeded maximum depth of %0">,
  DefaultFatal, NoSFINAE;
def note_template_recursion_depth : Note<
  "use -ftemplate-depth-N to increase recursive template instantiation depth">;

def err_template_instantiate_undefined : Error<
  "%select{implicit|explicit}0 instantiation of undefined template %1">;
def err_implicit_instantiate_member_undefined : Error<
  "implicit instantiation of undefined member %0">;
def note_template_class_instantiation_here : Note<
  "in instantiation of template class %0 requested here">;
def note_template_member_class_here : Note<
  "in instantiation of member class %0 requested here">;
def note_template_member_function_here : Note<
  "in instantiation of member function %q0 requested here">;
def note_function_template_spec_here : Note<
  "in instantiation of function template specialization %q0 requested here">;
def note_template_static_data_member_def_here : Note<
  "in instantiation of static data member %q0 requested here">;
  
def note_default_arg_instantiation_here : Note<
  "in instantiation of default argument for '%0' required here">;
def note_default_function_arg_instantiation_here : Note<
  "in instantiation of default function argument expression "
  "for '%0' required here">;
def note_explicit_template_arg_substitution_here : Note<
  "while substituting explicitly-specified template arguments into function "
  "template %0 %1">;
def note_function_template_deduction_instantiation_here : Note<
  "while substituting deduced template arguments into function template %0 "
  "%1">;
def note_partial_spec_deduct_instantiation_here : Note<
  "during template argument deduction for class template partial "
  "specialization %0 %1">;
def note_prior_template_arg_substitution : Note<
  "while substituting prior template arguments into %select{non-type|template}0"
  " template parameter%1 %2">;
def note_template_default_arg_checking : Note<
  "while checking a default template argument used here">;
def note_instantiation_contexts_suppressed : Note<
  "(skipping %0 context%s0 in backtrace; use -ftemplate-backtrace-limit=0 to "
  "see all)">;

def err_field_instantiates_to_function : Error<
  "data member instantiated with function type %0">;
def err_variable_instantiates_to_function : Error<
  "%select{variable|static data member}0 instantiated with function type %1">;
def err_nested_name_spec_non_tag : Error<
  "type %0 cannot be used prior to '::' because it has no members">;

// C++ Explicit Instantiation
def err_explicit_instantiation_duplicate : Error<
    "duplicate explicit instantiation of %0">;
def note_previous_explicit_instantiation : Note<
    "previous explicit instantiation is here">;
def ext_explicit_instantiation_after_specialization : Extension<
    "explicit instantiation of %0 that occurs after an explicit "
    "specialization will be ignored (C++0x extension)">,
    InGroup<CXX0x>;
def note_previous_template_specialization : Note<
    "previous template specialization is here">;
def err_explicit_instantiation_enum : Error<
    "explicit instantiation of enumeration type %0">;
def err_explicit_instantiation_nontemplate_type : Error<
    "explicit instantiation of non-templated type %0">;
def note_nontemplate_decl_here : Note<
    "non-templated declaration is here">;
def err_explicit_instantiation_in_class : Error<
  "explicit instantiation of %0 in class scope">;
def err_explicit_instantiation_out_of_scope : Error<
  "explicit instantiation of %0 not in a namespace enclosing %1">;
def err_explicit_instantiation_must_be_global : Error<
  "explicit instantiation of %0 must occur at global scope">;
def warn_explicit_instantiation_out_of_scope_0x : Warning<
  "explicit instantiation of %0 not in a namespace enclosing %1">, 
  InGroup<DiagGroup<"-Wc++0x-compat"> >;
def warn_explicit_instantiation_must_be_global_0x : Warning<
  "explicit instantiation of %0 must occur at global scope">, 
  InGroup<DiagGroup<"-Wc++0x-compat"> >;
  
def err_explicit_instantiation_requires_name : Error<
  "explicit instantiation declaration requires a name">;
def err_explicit_instantiation_of_typedef : Error<
  "explicit instantiation of typedef %0">;
def err_explicit_instantiation_not_known : Error<
  "explicit instantiation of %0 does not refer to a function template, member "
  "function, member class, or static data member">;
def note_explicit_instantiation_here : Note<
  "explicit instantiation refers here">;
def err_explicit_instantiation_data_member_not_instantiated : Error<
  "explicit instantiation refers to static data member %q0 that is not an "
  "instantiation">;
def err_explicit_instantiation_member_function_not_instantiated : Error<
  "explicit instantiation refers to member function %q0 that is not an "
  "instantiation">;
def err_explicit_instantiation_ambiguous : Error<
  "partial ordering for explicit instantiation of %0 is ambiguous">;
def note_explicit_instantiation_candidate : Note<
  "explicit instantiation candidate function template here %0">;
def err_explicit_instantiation_inline : Error<
  "explicit instantiation cannot be 'inline'">;
def ext_explicit_instantiation_without_qualified_id : ExtWarn<
  "qualifier in explicit instantiation of %q0 requires a template-id "
  "(a typedef is not permitted)">;
def err_explicit_instantiation_unqualified_wrong_namespace : Error<
  "explicit instantiation of %q0 must occur in %1">;
def warn_explicit_instantiation_unqualified_wrong_namespace_0x : Warning<
  "explicit instantiation of %q0 must occur in %1">, 
  InGroup<DiagGroup<"c++0x-compat"> >;
def err_explicit_instantiation_undefined_member : Error<
  "explicit instantiation of undefined %select{member class|member function|"
  "static data member}0 %1 of class template %2">;
def err_explicit_instantiation_undefined_func_template : Error<
  "explicit instantiation of undefined function template %0">;
def err_explicit_instantiation_declaration_after_definition : Error<
  "explicit instantiation declaration (with 'extern') follows explicit "
  "instantiation definition (without 'extern')">;
def note_explicit_instantiation_definition_here : Note<
  "explicit instantiation definition is here">;
  
// C++ typename-specifiers
def err_typename_nested_not_found : Error<"no type named %0 in %1">;
def err_typename_nested_not_type : Error<
    "typename specifier refers to non-type member %0 in %1">;
def note_typename_refers_here : Note<
    "referenced member %0 is declared here">;
def err_typename_missing : Error<
  "missing 'typename' prior to dependent type name '%0%1'">;
def ext_typename_outside_of_template : ExtWarn<
  "'typename' occurs outside of a template">, InGroup<CXX0x>;
def err_typename_refers_to_using_value_decl : Error<
  "typename specifier refers to a dependent using declaration for a value "
  "%0 in %1">;
def note_using_value_decl_missing_typename : Note<
  "add 'typename' to treat this using declaration as a type">;

def err_template_kw_refers_to_non_template : Error<
    "%0 following the 'template' keyword does not refer to a template">;
def err_template_kw_refers_to_function_template : Error<
    "%0 following the 'template' keyword refers to a function template">;
def err_template_kw_refers_to_class_template : Error<
    "'%0%1' instantiated to a class template, not a function template">;
def note_referenced_class_template : Error<
    "class template declared here">;
def err_template_kw_missing : Error<
  "missing 'template' keyword prior to dependent template name '%0%1'">;
def ext_template_outside_of_template : ExtWarn<
  "'template' keyword outside of a template">, InGroup<CXX0x>;

def err_non_type_template_in_nested_name_specifier : Error<
  "qualified name refers into a specialization of function template '%0'">;
def err_template_id_not_a_type : Error<
  "template name refers to non-type template '%0'">;
def note_template_declared_here : Note<
  "%select{function template|class template|template template parameter}0 "
  "%1 declared here">;

// C++0x Variadic Templates
def err_template_param_pack_default_arg : Error<
  "template parameter pack cannot have a default argument">;
def err_template_param_pack_must_be_last_template_parameter : Error<
  "template parameter pack must be the last template parameter">;

def err_template_parameter_pack_non_pack : Error<
  "%select{template type|non-type template|template template}0 parameter"
  "%select{| pack}1 conflicts with previous %select{template type|"
  "non-type template|template template}0 parameter%select{ pack|}1">;
def note_template_parameter_pack_non_pack : Note<
  "%select{template type|non-type template|template template}0 parameter"
  "%select{| pack}1 does not match %select{template type|non-type template"
  "|template template}0 parameter%select{ pack|}1 in template argument">;
def note_template_parameter_pack_here : Note<
  "previous %select{template type|non-type template|template template}0 "
  "parameter%select{| pack}1 declared here">;
  
def err_unexpanded_parameter_pack_0 : Error<
  "%select{expression|base type|declaration type|data member type|bit-field "
  "size|static assertion|fixed underlying type|enumerator value|"
  "using declaration|friend declaration|qualifier|initializer|default argument|"
  "non-type template parameter type|exception type|partial specialization}0 "
  "contains an unexpanded parameter pack">;
def err_unexpanded_parameter_pack_1 : Error<
  "%select{expression|base type|declaration type|data member type|bit-field "
  "size|static assertion|fixed underlying type|enumerator value|"
  "using declaration|friend declaration|qualifier|initializer|default argument|"
  "non-type template parameter type|exception type|partial specialization}0 "
  "contains unexpanded parameter pack %1">;
def err_unexpanded_parameter_pack_2 : Error<
  "%select{expression|base type|declaration type|data member type|bit-field "
  "size|static assertion|fixed underlying type|enumerator value|"
  "using declaration|friend declaration|qualifier|initializer|default argument|"
  "non-type template parameter type|exception type|partial specialization}0 "
  "contains unexpanded parameter packs %1 and %2">;
def err_unexpanded_parameter_pack_3_or_more : Error<
  "%select{expression|base type|declaration type|data member type|bit-field "
  "size|static assertion|fixed underlying type|enumerator value|"
  "using declaration|friend declaration|qualifier|initializer|default argument|"
  "non-type template parameter type|exception type|partial specialization}0 "
  "contains unexpanded parameter packs %1, %2, ...">;

def err_pack_expansion_without_parameter_packs : Error<
  "pack expansion does not contain any unexpanded parameter packs">;
def err_pack_expansion_length_conflict : Error<
  "pack expansion contains parameter packs %0 and %1 that have different "
  "lengths (%2 vs. %3)">;
def err_pack_expansion_length_conflict_multilevel : Error<
  "pack expansion contains parameter pack %0 that has a different "
  "length (%1 vs. %2) from outer parameter packs">;
def err_pack_expansion_member_init : Error<
  "pack expansion for initialization of member %0">;

def err_function_parameter_pack_without_parameter_packs : Error<
  "type %0 of function parameter pack does not contain any unexpanded "
  "parameter packs">;
def err_ellipsis_in_declarator_not_parameter : Error<
  "only function and template parameters can be parameter packs">;

def err_sizeof_pack_no_pack_name : Error<
  "%0 does not refer to the name of a parameter pack">;

def err_unexpected_typedef : Error<
  "unexpected type name %0: expected expression">;
def err_unexpected_namespace : Error<
  "unexpected namespace name %0: expected expression">;
def err_undeclared_var_use : Error<"use of undeclared identifier %0">;
def note_dependent_var_use : Note<"must qualify identifier to find this "
    "declaration in dependent base class">;
def err_undeclared_use : Error<"use of undeclared %0">;
def warn_deprecated : Warning<"%0 is deprecated">,
    InGroup<DeprecatedDeclarations>;
def warn_deprecated_message : Warning<"%0 is deprecated: %1">,
    InGroup<DeprecatedDeclarations>;
def warn_deprecated_fwdclass_message : Warning<
    "%0 maybe deprecated because receiver type is unknown">,
    InGroup<DeprecatedDeclarations>;
def warn_deprecated_def : Warning<
    "Implementing deprecated %select{method|class|category}0">,
    InGroup<DeprecatedImplementations>, DefaultIgnore;
def err_unavailable : Error<"%0 is unavailable">;
def err_unavailable_message : Error<"%0 is unavailable: %1">;
def warn_unavailable_fwdclass_message : Warning<
    "%0 maybe unavailable because receiver type is unknown">;
def note_unavailable_here : Note<
  "function has been explicitly marked %select{unavailable|deleted|deprecated}0 here">;
def warn_not_enough_argument : Warning<
  "not enough variable arguments in %0 declaration to fit a sentinel">;
def warn_missing_sentinel : Warning <
  "missing sentinel in %select{function call|method dispatch|block call}0">;
def note_sentinel_here : Note<
  "%select{function|method|block}0 has been explicitly marked sentinel here">;
def warn_missing_prototype : Warning<
  "no previous prototype for function %0">,
  InGroup<DiagGroup<"missing-prototypes">>, DefaultIgnore;
def err_redefinition : Error<"redefinition of %0">;
def err_definition_of_implicitly_declared_member : Error<
  "definition of implicitly declared %select{constructor|copy constructor|"
  "copy assignment operator|destructor}1">;
def err_redefinition_extern_inline : Error<
  "redefinition of a 'extern inline' function %0 is not supported in "
  "%select{C99 mode|C++}1">;

// This should eventually be an error.
def warn_undefined_internal : Warning<
  "%select{function|variable}0 %q1 has internal linkage but is not defined">,
  DiagGroup<"undefined-internal">;
def note_used_here : Note<"used here">;

def warn_redefinition_of_typedef : Warning<
  "redefinition of typedef %0 is invalid in C">,
  InGroup<DiagGroup<"typedef-redefinition"> >, DefaultError;

def err_inline_declaration_block_scope : Error<
  "inline declaration of %0 not allowed in block scope">;
def err_static_non_static : Error<
  "static declaration of %0 follows non-static declaration">;
def err_non_static_static : Error<
  "non-static declaration of %0 follows static declaration">;
def err_extern_non_extern : Error<
  "extern declaration of %0 follows non-extern declaration">;
def err_non_extern_extern : Error<
  "non-extern declaration of %0 follows extern declaration">;
def err_non_thread_thread : Error<
  "non-thread-local declaration of %0 follows thread-local declaration">;
def err_thread_non_thread : Error<
  "thread-local declaration of %0 follows non-thread-local declaration">;
def err_redefinition_different_type : Error<
  "redefinition of %0 with a different type">;
def err_redefinition_different_kind : Error<
  "redefinition of %0 as different kind of symbol">;
def err_redefinition_different_typedef : Error<
  "typedef redefinition with different types (%0 vs %1)">;
def err_tag_reference_non_tag : Error<
  "elaborated type refers to %select{a non-tag type|a typedef|a template}0">;
def err_tag_reference_conflict : Error<
  "implicit declaration introduced by elaborated type conflicts with "
  "%select{a declaration|a typedef|a template}0 of the same name">;
def err_dependent_tag_decl : Error<
  "%select{declaration|definition}0 of %select{struct|union|class|enum}1 "
  "in a dependent scope">;
def err_tag_definition_of_typedef : Error<
  "definition of type %0 conflicts with typedef of the same name">;
def err_conflicting_types : Error<"conflicting types for %0">;
def err_nested_redefinition : Error<"nested redefinition of %0">;
def err_use_with_wrong_tag : Error<
  "use of %0 with tag type that does not match previous declaration">;
def warn_struct_class_tag_mismatch : Warning<
    "%select{struct|class}0 %select{|template}1 %2 was previously declared "
    "as a %select{class|struct}0 %select{|template}1">,
    InGroup<MismatchedTags>, DefaultIgnore;
def ext_forward_ref_enum : Extension<
  "ISO C forbids forward references to 'enum' types">;
def err_forward_ref_enum : Error<
  "ISO C++ forbids forward references to 'enum' types">;
def ext_ms_forward_ref_enum : Extension<
  "forward references to 'enum' types are a Microsoft extension">, InGroup<Microsoft>;
def ext_forward_ref_enum_def : Extension<
  "redeclaration of already-defined enum %0 is a GNU extension">, InGroup<GNU>;
  
def err_redefinition_of_enumerator : Error<"redefinition of enumerator %0">;
def err_duplicate_member : Error<"duplicate member %0">;
def err_misplaced_ivar : Error<
  "ivars may not be placed in %select{categories|class extension}0">;
def ext_enum_value_not_int : Extension<
  "ISO C restricts enumerator values to range of 'int' (%0 is too "
  "%select{small|large}1)">;
def warn_enum_too_large : Warning<
  "enumeration values exceed range of largest integer">;
def warn_enumerator_too_large : Warning<
  "enumerator value %0 is not representable in the largest integer type">;
  
def warn_illegal_constant_array_size : Extension<
  "size of static array must be an integer constant expression">;
def err_vm_decl_in_file_scope : Error<
  "variably modified type declaration not allowed at file scope">;
def err_vm_decl_has_extern_linkage : Error<
  "variably modified type declaration can not have 'extern' linkage">;
def err_typecheck_field_variable_size : Error<
  "fields must have a constant size: 'variable length array in structure' "
  "extension will never be supported">;
def err_vm_func_decl : Error<
  "function declaration cannot have variably modified type">;
def err_array_too_large : Error<
  "array is too large (%0 elements)">;

// -Wpadded, -Wpacked
def warn_padded_struct_field : Warning<
  "padding %select{struct|class}0 %1 with %2 %select{byte|bit}3%select{|s}4 "
  "to align %5">, InGroup<Padded>, DefaultIgnore;
def warn_padded_struct_anon_field : Warning<
  "padding %select{struct|class}0 %1 with %2 %select{byte|bit}3%select{|s}4 "
  "to align anonymous bit-field">, InGroup<Padded>, DefaultIgnore;
def warn_padded_struct_size : Warning<
  "padding size of %0 with %1 %select{byte|bit}2%select{|s}3 "
  "to alignment boundary">, InGroup<Padded>, DefaultIgnore;
def warn_unnecessary_packed : Warning<
  "packed attribute is unnecessary for %0">, InGroup<Packed>, DefaultIgnore;

def err_typecheck_negative_array_size : Error<"array size is negative">;
def warn_typecheck_function_qualifiers : Warning<
  "qualifier on function type %0 has unspecified behavior">;
def err_typecheck_invalid_restrict_not_pointer : Error<
  "restrict requires a pointer or reference (%0 is invalid)">;
def err_typecheck_invalid_restrict_not_pointer_noarg : Error<
  "restrict requires a pointer or reference">;
def err_typecheck_invalid_restrict_invalid_pointee : Error<
  "pointer to function type %0 may not be 'restrict' qualified">;
def ext_typecheck_zero_array_size : Extension<
  "zero size arrays are an extension">;
def err_typecheck_zero_array_size : Error<
  "zero-length arrays are not permitted in C++">;
def err_at_least_one_initializer_needed_to_size_array : Error<
  "at least one initializer value required to size array">;
def err_array_size_non_int : Error<"size of array has non-integer type %0">;
def err_init_element_not_constant : Error<
  "initializer element is not a compile-time constant">;
def err_block_extern_cant_init : Error<
  "'extern' variable cannot have an initializer">;
def warn_extern_init : Warning<"'extern' variable has an initializer">;
def err_variable_object_no_init : Error<
  "variable-sized object may not be initialized">;
def err_array_init_list_required : Error<
  "initialization with '{...}' expected for array">;
def err_excess_initializers : Error<
  "excess elements in %select{array|vector|scalar|union|struct}0 initializer">;
def warn_excess_initializers : ExtWarn<
  "excess elements in %select{array|vector|scalar|union|struct}0 initializer">;
def err_excess_initializers_in_char_array_initializer : Error<
  "excess elements in char array initializer">;
def warn_excess_initializers_in_char_array_initializer : ExtWarn<
  "excess elements in char array initializer">;
def warn_initializer_string_for_char_array_too_long : ExtWarn<
  "initializer-string for char array is too long">;
def warn_missing_field_initializers : Warning<
  "missing field '%0' initializer">,
  InGroup<MissingFieldInitializers>, DefaultIgnore;
def warn_braces_around_scalar_init : Warning<
  "braces around scalar initializer">;
def warn_many_braces_around_scalar_init : ExtWarn<
  "too many braces around scalar initializer">;
def err_empty_scalar_initializer : Error<"scalar initializer cannot be empty">;
def err_illegal_initializer : Error<
  "illegal initializer (only variables can be initialized)">;
def err_illegal_initializer_type : Error<"illegal initializer type %0">;
def err_init_objc_class : Error<
  "cannot initialize Objective-C class type %0">;
def err_implicit_empty_initializer : Error<
  "initializer for aggregate with no elements requires explicit braces">;
def err_bitfield_has_negative_width : Error<
  "bit-field %0 has negative width (%1)">;
def err_anon_bitfield_has_negative_width : Error<
  "anonymous bit-field has negative width (%0)">;
def err_bitfield_has_zero_width : Error<"named bit-field %0 has zero width">;
def err_bitfield_width_exceeds_type_size : Error<
  "size of bit-field %0 (%1 bits) exceeds size of its type (%2 bits)">;
def err_anon_bitfield_width_exceeds_type_size : Error<
  "size of anonymous bit-field (%0 bits) exceeds size of its type (%1 bits)">;
  
// Used by C++ which allows bit-fields that are wider than the type.
def warn_bitfield_width_exceeds_type_size: Warning<
  "size of bit-field %0 (%1 bits) exceeds the size of its type; value will be "
  "truncated to %2 bits">;
def warn_anon_bitfield_width_exceeds_type_size : Warning<
  "size of anonymous bit-field (%0 bits) exceeds size of its type; value will "
  "be truncated to %1 bits">;

def warn_missing_braces : Warning<
  "suggest braces around initialization of subobject">,
  InGroup<DiagGroup<"missing-braces">>, DefaultIgnore;

def err_redefinition_of_label : Error<"redefinition of label %0">;
def err_undeclared_label_use : Error<"use of undeclared label %0">;
def warn_unused_label : Warning<"unused label %0">,
  InGroup<UnusedLabel>, DefaultIgnore;

def err_goto_into_protected_scope : Error<"goto into protected scope">;
def err_switch_into_protected_scope : Error<
  "switch case is in protected scope">;
def err_indirect_goto_without_addrlabel : Error<
  "indirect goto in function with no address-of-label expressions">;
def err_indirect_goto_in_protected_scope : Error<
  "indirect goto might cross protected scopes">;
def note_indirect_goto_target : Note<"possible target of indirect goto">;
def note_protected_by_variable_init : Note<
  "jump bypasses variable initialization">;
def note_protected_by_cleanup : Note<
  "jump bypasses initialization of variable with __attribute__((cleanup))">;
def note_protected_by_vla_typedef : Note<
  "jump bypasses initialization of VLA typedef">;
def note_protected_by_vla : Note<
  "jump bypasses initialization of variable length array">;
def note_protected_by_objc_try : Note<
  "jump bypasses initialization of @try block">;
def note_protected_by_objc_catch : Note<
  "jump bypasses initialization of @catch block">;
def note_protected_by_objc_finally : Note<
  "jump bypasses initialization of @finally block">;
def note_protected_by_objc_synchronized : Note<
  "jump bypasses initialization of @synchronized block">;
def note_protected_by_cxx_try : Note<
  "jump bypasses initialization of try block">;
def note_protected_by_cxx_catch : Note<
  "jump bypasses initialization of catch block">;
def note_protected_by___block : Note<
  "jump bypasses setup of __block variable">;

def note_exits_cleanup : Note<
  "jump exits scope of variable with __attribute__((cleanup))">;
def note_exits_dtor : Note<
  "jump exits scope of variable with non-trivial destructor">;
def note_exits___block : Note<
  "jump exits scope of __block variable">;
def note_exits_objc_try : Note<
  "jump exits @try block">;
def note_exits_objc_catch : Note<
  "jump exits @catch block">;
def note_exits_objc_finally : Note<
  "jump exits @finally block">;
def note_exits_objc_synchronized : Note<
  "jump exits @synchronized block">;
def note_exits_cxx_try : Note<
  "jump exits try block">;
def note_exits_cxx_catch : Note<
  "jump exits catch block">;

def err_func_returning_array_function : Error<
  "function cannot return %select{array|function}0 type %1">;
def err_field_declared_as_function : Error<"field %0 declared as a function">;
def err_field_incomplete : Error<"field has incomplete type %0">;
def ext_variable_sized_type_in_struct : ExtWarn<
  "field %0 with variable sized type %1 not at the end of a struct or class is"
  " a GNU extension">, InGroup<GNU>;

def err_flexible_array_empty_struct : Error<
  "flexible array %0 not allowed in otherwise empty struct">;
def err_flexible_array_has_nonpod_type : Error<
  "flexible array member %0 of non-POD element type %1">;
def ext_flexible_array_in_struct : Extension<
  "%0 may not be nested in a struct due to flexible array member">;
def ext_flexible_array_in_array : Extension<
  "%0 may not be used as an array element due to flexible array member">;
def err_flexible_array_init_nonempty : Error<
  "non-empty initialization of flexible array member inside subobject">;
def ext_flexible_array_empty_aggregate_ms : Extension<
  "flexible array member %0 in otherwise empty %select{struct|class}1 "
  "is a Microsoft extension">, InGroup<Microsoft>;
def ext_flexible_array_union_ms : Extension<
  "flexible array member %0 in a union is a Microsoft extension">,
  InGroup<Microsoft>;
def ext_flexible_array_empty_aggregate_gnu : Extension<
  "flexible array member %0 in otherwise empty %select{struct|class}1 "
  "is a GNU extension">, InGroup<GNU>;
def ext_flexible_array_union_gnu : Extension<
  "flexible array member %0 in a union is a GNU extension">, InGroup<GNU>;

def err_flexible_array_init_needs_braces : Error<
  "flexible array requires brace-enclosed initializer">;
def err_illegal_decl_array_of_functions : Error<
  "'%0' declared as array of functions of type %1">;
def err_illegal_decl_array_incomplete_type : Error<
  "array has incomplete element type %0">;
def err_illegal_message_expr_incomplete_type : Error<
  "objective-c message has incomplete result type %0">;
def err_illegal_decl_array_of_references : Error<
  "'%0' declared as array of references of type %1">;
def err_decl_negative_array_size : Error<
  "'%0' declared as an array with a negative size">;
def err_array_star_outside_prototype : Error<
  "star modifier used outside of function prototype">;
def err_illegal_decl_pointer_to_reference : Error<
  "'%0' declared as a pointer to a reference of type %1">;
def err_illegal_decl_mempointer_to_reference : Error<
  "'%0' declared as a member pointer to a reference of type %1">;
def err_illegal_decl_mempointer_to_void : Error<
  "'%0' declared as a member pointer to void">;
def err_illegal_decl_mempointer_in_nonclass : Error<
  "'%0' does not point into a class">;
def err_mempointer_in_nonclass_type : Error<
  "member pointer refers into non-class type %0">;
def err_reference_to_void : Error<"cannot form a reference to 'void'">;
def err_qualified_block_pointer_type : Error<
  "qualifier specification on block pointer type not allowed">;
def err_nonfunction_block_type : Error<
  "block pointer to non-function type is invalid">;
def err_return_block_has_expr : Error<"void block should not return a value">;
def err_block_return_missing_expr : Error<
  "non-void block should return a value">;
def err_block_with_return_type_requires_args : Error<
  "block with explicit return type requires argument list">;
def err_func_def_incomplete_result : Error<
  "incomplete result type %0 in function definition">;

// Expressions.
def ext_sizeof_function_type : Extension<
  "invalid application of 'sizeof' to a function type">, InGroup<PointerArith>;
def err_sizeof_alignof_overloaded_function_type : Error<
  "invalid application of '%select{sizeof|__alignof}0' to an overloaded "
  "function">;
def ext_sizeof_void_type : Extension<
  "invalid application of '%0' to a void type">, InGroup<PointerArith>;
def err_sizeof_alignof_incomplete_type : Error<
  "invalid application of '%select{sizeof|__alignof}0' to an incomplete type %1">;
def err_sizeof_alignof_bitfield : Error<
  "invalid application of '%select{sizeof|__alignof}0' to bit-field">;
def err_offsetof_incomplete_type : Error<
  "offsetof of incomplete type %0">;
def err_offsetof_record_type : Error<
  "offsetof requires struct, union, or class type, %0 invalid">;
def err_offsetof_array_type : Error<"offsetof requires array type, %0 invalid">;
def ext_offsetof_extended_field_designator : Extension<
  "using extended field designator is an extension">;
def warn_offsetof_non_pod_type : ExtWarn<"offset of on non-POD type %0">,
  InGroup<InvalidOffsetof>;
def err_offsetof_bitfield : Error<"cannot compute offset of bit-field %0">;

def warn_floatingpoint_eq : Warning<
  "comparing floating point with == or != is unsafe">,
  InGroup<DiagGroup<"float-equal">>, DefaultIgnore;

def warn_division_by_zero : Warning<"division by zero is undefined">;
def warn_remainder_by_zero : Warning<"remainder by zero is undefined">;
def warn_shift_negative : Warning<"shift count is negative">;
def warn_shift_gt_typewidth : Warning<"shift count >= width of type">;
def warn_shift_result_gt_typewidth : Warning<
  "shift result (%0) requires %1 bits to represent, but %2 only has %3 bits">,
   InGroup<DiagGroup<"shift-overflow">>;
def warn_shift_result_overrides_sign_bit : Warning<
  "shift result (%0) overrides the sign bit of the shift expression's type "
  "(%1) and becomes negative">,
  InGroup<DiagGroup<"shift-sign-overflow">>, DefaultIgnore;

def warn_precedence_bitwise_rel : Warning<
  "%0 has lower precedence than %1; %1 will be evaluated first">,
  InGroup<Parentheses>;
def note_precedence_bitwise_first : Note<
  "place parentheses around the %0 expression to evaluate it first">;
def note_precedence_bitwise_silence : Note<
  "place parentheses around the %0 expression to silence this warning">;

def warn_logical_instead_of_bitwise : Warning<
  "use of logical %0 with constant operand; switch to bitwise %1 or "
  "remove constant">, InGroup<DiagGroup<"constant-logical-operand">>;

def warn_logical_and_in_logical_or : Warning<
  "'&&' within '||'">, InGroup<LogicalOpParentheses>;
def note_logical_and_in_logical_or_silence : Note<
  "place parentheses around the '&&' expression to silence this warning">;

def warn_self_assignment : Warning<
  "explicitly assigning a variable of type %0 to itself">,
  InGroup<SelfAssignment>, DefaultIgnore;

def err_sizeof_nonfragile_interface : Error<
  "invalid application of '%select{alignof|sizeof}1' to interface %0 in "
  "non-fragile ABI">;
def err_atdef_nonfragile_interface : Error<
  "invalid application of @defs in non-fragile ABI">;
def err_subscript_nonfragile_interface : Error<
  "subscript requires size of interface %0, which is not constant in "
  "non-fragile ABI">;

def err_arithmetic_nonfragile_interface : Error<
  "arithmetic on pointer to interface %0, which is not a constant size in "
  "non-fragile ABI">;


def ext_subscript_non_lvalue : Extension<
  "ISO C90 does not allow subscripting non-lvalue array">;
def err_typecheck_subscript_value : Error<
  "subscripted value is not an array, pointer, or vector">;
def err_typecheck_subscript_not_integer : Error<
  "array subscript is not an integer">;
def err_subscript_function_type : Error<
  "subscript of pointer to function type %0">;
def err_subscript_incomplete_type : Error<
  "subscript of pointer to incomplete type %0">;
def err_typecheck_member_reference_struct_union : Error<
  "member reference base type %0 is not a structure or union">;
def err_typecheck_member_reference_ivar : Error<
  "%0 does not have a member named %1">;
def err_typecheck_member_reference_arrow : Error<
  "member reference type %0 is not a pointer">;
def err_typecheck_member_reference_suggestion : Error<
  "member reference type %0 is %select{a|not a}1 pointer; maybe you meant to use '%select{->|.}1'?">;
def err_typecheck_member_reference_type : Error<
  "cannot refer to type member %0 in %1 with '%select{.|->}2'">;
def err_typecheck_member_reference_unknown : Error<
  "cannot refer to member %0 in %1 with '%select{.|->}2'">;
def err_member_reference_needs_call : Error<
  "base of member reference is %select{a function|an overloaded function}0; "
  "perhaps you meant to call it%select{| with no arguments}1?">;
def note_member_ref_possible_intended_overload : Note<
  "possibly valid overload here">;
def warn_subscript_is_char : Warning<"array subscript is of type 'char'">,
  InGroup<CharSubscript>, DefaultIgnore;

def err_typecheck_incomplete_tag : Error<"incomplete definition of type %0">;
def err_no_member : Error<"no member named %0 in %1">;

def err_member_not_yet_instantiated : Error<
  "no member %0 in %1; it has not yet been instantiated">;
def note_non_instantiated_member_here : Note<
  "not-yet-instantiated member is declared here">;

def err_member_redeclared : Error<"class member cannot be redeclared">;
def err_member_name_of_class : Error<"member %0 has the same name as its class">;
def err_member_def_undefined_record : Error<
  "out-of-line definition of %0 from class %1 without definition">;
def err_member_def_does_not_match : Error<
  "out-of-line definition of %0 does not match any declaration in %1">;
def err_member_def_does_not_match_ret_type : Error<
  "out-of-line definition of %q0 differ from the declaration in the return type">;
def err_nonstatic_member_out_of_line : Error<
  "non-static data member defined out-of-line">;
def err_nonstatic_flexible_variable : Error<
  "non-static initialization of a variable with flexible array member">;
def err_qualified_typedef_declarator : Error<
  "typedef declarator cannot be qualified">;
def err_qualified_param_declarator : Error<
  "parameter declarator cannot be qualified">;
def ext_out_of_line_declaration : ExtWarn<
  "out-of-line declaration of a member must be a definition">,
  InGroup<OutOfLineDeclaration>, DefaultError;
def warn_member_extra_qualification : Warning<
  "extra qualification on member %0">;
def err_member_qualification : Error<
  "non-friend class member %0 cannot have a qualified name">;  
def note_member_def_close_match : Note<"member declaration nearly matches">;
def err_typecheck_ivar_variable_size : Error<
  "instance variables must have a constant size">;
def err_ivar_reference_type : Error<
  "instance variables cannot be of reference type">;
def err_typecheck_illegal_increment_decrement : Error<
  "cannot %select{decrement|increment}1 value of type %0">;
def err_typecheck_arithmetic_incomplete_type : Error<
  "arithmetic on pointer to incomplete type %0">;
def err_typecheck_pointer_arith_function_type : Error<
  "arithmetic on pointer to function type %0">;
def err_typecheck_pointer_arith_void_type : Error<
  "arithmetic on pointer to void type">;
def err_typecheck_decl_incomplete_type : Error<
  "variable has incomplete type %0">;
def ext_typecheck_decl_incomplete_type : ExtWarn<
  "tentative definition of variable with internal linkage has incomplete non-array type %0">;
def err_tentative_def_incomplete_type : Error<
  "tentative definition has type %0 that is never completed">;
def err_tentative_def_incomplete_type_arr : Error<
  "tentative definition has array of type %0 that is never completed">;
def warn_tentative_incomplete_array : Warning<
  "tentative array definition assumed to have one element">;
def err_typecheck_incomplete_array_needs_initializer : Error<
  "definition of variable with array type needs an explicit size "
  "or an initializer">;
def err_array_init_not_init_list : Error<
  "array initializer must be an initializer "
  "list%select{| or string literal}0">;
def err_array_init_different_type : Error<
  "cannot initialize array of type %0 with array of type %1">;
def err_array_init_non_constant_array : Error<
  "cannot initialize array of type %0 with non-constant array of type %1">;
def ext_array_init_copy : Extension<
  "initialization of an array of type %0 from a compound literal of type %1 is "
  "a GNU extension">;
def warn_deprecated_string_literal_conversion : Warning<
  "conversion from string literal to %0 is deprecated">, InGroup<DeprecatedWritableStr>;
def err_realimag_invalid_type : Error<"invalid type %0 to %1 operator">;
def err_typecheck_sclass_fscope : Error<
  "illegal storage class on file-scoped variable">;
def err_unsupported_global_register : Error<
  "global register variables are not supported">;
def err_typecheck_sclass_func : Error<"illegal storage class on function">;
def err_static_block_func : Error<
  "function declared in block scope cannot have 'static' storage class">;
def err_typecheck_address_of : Error<"address of %0 requested">;
def ext_typecheck_addrof_void : Extension<
  "ISO C forbids taking the address of an expression of type 'void'">;
def err_unqualified_pointer_member_function : Error<
  "must explicitly qualify name of member function when taking its address">;
def err_invalid_form_pointer_member_function : Error<
  "cannot create a non-constant pointer to member function">;
def err_parens_pointer_member_function : Error<
  "cannot parenthesize the name of a method when forming a member pointer">;
def err_typecheck_invalid_lvalue_addrof : Error<
  "address expression must be an lvalue or a function designator">;
def ext_typecheck_addrof_class_temporary : ExtWarn<
  "taking the address of a temporary object of type %0">, 
  InGroup<DiagGroup<"address-of-temporary">>, DefaultError;
def err_typecheck_addrof_class_temporary : Error<
  "taking the address of a temporary object of type %0">;
def err_typecheck_unary_expr : Error<
  "invalid argument type %0 to unary expression">;
def err_typecheck_indirection_requires_pointer : Error<
  "indirection requires pointer operand (%0 invalid)">;
def warn_indirection_through_null : Warning<
  "indirection of non-volatile null pointer will be deleted, not trap">, InGroup<NullDereference>;
def note_indirection_through_null : Note<
  "consider using __builtin_trap() or qualifying pointer with 'volatile'">;

def err_assignment_requires_nonfragile_object : Error<
  "cannot assign to class object in non-fragile ABI (%0 invalid)">;
def err_direct_interface_unsupported : Error<
  "indirection to an interface is not supported (%0 invalid)">;
def err_typecheck_invalid_operands : Error<
  "invalid operands to binary expression (%0 and %1)">;
def err_typecheck_sub_ptr_object : Error<
  "subtraction of pointer %0 requires pointee to be a complete object type">;
def err_typecheck_sub_ptr_compatible : Error<
  "%0 and %1 are not pointers to compatible types">;
def ext_typecheck_ordered_comparison_of_pointer_integer : ExtWarn<
  "ordered comparison between pointer and integer (%0 and %1)">;
def ext_typecheck_ordered_comparison_of_pointer_and_zero : Extension<
  "ordered comparison between pointer and zero (%0 and %1) is an extension">;
def ext_typecheck_ordered_comparison_of_function_pointers : ExtWarn<
  "ordered comparison of function pointers (%0 and %1)">;
def ext_typecheck_comparison_of_fptr_to_void : Extension<
  "equality comparison between function pointer and void pointer (%0 and %1)">;
def err_typecheck_comparison_of_fptr_to_void : Error<
  "equality comparison between function pointer and void pointer (%0 and %1)">;
def ext_typecheck_comparison_of_pointer_integer : ExtWarn<
  "comparison between pointer and integer (%0 and %1)">;
def err_typecheck_comparison_of_pointer_integer : Error<
  "comparison between pointer and integer (%0 and %1)">;
def ext_typecheck_comparison_of_distinct_pointers : ExtWarn<
  "comparison of distinct pointer types (%0 and %1)">;
def ext_typecheck_cond_incompatible_operands : ExtWarn<
  "incompatible operand types (%0 and %1)">;
def err_typecheck_comparison_of_distinct_pointers : Error<
  "comparison of distinct pointer types (%0 and %1)">;
def ext_typecheck_comparison_of_distinct_pointers_nonstandard : ExtWarn<
  "comparison of distinct pointer types (%0 and %1) uses non-standard "
  "composite pointer type %2">;
def err_typecheck_vector_comparison : Error<
  "comparison of vector types (%0 and %1) not supported yet">;
def err_typecheck_assign_const : Error<"read-only variable is not assignable">;
def err_stmtexpr_file_scope : Error<
  "statement expression not allowed at file scope">;
def warn_mixed_sign_comparison : Warning<
  "comparison of integers of different signs: %0 and %1">,
  InGroup<SignCompare>, DefaultIgnore;
def warn_mixed_sign_conditional : Warning<
  "operands of ? are integers of different signs: %0 and %1">,
  InGroup<SignCompare>, DefaultIgnore;
def warn_lunsigned_always_true_comparison : Warning<
  "comparison of unsigned%select{| enum}2 expression %0 is always %1">,
  InGroup<TautologicalCompare>;
def warn_runsigned_always_true_comparison : Warning<
  "comparison of %0 unsigned%select{| enum}2 expression is always %1">,
  InGroup<TautologicalCompare>;
def warn_comparison_of_mixed_enum_types : Warning<
  "comparison of two values with different enumeration types (%0 and %1)">,
  InGroup<DiagGroup<"enum-compare">>;

def err_invalid_this_use : Error<
  "invalid use of 'this' outside of a nonstatic member function">;
def err_invalid_member_use_in_static_method : Error<
  "invalid use of member %0 in static member function">;
def err_invalid_qualified_function_type : Error<
  "type qualifier is not allowed on this function">;
def err_invalid_ref_qualifier_function_type : Error<
  "ref-qualifier '%select{&&|&}0' is only allowed on non-static member functions,"
  " member function pointers, and typedefs of function types">;
def ext_qualified_function_type_template_arg : ExtWarn<
  "template argument of '%0' qualified function type is a GNU extension">, 
  InGroup<GNU>;

def err_invalid_qualified_function_pointer : Error<
  "type qualifier is not allowed on this function %select{pointer|reference}0">;
def err_invalid_qualified_typedef_function_type_use : Error<
  "a qualified function type cannot be used to declare a "
  "%select{static member|nonmember}0 function">;
def err_invalid_ref_qualifier_typedef_function_type_use : Error<
  "%select{static member|nonmember}0 function cannot have a ref-qualifier "
  "'%select{&&|&}1'">;

def err_ref_qualifier_overload : Error<
  "cannot overload a member function %select{without a ref-qualifier|with "
  "ref-qualifier '&'|with ref-qualifier '&&'}0 with a member function %select{"
  "without a ref-qualifier|with ref-qualifier '&'|with ref-qualifier '&&'}1">;

def err_invalid_non_static_member_use : Error<
  "invalid use of nonstatic data member %0">;
def err_invalid_incomplete_type_use : Error<
  "invalid use of incomplete type %0">;
def err_builtin_func_cast_more_than_one_arg : Error<
  "function-style cast to a builtin type can only take one argument">;
def err_builtin_direct_init_more_than_one_arg : Error<
  "initializer of a builtin type can only take one argument">;
def err_value_init_for_array_type : Error<
  "array types cannot be value-initialized">;
def warn_format_nonliteral_noargs : Warning<
  "format string is not a string literal (potentially insecure)">,
  InGroup<FormatSecurity>;
def warn_format_nonliteral : Warning<
  "format string is not a string literal">,
  InGroup<FormatNonLiteral>, DefaultIgnore;

def err_unexpected_interface : Error<
  "unexpected interface name %0: expected expression">;
def err_ref_non_value : Error<"%0 does not refer to a value">;
def err_ref_vm_type : Error<
  "cannot refer to declaration with a variably modified type inside block">;
def err_ref_array_type : Error<
  "cannot refer to declaration with an array type inside block">;
def err_property_not_found : Error<
  "property %0 not found on object of type %1">;
def err_getter_not_found : Error<
  "expected getter method not found on object of type %0">;
def err_property_not_found_forward_class : Error<
  "property %0 cannot be found in forward class object %1">;
def err_property_not_as_forward_class : Error<
  "property %0 refers to an incomplete Objective-C class %1 "
  "(with no @interface available)">;
def note_forward_class : Note<
  "forward class is declared here">;
def err_duplicate_property : Error<
  "property has a previous declaration">;
def ext_gnu_void_ptr : Extension<
  "use of GNU void* extension">, InGroup<PointerArith>;
def ext_gnu_ptr_func_arith : Extension<
  "arithmetic on pointer to function type %0 is a GNU extension">,
  InGroup<PointerArith>;
def error_readonly_property_assignment : Error<
  "assigning to property with 'readonly' attribute not allowed">;
def ext_integer_increment_complex : Extension<
  "ISO C does not support '++'/'--' on complex integer type %0">;
def ext_integer_complement_complex : Extension<
  "ISO C does not support '~' for complex conjugation of %0">;
def error_nosetter_property_assignment : Error<
  "setter method is needed to assign to object using property" " assignment syntax">;
def error_no_subobject_property_setting : Error<
  "expression is not assignable">;

def ext_freestanding_complex : Extension<
  "complex numbers are an extension in a freestanding C99 implementation">;

// FIXME: Remove when we support imaginary.
def err_imaginary_not_supported : Error<"imaginary types are not supported">;

// Obj-c expressions
def warn_root_inst_method_not_found : Warning<
  "instance method %0 is being used on 'Class' which is not in the root class">;
def warn_class_method_not_found : Warning<
  "method %objcclass0 not found (return type defaults to 'id')">;
def warn_inst_method_not_found : Warning<
  "method %objcinstance0 not found (return type defaults to 'id')">;
def error_no_super_class_message : Error<
  "no @interface declaration found in class messaging of %0">;
def error_root_class_cannot_use_super : Error<
  "%0 cannot use 'super' because it is a root class">;
def err_invalid_receiver_to_message : Error<
  "invalid receiver to message expression">;
def err_invalid_receiver_to_message_super : Error<
  "'super' is only valid in a method body">;
def err_invalid_receiver_class_message : Error<
  "receiver type %0 is not an Objective-C class">;
def err_missing_open_square_message_send : Error<
  "missing '[' at start of message send expression">;
def warn_bad_receiver_type : Warning<
  "receiver type %0 is not 'id' or interface pointer, consider "
  "casting it to 'id'">;
def err_bad_receiver_type : Error<"bad receiver type %0">;
def err_unknown_receiver_suggest : Error<
  "unknown receiver %0; did you mean %1?">;
def error_objc_throw_expects_object : Error<
  "@throw requires an Objective-C object type (%0 invalid)">;
def error_objc_synchronized_expects_object : Error<
  "@synchronized requires an Objective-C object type (%0 invalid)">;
def error_rethrow_used_outside_catch : Error<
  "@throw (rethrow) used outside of a @catch block">;
def err_attribute_multiple_objc_gc : Error<
  "multiple garbage collection attributes specified for type">;
def err_catch_param_not_objc_type : Error<
  "@catch parameter is not a pointer to an interface type">;
def err_illegal_qualifiers_on_catch_parm : Error<
  "illegal qualifiers on @catch parameter">;
def err_storage_spec_on_catch_parm : Error<
  "@catch parameter cannot have storage specifier %select{|'typedef'|'extern'|"
  "'static'|'auto'|'register'|'__private_extern__'|'mutable'}0">;
def warn_register_objc_catch_parm : Warning<
  "'register' storage specifier on @catch parameter will be ignored">;
def err_qualified_objc_catch_parm : Error<
  "@catch parameter declarator cannot be qualified">;
def err_objc_pointer_cxx_catch_gnu : Error<
  "can't catch Objective C exceptions in C++ in the GNU runtime">;
def err_objc_pointer_cxx_catch_fragile : Error<
  "can't catch Objective C exceptions in C++ in the non-unified "
  "exception model">;
def err_objc_object_catch : Error<
  "can't catch an Objective C object by value">;

def warn_setter_getter_impl_required : Warning<
  "property %0 requires method %1 to be defined - "
  "use @synthesize, @dynamic or provide a method implementation">;
def warn_setter_getter_impl_required_in_category : Warning<
  "property %0 requires method %1 to be defined - "
  "use @dynamic or provide a method implementation in category">;
def note_property_impl_required : Note<
  "implementation is here">;
def note_parameter_named_here : Note<
  "passing argument to parameter %0 here">;
def note_parameter_here : Note<
  "passing argument to parameter here">;

// C++ casts
// These messages adhere to the TryCast pattern: %0 is an int specifying the
// cast type, %1 is the source type, %2 is the destination type.
def err_bad_reinterpret_cast_overload : Error<
  "reinterpret_cast cannot resolve overloaded function %0 to type %1">;

def err_bad_static_cast_overload : Error<
  "address of overloaded function %0 cannot be static_cast to type %1">;

def err_bad_cstyle_cast_overload : Error<
  "address of overloaded function %0 cannot be cast to type %1">;


def err_bad_cxx_cast_generic : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from %1 to %2 is not allowed">;
def err_bad_cxx_cast_rvalue : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from rvalue to reference type %2">;
def err_bad_cxx_cast_const_away : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from %1 to %2 casts away constness">;
def err_bad_const_cast_dest : Error<
  "%select{const_cast||||C-style cast|functional-style cast}0 to %2, "
  "which is not a reference, pointer-to-object, or pointer-to-data-member">;
def ext_cast_fn_obj : Extension<
  "cast between pointer-to-function and pointer-to-object is an extension">;
def err_bad_reinterpret_cast_small_int : Error<
  "cast from pointer to smaller type %2 loses information">;
def err_bad_cxx_cast_vector_to_scalar_different_size : Error<
  "%select{||reinterpret_cast||C-style cast|}0 from vector %1 " 
  "to scalar %2 of different size">;
def err_bad_cxx_cast_scalar_to_vector_different_size : Error<
  "%select{||reinterpret_cast||C-style cast|}0 from scalar %1 " 
  "to vector %2 of different size">;
def err_bad_cxx_cast_vector_to_vector_different_size : Error<
  "%select{||reinterpret_cast||C-style cast|}0 from vector %1 " 
  "to vector %2 of different size">;
def err_bad_lvalue_to_rvalue_cast : Error<
  "cannot cast from lvalue of type %1 to rvalue reference type %2; types are "
  "not compatible">;
def err_bad_static_cast_pointer_nonpointer : Error<
  "cannot cast from type %1 to pointer type %2">;
def err_bad_static_cast_member_pointer_nonmp : Error<
  "cannot cast from type %1 to member pointer type %2">;
def err_bad_cxx_cast_member_pointer_size : Error<
  "cannot %select{||reinterpret_cast||C-style cast|}0 from member pointer "
  "type %1 to member pointer type %2 of different size">;
def err_bad_static_cast_incomplete : Error<"%0 is an incomplete type">;

// These messages don't adhere to the pattern.
// FIXME: Display the path somehow better.
def err_ambiguous_base_to_derived_cast : Error<
  "ambiguous cast from base %0 to derived %1:%2">;
def err_static_downcast_via_virtual : Error<
  "cannot cast %0 to %1 via virtual base %2">;
def err_downcast_from_inaccessible_base : Error<
  "cannot cast %select{private|protected}2 base class %1 to %0">;
def err_upcast_to_inaccessible_base : Error<
  "cannot cast %0 to its %select{private|protected}2 base class %1">;
def err_bad_dynamic_cast_not_ref_or_ptr : Error<
  "%0 is not a reference or pointer">;
def err_bad_dynamic_cast_not_class : Error<"%0 is not a class">;
def err_bad_dynamic_cast_incomplete : Error<"%0 is an incomplete type">;
def err_bad_dynamic_cast_not_ptr : Error<"%0 is not a pointer">;
def err_bad_dynamic_cast_not_polymorphic : Error<"%0 is not polymorphic">;

// Other C++ expressions
def err_need_header_before_typeid : Error<
  "you need to include <typeinfo> before using the 'typeid' operator">;
def err_need_header_before_ms_uuidof : Error<
  "you need to include <guiddef.h> before using the '__uuidof' operator">;
def err_uuidof_without_guid : Error<
  "cannot call operator __uuidof on a type with no GUID">;
def err_incomplete_typeid : Error<"'typeid' of incomplete type %0">;
def err_static_illegal_in_new : Error<
  "the 'static' modifier for the array size is not legal in new expressions">;
def err_array_new_needs_size : Error<
  "array size must be specified in new expressions">;
def err_bad_new_type : Error<
  "cannot allocate %select{function|reference}1 type %0 with new">;
def err_new_incomplete_type : Error<
  "allocation of incomplete type %0">;
def err_new_array_nonconst : Error<
  "only the first dimension of an allocated array may have dynamic size">;
def err_new_array_init_args : Error<
  "array 'new' cannot have initialization arguments">;
def ext_new_paren_array_nonconst : ExtWarn<
  "when type is in parentheses, array cannot have dynamic size">;
def err_placement_new_non_placement_delete : Error<
  "'new' expression with placement arguments refers to non-placement "
  "'operator delete'">;
def err_array_size_not_integral : Error<
  "array size expression must have integral or enumerated type, not %0">;
def err_array_size_incomplete_type : Error<
  "array size expression has incomplete class type %0">;
def err_array_size_explicit_conversion : Error<
  "array size expression of type %0 requires explicit conversion to type %1">;
def note_array_size_conversion : Note<
  "conversion to %select{integral|enumeration}0 type %1 declared here">;
def err_array_size_ambiguous_conversion : Error<
  "ambiguous conversion of array size expression of type %0 to an integral or "
  "enumeration type">;
def ext_array_size_conversion : Extension<
  "implicit conversion from array size expression of type %0 to "
  "%select{integral|enumeration}1 type %2 is a C++0x extension">,
  InGroup<CXX0x>;
  
def err_default_init_const : Error<
  "default initialization of an object of const type %0"
  "%select{| requires a user-provided default constructor}1">;
def err_delete_operand : Error<"cannot delete expression of type %0">;
def ext_delete_void_ptr_operand : ExtWarn<
  "cannot delete expression with pointer-to-'void' type %0">;
def err_ambiguous_delete_operand : Error<"ambiguous conversion of delete "
                                         "expression of type %0 to a pointer">;
def warn_delete_incomplete : Warning<
  "deleting pointer to incomplete type %0 may cause undefined behaviour">;
def err_delete_incomplete_class_type : Warning<
  "deleting incomplete class type %0; no conversions to pointer type">;
def warn_delete_array_type : Warning<
  "'delete' applied to a pointer-to-array type %0 treated as delete[]">;
def err_no_suitable_delete_member_function_found : Error<
  "no suitable member %0 in %1">;
def err_ambiguous_suitable_delete_member_function_found : Error<
  "multiple suitable %0 functions in %1">;
def note_member_declared_here : Note<
  "member %0 declared here">;
def err_decrement_bool : Error<"cannot decrement expression of type bool">;
def warn_increment_bool : Warning<
  "incrementing expression of type bool is deprecated">, InGroup<Deprecated>;
def ext_catch_incomplete_ptr : ExtWarn<
  "ISO C++ forbids catching a pointer to incomplete type %0">;
def ext_catch_incomplete_ref : ExtWarn<
  "ISO C++ forbids catching a reference to incomplete type %0">;
def err_catch_incomplete : Error<"cannot catch incomplete type %0">;
def err_catch_rvalue_ref : Error<"cannot catch exceptions by rvalue reference">;
def err_qualified_catch_declarator : Error<
  "exception declarator cannot be qualified">;
def err_early_catch_all : Error<"catch-all handler must come last">;
def err_bad_memptr_rhs : Error<
  "right hand operand to %0 has non pointer-to-member type %1">;
def err_memptr_rhs_to_incomplete : Error<
  "cannot dereference pointer into incomplete class type %0">;
def err_bad_memptr_lhs : Error<
  "left hand operand to %0 must be a %select{|pointer to }1class "
  "compatible with the right hand operand, but is %2">;
def warn_exception_caught_by_earlier_handler : Warning<
  "exception of type %0 will be caught by earlier handler">;
def note_previous_exception_handler : Note<"for type %0">;
def err_exceptions_disabled : Error<
  "cannot use '%0' with exceptions disabled">;
def err_objc_exceptions_disabled : Error<
  "cannot use '%0' with Objective-C exceptions disabled">;
def warn_non_virtual_dtor : Warning<
  "%0 has virtual functions but non-virtual destructor">,
  InGroup<NonVirtualDtor>, DefaultIgnore;
def warn_overloaded_virtual : Warning<
  "%q0 hides overloaded virtual %select{function|functions}1">,
  InGroup<OverloadedVirtual>, DefaultIgnore;
def note_hidden_overloaded_virtual_declared_here : Note<
  "hidden overloaded virtual function %q0 declared here">;

def err_conditional_void_nonvoid : Error<
  "%select{left|right}1 operand to ? is void, but %select{right|left}1 operand "
  "is of type %0">;
def err_conditional_ambiguous : Error<
  "conditional expression is ambiguous; %0 can be converted to %1 "
  "and vice versa">;
def err_conditional_ambiguous_ovl : Error<
  "conditional expression is ambiguous; %0 and %1 can be converted to several "
  "common types">;

def err_throw_incomplete : Error<
  "cannot throw object of incomplete type %0">;
def err_throw_incomplete_ptr : Error<
  "cannot throw pointer to object of incomplete type %0">;
def err_return_in_constructor_handler : Error<
  "return in the catch of a function try block of a constructor is illegal">;

def err_operator_arrow_circular : Error<
  "circular pointer delegation detected">;
def err_pseudo_dtor_base_not_scalar : Error<
  "object expression of non-scalar type %0 cannot be used in a "
  "pseudo-destructor expression">;
def err_pseudo_dtor_type_mismatch : Error<
  "the type of object expression (%0) does not match the type being destroyed "
  "(%1) in pseudo-destructor expression">;
def err_pseudo_dtor_call_with_args : Error<
  "call to pseudo-destructor cannot have any arguments">;
def err_dtor_expr_without_call : Error<
  "%select{destructor reference|pseudo-destructor expression}0 must be "
  "called immediately with '()'">;
def err_pseudo_dtor_destructor_non_type : Error<
  "%0 does not refer to a type name in pseudo-destructor expression; expected "
  "the name of type %1">;
def err_pseudo_dtor_template : Error<
  "specialization of template %0 does not refer to a scalar type in pseudo-"
  "destructor expression">;
def err_invalid_use_of_function_type : Error<
  "a function type is not allowed here">;
def err_invalid_use_of_array_type : Error<"an array type is not allowed here">;
def err_type_defined_in_condition : Error<
  "types may not be defined in conditions">;
def err_typecheck_bool_condition : Error<
  "value of type %0 is not contextually convertible to 'bool'">;
def err_typecheck_ambiguous_condition : Error<
  "conversion from %0 to %1 is ambiguous">;
def err_typecheck_nonviable_condition : Error<
  "no viable conversion from %0 to %1">;
def err_typecheck_deleted_function : Error<
  "conversion function from %0 to %1 invokes a deleted function">;
  
def err_expected_class_or_namespace : Error<"expected a class or namespace">;
def err_missing_qualified_for_redecl : Error<
  "must qualify the name %0 to declare %q1 in this scope">;
def err_invalid_declarator_scope : Error<
  "definition or redeclaration of %0 not in a namespace enclosing %1">;
def err_invalid_declarator_global_scope : Error<
  "definition or redeclaration of %0 cannot name the global scope">;
def err_invalid_declarator_in_function : Error<
  "definition or redeclaration of %0 not allowed inside a function">;
def err_not_tag_in_scope : Error<
  "no %select{struct|union|class|enum}0 named %1 in %2">;

def err_cannot_form_pointer_to_member_of_reference_type : Error<
  "cannot form a pointer-to-member to member %0 of reference type %1">;
def err_invalid_use_of_bound_member_func : Error<
  "a bound member function may only be called">;
def err_incomplete_object_call : Error<
  "incomplete type in call to object of type %0">;
def err_incomplete_pointer_to_member_return : Error<
  "incomplete return type %0 of pointer-to-member constant">;

def warn_condition_is_assignment : Warning<"using the result of an "
  "assignment as a condition without parentheses">,
  InGroup<Parentheses>;
// Completely identical except off by default.
def warn_condition_is_idiomatic_assignment : Warning<"using the result "
  "of an assignment as a condition without parentheses">,
  InGroup<DiagGroup<"idiomatic-parentheses">>, DefaultIgnore;
def note_condition_assign_to_comparison : Note<
  "use '==' to turn this assignment into an equality comparison">;
def note_condition_or_assign_to_comparison : Note<
  "use '!=' to turn this compound assignment into an inequality comparison">;
def note_condition_assign_silence : Note<
  "place parentheses around the assignment to silence this warning">;

def warn_equality_with_extra_parens : Warning<"equality comparison with "
  "extraneous parentheses">, InGroup<Parentheses>;
def note_equality_comparison_to_assign : Note<
  "use '=' to turn this equality comparison into an assignment">;
def note_equality_comparison_silence : Note<
  "remove extraneous parentheses around the comparison to silence this warning">;

def warn_synthesized_ivar_access : Warning<
  "direct access of synthesized ivar by using property access %0">,
  InGroup<NonfragileAbi2>, DefaultIgnore;

def note_global_declared_at : Note<"global variable declared here">;

// assignment related diagnostics (also for argument passing, returning, etc).
// In most of these diagnostics the %2 is a value from the
// Sema::AssignmentAction enumeration
def err_typecheck_convert_incompatible : Error<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from incompatible type|to parameter of incompatible type|"
  "from a function with incompatible result type|to incompatible type|"
  "with an expression of incompatible type|to parameter of incompatible type|"
  "to incompatible type}2 %1">;
def warn_incompatible_qualified_id : Warning<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from incompatible type|to parameter of incompatible type|"
  "from a function with incompatible result type|to incompatible type|"
  "with an expression of incompatible type|to parameter of incompatible type|"
  "to incompatible type}2 %1">;
def ext_typecheck_convert_pointer_int : ExtWarn<
  "incompatible pointer to integer conversion "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">;
def ext_typecheck_convert_int_pointer : ExtWarn<
  "incompatible integer to pointer conversion "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">;
def ext_typecheck_convert_pointer_void_func : Extension<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1 "
  "converts between void pointer and function pointer">;
def ext_typecheck_convert_incompatible_pointer_sign : ExtWarn<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1 "
  "converts between pointers to integer types with different sign">,
  InGroup<DiagGroup<"pointer-sign">>;
def ext_typecheck_convert_incompatible_pointer : ExtWarn<
  "incompatible pointer types "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">;
def ext_typecheck_convert_discards_qualifiers : ExtWarn<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1 discards "
  "qualifiers">;
def ext_nested_pointer_qualifier_mismatch : ExtWarn<
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1 discards "
  "qualifiers in nested pointer types">;
def warn_incompatible_vectors : Warning<
  "incompatible vector types "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">,
  InGroup<VectorConversions>, DefaultIgnore;
def err_int_to_block_pointer : Error<
  "invalid block pointer conversion "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">;
def err_typecheck_convert_incompatible_block_pointer : Error<
  "incompatible block pointer types "
  "%select{assigning to|passing|returning|converting|initializing|sending|casting}2"
  " %0 "
  "%select{from|to parameter of type|from a function with result type|to type|"
  "with an expression of type|to parameter of type|to type}2 %1">;
def err_typecheck_incompatible_address_space : Error<
  "%select{assigning %1 to %0"
  "|passing %0 to parameter of type %1"
  "|returning %0 from a function with result type %1"
  "|converting %0 to type %1"
  "|initializing %0 with an expression of type %1"
  "|sending %0 to parameter of type %1"
  "|casting %0 to type %1}2"
  " changes address space of pointer">;
def err_typecheck_convert_ambiguous : Error<
  "ambiguity in initializing value of type %0 with initializer of type %1">;
def err_cannot_initialize_decl_noname : Error<
  "cannot initialize a value of type %0 with an %select{rvalue|lvalue}1 "
  "of type %2">;
def err_cannot_initialize_decl : Error<
  "cannot initialize %0 with an %select{rvalue|lvalue}1 of type %2">;
def err_typecheck_comparison_of_distinct_blocks : Error<
  "comparison of distinct block types (%0 and %1)">;

def err_typecheck_array_not_modifiable_lvalue : Error<
  "array type %0 is not assignable">;
def err_typecheck_non_object_not_modifiable_lvalue : Error<
  "non-object type %0 is not assignable">;
def err_typecheck_expression_not_modifiable_lvalue : Error<
  "expression is not assignable">;
def err_typecheck_incomplete_type_not_modifiable_lvalue : Error<
  "incomplete type %0 is not assignable">;
def err_typecheck_lvalue_casts_not_supported : Error<
  "assignment to cast is illegal, lvalue casts are not supported">;

def err_typecheck_duplicate_vector_components_not_mlvalue : Error<
  "vector is not assignable (contains duplicate components)">;
def err_block_decl_ref_not_modifiable_lvalue : Error<
  "variable is not assignable (missing __block type specifier)">;
def err_typecheck_call_not_function : Error<
  "called object type %0 is not a function or function pointer">;
def err_call_incomplete_return : Error<
  "calling function with incomplete return type %0">;
def err_call_function_incomplete_return : Error<
  "calling %0 with incomplete return type %1">;
def note_function_with_incomplete_return_type_declared_here : Note<
  "%0 declared here">;
def err_call_incomplete_argument : Error<
  "argument type %0 is incomplete">;
def err_typecheck_call_too_few_args : Error<
  "too few arguments to %select{function|block|method}0 call, "
  "expected %1, have %2">;
def err_typecheck_call_too_few_args_at_least : Error<
  "too few arguments to %select{function|block|method}0 call, "
  "expected at least %1, have %2">;
def err_typecheck_call_too_many_args : Error<
  "too many arguments to %select{function|block|method}0 call, "
  "expected %1, have %2">;
def err_typecheck_call_too_many_args_at_most : Error<
  "too many arguments to %select{function|block|method}0 call, "
  "expected at most %1, have %2">;
def warn_call_wrong_number_of_arguments : Warning<
  "too %select{few|many}0 arguments in call to %1">;
def err_atomic_builtin_must_be_pointer : Error<
  "first argument to atomic builtin must be a pointer (%0 invalid)">;
def err_atomic_builtin_must_be_pointer_intptr : Error<
  "first argument to atomic builtin must be a pointer to integer or pointer"
  " (%0 invalid)">;
def err_atomic_builtin_pointer_size : Error<
  "first argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte "
  "type (%0 invalid)">;

def err_deleted_function_use : Error<"attempt to use a deleted function">;

def err_kern_type_not_void_return : Error<
  "kernel function type %0 must have void return type">;
def err_config_scalar_return : Error<
  "CUDA special function 'cudaConfigureCall' must have scalar return type">;
def err_kern_call_not_global_function : Error<
  "kernel call to non-global function %0">;


def err_cannot_pass_objc_interface_to_vararg : Error<
  "cannot pass object with interface type %0 by-value through variadic "
  "%select{function|block|method}1">;

def warn_cannot_pass_non_pod_arg_to_vararg : Warning<
  "cannot pass object of non-POD type %0 through variadic "
  "%select{function|block|method|constructor}1; call will abort at runtime">,
  InGroup<DiagGroup<"non-pod-varargs">>, DefaultError;

def err_typecheck_call_invalid_ordered_compare : Error<
  "ordered compare requires two args of floating point type (%0 and %1)">;
def err_typecheck_call_invalid_unary_fp : Error<
  "floating point classification requires argument of floating point type "
  "(passed in %0)">;
def err_typecheck_cond_expect_scalar : Error<
  "used type %0 where arithmetic or pointer type is required">;
def ext_typecheck_cond_one_void : Extension<
  "C99 forbids conditional expressions with only one void side">;
def err_typecheck_cond_expect_scalar_or_vector : Error<
  "used type %0 where arithmetic, pointer, or vector type is required">;
def err_typecheck_cast_to_incomplete : Error<
  "cast to incomplete type %0">;
def ext_typecheck_cast_nonscalar : Extension<
  "C99 forbids casting nonscalar type %0 to the same type">;
def ext_typecheck_cast_to_union : Extension<"C99 forbids casts to union type">;
def err_typecheck_cast_to_union_no_type : Error<
  "cast to union type from type %0 not present in union">;
def err_cast_pointer_from_non_pointer_int : Error<
  "operand of type %0 cannot be cast to a pointer type">;
def err_cast_pointer_to_non_pointer_int : Error<
  "pointer cannot be cast to type %0">;
def err_typecheck_expect_scalar_operand : Error<
  "operand of type %0 where arithmetic or pointer type is required">;
def err_typecheck_cond_incompatible_operands : Error<
  "incompatible operand types (%0 and %1)">;
def ext_typecheck_cond_incompatible_operands_nonstandard : ExtWarn<
  "incompatible operand types (%0 and %1) use non-standard composite pointer "
  "type %2">;
def err_cast_selector_expr : Error<
  "cannot type cast @selector expression">;
def warn_typecheck_cond_incompatible_pointers : ExtWarn<
  "pointer type mismatch (%0 and %1)">;
def warn_typecheck_cond_pointer_integer_mismatch : ExtWarn<
  "pointer/integer type mismatch in conditional expression (%0 and %1)">;
def err_typecheck_choose_expr_requires_constant : Error<
  "'__builtin_choose_expr' requires a constant expression">;
def ext_typecheck_expression_not_constant_but_accepted : Extension<
  "expression is not a constant, but is accepted as one by GNU extensions">, 
  InGroup<GNU>;
def warn_unused_expr : Warning<"expression result unused">,
  InGroup<UnusedValue>;
def warn_unused_voidptr : Warning<
  "expression result unused; should this cast be to 'void'?">,
  InGroup<UnusedValue>;
def warn_unused_property_expr : Warning<
 "property access result unused - getters should not be used for side effects">,
  InGroup<UnusedValue>;
def warn_unused_call : Warning<
  "ignoring return value of function declared with %0 attribute">,
  InGroup<UnusedValue>;

def err_incomplete_type_used_in_type_trait_expr : Error<
  "incomplete type %0 used in type trait expression">;
def err_expected_ident_or_lparen : Error<"expected identifier or '('">;

def err_typecheck_cond_incompatible_operands_null : Error<
  "non-pointer operand type %0 incompatible with %select{NULL|nullptr}1">;
} // End of general sema category.

// inline asm.
let CategoryName = "Inline Assembly Issue" in {
  def err_asm_wide_character : Error<"wide string is invalid in 'asm'">;
  def err_asm_invalid_lvalue_in_output : Error<"invalid lvalue in asm output">;
  def err_asm_invalid_output_constraint : Error<
    "invalid output constraint '%0' in asm">;
  def err_asm_invalid_lvalue_in_input : Error<
    "invalid lvalue in asm input for constraint '%0'">;
  def err_asm_invalid_input_constraint : Error<
    "invalid input constraint '%0' in asm">;
  def err_asm_invalid_type_in_input : Error<
    "invalid type %0 in asm input for constraint '%1'">;
  def err_asm_tying_incompatible_types : Error<
    "unsupported inline asm: input with type %0 matching output with type %1">;
  def err_asm_unknown_register_name : Error<"unknown register name '%0' in asm">;
  def warn_asm_label_on_auto_decl : Warning<
    "ignored asm label '%0' on automatic variable">;
  def err_invalid_asm_cast_lvalue : Error<
    "invalid use of a cast in a inline asm context requiring an l-value: "
    "remove the cast or build with -fheinous-gnu-extensions">;

  def warn_invalid_asm_cast_lvalue : Warning<
    "invalid use of a cast in a inline asm context requiring an l-value: "
    "accepted due to -fheinous-gnu-extensions, but clang may remove support "
    "for this in the future">;
}

let CategoryName = "Semantic Issue" in {

def err_invalid_conversion_between_vectors : Error<
  "invalid conversion between vector type %0 and %1 of different size">;
def err_invalid_conversion_between_vector_and_integer : Error<
  "invalid conversion between vector type %0 and integer type %1 "
  "of different size">;

def err_invalid_conversion_between_vector_and_scalar : Error<
  "invalid conversion between vector type %0 and scalar type %1">;
def err_overload_expr_requires_non_zero_constant : Error<
  "overload requires a non-zero constant expression as first argument">;
def err_overload_incorrect_fntype : Error<
  "argument is not a function, or has wrong number of parameters">;

// C++ member initializers.
def err_only_constructors_take_base_inits : Error<
  "only constructors take base initializers">;

def err_multiple_mem_initialization : Error <
  "multiple initializations given for non-static member %0">;
def err_multiple_mem_union_initialization : Error <
  "initializing multiple members of anonymous union">;
def err_multiple_base_initialization : Error <
  "multiple initializations given for base %0">;

def err_mem_init_not_member_or_class : Error<
  "member initializer %0 does not name a non-static data member or base "
  "class">;

def warn_initializer_out_of_order : Warning<
  "%select{field|base class}0 %1 will be initialized after "
  "%select{field|base}2 %3">,
  InGroup<Reorder>, DefaultIgnore;

def err_base_init_does_not_name_class : Error<
  "constructor initializer %0 does not name a class">;
def err_base_init_direct_and_virtual : Error<
  "base class initializer %0 names both a direct base class and an "
  "inherited virtual base class">;
def err_not_direct_base_or_virtual : Error<
  "type %0 is not a direct or virtual base of %1">;

def err_in_class_initializer_non_const : Error<
  "non-const static data member must be initialized out of line">;
def err_in_class_initializer_bad_type : Error<
  "static data member of type %0 must be initialized out of line">;
def ext_in_class_initializer_float_type : ExtWarn<
  "in-class initializer for static data member of type %0 "
  "is a C++0x extension">, InGroup<CXX0xStaticNonIntegralInitializer>;
def err_in_class_initializer_non_constant : Error<
  "in-class initializer is not a constant expression">;

// C++ anonymous unions and GNU anonymous structs/unions
def ext_anonymous_union : Extension<
  "anonymous unions are a GNU extension in C">, InGroup<GNU>;
def ext_anonymous_struct : Extension<
  "anonymous structs are a GNU extension">, InGroup<GNU>;
def err_anonymous_union_not_static : Error<
  "anonymous unions at namespace or global scope must be declared 'static'">;
def err_anonymous_union_with_storage_spec : Error<
  "anonymous union at class scope must not have a storage specifier">;
def err_anonymous_struct_not_member : Error<
  "anonymous %select{structs|structs and classes}0 must be "
  "%select{struct or union|class}0 members">;
def err_anonymous_union_member_redecl : Error<
  "member of anonymous union redeclares %0">;
def err_anonymous_struct_member_redecl : Error<
  "member of anonymous struct redeclares %0">;
def err_anonymous_record_with_type : Error<
  "types cannot be declared in an anonymous %select{struct|union}0">;
def ext_anonymous_record_with_type : Extension<
  "types declared in an anonymous %select{struct|union}0 are a Microsoft "
  "extension">, InGroup<Microsoft>;
def err_anonymous_record_with_function : Error<
  "functions cannot be declared in an anonymous %select{struct|union}0">;
def err_anonymous_record_with_static : Error<
  "static members cannot be declared in an anonymous %select{struct|union}0">;
def err_anonymous_record_bad_member : Error<
  "anonymous %select{struct|union}0 can only contain non-static data members">;
def err_anonymous_record_nonpublic_member : Error<
  "anonymous %select{struct|union}0 cannot contain a "
  "%select{private|protected}1 data member">;
def ext_ms_anonymous_struct : ExtWarn<
  "anonymous structs are a Microsoft extension">, InGroup<Microsoft>;

// C++ local classes
def err_reference_to_local_var_in_enclosing_function : Error<
  "reference to local variable %0 declared in enclosed function %1">;
def note_local_variable_declared_here : Note<
  "%0 declared here">;
def err_static_data_member_not_allowed_in_local_class : Error<
  "static data member %0 not allowed in local class %1">; 
  
// C++ derived classes
def err_base_clause_on_union : Error<"unions cannot have base classes">;
def err_base_must_be_class : Error<"base specifier must name a class">;
def err_union_as_base_class : Error<"unions cannot be base classes">;
def err_incomplete_base_class : Error<"base class has incomplete type">;
def err_duplicate_base_class : Error<
  "base class %0 specified more than once as a direct base class">;
// FIXME: better way to display derivation?  Pass entire thing into diagclient?
def err_ambiguous_derived_to_base_conv : Error<
  "ambiguous conversion from derived class %0 to base class %1:%2">;
def err_ambiguous_memptr_conv : Error<
  "ambiguous conversion from pointer to member of %select{base|derived}0 "
  "class %1 to pointer to member of %select{derived|base}0 class %2:%3">;

def err_memptr_conv_via_virtual : Error<
  "conversion from pointer to member of class %0 to pointer to member "
  "of class %1 via virtual base %2 is not allowed">;

// C++ access control
def err_conv_to_inaccessible_base : Error<
  "conversion from %0 to inaccessible base class %1">, AccessControl;
def note_inheritance_specifier_here : Note<
  "'%0' inheritance specifier here">;
def note_inheritance_implicitly_private_here : Note<
  "inheritance is implicitly 'private'">;

// C++ member name lookup
def err_ambiguous_member_multiple_subobjects : Error<
  "non-static member %0 found in multiple base-class subobjects of type %1:%2">;
def err_ambiguous_member_multiple_subobject_types : Error<
  "member %0 found in multiple base classes of different types">;
def note_ambiguous_member_found : Note<"member found by ambiguous name lookup">;
def err_ambiguous_reference : Error<"reference to %0 is ambiguous">;
def note_ambiguous_candidate : Note<"candidate found by name lookup is %q0">;
def err_ambiguous_tag_hiding : Error<"a type named %0 is hidden by a "
  "declaration in a different namespace">;
def note_hidden_tag : Note<"type declaration hidden">;
def note_hiding_object : Note<"declaration hides type">;

// C++ operator overloading
def err_operator_overload_needs_class_or_enum : Error<
  "overloaded %0 must have at least one parameter of class "
  "or enumeration type">;

def err_operator_overload_variadic : Error<"overloaded %0 cannot be variadic">;
def err_operator_overload_static : Error<
  "overloaded %0 cannot be a static member function">;
def err_operator_overload_default_arg : Error<
  "parameter of overloaded %0 cannot have a default argument">;
def err_operator_overload_must_be : Error<
  "overloaded %0 must be a %select{unary|binary|unary or binary}2 operator "
  "(has %1 parameter%s1)">;

def err_operator_overload_must_be_member : Error<
  "overloaded %0 must be a non-static member function">;
def err_operator_overload_post_incdec_must_be_int : Error<
  "parameter of overloaded post-%select{increment|decrement}1 operator must "
  "have type 'int' (not %0)">;

// C++ allocation and deallocation functions.
def err_operator_new_delete_declared_in_namespace : Error<
  "%0 cannot be declared inside a namespace">;
def err_operator_new_delete_declared_static : Error<
  "%0 cannot be declared static in global scope">;
def err_operator_new_delete_invalid_result_type : Error<
  "%0 must return type %1">;
def err_operator_new_delete_dependent_result_type : Error<
  "%0 cannot have a dependent return type; use %1 instead">;
def err_operator_new_delete_too_few_parameters : Error<
  "%0 must have at least one parameter.">;
def err_operator_new_delete_template_too_few_parameters : Error<
  "%0 template must have at least two parameters.">;

def err_operator_new_dependent_param_type : Error<
  "%0 cannot take a dependent type as first parameter; "
  "use size_t (%1) instead">;
def err_operator_new_param_type : Error<
  "%0 takes type size_t (%1) as first parameter">;
def err_operator_new_default_arg: Error<
  "parameter of %0 cannot have a default argument">;
def err_operator_delete_dependent_param_type : Error<
  "%0 cannot take a dependent type as first parameter; use %1 instead">;
def err_operator_delete_param_type : Error<
  "first parameter of %0 must have type %1">;

// C++ literal operators
def err_literal_operator_outside_namespace : Error<
  "literal operator %0 must be in a namespace or global scope">;
// FIXME: This diagnostic sucks
def err_literal_operator_params : Error<
  "parameter declaration for literal operator %0 is not valid">;

// C++ conversion functions
def err_conv_function_not_member : Error<
  "conversion function must be a non-static member function">;
def err_conv_function_return_type : Error<
  "conversion function cannot have a return type">;
def err_conv_function_with_params : Error<
  "conversion function cannot have any parameters">;
def err_conv_function_variadic : Error<
  "conversion function cannot be variadic">;
def err_conv_function_to_array : Error<
  "conversion function cannot convert to an array type">;
def err_conv_function_to_function : Error<
  "conversion function cannot convert to a function type">;
def err_conv_function_with_complex_decl : Error<
  "must use a typedef to declare a conversion to %0">;
def err_conv_function_redeclared : Error<
  "conversion function cannot be redeclared">;
def warn_conv_to_self_not_used : Warning<
  "conversion function converting %0 to itself will never be used">;
def warn_conv_to_base_not_used : Warning<
  "conversion function converting %0 to its base class %1 will never be used">;
def warn_conv_to_void_not_used : Warning<
  "conversion function converting %0 to %1 will never be used">;

def warn_not_compound_assign : Warning<
  "use of unary operator that may be intended as compound assignment (%0=)">;

// C++0x explicit conversion operators
def warn_explicit_conversion_functions : Warning<
  "explicit conversion functions are a C++0x extension">, InGroup<CXX0x>;

def warn_array_index_precedes_bounds : Warning<
  "array index of '%0' indexes before the beginning of the array">,
  InGroup<DiagGroup<"array-bounds">>;
def warn_array_index_exceeds_bounds : Warning<
  "array index of '%0' indexes past the end of an array (that contains %1 elements)">,
  InGroup<DiagGroup<"array-bounds">>;
def note_array_index_out_of_bounds : Note<
  "array %0 declared here">;

def warn_printf_write_back : Warning<
  "use of '%%n' in format string discouraged (potentially insecure)">,
  InGroup<FormatSecurity>;
def warn_printf_insufficient_data_args : Warning<
  "more '%%' conversions than data arguments">, InGroup<Format>;
def warn_printf_data_arg_not_used : Warning<
  "data argument not used by format string">, InGroup<FormatExtraArgs>;
def warn_format_invalid_conversion : Warning<
  "invalid conversion specifier '%0'">, InGroup<Format>;
def warn_printf_incomplete_specifier : Warning<
  "incomplete format specifier">, InGroup<Format>;
def warn_missing_format_string : Warning<
  "format string missing">, InGroup<Format>;
def warn_scanf_nonzero_width : Warning<
  "zero field width in scanf format string is unused">,
  InGroup<Format>;
def warn_printf_conversion_argument_type_mismatch : Warning<
  "conversion specifies type %0 but the argument has type %1">,
  InGroup<Format>;
def warn_printf_positional_arg_exceeds_data_args : Warning <
  "data argument position '%0' exceeds the number of data arguments (%1)">,
  InGroup<Format>;
def warn_format_zero_positional_specifier : Warning<
  "position arguments in format strings start counting at 1 (not 0)">,
  InGroup<Format>;
def warn_format_invalid_positional_specifier : Warning<
  "invalid position specified for %select{field width|field precision}0">,
  InGroup<Format>;
def warn_format_mix_positional_nonpositional_args : Warning<
  "cannot mix positional and non-positional arguments in format string">,
  InGroup<Format>;
def warn_null_arg : Warning<
  "null passed to a callee which requires a non-null argument">,
  InGroup<NonNull>;
def warn_empty_format_string : Warning<
  "format string is empty">, InGroup<FormatZeroLength>;
def warn_format_string_is_wide_literal : Warning<
  "format string should not be a wide string">, InGroup<Format>;
def warn_printf_format_string_contains_null_char : Warning<
  "format string contains '\\0' within the string body">, InGroup<Format>;
def warn_printf_asterisk_missing_arg : Warning<
  "'%select{*|.*}0' specified field %select{width|precision}0 is missing a matching 'int' argument">;
def warn_printf_asterisk_wrong_type : Warning<
  "field %select{width|precision}0 should have type %1, but argument has type %2">,
  InGroup<Format>;
def warn_printf_nonsensical_optional_amount: Warning<
  "%select{field width|precision}0 used with '%1' conversion specifier, resulting in undefined behavior">,
  InGroup<Format>;
def warn_printf_nonsensical_flag: Warning<
  "flag '%0' results in undefined behavior with '%1' conversion specifier">,
  InGroup<Format>;
def warn_format_nonsensical_length: Warning<
  "length modifier '%0' results in undefined behavior or no effect with '%1' conversion specifier">,
  InGroup<Format>;
def warn_printf_ignored_flag: Warning<
  "flag '%0' is ignored when flag '%1' is present">,
  InGroup<Format>;
def warn_scanf_scanlist_incomplete : Warning<
  "no closing ']' for '%%[' in scanf format string">,
  InGroup<Format>;
  
// CHECK: returning address/reference of stack memory
def warn_ret_stack_addr : Warning<
  "address of stack memory associated with local variable %0 returned">;
def warn_ret_stack_ref : Warning<
  "reference to stack memory associated with local variable %0 returned">;
def warn_ret_local_temp_addr : Warning<
  "returning address of local temporary object">;
def warn_ret_local_temp_ref : Warning<
  "returning reference to local temporary object">;
def warn_ret_addr_label : Warning<
  "returning address of label, which is local">;
def err_ret_local_block : Error<
  "returning block that lives on the local stack">;
def note_ref_var_local_bind : Note<
  "binding reference variable %0 here">;


// For non-floating point, expressions of the form x == x or x != x
// should result in a warning, since these always evaluate to a constant.
// Array comparisons have similar warnings
def warn_comparison_always : Warning<
  "%select{self-|array }0comparison always evaluates to %select{false|true|a constant}1">,
  InGroup<TautologicalCompare>;

def warn_stringcompare : Warning<
  "result of comparison against %select{a string literal|@encode}0 is "
  "unspecified (use strncmp instead)">;



// Blocks
def err_blocks_disable : Error<"blocks support disabled - compile with -fblocks"
  " or pick a deployment target that supports them">;
def err_expected_block_lbrace : Error<"expected '{' in block literal">;
def err_return_in_block_expression : Error<
  "return not allowed in block expression literal">;
def err_block_returning_array_function : Error<
  "block cannot return %select{array|function}0 type %1">;


// CFString checking
def err_cfstring_literal_not_string_constant : Error<
  "CFString literal is not a string constant">;
def warn_cfstring_literal_contains_nul_character : Warning<
  "CFString literal contains NUL character">;
def warn_cfstring_truncated : Warning<
  "input conversion stopped due to an input byte that does not "
  "belong to the input codeset UTF-8">;

// Statements.
def err_continue_not_in_loop : Error<
  "'continue' statement not in loop statement">;
def err_break_not_in_loop_or_switch : Error<
  "'break' statement not in loop or switch statement">;
def err_default_not_in_switch : Error<
  "'default' statement not in switch statement">;
def err_case_not_in_switch : Error<"'case' statement not in switch statement">;
def warn_bool_switch_condition : Warning<
  "switch condition has boolean value">;
def warn_case_value_overflow : Warning<
  "overflow converting case value to switch condition type (%0 to %1)">,
  InGroup<DiagGroup<"switch">>;
def err_duplicate_case : Error<"duplicate case value '%0'">;
def warn_case_empty_range : Warning<"empty case range specified">;
def warn_missing_case_for_condition :
  Warning<"no case matching constant switch condition '%0'">;
def warn_missing_case1 : Warning<"enumeration value %0 not handled in switch">,
  InGroup<DiagGroup<"switch-enum"> >;
def warn_missing_case2 : Warning<
  "enumeration values %0 and %1 not handled in switch">,
  InGroup<DiagGroup<"switch-enum"> >;
def warn_missing_case3 : Warning<
  "enumeration values %0, %1, and %2 not handled in switch">,
  InGroup<DiagGroup<"switch-enum"> >;
def warn_missing_cases : Warning<
  "%0 enumeration values not handled in switch: %1, %2, %3...">,
  InGroup<DiagGroup<"switch-enum"> >;

def warn_not_in_enum : Warning<"case value not in enumerated type %0">,
  InGroup<DiagGroup<"switch-enum"> >; 
def err_typecheck_statement_requires_scalar : Error<
  "statement requires expression of scalar type (%0 invalid)">;
def err_typecheck_statement_requires_integer : Error<
  "statement requires expression of integer type (%0 invalid)">;
def err_multiple_default_labels_defined : Error<
  "multiple default labels in one switch">;
def err_switch_multiple_conversions : Error<
  "multiple conversions from switch condition type %0 to an integral or "
  "enumeration type">;
def note_switch_conversion : Note<
  "conversion to %select{integral|enumeration}0 type %1">;
def err_switch_explicit_conversion : Error<
  "switch condition type %0 requires explicit conversion to %1">;
def err_switch_incomplete_class_type : Error<
  "switch condition has incomplete class type %0">;
def warn_empty_if_body : Warning<
  "if statement has empty body">, InGroup<EmptyBody>;
def err_va_start_used_in_non_variadic_function : Error<
  "'va_start' used in function with fixed args">;
def warn_second_parameter_of_va_start_not_last_named_argument : Warning<
  "second parameter of 'va_start' not last named argument">;
def err_first_argument_to_va_arg_not_of_type_va_list : Error<
  "first argument to 'va_arg' is of type %0 and not 'va_list'">;

def warn_return_missing_expr : Warning<
  "non-void %select{function|method}1 %0 should return a value">, DefaultError,
  InGroup<ReturnType>;
def ext_return_missing_expr : ExtWarn<
  "non-void %select{function|method}1 %0 should return a value">, DefaultError,
  InGroup<ReturnType>;
def ext_return_has_expr : ExtWarn<
  "void %select{function|method}1 %0 should not return a value">, DefaultError,
  InGroup<ReturnType>;
def ext_return_has_void_expr : Extension<
  "void %select{function|method}1 %0 should not return void expression">;
def warn_noreturn_function_has_return_expr : Warning<
  "function %0 declared 'noreturn' should not return">,
  InGroup<DiagGroup<"invalid-noreturn">>;
def warn_falloff_noreturn_function : Warning<
  "function declared 'noreturn' should not return">,
  InGroup<DiagGroup<"invalid-noreturn">>;
def err_noreturn_block_has_return_expr : Error<
  "block declared 'noreturn' should not return">;
def err_block_on_nonlocal : Error<
  "__block attribute not allowed, only allowed on local variables">;
def err_block_on_vm : Error<
  "__block attribute not allowed on declaration with a variably modified type">;

def err_shufflevector_non_vector : Error<
  "first two arguments to __builtin_shufflevector must be vectors">;
def err_shufflevector_incompatible_vector : Error<
  "first two arguments to __builtin_shufflevector must have the same type">;
def err_shufflevector_nonconstant_argument : Error<
  "index for __builtin_shufflevector must be a constant integer">;
def err_shufflevector_argument_too_large : Error<
  "index for __builtin_shufflevector must be less than the total number "
  "of vector elements">;

def err_vector_incorrect_num_initializers : Error<
  "%select{too many|too few}0 elements in vector initialization (expected %1 elements, have %2)">;
def err_altivec_empty_initializer : Error<"expected initializer">;

def err_invalid_neon_type_code : Error<
  "incompatible constant for this __builtin_neon function">; 
def err_argument_invalid_range : Error<
  "argument should be a value from %0 to %1">;

def err_builtin_longjmp_invalid_val : Error<
  "argument to __builtin_longjmp must be a constant 1">;

def err_constant_integer_arg_type : Error<
  "argument to %0 must be a constant integer">;

def ext_mixed_decls_code : Extension<
  "ISO C90 forbids mixing declarations and code">,
  InGroup<DiagGroup<"declaration-after-statement">>;
  
def err_non_variable_decl_in_for : Error<
  "declaration of non-local variable in 'for' loop">;
def err_toomany_element_decls : Error<
  "only one element declaration is allowed">;
def err_selector_element_not_lvalue : Error<
  "selector element is not a valid lvalue">;
def err_selector_element_type : Error<
  "selector element type %0 is not a valid object">;
def err_collection_expr_type : Error<
  "collection expression type %0 is not a valid object">;
def warn_collection_expr_type : Warning<
  "collection expression type %0 may not respond to %1">;

def err_invalid_conversion_between_ext_vectors : Error<
  "invalid conversion between ext-vector type %0 and %1">;

// Type
def ext_invalid_sign_spec : Extension<"'%0' cannot be signed or unsigned">;
def warn_receiver_forward_class : Warning<
    "receiver %0 is a forward class and corresponding @interface may not exist">;
def note_method_sent_forward_class : Note<"method %0 is used for the forward class">;
def ext_missing_declspec : ExtWarn<
  "declaration specifier missing, defaulting to 'int'">;
def ext_missing_type_specifier : ExtWarn<
  "type specifier missing, defaults to 'int'">,
  InGroup<ImplicitInt>;
def err_decimal_unsupported : Error<
  "GNU decimal type extension not supported">;
def err_missing_type_specifier : Error<
  "C++ requires a type specifier for all declarations">;
def err_missing_param_declspec : Error<
  "parameter requires a declaration specifier">;
def err_objc_array_of_interfaces : Error<
  "array of interface %0 is invalid (probably should be an array of pointers)">;
def ext_c99_array_usage : Extension<
  "use of C99-specific array features, accepted as an extension">;
def err_c99_array_usage_cxx : Error<
  "C99-specific array features are not permitted in C++">;
def err_double_requires_fp64 : Error<
  "use of type 'double' requires cl_khr_fp64 extension to be enabled">;
  
def note_getter_unavailable : Note<
  "or because setter is declared here, but no getter method %0 is found">;
def err_invalid_protocol_qualifiers : Error<
  "invalid protocol qualifiers on non-ObjC type">;
def warn_ivar_use_hidden : Warning<
  "local declaration of %0 hides instance variable">;
def error_ivar_use_in_class_method : Error<
  "instance variable %0 accessed in class method">;
def error_private_ivar_access : Error<"instance variable %0 is private">,
  AccessControl;
def error_protected_ivar_access : Error<"instance variable %0 is protected">,
  AccessControl;
def warn_maynot_respond : Warning<"%0 may not respond to %1">;
def warn_attribute_method_def : Warning<
  "method attribute can only be specified on method declarations">;
def ext_typecheck_base_super : Warning<
  "method parameter type %0 does not match "
  "super class method parameter type %1">, InGroup<SuperSubClassMismatch>, DefaultIgnore;

// Spell-checking diagnostics
def err_unknown_typename_suggest : Error<
  "unknown type name %0; did you mean %1?">;
def err_unknown_nested_typename_suggest : Error<
  "no type named %0 in %1; did you mean %2?">;
def err_no_member_suggest : Error<"no member named %0 in %1; did you mean %2?">;
def err_undeclared_use_suggest : Error<
  "use of undeclared %0; did you mean %1?">;
def err_undeclared_var_use_suggest : Error<
  "use of undeclared identifier %0; did you mean %1?">;
def err_no_template_suggest : Error<"no template named %0; did you mean %1?">;
def err_no_member_template_suggest : Error<
  "no template named %0 in %1; did you mean %2?">;
def err_mem_init_not_member_or_class_suggest : Error<
  "initializer %0 does not name a non-static data member or base "
  "class; did you mean the %select{base class|member}1 %2?">;
def err_field_designator_unknown_suggest : Error<
  "field designator %0 does not refer to any field in type %1; did you mean "
  "%2?">;
def err_typecheck_member_reference_ivar_suggest : Error<
  "%0 does not have a member named %1; did you mean %2?">;
def err_property_not_found_suggest : Error<
  "property %0 not found on object of type %1; did you mean %2?">;
def err_undef_interface_suggest : Error<
  "cannot find interface declaration for %0; did you mean %1?">;
def warn_undef_interface_suggest : Warning<
  "cannot find interface declaration for %0; did you mean %1?">;
def err_undef_superclass_suggest : Error<
  "cannot find interface declaration for %0, superclass of %1; did you mean "
  "%2?">;
def err_undeclared_protocol_suggest : Error<
  "cannot find protocol declaration for %0; did you mean %1?">;
def note_base_class_specified_here : Note<
  "base class %0 specified here">;
def err_using_directive_suggest : Error<
  "no namespace named %0; did you mean %1?">;
def err_using_directive_member_suggest : Error<
  "no namespace named %0 in %1; did you mean %2?">;
def note_namespace_defined_here : Note<"namespace %0 defined here">;
def err_sizeof_pack_no_pack_name_suggest : Error<
  "%0 does not refer to the name of a parameter pack; did you mean %1?">;
def note_parameter_pack_here : Note<"parameter pack %0 declared here">;

} // end of sema category
} // end of sema component.


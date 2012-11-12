// RUN: %clang_cc1 -E -std=c++11 %s -o - | FileCheck --check-prefix=CHECK-0X %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-0X %s

#if __has_feature(cxx_atomic)
int has_atomic();
#else
int no_atomic();
#endif

// CHECK-0X: has_atomic
// CHECK-NO-0X: no_atomic

#if __has_feature(cxx_lambdas)
int has_lambdas();
#else
int no_lambdas();
#endif

// CHECK-0X: has_lambdas
// CHECK-NO-0X: no_lambdas


#if __has_feature(cxx_nullptr)
int has_nullptr();
#else
int no_nullptr();
#endif

// CHECK-0X: has_nullptr
// CHECK-NO-0X: no_nullptr


#if __has_feature(cxx_decltype)
int has_decltype();
#else
int no_decltype();
#endif

// CHECK-0X: has_decltype
// CHECK-NO-0X: no_decltype


#if __has_feature(cxx_decltype_incomplete_return_types)
int has_decltype_incomplete_return_types();
#else
int no_decltype_incomplete_return_types();
#endif

// CHECK-0X: has_decltype_incomplete_return_types
// CHECK-NO-0X: no_decltype_incomplete_return_types


#if __has_feature(cxx_auto_type)
int has_auto_type();
#else
int no_auto_type();
#endif

// CHECK-0X: has_auto_type
// CHECK-NO-0X: no_auto_type


#if __has_feature(cxx_trailing_return)
int has_trailing_return();
#else
int no_trailing_return();
#endif

// CHECK-0X: has_trailing_return
// CHECK-NO-0X: no_trailing_return


#if __has_feature(cxx_attributes)
int has_attributes();
#else
int no_attributes();
#endif

// CHECK-0X: has_attributes
// CHECK-NO-0X: no_attributes


#if __has_feature(cxx_static_assert)
int has_static_assert();
#else
int no_static_assert();
#endif

// CHECK-0X: has_static_assert
// CHECK-NO-0X: no_static_assert

#if __has_feature(cxx_deleted_functions)
int has_deleted_functions();
#else
int no_deleted_functions();
#endif

// CHECK-0X: has_deleted_functions
// CHECK-NO-0X: no_deleted_functions

#if __has_feature(cxx_defaulted_functions)
int has_defaulted_functions();
#else
int no_defaulted_functions();
#endif

// CHECK-0X: has_defaulted_functions
// CHECK-NO-0X: no_defaulted_functions

#if __has_feature(cxx_rvalue_references)
int has_rvalue_references();
#else
int no_rvalue_references();
#endif

// CHECK-0X: has_rvalue_references
// CHECK-NO-0X: no_rvalue_references


#if __has_feature(cxx_variadic_templates)
int has_variadic_templates();
#else
int no_variadic_templates();
#endif

// CHECK-0X: has_variadic_templates
// CHECK-NO-0X: no_variadic_templates


#if __has_feature(cxx_inline_namespaces)
int has_inline_namespaces();
#else
int no_inline_namespaces();
#endif

// CHECK-0X: has_inline_namespaces
// CHECK-NO-0X: no_inline_namespaces


#if __has_feature(cxx_range_for)
int has_range_for();
#else
int no_range_for();
#endif

// CHECK-0X: has_range_for
// CHECK-NO-0X: no_range_for


#if __has_feature(cxx_reference_qualified_functions)
int has_reference_qualified_functions();
#else
int no_reference_qualified_functions();
#endif

// CHECK-0X: has_reference_qualified_functions
// CHECK-NO-0X: no_reference_qualified_functions

#if __has_feature(cxx_default_function_template_args)
int has_default_function_template_args();
#else
int no_default_function_template_args();
#endif

// CHECK-0X: has_default_function_template_args
// CHECK-NO-0X: no_default_function_template_args

#if __has_feature(cxx_noexcept)
int has_noexcept();
#else
int no_noexcept();
#endif

// CHECK-0X: has_noexcept
// CHECK-NO-0X: no_noexcept

#if __has_feature(cxx_override_control)
int has_override_control();
#else
int no_override_control();
#endif

// CHECK-0X: has_override_control
// CHECK-NO-0X: no_override_control

#if __has_feature(cxx_alias_templates)
int has_alias_templates();
#else
int no_alias_templates();
#endif

// CHECK-0X: has_alias_templates
// CHECK-NO-0X: no_alias_templates

#if __has_feature(cxx_implicit_moves)
int has_implicit_moves();
#else
int no_implicit_moves();
#endif

// CHECK-0X: has_implicit_moves
// CHECK-NO-0X: no_implicit_moves

#if __has_feature(cxx_alignas)
int has_alignas();
#else
int no_alignas();
#endif

// CHECK-0X: has_alignas
// CHECK-NO-0X: no_alignas

#if __has_feature(cxx_raw_string_literals)
int has_raw_string_literals();
#else
int no_raw_string_literals();
#endif

// CHECK-0X: has_raw_string_literals
// CHECK-NO-0X: no_raw_string_literals

#if __has_feature(cxx_unicode_literals)
int has_unicode_literals();
#else
int no_unicode_literals();
#endif

// CHECK-0X: has_unicode_literals
// CHECK-NO-0X: no_unicode_literals

#if __has_feature(cxx_constexpr)
int has_constexpr();
#else
int no_constexpr();
#endif

// CHECK-0X: has_constexpr
// CHECK-NO-0X: no_constexpr

#if __has_feature(cxx_generalized_initializers)
int has_generalized_initializers();
#else
int no_generalized_initializers();
#endif

// CHECK-0X: has_generalized_initializers
// CHECK-NO-0X: no_generalized_initializers

#if __has_feature(cxx_unrestricted_unions)
int has_unrestricted_unions();
#else
int no_unrestricted_unions();
#endif

// CHECK-0X: has_unrestricted_unions
// CHECK-NO-0X: no_unrestricted_unions

#if __has_feature(cxx_user_literals)
int has_user_literals();
#else
int no_user_literals();
#endif

// CHECK-0X: has_user_literals
// CHECK-NO-0X: no_user_literals

#if __has_feature(cxx_local_type_template_args)
int has_local_type_template_args();
#else
int no_local_type_template_args();
#endif

// CHECK-0X: has_local_type_template_args
// CHECK-NO-0X: no_local_type_template_args

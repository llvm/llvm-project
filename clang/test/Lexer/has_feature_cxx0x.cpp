// RUN: %clang_cc1 -E -triple x86_64-linux-gnu -std=c++11 %s -o - | FileCheck --check-prefix=CHECK-11 %s
// RUN: %clang_cc1 -E -triple armv7-apple-darwin -std=c++11 %s -o - | FileCheck --check-prefix=CHECK-NO-TLS %s
// RUN: %clang_cc1 -E -triple x86_64-linux-gnu %s -o - | FileCheck --check-prefix=CHECK-NO-11 %s
// RUN: %clang_cc1 -E -triple x86_64-linux-gnu -std=c++1y %s -o - | FileCheck --check-prefix=CHECK-1Y %s

#if __has_feature(cxx_atomic)
int has_atomic();
#else
int no_atomic();
#endif

// CHECK-1Y: has_atomic
// CHECK-11: has_atomic
// CHECK-NO-11: no_atomic

#if __has_feature(cxx_lambdas)
int has_lambdas();
#else
int no_lambdas();
#endif

// CHECK-1Y: has_lambdas
// CHECK-11: has_lambdas
// CHECK-NO-11: no_lambdas


#if __has_feature(cxx_nullptr)
int has_nullptr();
#else
int no_nullptr();
#endif

// CHECK-1Y: has_nullptr
// CHECK-11: has_nullptr
// CHECK-NO-11: no_nullptr


#if __has_feature(cxx_decltype)
int has_decltype();
#else
int no_decltype();
#endif

// CHECK-1Y: has_decltype
// CHECK-11: has_decltype
// CHECK-NO-11: no_decltype


#if __has_feature(cxx_decltype_incomplete_return_types)
int has_decltype_incomplete_return_types();
#else
int no_decltype_incomplete_return_types();
#endif

// CHECK-1Y: has_decltype_incomplete_return_types
// CHECK-11: has_decltype_incomplete_return_types
// CHECK-NO-11: no_decltype_incomplete_return_types


#if __has_feature(cxx_auto_type)
int has_auto_type();
#else
int no_auto_type();
#endif

// CHECK-1Y: has_auto_type
// CHECK-11: has_auto_type
// CHECK-NO-11: no_auto_type


#if __has_feature(cxx_trailing_return)
int has_trailing_return();
#else
int no_trailing_return();
#endif

// CHECK-1Y: has_trailing_return
// CHECK-11: has_trailing_return
// CHECK-NO-11: no_trailing_return


#if __has_feature(cxx_attributes)
int has_attributes();
#else
int no_attributes();
#endif

// CHECK-1Y: has_attributes
// CHECK-11: has_attributes
// CHECK-NO-11: no_attributes


#if __has_feature(cxx_static_assert)
int has_static_assert();
#else
int no_static_assert();
#endif

// CHECK-1Y: has_static_assert
// CHECK-11: has_static_assert
// CHECK-NO-11: no_static_assert

#if __has_feature(cxx_deleted_functions)
int has_deleted_functions();
#else
int no_deleted_functions();
#endif

// CHECK-1Y: has_deleted_functions
// CHECK-11: has_deleted_functions
// CHECK-NO-11: no_deleted_functions

#if __has_feature(cxx_defaulted_functions)
int has_defaulted_functions();
#else
int no_defaulted_functions();
#endif

// CHECK-1Y: has_defaulted_functions
// CHECK-11: has_defaulted_functions
// CHECK-NO-11: no_defaulted_functions

#if __has_feature(cxx_rvalue_references)
int has_rvalue_references();
#else
int no_rvalue_references();
#endif

// CHECK-1Y: has_rvalue_references
// CHECK-11: has_rvalue_references
// CHECK-NO-11: no_rvalue_references


#if __has_feature(cxx_variadic_templates)
int has_variadic_templates();
#else
int no_variadic_templates();
#endif

// CHECK-1Y: has_variadic_templates
// CHECK-11: has_variadic_templates
// CHECK-NO-11: no_variadic_templates


#if __has_feature(cxx_inline_namespaces)
int has_inline_namespaces();
#else
int no_inline_namespaces();
#endif

// CHECK-1Y: has_inline_namespaces
// CHECK-11: has_inline_namespaces
// CHECK-NO-11: no_inline_namespaces


#if __has_feature(cxx_range_for)
int has_range_for();
#else
int no_range_for();
#endif

// CHECK-1Y: has_range_for
// CHECK-11: has_range_for
// CHECK-NO-11: no_range_for


#if __has_feature(cxx_reference_qualified_functions)
int has_reference_qualified_functions();
#else
int no_reference_qualified_functions();
#endif

// CHECK-1Y: has_reference_qualified_functions
// CHECK-11: has_reference_qualified_functions
// CHECK-NO-11: no_reference_qualified_functions

#if __has_feature(cxx_default_function_template_args)
int has_default_function_template_args();
#else
int no_default_function_template_args();
#endif

// CHECK-1Y: has_default_function_template_args
// CHECK-11: has_default_function_template_args
// CHECK-NO-11: no_default_function_template_args

#if __has_feature(cxx_noexcept)
int has_noexcept();
#else
int no_noexcept();
#endif

// CHECK-1Y: has_noexcept
// CHECK-11: has_noexcept
// CHECK-NO-11: no_noexcept

#if __has_feature(cxx_override_control)
int has_override_control();
#else
int no_override_control();
#endif

// CHECK-1Y: has_override_control
// CHECK-11: has_override_control
// CHECK-NO-11: no_override_control

#if __has_feature(cxx_alias_templates)
int has_alias_templates();
#else
int no_alias_templates();
#endif

// CHECK-1Y: has_alias_templates
// CHECK-11: has_alias_templates
// CHECK-NO-11: no_alias_templates

#if __has_feature(cxx_implicit_moves)
int has_implicit_moves();
#else
int no_implicit_moves();
#endif

// CHECK-1Y: has_implicit_moves
// CHECK-11: has_implicit_moves
// CHECK-NO-11: no_implicit_moves

#if __has_feature(cxx_alignas)
int has_alignas();
#else
int no_alignas();
#endif

// CHECK-1Y: has_alignas
// CHECK-11: has_alignas
// CHECK-NO-11: no_alignas

#if __has_feature(cxx_raw_string_literals)
int has_raw_string_literals();
#else
int no_raw_string_literals();
#endif

// CHECK-1Y: has_raw_string_literals
// CHECK-11: has_raw_string_literals
// CHECK-NO-11: no_raw_string_literals

#if __has_feature(cxx_unicode_literals)
int has_unicode_literals();
#else
int no_unicode_literals();
#endif

// CHECK-1Y: has_unicode_literals
// CHECK-11: has_unicode_literals
// CHECK-NO-11: no_unicode_literals

#if __has_feature(cxx_constexpr)
int has_constexpr();
#else
int no_constexpr();
#endif

// CHECK-1Y: has_constexpr
// CHECK-11: has_constexpr
// CHECK-NO-11: no_constexpr

#if __has_feature(cxx_generalized_initializers)
int has_generalized_initializers();
#else
int no_generalized_initializers();
#endif

// CHECK-1Y: has_generalized_initializers
// CHECK-11: has_generalized_initializers
// CHECK-NO-11: no_generalized_initializers

#if __has_feature(cxx_unrestricted_unions)
int has_unrestricted_unions();
#else
int no_unrestricted_unions();
#endif

// CHECK-1Y: has_unrestricted_unions
// CHECK-11: has_unrestricted_unions
// CHECK-NO-11: no_unrestricted_unions

#if __has_feature(cxx_user_literals)
int has_user_literals();
#else
int no_user_literals();
#endif

// CHECK-1Y: has_user_literals
// CHECK-11: has_user_literals
// CHECK-NO-11: no_user_literals

#if __has_feature(cxx_local_type_template_args)
int has_local_type_template_args();
#else
int no_local_type_template_args();
#endif

// CHECK-1Y: has_local_type_template_args
// CHECK-11: has_local_type_template_args
// CHECK-NO-11: no_local_type_template_args

#if __has_feature(cxx_inheriting_constructors)
int has_inheriting_constructors();
#else
int no_inheriting_constructors();
#endif

// CHECK-1Y: has_inheriting_constructors
// CHECK-11: has_inheriting_constructors
// CHECK-NO-11: no_inheriting_constructors

#if __has_feature(cxx_thread_local)
int has_thread_local();
#else
int no_thread_local();
#endif

// CHECK-1Y: has_thread_local
// CHECK-11: has_thread_local
// CHECK-NO-11: no_thread_local
// CHECK-NO-TLS: no_thread_local

// === C++1y features ===

#if __has_feature(cxx_binary_literals)
int has_binary_literals();
#else
int no_binary_literals();
#endif

// CHECK-1Y: has_binary_literals
// CHECK-11: no_binary_literals
// CHECK-NO-11: no_binary_literals

#if __has_feature(cxx_aggregate_nsdmi)
int has_aggregate_nsdmi();
#else
int no_aggregate_nsdmi();
#endif

// CHECK-1Y: has_aggregate_nsdmi
// CHECK-11: no_aggregate_nsdmi
// CHECK-NO-11: no_aggregate_nsdmi

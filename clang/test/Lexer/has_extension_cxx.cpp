// RUN: %clang_cc1 -std=c++98 -E %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -E %s -o - | FileCheck %s --check-prefix=CHECK11

// CHECK: c_static_assert
#if __has_extension(c_static_assert)
int c_static_assert();
#endif

// CHECK: c_generic_selections
#if __has_extension(c_generic_selections)
int c_generic_selections();
#endif

// CHECK: has_default_function_template_args
#if __has_extension(cxx_default_function_template_args)
int has_default_function_template_args();
#endif

// CHECK: has_defaulted_functions
#if __has_extension(cxx_defaulted_functions)
int has_defaulted_functions();
#endif

// CHECK: has_deleted_functions
#if __has_extension(cxx_deleted_functions)
int has_deleted_functions();
#endif

// CHECK: has_inline_namespaces
#if __has_extension(cxx_inline_namespaces)
int has_inline_namespaces();
#endif

// CHECK: has_override_control
#if __has_extension(cxx_override_control)
int has_override_control();
#endif

// CHECK: has_range_for
#if __has_extension(cxx_range_for)
int has_range_for();
#endif

// CHECK: has_reference_qualified_functions
#if __has_extension(cxx_reference_qualified_functions)
int has_reference_qualified_functions();
#endif

// CHECK: has_rvalue_references
#if __has_extension(cxx_rvalue_references)
int has_rvalue_references();
#endif

// CHECK: has_variadic_templates
#if __has_extension(cxx_variadic_templates)
int has_variadic_templates();
#endif

// CHECK: has_local_type_template_args
#if __has_extension(cxx_local_type_template_args)
int has_local_type_template_args();
#endif

// CHECK: has_binary_literals
#if __has_extension(cxx_binary_literals)
int has_binary_literals();
#endif

// CHECK: has_variable_templates
#if __has_extension(cxx_variable_templates)
int has_variable_templates();
#endif

// CHECK-NOT: has_init_captures
// CHECK11: has_init_captures
#if __has_extension(cxx_init_captures)
int has_init_captures();
#endif

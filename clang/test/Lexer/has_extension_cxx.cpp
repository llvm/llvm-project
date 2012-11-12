// RUN: %clang_cc1 -E %s -o - | FileCheck %s

// CHECK: c_static_assert
#if __has_extension(c_static_assert)
int c_static_assert();
#endif

// CHECK: c_generic_selections
#if __has_extension(c_generic_selections)
int c_generic_selections();
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

#if __has_extension(cxx_local_type_template_args)
int has_local_type_template_args();
#else
int no_local_type_template_args();
#endif

// CHECK: has_local_type_template_args

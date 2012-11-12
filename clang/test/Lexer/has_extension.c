// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-PED-NONE %s
// RUN: %clang_cc1 -pedantic-errors -E %s -o - | FileCheck --check-prefix=CHECK-PED-ERR %s

// CHECK-PED-NONE: no_dummy_extension
#if !__has_extension(dummy_extension)
int no_dummy_extension();
#endif

// Arbitrary feature to test that has_extension is a superset of has_feature
// CHECK-PED-NONE: attribute_overloadable
#if __has_extension(attribute_overloadable)
int attribute_overloadable();
#endif

// CHECK-PED-NONE: has_c_static_assert
// CHECK-PED-ERR: no_c_static_assert
#if __has_extension(c_static_assert)
int has_c_static_assert();
#else
int no_c_static_assert();
#endif

// CHECK-PED-NONE: has_c_generic_selections
// CHECK-PED-ERR: no_c_generic_selections
#if __has_extension(c_generic_selections)
int has_c_generic_selections();
#else
int no_c_generic_selections();
#endif

// CHECK-PED-NONE: has_c_alignas
// CHECK-PED-ERR: no_c_alignas
#if __has_extension(c_alignas)
int has_c_alignas();
#else
int no_c_alignas();
#endif

// Arbitrary feature to test that the extension name can be surrounded with
// double underscores.
// CHECK-PED-NONE: has_double_underscores
#if __has_extension(__c_alignas__)
int has_double_underscores();
#endif

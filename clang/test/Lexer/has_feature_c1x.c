// RUN: %clang_cc1 -E -std=c1x %s -o - | FileCheck --check-prefix=CHECK-1X %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-1X %s

#if __has_feature(c_atomic)
int has_atomic();
#else
int no_atomic();
#endif

// CHECK-1X: has_atomic
// CHECK-NO-1X: no_atomic

#if __has_feature(c_static_assert)
int has_static_assert();
#else
int no_static_assert();
#endif

// CHECK-1X: has_static_assert
// CHECK-NO-1X: no_static_assert

#if __has_feature(c_generic_selections)
int has_generic_selections();
#else
int no_generic_selections();
#endif

// CHECK-1X: has_generic_selections
// CHECK-NO-1X: no_generic_selections

#if __has_feature(c_alignas)
int has_alignas();
#else
int no_alignas();
#endif

// CHECK-1X: has_alignas
// CHECK-NO-1X: no_alignas

#if __STDC_VERSION__ > 199901L
int is_c1x();
#else
int is_not_c1x();
#endif

// CHECK-1X: is_c1x
// CHECK-NO-1X: is_not_c1x

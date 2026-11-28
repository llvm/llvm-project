// RUN: %clang_cc1 -E %s -fexperimental-bounds-safety | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 -E %s                              | FileCheck %s --check-prefix=DISABLED

#if __has_feature(bounds_safety)
// ENABLED: has_bounds_safety
void has_bounds_safety() {}
#else
// DISABLED: no_bounds_safety
void no_bounds_safety() {}
#endif

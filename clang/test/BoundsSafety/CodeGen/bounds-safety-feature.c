

// RUN: %clang_cc1 %s -E -fbounds-safety | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E | FileCheck %s --check-prefix=DISABLED
// RUN: %clang_cc1 %s -E -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E -x objective-c | FileCheck %s --check-prefix=DISABLED

#if __has_feature(bounds_safety)
// ENABLED: has_bounds_safety
void has_bounds_safety() {}
#else
// DISABLED: no_bounds_safety
void no_bounds_safety() {}
#endif

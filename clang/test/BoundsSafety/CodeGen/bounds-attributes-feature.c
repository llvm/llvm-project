

// RUN: %clang_cc1 %s -E -fbounds-safety | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E -fbounds-safety | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E | FileCheck %s --check-prefix=DISABLED
// RUN: %clang_cc1 %s -E -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E -x objective-c | FileCheck %s --check-prefix=DISABLED

#if __has_feature(bounds_attributes)
// ENABLED: has_bounds_attributes
void has_bounds_attributes() {}
#else
// DISABLED: no_bounds_attributes
void no_bounds_attributes() {}
#endif

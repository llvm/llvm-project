// RUN: %clang_cc1 -E %s -o - | FileCheck %s

#if __has_feature(objc_boxed_nsvalue_expressions)
int has_objc_boxed_nsvalue_expressions();
#endif

// CHECK: has_objc_boxed_nsvalue_expressions


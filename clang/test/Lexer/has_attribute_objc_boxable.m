// RUN: %clang_cc1 -E %s -o - | FileCheck %s

#if __has_attribute(objc_boxable)
int has_objc_boxable_attribute();
#endif

// CHECK: has_objc_boxable_attribute


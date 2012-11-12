// RUN: %clang_cc1 -E %s -fobjc-arc "-triple" "x86_64-apple-macosx10.7.0"  -fobjc-runtime-has-weak | FileCheck --check-prefix=CHECK-ARC %s
// RUN: %clang_cc1 -E %s -fobjc-arc "-triple" "x86_64-apple-macosx10.6.0" | FileCheck --check-prefix=CHECK-ARCLITE %s

#if __has_feature(objc_arc)
void has_objc_arc_feature();
#else
void no_objc_arc_feature();
#endif

#if __has_feature(objc_arc_weak)
void has_objc_arc_weak_feature();
#else
void no_objc_arc_weak_feature();
#endif

// CHECK-ARC: void has_objc_arc_feature();
// CHECK-ARC: void has_objc_arc_weak_feature();

// CHECK-ARCLITE: void has_objc_arc_feature();
// CHECK-ARCLITE: void no_objc_arc_weak_feature();

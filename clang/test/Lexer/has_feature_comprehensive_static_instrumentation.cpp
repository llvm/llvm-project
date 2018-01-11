// RUN: %clang_cc1 -E -fcsi %s -o - | FileCheck --check-prefix=CHECK-CSI %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-CSI %s

#if __has_feature(comprehensive_static_instrumentation)
int CsiEnabled();
#else
int CsiDisabled();
#endif

// CHECK-CSI: CsiEnabled
// CHECK-NO-CSI: CsiDisabled

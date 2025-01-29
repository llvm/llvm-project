// RUN: %clang_cc1 -E -fsanitize=numerical %s -o - | FileCheck --check-prefix=CHECK-NSAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-NSAN %s

#if __has_feature(numerical_stability_sanitizer)
int NumericalStabilitySanitizerEnabled();
#else
int NumericalStabilitySanitizerDisabled();
#endif

// CHECK-NSAN: NumericalStabilitySanitizerEnabled
// CHECK-NO-NSAN: NumericalStabilitySanitizerDisabled

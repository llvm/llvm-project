// RUN: %clang_cc1 -E -fsanitize=efficiency-cache-frag %s -o - | FileCheck --check-prefix=CHECK-ESAN %s
// RUN: %clang_cc1 -E -fsanitize=efficiency-working-set %s -o - | FileCheck --check-prefix=CHECK-ESAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-ESAN %s

#if __has_feature(efficiency_sanitizer)
int EfficiencySanitizerEnabled();
#else
int EfficiencySanitizerDisabled();
#endif

// CHECK-ESAN: EfficiencySanitizerEnabled
// CHECK-NO-ESAN: EfficiencySanitizerDisabled

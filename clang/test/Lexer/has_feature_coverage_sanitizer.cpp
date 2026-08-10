// RUN: %clang -E -fsanitize-coverage=indirect-calls %s -o - | FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: %clang -E -fsanitize-coverage=inline-8bit-counters %s -o - | FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: %clang -E -fsanitize-coverage=trace-cmp %s -o - | FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: %clang -E -fsanitize-coverage=trace-pc %s -o - | FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: %clang -E -fsanitize-coverage=trace-pc-guard %s -o - | FileCheck --check-prefix=CHECK-SANCOV %s
// RUN: %clang -E  %s -o - | FileCheck --check-prefix=CHECK-NO-SANCOV %s

#if __has_feature(coverage_sanitizer)
int SancovEnabled();
#else
int SancovDisabled();
#endif

// CHECK-SANCOV: SancovEnabled
// CHECK-NO-SANCOV: SancovDisabled

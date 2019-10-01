// RUN: %clang_cc1 -E -fsanitize=leak %s -o - | FileCheck --check-prefix=CHECK-LSAN %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-LSAN %s

#if __has_feature(leak_sanitizer)
int LeakSanitizerEnabled();
#else
int LeakSanitizerDisabled();
#endif

// CHECK-LSAN: LeakSanitizerEnabled
// CHECK-NO-LSAN: LeakSanitizerDisabled

// RUN: %clang_cc1 -E -fsanitize=realtime %s -o - | FileCheck --check-prefix=CHECK-RTSAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-RTSAN %s

#if __has_feature(realtime_sanitizer)
int RealtimeSanitizerEnabled();
#else
int RealtimeSanitizerDisabled();
#endif

// CHECK-RTSAN: RealtimeSanitizerEnabled

// CHECK-NO-RTSAN: RealtimeSanitizerDisabled

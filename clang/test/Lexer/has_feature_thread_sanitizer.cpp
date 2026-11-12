// RUN: %clang_cc1 -E -fsanitize=thread %s -o - | FileCheck --check-prefix=CHECK-TSAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-TSAN %s

#if __has_feature(thread_sanitizer)
int ThreadSanitizerEnabled();
#else
int ThreadSanitizerDisabled();
#endif

// CHECK-TSAN: ThreadSanitizerEnabled
// CHECK-NO-TSAN: ThreadSanitizerDisabled

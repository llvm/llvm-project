// RUN: %clang_cc1 -E -fsanitize=memory %s -o - | FileCheck --check-prefix=CHECK-MSAN %s
// RUN: %clang_cc1 -E -fsanitize=kernel-memory %s -o - | FileCheck --check-prefix=CHECK-MSAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-MSAN %s

#if __has_feature(memory_sanitizer)
int MemorySanitizerEnabled();
#else
int MemorySanitizerDisabled();
#endif

// CHECK-MSAN: MemorySanitizerEnabled
// CHECK-NO-MSAN: MemorySanitizerDisabled

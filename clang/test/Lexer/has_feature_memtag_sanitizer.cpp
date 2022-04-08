// RUN: %clang_cc1 -E -fsanitize=memtag %s -o - | FileCheck --check-prefix=CHECK-MEMTAG %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-MEMTAG %s

#if __has_feature(memtag_sanitizer)
int MemTagSanitizerEnabled();
#else
int MemTagSanitizerDisabled();
#endif

// CHECK-MEMTAG: MemTagSanitizerEnabled
// CHECK-NO-MEMTAG: MemTagSanitizerDisabled

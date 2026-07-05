// RUN: %clang_cc1 -E -fsanitize=memtag-stack %s -o - | FileCheck --check-prefix=CHECK-MEMTAG-STACK %s
// RUN: %clang_cc1 -E -fsanitize=memtag-heap %s -o -  | FileCheck --check-prefix=CHECK-MEMTAG-HEAP %s
// RUN: %clang -E -fsanitize=memtag --target=aarch64-unknown-linux -march=armv8a+memtag %s -o - \
// RUN:     | FileCheck --check-prefixes=CHECK-MEMTAG-STACK,CHECK-MEMTAG-HEAP %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-MEMTAG %s

#if __has_feature(memtag_stack)
int MemTagSanitizerStack();
#else
int MemTagSanitizerNoStack();
#endif

#if __has_feature(memtag_heap)
int MemTagSanitizerHeap();
#else
int MemTagSanitizerNoHeap();
#endif

// CHECK-MEMTAG-STACK: MemTagSanitizerStack
// CHECK-MEMTAG-HEAP: MemTagSanitizerHeap

// CHECK-NO-MEMTAG: MemTagSanitizerNoStack
// CHECK-NO-MEMTAG: MemTagSanitizerNoHeap

// Make sure the sanitize_memtag attribute is emitted when using MemTag sanitizer.
// Make sure __attribute__((no_sanitize("memtag")) disables instrumentation.

// RUN: %clang_cc1 -triple aarch64-unknown-linux -disable-O0-optnone \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NO %s

// RUN: %clang_cc1 -triple aarch64-unknown-linux -fsanitize=memtag \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-MEMTAG %s

int HasSanitizeMemTag() { return 1; }
// CHECK-NO: {{Function Attrs: noinline nounwind mustprogress$}}
// CHECK-MEMTAG: Function Attrs: noinline nounwind sanitize_memtag

__attribute__((no_sanitize("memtag"))) int NoSanitizeQuoteAddress() {
  return 0;
}
// CHECK-NO: {{Function Attrs: noinline nounwind mustprogress$}}
// CHECK-MEMTAG: {{Function Attrs: noinline nounwind mustprogress$}}

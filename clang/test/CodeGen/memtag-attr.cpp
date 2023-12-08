// Make sure the sanitize_memtag attribute is emitted when using MemTag sanitizer.
// Make sure __attribute__((no_sanitize("memtag")) disables instrumentation.

// RUN: %clang_cc1 -triple aarch64-unknown-linux -disable-O0-optnone \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NO %s

// RUN: %clang_cc1 -triple aarch64-unknown-linux -fsanitize=memtag-stack \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-MEMTAG %s

// RUN: %clang --target=aarch64-unknown-linux -march=armv8a+memtag \
// RUN:   -fsanitize=memtag -disable-O0-optnone -S -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-MEMTAG %s

extern "C" int HasSanitizeMemTag() { return 1; }
// CHECK-NO: Function Attrs
// CHECK-NO-NOT: sanitize_memtag
// CHECK-NO-NEXT: define {{.*}} @HasSanitizeMemTag(
// CHECK-MEMTAG: Function Attrs: {{.*}} sanitize_memtag
// CHECK-MEMTAG-NEXT: define {{.*}} @HasSanitizeMemTag(

extern "C" __attribute__((no_sanitize("memtag"))) int NoSanitizeQuoteAddress() {
  return 0;
}
// CHECK-NO: Function Attrs
// CHECK-NO-NOT: sanitize_memtag
// CHECK-NO-NEXT: define {{.*}} @NoSanitizeQuoteAddress(
// CHECK-MEMTAG: Function Attrs
// CHECK-MEMTAG-NOT: sanitize_memtag
// CHECK-MEMTAG-NEXT: define {{.*}} @NoSanitizeQuoteAddress(

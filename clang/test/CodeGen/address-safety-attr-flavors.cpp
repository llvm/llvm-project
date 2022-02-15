// Make sure the sanitize_address attribute is emitted when using ASan, KASan or
// HWASan. Either __attribute__((no_sanitize("address")) or
// __attribute__((no_sanitize("kernel-address")) disables both ASan and KASan
// instrumentation.
// Same for __attribute__((no_sanitize("hwaddress")) and
// __attribute__((no_sanitize("kernel-hwddress")) and HWASan and KHWASan.

// RUN: %clang_cc1 -triple i386-unknown-linux -disable-O0-optnone \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NOASAN %s

// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=address \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-ASAN %s

// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=kernel-address \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-KASAN %s

// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=hwaddress \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-HWASAN %s

// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=kernel-hwaddress \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-KHWASAN %s

int HasSanitizeAddress() { return 1; }
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: Function Attrs: mustprogress noinline nounwind sanitize_address
// CHECK-KASAN: Function Attrs: mustprogress noinline nounwind sanitize_address
// CHECK-HWASAN: Function Attrs: mustprogress noinline nounwind sanitize_hwaddress
// CHECK-KHWASAN: Function Attrs: mustprogress noinline nounwind sanitize_hwaddress

__attribute__((no_sanitize("address"))) int NoSanitizeQuoteAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-KASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-HWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}
// CHECK-KHWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize_address)) int NoSanitizeAddress() { return 0; }
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-KASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-HWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}
// CHECK-KHWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize("kernel-address"))) int NoSanitizeKernelAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-KASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-HWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}
// CHECK-KHWASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize("hwaddress"))) int NoSanitizeHWAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_address$}}
// CHECK-KASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_address$}}
// CHECK-HWASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-KHWASAN: {{Function Attrs: mustprogress noinline nounwind$}}

__attribute__((no_sanitize("kernel-hwaddress"))) int NoSanitizeKernelHWAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_address$}}
// CHECK-KASAN: {{Function Attrs: mustprogress noinline nounwind sanitize_address$}}
// CHECK-HWASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-KHWASAN: {{Function Attrs: mustprogress noinline nounwind$}}

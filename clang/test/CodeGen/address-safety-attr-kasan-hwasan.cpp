// Make sure the sanitize_address attribute is emitted when using both ASan and KASan.
// Also document that __attribute__((no_sanitize_address)) doesn't disable KASan instrumentation.

/// RUN: %clang_cc1 -triple i386-unknown-linux -disable-O0-optnone -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NOASAN %s
/// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=address -disable-O0-optnone -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-ASAN %s
/// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=kernel-address -disable-O0-optnone -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-KASAN %s
/// RUN: %clang_cc1 -triple i386-unknown-linux -fsanitize=hwaddress -disable-O0-optnone -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-HWASAN %s

int HasSanitizeAddress() {
  return 1;
}
// CHECK-NOASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-ASAN: Function Attrs: noinline nounwind sanitize_address
// CHECK-KASAN: Function Attrs: noinline nounwind sanitize_address
// CHECK-HWASAN: Function Attrs: noinline nounwind sanitize_hwaddress

__attribute__((no_sanitize("address")))
int NoSanitizeQuoteAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-KASAN: {{Function Attrs: noinline nounwind sanitize_address$}}
// CHECK-HWASAN: {{Function Attrs: noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize_address))
int NoSanitizeAddress() {
  return 0;
}
// CHECK-NOASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-KASAN: {{Function Attrs: noinline nounwind sanitize_address$}}
// CHECK-HWASAN: {{Function Attrs: noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize("kernel-address")))
int NoSanitizeKernelAddress() {
  return 0;
}

// CHECK-NOASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: noinline nounwind sanitize_address$}}
// CHECK-KASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-HWASAN: {{Function Attrs: noinline nounwind sanitize_hwaddress$}}

__attribute__((no_sanitize("hwaddress")))
int NoSanitizeHWAddress() {
  return 0;
}

// CHECK-NOASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-ASAN: {{Function Attrs: noinline nounwind sanitize_address$}}
// CHECK-KASAN: {{Function Attrs: noinline nounwind sanitize_address$}}
// CHECK-HWASAN: {{Function Attrs: noinline nounwind$}}

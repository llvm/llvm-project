// RUN: %clang_cc1 -triple x86_64-unknown-linux -disable-O0-optnone \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-NODASAN %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=daddress \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-DASAN %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsanitize=daddress \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-DEVICE %s

// RUN: echo "fun:*NoSanitizeIgnorelisted*" > %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=daddress \
// RUN:   -fsanitize-ignorelist=%t -disable-O0-optnone -emit-llvm -o - %s | \
// RUN:   FileCheck -check-prefix=CHECK-IGNORELIST %s

int HasSanitizeDeviceAddress(int *a) { return *a; }
// CHECK-NODASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-DASAN: Function Attrs: mustprogress noinline nounwind sanitize_device_address
// CHECK-DEVICE: Function Attrs: convergent mustprogress noinline nounwind sanitize_device_address
// CHECK-IGNORELIST: Function Attrs: mustprogress noinline nounwind sanitize_device_address

__attribute__((no_sanitize("daddress"))) int NoSanitizeDeviceAddress(int *a) {
  return *a;
}
// CHECK-NODASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-DASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-DEVICE: {{Function Attrs: convergent mustprogress noinline nounwind$}}
// CHECK-IGNORELIST: {{Function Attrs: mustprogress noinline nounwind$}}

int NoSanitizeIgnorelisted(int *a) { return *a; }
// CHECK-NODASAN: {{Function Attrs: mustprogress noinline nounwind$}}
// CHECK-DASAN: Function Attrs: mustprogress noinline nounwind sanitize_device_address
// CHECK-DEVICE: Function Attrs: convergent mustprogress noinline nounwind sanitize_device_address
// CHECK-IGNORELIST: {{Function Attrs: mustprogress noinline nounwind$}}

__attribute__((disable_sanitizer_instrumentation)) int NoInstrumentation(int *a) {
  return *a;
}
// CHECK-NODASAN: {{Function Attrs: disable_sanitizer_instrumentation mustprogress noinline nounwind$}}
// CHECK-DASAN: {{Function Attrs: disable_sanitizer_instrumentation mustprogress noinline nounwind$}}
// CHECK-DEVICE: {{Function Attrs: convergent disable_sanitizer_instrumentation mustprogress noinline nounwind$}}
// CHECK-IGNORELIST: {{Function Attrs: disable_sanitizer_instrumentation mustprogress noinline nounwind$}}

int global = *(int *)0x1000;
// CHECK-NODASAN: {{Function Attrs: noinline nounwind$}}
// CHECK-NODASAN-NEXT: define internal void @__cxx_global_var_init()
// CHECK-DASAN: Function Attrs: noinline nounwind sanitize_device_address
// CHECK-DASAN-NEXT: define internal void @__cxx_global_var_init()
// CHECK-DEVICE: Function Attrs: convergent noinline nounwind sanitize_device_address
// CHECK-DEVICE-NEXT: define internal void @__cxx_global_var_init()
// CHECK-IGNORELIST: Function Attrs: noinline nounwind sanitize_device_address
// CHECK-IGNORELIST-NEXT: define internal void @__cxx_global_var_init()

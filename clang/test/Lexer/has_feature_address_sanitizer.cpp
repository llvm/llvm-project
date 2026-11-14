// RUN: %clang_cc1 -E -fsanitize=address %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E -fsanitize=kernel-address %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E -fsanitize=hwaddress %s -o - | FileCheck --check-prefix=CHECK-HWASAN %s
// RUN: %clang_cc1 -E -fsanitize=kernel-hwaddress %s -o - | FileCheck --check-prefix=CHECK-HWASAN %s
// RUN: %clang_cc1 -E -fsanitize=daddress %s -o - | FileCheck --check-prefix=CHECK-DASAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-ASAN %s

#if __has_feature(address_sanitizer)
int AddressSanitizerEnabled();
#else
int AddressSanitizerDisabled();
#endif

#if __has_feature(hwaddress_sanitizer)
int HWAddressSanitizerEnabled();
#else
int HWAddressSanitizerDisabled();
#endif

#if __has_feature(device_address_sanitizer)
int DeviceAddressSanitizerEnabled();
#else
int DeviceAddressSanitizerDisabled();
#endif

// CHECK-ASAN: AddressSanitizerEnabled
// CHECK-ASAN: HWAddressSanitizerDisabled
// CHECK-ASAN: DeviceAddressSanitizerDisabled

// CHECK-HWASAN: AddressSanitizerDisabled
// CHECK-HWASAN: HWAddressSanitizerEnabled
// CHECK-HWASAN: DeviceAddressSanitizerDisabled

// CHECK-DASAN: AddressSanitizerDisabled
// CHECK-DASAN: HWAddressSanitizerDisabled
// CHECK-DASAN: DeviceAddressSanitizerEnabled

// CHECK-NO-ASAN: AddressSanitizerDisabled
// CHECK-NO-ASAN: HWAddressSanitizerDisabled
// CHECK-NO-ASAN: DeviceAddressSanitizerDisabled

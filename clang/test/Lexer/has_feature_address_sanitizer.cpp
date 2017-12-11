// RUN: %clang_cc1 -E -fsanitize=address %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E -fsanitize=kernel-address %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E -fsanitize=hwaddress %s -o - | FileCheck --check-prefix=CHECK-HWASAN %s
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

// CHECK-ASAN: AddressSanitizerEnabled
// CHECK-ASAN: HWAddressSanitizerDisabled

// CHECK-HWASAN: AddressSanitizerDisabled
// CHECK-HWASAN: HWAddressSanitizerEnabled

// CHECK-NO-ASAN: AddressSanitizerDisabled
// CHECK-NO-ASAN: HWAddressSanitizerDisabled

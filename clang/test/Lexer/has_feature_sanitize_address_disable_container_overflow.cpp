// RUN: %clang_cc1 -E -fsanitize=address %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E -fsanitize=address -fsanitize-address-disable-container-overflow %s -o - | FileCheck --check-prefix=CHECK-ASAN-DISABLE-CONTAINER %s
// RUN: %clang_cc1 -E -fsanitize=kernel-address -fsanitize-address-disable-container-overflow %s -o - | FileCheck --check-prefix=CHECK-ASAN-DISABLE-CONTAINER %s
// RUN: %clang_cc1 -E -fsanitize=address -fno-sanitize-address-disable-container-overflow %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-ASAN %s

#if __has_feature(address_sanitizer)
int AddressSanitizerEnabled();
#else
int AddressSanitizerDisabled();
#endif

#if __has_feature(sanitize_address_disable_container_overflow)
int AddressSanitizerContainerOverflowDisabled();
#else
int AddressSanitizerContainerOverflowEnabled();
#endif

// CHECK-ASAN: AddressSanitizerEnabled
// CHECK-ASAN: AddressSanitizerContainerOverflowEnabled

// CHECK-ASAN-DISABLE-CONTAINER: AddressSanitizerEnabled
// CHECK-ASAN-DISABLE-CONTAINER: AddressSanitizerContainerOverflowDisabled

// CHECK-NO-ASAN: AddressSanitizerDisabled
// CHECK-NO-ASAN: AddressSanitizerContainerOverflowEnabled
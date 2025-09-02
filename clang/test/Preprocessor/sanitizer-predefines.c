// RUN: %clang_cc1 -E -dM -triple aarch64-unknown-linux -fsanitize=address %s | FileCheck %s --check-prefix=ASAN
// ASAN: #define __SANITIZE_ADDRESS__ 1

// RUN: %clang_cc1 -E -dM -triple aarch64-unknown-linux -fsanitize=hwaddress %s | FileCheck %s --check-prefix=HWASAN
// HWASAN: #define __SANITIZE_HWADDRESS__ 1

// RUN: %clang_cc1 -E -dM -triple aarch64-unknown-linux -fsanitize=thread %s | FileCheck %s --check-prefix=TSAN
// TSAN: #define __SANITIZE_THREAD__ 1

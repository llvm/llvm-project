// __ARM_SIZEOF_WCHAR_T is the size of wchar_t, which is 2 bytes on Windows.

// RUN: %clang_cc1 -E -dM -triple=thumbv7-windows-msvc %s | FileCheck %s --check-prefix=TWO
// RUN: %clang_cc1 -E -dM -triple=aarch64-windows-msvc %s | FileCheck %s --check-prefix=TWO
// RUN: %clang_cc1 -E -dM -triple=armv7-linux-gnueabihf %s | FileCheck %s --check-prefix=FOUR

// The host target of an offloading compile is not adjusted for -fwchar-type.
// RUN: %clang_cc1 -E -dM -triple=nvptx64-nvidia-cuda -aux-triple aarch64-linux-gnu -fcuda-is-device -fwchar-type=short -fno-signed-wchar -x cuda %s | FileCheck %s --check-prefix=TWO

// TWO: #define __ARM_SIZEOF_WCHAR_T 2
// FOUR: #define __ARM_SIZEOF_WCHAR_T 4

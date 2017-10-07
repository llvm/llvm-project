// RUN: %clang_cc1 -triple i386-pc-solaris -dM -E %s -o - | FileCheck %s -check-prefix CHECK-SOLARIS
// CHECK-SOLARIS-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-SOLARIS-DAG: #define __WCHAR_TYPE__ int
// CHECK-SOLARIS-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple avr-unknown-unknown -fwchar-type=int -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-AVR
// CHECK-AVR-DAG: #define __WCHAR_MAX__ 32767
// CHECK-AVR-DAG: #define __WCHAR_TYPE__ int
// CHECK-AVR-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple arm-unknown-none-gnu -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM-APCS
// CHECK-ARM-APCS-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM-APCS-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM-APCS-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple arm-unknown-netbsd-gnu -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM-NETBSD-AAPCS
// CHECK-ARM-NETBSD-AAPCS-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM-NETBSD-AAPCS-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM-NETBSD-AAPCS-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple arm-unknown-openbsd -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM-OPENBSD
// CHECK-ARM-OPENBSD-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM-OPENBSD-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM-OPENBSD-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple arm64-apple-ios -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM64-DARWIN
// CHECK-ARM64-DARWIN-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM64-DARWIN-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM64-DARWIN-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple aarch64-unknown-netbsd -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM64-NETBSD
// CHECK-ARM64-NETBSD-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM64-NETBSD-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM64-NETBSD-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple aarch64-unknown-openbsd -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM64-OPENBSD
// CHECK-ARM64-OPENBSD-DAG: #define __WCHAR_MAX__ 2147483647
// CHECK-ARM64-OPENBSD-DAG: #define __WCHAR_TYPE__ int
// CHECK-ARM64-OPENBSD-NOT: #define __WCHAR_UNSIGNED__ 0

// RUN: %clang_cc1 -triple aarch64-unknown-none -fwchar-type=int -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-ARM64-AAPCS64
// CHECK-ARM64-AAPCS64-DAG: #define __WCHAR_MAX__ 4294967295U
// CHECK-ARM64-AAPCS64-DAG: #define __WCHAR_TYPE__ unsigned int
// CHECK-ARM64-AAPCS64-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple xcore-unknown-unknown -fwchar-type=char -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-XCORE
// CHECK-XCORE-DAG: #define __WCHAR_MAX__ 255
// CHECK-XCORE-DAG: #define __WCHAR_TYPE__ unsigned char
// CHECK-XCORE-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple x86_64-unknown-windows-cygnus -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-CYGWIN-X64
// CHECK-CYGWIN-X64-DAG: #define __WCHAR_MAX__ 65535
// CHECK-CYGWIN-X64-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-CYGWIN-X64-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-MSVC-X64
// CHECK-MSVC-X64-DAG: #define __WCHAR_MAX__ 65535
// CHECK-MSVC-X64-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-MSVC-X64-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple i686-unknown-windows-cygnus -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-CYGWIN-X86
// CHECK-CYGWIN-X86-DAG: #define __WCHAR_MAX__ 65535
// CHECK-CYGWIN-X86-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-CYGWIN-X86-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-MSVC-X86
// CHECK-MSVC-X86-DAG: #define __WCHAR_MAX__ 65535
// CHECK-MSVC-X86-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-MSVC-X86-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple x86_64-scei-ps4 -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-PS4
// CHECK-PS4-DAG: #define __WCHAR_MAX__ 65535
// CHECK-PS4-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-PS4-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple thumbv7-unknown-windows-cygnus -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-CYGWIN-ARM
// CHECK-CYGWIN-ARM-DAG: #define __WCHAR_MAX__ 65535
// CHECK-CYGWIN-ARM-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-CYGWIN-ARM-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-MSVC-ARM
// CHECK-MSVC-ARM-DAG: #define __WCHAR_MAX__ 65535
// CHECK-MSVC-ARM-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-MSVC-ARM-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fwchar-type=short -fno-signed-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-MSVC-ARM64
// CHECK-MSVC-ARM64-DAG: #define __WCHAR_MAX__ 65535
// CHECK-MSVC-ARM64-DAG: #define __WCHAR_TYPE__ unsigned short
// CHECK-MSVC-ARM64-DAG: #define __WCHAR_UNSIGNED__ 1

// RUN: %clang_cc1 -triple i386-apple-macosx -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple x86_64-apple-macosx -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple ppc64-apple-macosx -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple i386-apple-ios -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple x86_64-apple-ios -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple armv7-apple-ios -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple aarch64-apple-ios -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple i386-apple-tvos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple x86_64-apple-tvos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple armv7-apple-tvos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple aarch64-apple-tvos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple i386-apple-watchos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple x86_64-apple-watchos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple armv7-apple-watchos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// RUN: %clang_cc1 -triple aarch64-apple-watchos -dM -E %s -o - | FileCheck %s -check-prefix CHECK-DARWIN
// CHECK-DARWIN: #define __WCHAR_TYPE__ int

// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fwchar-type=int -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-WINDOWS-ISO10646
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fwchar-type=int -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-WINDOWS-ISO10646
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fwchar-type=int -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-WINDOWS-ISO10646
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fwchar-type=int -fsigned-wchar -dM -E %s -o - | FileCheck %s -check-prefix CHECK-WINDOWS-ISO10646
// CHECK-WINDOWS-ISO10646: #define __WCHAR_TYPE__ int


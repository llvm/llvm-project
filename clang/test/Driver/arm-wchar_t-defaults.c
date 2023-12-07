// RUN: %clang -### -target armv7-unknown-none-eabi -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-none-eabi -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s
// RUN: %clang -### -target armv7-unknown-none-eabi -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-none-eabi -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s
// RUN: %clang -### -target armv7eb-unknown-none-eabi -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-none-eabi -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s
// RUN: %clang -### -target armv7eb-unknown-none-eabi -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-none-eabi -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s
// RUN: %clang -### -target aarch64 -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64 -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s
// RUN: %clang -### -target aarch64_be -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64_be -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-UNSIGNED %s

// RUN: %clang -### -target armv7-unknown-netbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-netbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7-unknown-netbsd -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-netbsd -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7eb-unknown-netbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-netbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7eb-unknown-netbsd -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-netbsd -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target aarch64-unknown-netbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64-unknown-netbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target aarch64_be-unknown-netbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64_be-unknown-netbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s

// RUN: %clang -### -target armv7-unknown-openbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-openbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7-unknown-openbsd -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-openbsd -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7eb-unknown-openbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-openbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target armv7eb-unknown-openbsd -mthumb -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7eb-unknown-openbsd -mthumb -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target aarch64-unknown-openbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64-unknown-openbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target aarch64_be-unknown-openbsd -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64_be-unknown-openbsd -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s

// RUN: %clang -### -target armv7-unknown-windows-msvc -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target armv7-unknown-windows-msvc -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s
// RUN: %clang -### -target aarch64-unknown-windows-msvc -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-SHORT %s
// RUN: %clang -### -target aarch64-unknown-windows-msvc -fno-short-wchar -c %s -o /dev/null 2>&1 | FileCheck -check-prefix CHECK-LONG-SIGNED %s

// CHECK-SHORT-NOT: "-fwchar-type=int"
// CHECK-SHORT-NOT: "-fno-signed-wchar"

// CHECK-LONG-UNSIGNED: "-fwchar-type=int"
// CHECK-LONG-UNSIGNED: "-fno-signed-wchar"

// CHECK-LONG-SIGNED: "-fwchar-type=int"
// CHECK-LONG-SIGNED: "-fsigned-wchar"


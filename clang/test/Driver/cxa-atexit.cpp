// RUN: %clang -### -target armv7-unknown-windows-msvc -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target armv7-unknown-windows-itanium -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target armv7-unknown-windows-gnu -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target armv7-unknown-windows-cygnus -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target i686-unknown-windows-msvc -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target i686-unknown-windows-itanium -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target i686-unknown-windows-gnu -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target i686-unknown-windows-cygnus -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target x86_64-unknown-windows-msvc -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target x86_64-unknown-windows-itanium -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target x86_64-unknown-windows-gnu -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target x86_64-unknown-windows-cygnus -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-WINDOWS
// RUN: %clang -### -target hexagon-unknown-none -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-HEXAGON
// RUN: %clang -### -target xcore-unknown-none -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-XCORE
// RUN: %clang -### -target armv7-mti-none -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MTI
// RUN: %clang -### -target mips-unknown-none -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MIPS
// RUN: %clang -### -target mips-unknown-none-gnu -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MIPS
// RUN: %clang -### -target mips-mti-none-gnu -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MIPS
// RUN: %clang -### -target sparc-sun-solaris -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-SOLARIS

// CHECK-WINDOWS: "-fno-use-cxa-atexit"
// CHECK-SOLARIS: "-fno-use-cxa-atexit"
// CHECK-HEXAGON: "-fno-use-cxa-atexit"
// CHECK-XCORE: "-fno-use-cxa-atexit"
// CHECK-MTI: "-fno-use-cxa-atexit"
// CHECK-MIPS-NOT: "-fno-use-cxa-atexit"


// RUN: %clang -target armv7-windows-msvc -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-msvc -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-msvc -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-msvc -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-msvc -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target armv7-windows-itanium -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-itanium -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-itanium -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-itanium -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT
// RUN: %clang -target armv7-windows-itanium -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target armv7-windows-msvc -fomit-frame-pointer -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-msvc -fomit-frame-pointer -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-msvc -fomit-frame-pointer -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-msvc -fomit-frame-pointer -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-msvc -fomit-frame-pointer -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO

// RUN: %clang -target armv7-windows-itanium -fomit-frame-pointer -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-itanium -fomit-frame-pointer -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-itanium -fomit-frame-pointer -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-itanium -fomit-frame-pointer -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO
// RUN: %clang -target armv7-windows-itanium -fomit-frame-pointer -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-FPO

// RUN: %clang -target armv7-windows-msvc -fno-omit-frame-pointer -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-msvc -fno-omit-frame-pointer -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-msvc -fno-omit-frame-pointer -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-msvc -fno-omit-frame-pointer -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-msvc -fno-omit-frame-pointer -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO

// RUN: %clang -target armv7-windows-itanium -fno-omit-frame-pointer -### -S %s -O0 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-itanium -fno-omit-frame-pointer -### -S %s -O1 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-itanium -fno-omit-frame-pointer -### -S %s -O2 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-itanium -fno-omit-frame-pointer -### -S %s -O3 -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO
// RUN: %clang -target armv7-windows-itanium -fno-omit-frame-pointer -### -S %s -Os -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-NO-FPO

// CHECK-DEFAULT: "-mframe-pointer=all"
// CHECK-FPO-NOT: "-mframe-pointer=all"
// CHECK-NO-FPO: "-mframe-pointer=all"


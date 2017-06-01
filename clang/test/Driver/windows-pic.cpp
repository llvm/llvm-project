// RUN: %clang -### -target i686-windows -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target i686-windows -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target i686-windows -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target i686-windows -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target i686-windows -fPIC -fno-pic %s
// RUN: %clang -### -target i686-windows -Fpic -fno-pic %s
// RUN: %clang -### -target i686-windows -fPIE -fno-pie %s
// RUN: %clang -### -target i686-windows -fpie -fno-pie %s

// RUN: %clang -### -target i686-windows-itanium -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target i686-windows-itanium -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target i686-windows-itanium -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target i686-windows-itanium -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target i686-windows-itanium -fPIC -fno-pic %s
// RUN: %clang -### -target i686-windows-itanium -Fpic -fno-pic %s
// RUN: %clang -### -target i686-windows-itanium -fPIE -fno-pie %s
// RUN: %clang -### -target i686-windows-itanium -fpie -fno-pie %s

// RUN: %clang -### -target i686-windows-gnu -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target i686-windows-gnu -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target i686-windows-gnu -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target i686-windows-gnu -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target i686-windows-gnu -fPIC -fno-pic %s
// RUN: %clang -### -target i686-windows-gnu -Fpic -fno-pic %s
// RUN: %clang -### -target i686-windows-gnu -fPIE -fno-pie %s
// RUN: %clang -### -target i686-windows-gnu -fpie -fno-pie %s

// RUN: %clang -### -target x86_64-windows -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target x86_64-windows -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target x86_64-windows -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target x86_64-windows -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target x86_64-windows -fPIC -fno-pic %s
// RUN: %clang -### -target x86_64-windows -Fpic -fno-pic %s
// RUN: %clang -### -target x86_64-windows -fPIE -fno-pie %s
// RUN: %clang -### -target x86_64-windows -fpie -fno-pie %s

// RUN: %clang -### -target x86_64-windows-itanium -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target x86_64-windows-itanium -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target x86_64-windows-itanium -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target x86_64-windows-itanium -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target x86_64-windows-itanium -fPIC -fno-pic %s
// RUN: %clang -### -target x86_64-windows-itanium -Fpic -fno-pic %s
// RUN: %clang -### -target x86_64-windows-itanium -fPIE -fno-pie %s
// RUN: %clang -### -target x86_64-windows-itanium -fpie -fno-pie %s

// RUN: %clang -### -target x86_64-windows-gnu -fPIC %s 2>&1 | FileCheck -check-prefix CHECK-PIC-ERROR %s
// RUN: %clang -### -target x86_64-windows-gnu -fpic %s 2>&1 | FileCheck -check-prefix CHECK-pic-ERROR %s
// RUN: %clang -### -target x86_64-windows-gnu -fPIE %s 2>&1 | FileCheck -check-prefix CHECK-PIE-ERROR %s
// RUN: %clang -### -target x86_64-windows-gnu -fpie %s 2>&1 | FileCheck -check-prefix CHECK-pie-ERROR %s
// RUN: %clang -### -target x86_64-windows-gnu -fPIC -fno-pic %s
// RUN: %clang -### -target x86_64-windows-gnu -Fpic -fno-pic %s
// RUN: %clang -### -target x86_64-windows-gnu -fPIE -fno-pie %s
// RUN: %clang -### -target x86_64-windows-gnu -fpie -fno-pie %s

// CHECK-PIC-ERROR: unsupported option '-fPIC' for target '{{.*}}
// CHECK-pic-ERROR: unsupported option '-fpic' for target '{{.*}}
// CHECK-PIE-ERROR: unsupported option '-fPIE' for target '{{.*}}
// CHECK-pie-ERROR: unsupported option '-fpie' for target '{{.*}}


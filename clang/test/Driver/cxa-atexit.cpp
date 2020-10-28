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
// RUN: %clang -### -target powerpc-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-AIX
// RUN: %clang -### -target powerpc64-ibm-aix-xcoff -c %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-AIX

// CHECK-WINDOWS: "-fno-use-cxa-atexit"
// CHECK-SOLARIS-NOT: "-fno-use-cxa-atexit"
// CHECK-HEXAGON-NOT: "-fno-use-cxa-atexit"
// CHECK-XCORE: "-fno-use-cxa-atexit"
// CHECK-MTI: "-fno-use-cxa-atexit"
// CHECK-MIPS-NOT: "-fno-use-cxa-atexit"
// CHECK-AIX: "-fno-use-cxa-atexit"

// RUN: %clang -target x86_64-apple-darwin -fregister-global-dtors-with-atexit -fno-register-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s
// RUN: %clang -target x86_64-apple-darwin -fno-register-global-dtors-with-atexit -fregister-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHATEXIT %s
// RUN: %clang -target x86_64-apple-darwin -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHATEXIT %s
// RUN: %clang -target x86_64-apple-darwin -c -mkernel -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s

// RUN: %clang -target x86_64-pc-linux-gnu -fregister-global-dtors-with-atexit -fno-register-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s
// RUN: %clang -target x86_64-pc-linux-gnu -fno-register-global-dtors-with-atexit -fregister-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHATEXIT %s
// RUN: %clang -target x86_64-pc-linux-gnu -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s

// RUN: %clang -target powerpc-ibm-aix-xcoff -fregister-global-dtors-with-atexit -fno-register-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s
// RUN: %clang -target powerpc-ibm-aix-xcoff -fno-register-global-dtors-with-atexit -fregister-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHATEXIT %s
// RUN: %clang -target powerpc-ibm-aix-xcoff -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s
// RUN: %clang -target powerpc64-ibm-aix-xcoff -fregister-global-dtors-with-atexit -fno-register-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s
// RUN: %clang -target powerpc64-ibm-aix-xcoff -fno-register-global-dtors-with-atexit -fregister-global-dtors-with-atexit -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHATEXIT %s
// RUN: %clang -target powerpc64-ibm-aix-xcoff -c -### %s 2>&1 | \
// RUN: FileCheck --check-prefix=WITHOUTATEXIT %s

// WITHATEXIT: -fregister-global-dtors-with-atexit
// WITHOUTATEXIT-NOT: -fregister-global-dtors-with-atexit

// RUN: %clang -### -target x86_64-unknown-linux -c %s 2>&1 | FileCheck -check-prefix=NO-CIR %s
// RUN: %clang -### -target x86_64-pc-win32 -c %s 2>&1 | FileCheck -check-prefix=NO-CIR %s
// RUN: %clang -### -target x86_64-scei-ps4 -c %s 2>&1 | FileCheck -check-prefix=NO-CIR %s
// RUN: %clang -### -target x86_64-linux-android21 -c %s 2>&1 | FileCheck -check-prefix=NO-CIR %s

// RUN: %clang -### -target x86_64-unknown-linux -c -fcir-warnings %s 2>&1 | FileCheck -check-prefix=CIR %s
// RUN: %clang -### -target x86_64-pc-win32 -c -fcir-warnings %s 2>&1 | FileCheck -check-prefix=CIR %s
// RUN: %clang -### -target x86_64-scei-ps4 -c -fcir-warnings %s 2>&1 | FileCheck -check-prefix=CIR %s
// RUN: %clang -### -target x86_64-linux-android21 -c -fcir-warnings %s 2>&1 | FileCheck -check-prefix=CIR %s

// CIR: -fcir-warnings
// NO-CIR-NOT: -fcir-warnings

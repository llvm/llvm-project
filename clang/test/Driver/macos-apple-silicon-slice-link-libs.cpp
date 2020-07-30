// RUN: %clang -### -target arm64-apple-macos10.7 %s 2>&1 | FileCheck -check-prefix=ARM64-10_7 %s
// RUN: %clang -### -target x86_64-apple-macos10.7 %s 2>&1 | FileCheck -check-prefix=x86_64-10_7 %s

// RUN: %clang -### -target arm64-apple-macos10.5 %s 2>&1 | FileCheck -check-prefix=ARM64-10_5 %s
// RUN: %clang -### -target x86_64-apple-macos10.5 %s 2>&1 | FileCheck -check-prefix=x86_64-10_5 %s

// RUN: %clang -### -target arm64-apple-macos10.4 %s 2>&1 | FileCheck -check-prefix=ARM64-10_4 %s
// RUN: %clang -### -target x86_64-apple-macos10.4 %s 2>&1 | FileCheck -check-prefix=x86_64-10_4 %s

// RUN: %clang -### -target arm64-apple-macos10.5 -bundle %s 2>&1 | FileCheck -check-prefix=ARM64-BUNDLE %s
// RUN: %clang -### -target x86_64-apple-macos10.5 -bundle %s 2>&1 | FileCheck -check-prefix=x86_64-BUNDLE %s

// RUN: %clang -### -target arm64-apple-macos10.5 -dynamiclib %s 2>&1 | FileCheck -check-prefix=ARM64-10_5-DYNAMICLIB %s
// RUN: %clang -### -target x86_64-apple-macos10.5 -dynamiclib %s 2>&1 | FileCheck -check-prefix=x86_64-10_5-DYNAMICLIB %s

// RUN: %clang -### -target arm64-apple-macos10.4 -dynamiclib %s 2>&1 | FileCheck -check-prefix=ARM64-10_4-DYNAMICLIB %s
// RUN: %clang -### -target arm64-apple-darwin8 -dynamiclib %s 2>&1 | FileCheck -check-prefix=ARM64-10_4-DYNAMICLIB %s
// RUN: %clang -### -target x86_64-apple-macos10.4 -dynamiclib %s 2>&1 | FileCheck -check-prefix=x86_64-10_4-DYNAMICLIB %s
// RUN: %clang -### -target x86_64-apple-darwin8 -dynamiclib %s 2>&1 | FileCheck -check-prefix=x86_64-10_4-DYNAMICLIB %s

// RUN: %clang -### -target arm64-apple-macos10.7 -static %s 2>&1 | FileCheck -check-prefix=STATIC %s
// RUN: %clang -### -target x86_64-apple-macos10.7 -static %s 2>&1 | FileCheck -check-prefix=STATIC %s

// ARM64-10_7-NOT: -lcrt1.10.6.o
// x86_64-10_7:    -lcrt1.10.6.o

// ARM64-10_5-NOT: -lcrt1.10.5.o
// x86_64-10_5:    -lcrt1.10.5.o

// ARM64-10_4-NOT: -lcrt1.o
// x86_64-10_4:    -lcrt1.o

// ARM64-BUNDLE-NOT: -lbundle1.o
// x86_64-BUNDLE:    -lbundle1.o

// ARM64-10_5-DYNAMICLIB-NOT: -ldylib1.10.5.o
// x86_64-10_5-DYNAMICLIB:    -ldylib1.10.5.o

// ARM64-10_4-DYNAMICLIB-NOT: -ldylib1.o
// x86_64-10_4-DYNAMICLIB:    -ldylib1.o

// STATIC: -lcrt0.o

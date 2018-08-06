// RUN: %clang -### -target x86_64-unknown-linux -c %s 2>&1 | FileCheck -check-prefix=ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-integrated-as -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-integrated-as -faddrsig -c %s 2>&1 | FileCheck -check-prefix=ADDRSIG %s
// RUN: %clang -### -target x86_64-unknown-linux -fno-addrsig -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s
// RUN: %clang -### -target x86_64-apple-darwin -c %s 2>&1 | FileCheck -check-prefix=NO-ADDRSIG %s

// ADDRSIG: -faddrsig
// NO-ADDRSIG-NOT: -faddrsig

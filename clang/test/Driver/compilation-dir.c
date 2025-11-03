// XFAIL: target={{.*}}-aix{{.*}}

// RUN: %clang -### -S -fdebug-compilation-dir . %s 2>&1 | FileCheck -check-prefix=CHECK-DEBUG-COMPILATION-DIR %s
// RUN: %clang -### -S -fdebug-compilation-dir=. %s 2>&1 | FileCheck -check-prefix=CHECK-DEBUG-COMPILATION-DIR %s
// RUN: %clang -### -integrated-as -fdebug-compilation-dir . -x assembler %s 2>&1 | FileCheck -check-prefix=CHECK-DEBUG-COMPILATION-DIR %s
// RUN: %clang -### -integrated-as -fdebug-compilation-dir=. -x assembler %s 2>&1 | FileCheck -check-prefix=CHECK-DEBUG-COMPILATION-DIR %s
// RUN: %clang -### -S -ffile-compilation-dir=. %s 2>&1 | FileCheck -check-prefix=CHECK-DEBUG-COMPILATION-DIR %s
// RUN: %clang -### -integrated-as -ffile-compilation-dir=. -x assembler %s 2>&1 | FileCheck -check-prefixes=CHECK-DEBUG-COMPILATION-DIR %s
// CHECK-DEBUG-COMPILATION-DIR: "-fdebug-compilation-dir=."
// CHECK-DEBUG-COMPILATION-DIR-NOT: "-ffile-compilation-dir=."

// RUN: %clang -### -S %s -working-directory %S 2>&1 | FileCheck -check-prefix=CHECK-CWD %s
// RUN: cd %S
// RUN: %clang -### -S %s 2>&1 | FileCheck -check-prefix=CHECK-CWD %s
// CHECK-CWD: -fdebug-compilation-dir={{.*}}Driver

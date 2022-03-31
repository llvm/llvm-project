// RUN: %clang -### %s --target=x86_64-pc-linux-gnu -c -Wa,--fatal-warnings 2>&1 | FileCheck %s --check-prefix=FATAL_WARNINGS

// FATAL_WARNINGS: "-massembler-fatal-warnings"

// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,--no-warn 2>&1 | FileCheck -check-prefix=CHECK-NOIAS %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,--no-warn 2>&1 | FileCheck %s

/// -W is alias for --no-warn.
// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-NOIASW %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,-W 2>&1 | FileCheck %s

/// Make sure warnings behave properly in integrated assembler.
// RUN: %clang %s -c -o %t.o -integrated-as -Wa,--no-warn 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: not %clang %s -c -o %t.o -integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck --check-prefix=CHECK-AS-FATAL %s

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: system-linux

// CHECK: "-cc1" {{.*}} "-massembler-no-warn"
// CHECK-NOIAS: "--no-warn"
// CHECK-NOIASW: "-W"
// CHECK-AS-NOWARN-NOT: warning:
// CHECK-AS-FATAL: error: .warning directive invoked in source file

__asm(".warning");

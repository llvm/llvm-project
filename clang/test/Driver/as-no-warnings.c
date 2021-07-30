// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,--no-warn 2>&1 | FileCheck -check-prefix=CHECK-NOIAS %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,--no-warn 2>&1 | FileCheck %s

/// -W is alias for --no-warn.
// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-NOIASW %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,-W 2>&1 | FileCheck %s

// RUN: %clang %s -c -o %t.o -integrated-as -Wa,--no-warn 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: %clang %s -c -o %t.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,--no-warn 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: %clang %s -c -o %t.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,-W 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: not %clang %s -c -o %t.o -target i686-pc-linux-gnu -integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck --check-prefix=CHECK-AS-FATAL %s
// RUN: not %clang %s -c -o %t.o -target i686-pc-linux-gnu -fno-integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck --check-prefix=CHECK-AS-FATAL %s

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: system-linux

// CHECK: "-cc1" {{.*}} "-massembler-no-warn"
// CHECK-NOIAS: "--no-warn"
// CHECK-NOIASW: "-W"
// CHECK-AS-NOWARN-NOT: warning:
// CHECK-AS-FATAL-NOT: warning:
// CHECK-AS-FATAL: error

__asm(".warning");

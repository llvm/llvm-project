// RUN: %clang -### %s -c -o tmp.o -fno-integrated-as -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-NOIAS %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-IAS %s
// RUN: %clang %s -c -o %t.o -integrated-as -Wa,-W 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: %clang %s -c -o %t.o -fno-integrated-as -Wa,-W 2>&1 | FileCheck -allow-empty --check-prefix=CHECK-AS-NOWARN %s
// RUN: not %clang %s -c -o %t.o -integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck --check-prefix=CHECK-AS-FATAL %s
// RUN: not %clang %s -c -o %t.o -fno-integrated-as -Wa,--fatal-warnings 2>&1 | FileCheck --check-prefix=CHECK-AS-FATAL %s

// REQUIRES: clang-driver
// REQUIRES: linux

// CHECK-IAS: "-cc1" {{.*}} "-massembler-no-warn"
// CHECK-NOIAS: "-W"
// CHECK-AS-NOWARN-NOT: warning:
// CHECK-AS-FATAL-NOT: warning:
// CHECK-AS-FATAL: error

__asm(".warning");

// RUN: %clang -### %s -c -o tmp.o -fno-integrated-as --target=x86_64-linux-gnu -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-NOIAS %s
// RUN: %clang -### %s -c -o tmp.o -integrated-as -Wa,-W 2>&1 | FileCheck -check-prefix=CHECK-IAS %s

// CHECK-IAS: "-cc1" {{.*}} "-massembler-no-warn"
// CHECK-NOIAS: "-W"

__asm(".warning");

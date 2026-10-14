// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mincremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST1
// TEST1: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mno-incremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST2
// TEST2: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mno-incremental-linker-compatible -mincremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST3
// TEST3: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mincremental-linker-compatible -mno-incremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST4
// TEST4: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-mingw32 -integrated-as 2>&1 | FileCheck %s --check-prefix=TEST5
// TEST5-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"
// TEST5-NOT: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-win32 -integrated-as 2>&1 | FileCheck %s --check-prefix=TEST6
// TEST6-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"
// TEST6-NOT: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target x86_64-uefi -integrated-as -mincremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST7
// TEST7: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target x86_64-uefi -integrated-as 2>&1 | FileCheck %s --check-prefix=TEST8
// TEST8-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"
// TEST8-NOT: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target x86_64-uefi -integrated-as -mno-incremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST9
// TEST9: "-cc1" {{.*}} "-mno-incremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mincremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST1
// TEST1: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mno-incremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST2
// TEST2-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mno-incremental-linker-compatible -mincremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST3
// TEST3: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -mincremental-linker-compatible -mno-incremental-linker-compatible 2>&1 | FileCheck %s --check-prefix=TEST4
// TEST4-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-mingw32 -integrated-as 2>&1 | FileCheck %s --check-prefix=TEST5
// TEST5-NOT: "-cc1" {{.*}} "-mincremental-linker-compatible"

// RUN: %clang '-###' %s -c -o tmp.o -target i686-pc-win32 -integrated-as 2>&1 | FileCheck %s --check-prefix=TEST6
// TEST6: "-cc1" {{.*}} "-mincremental-linker-compatible"

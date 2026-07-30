// RUN: %clang -### %s -c -o tmp.o -target i686-pc-linux-gnu -integrated-as -Wa,--noexecstack 2>&1 | FileCheck %s

// CHECK: "-cc1" {{.*}} "-mnoexecstack"

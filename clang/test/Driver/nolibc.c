// RUN: %clang -target i686-pc-linux-gnu -### -rtlib=libgcc --unwindlib=libgcc -nolibc %s 2>&1 | FileCheck %s
// CHECK: crtbegin
// CHECK: "-lgcc"
// CHECK-NOT: "-lc"
// CHECK: crtend

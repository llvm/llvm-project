// RUN: %clang -target i686-pc-linux-gnu -### -nodefaultlibs %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK-NOT: start-group
// CHECK-NOT: "-lgcc"
// CHECK-NOT: "-lc"
// CHECK: crtbegin
// CHECK: crtend

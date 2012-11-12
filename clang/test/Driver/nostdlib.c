// RUN: %clang -target i686-pc-linux-gnu -### -nostdlib %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK-NOT: start-group

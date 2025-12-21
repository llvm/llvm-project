// RUN: %clang -target i386-pc-elfiamcu -c -v -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK-NOT: /usr/include
// CHECK-NOT: /usr/local/include


// RUN: %clang -target i386-apple-darwin10 -### -fsyntax-only -fgnu-runtime %s 2>&1 | FileCheck %s
// RUN: %clang -target i386-apple-darwin10 -### -x objective-c++ -fsyntax-only -fgnu-runtime %s 2>&1 | FileCheck %s
// CHECK: -fobjc-runtime=gcc
// CHECK-NOT: fragile

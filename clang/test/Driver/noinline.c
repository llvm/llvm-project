// Make sure the driver is correctly passing -fno-inline-functions
// rdar://10972766

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -fno-inline -fno-inline-functions -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK < %t %s

// CHECK: clang
// CHECK: "-fno-inline"
// CHECK: "-fno-inline-functions"

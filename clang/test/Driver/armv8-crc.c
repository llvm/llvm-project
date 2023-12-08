// RUN: %clang -target armv8 -mcrc -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-V8-CRC < %t %s
// CHECK-V8-CRC: "-target-feature" "+crc"

// RUN: %clang -target armv8 -mnocrc -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-V8-NOCRC < %t %s
// CHECK-V8-NOCRC: "-target-feature" "-crc"


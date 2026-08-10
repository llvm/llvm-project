// RUN: %clang -target arm-none-gnueabi -ffixed-r9 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R9 < %t %s

// CHECK-FIXED-R9: "-target-feature" "+reserve-r9"

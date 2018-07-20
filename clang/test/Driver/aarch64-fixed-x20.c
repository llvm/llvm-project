// RUN: %clang -target aarch64-none-gnu -ffixed-x20 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X20 < %t %s

// CHECK-FIXED-X20: "-target-feature" "+reserve-x20"

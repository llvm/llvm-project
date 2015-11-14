// RUN: %clang -target aarch64-none-gnu -ffixed-x18 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X18 < %t %s

// CHECK-FIXED-X18: "-target-feature" "+reserve-x18"

// RUN: %clang -target aarch64-none-gnu -ffixed-x18 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X18 < %t %s
// RUN: %clang -target aarch64-none-gnu -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FIXED-X18 < %t %s
// RUN: %clang -target -arm64-apple-ios -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X18 < %t %s

// CHECK-FIXED-X18: "-target-feature" "+reserve-x18"
// CHECK-NO-FIXED-X18-NOT: "-target-feature" "+reserve-x18"

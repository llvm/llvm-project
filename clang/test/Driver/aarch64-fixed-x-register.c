// RUN: %clang -target aarch64-none-gnu -ffixed-x1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X1 < %t %s
// CHECK-FIXED-X1: "-target-feature" "+reserve-x1"

// RUN: %clang -target aarch64-none-gnu -ffixed-x2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X2 < %t %s
// CHECK-FIXED-X2: "-target-feature" "+reserve-x2"

// RUN: %clang -target aarch64-none-gnu -ffixed-x3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X3 < %t %s
// CHECK-FIXED-X3: "-target-feature" "+reserve-x3"

// RUN: %clang -target aarch64-none-gnu -ffixed-x4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X4 < %t %s
// CHECK-FIXED-X4: "-target-feature" "+reserve-x4"

// RUN: %clang -target aarch64-none-gnu -ffixed-x5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X5 < %t %s
// CHECK-FIXED-X5: "-target-feature" "+reserve-x5"

// RUN: %clang -target aarch64-none-gnu -ffixed-x6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X6 < %t %s
// CHECK-FIXED-X6: "-target-feature" "+reserve-x6"

// RUN: %clang -target aarch64-none-gnu -ffixed-x7 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X7 < %t %s
// CHECK-FIXED-X7: "-target-feature" "+reserve-x7"

// RUN: %clang -target aarch64-none-gnu -ffixed-x18 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X18 < %t %s
// CHECK-FIXED-X18: "-target-feature" "+reserve-x18"

// RUN: %clang -target aarch64-none-gnu -ffixed-x20 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X20 < %t %s
// CHECK-FIXED-X20: "-target-feature" "+reserve-x20"

// Test multiple of reserve-x# options together.
// RUN: %clang -target aarch64-none-gnu \
// RUN: -ffixed-x1 \
// RUN: -ffixed-x2 \
// RUN: -ffixed-x18 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-X1 \
// RUN: --check-prefix=CHECK-FIXED-X2 \
// RUN: --check-prefix=CHECK-FIXED-X18 \
// RUN: < %t %s

// Test all reserve-x# options together.
// RUN: %clang -target aarch64-none-gnu \
// RUN: -ffixed-x1 \
// RUN: -ffixed-x2 \
// RUN: -ffixed-x3 \
// RUN: -ffixed-x4 \
// RUN: -ffixed-x5 \
// RUN: -ffixed-x6 \
// RUN: -ffixed-x7 \
// RUN: -ffixed-x18 \
// RUN: -ffixed-x20 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-X1 \
// RUN: --check-prefix=CHECK-FIXED-X2 \
// RUN: --check-prefix=CHECK-FIXED-X3 \
// RUN: --check-prefix=CHECK-FIXED-X4 \
// RUN: --check-prefix=CHECK-FIXED-X5 \
// RUN: --check-prefix=CHECK-FIXED-X6 \
// RUN: --check-prefix=CHECK-FIXED-X7 \
// RUN: --check-prefix=CHECK-FIXED-X18 \
// RUN: --check-prefix=CHECK-FIXED-X20 \
// RUN: < %t %s

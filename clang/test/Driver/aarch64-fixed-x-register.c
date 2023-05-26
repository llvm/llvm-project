// RUN: %clang --target=aarch64-none-gnu -ffixed-x1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X1 < %t %s
// CHECK-FIXED-X1: "-target-feature" "+reserve-x1"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X2 < %t %s
// CHECK-FIXED-X2: "-target-feature" "+reserve-x2"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X3 < %t %s
// CHECK-FIXED-X3: "-target-feature" "+reserve-x3"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X4 < %t %s
// CHECK-FIXED-X4: "-target-feature" "+reserve-x4"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X5 < %t %s
// CHECK-FIXED-X5: "-target-feature" "+reserve-x5"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X6 < %t %s
// CHECK-FIXED-X6: "-target-feature" "+reserve-x6"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x7 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X7 < %t %s
// CHECK-FIXED-X7: "-target-feature" "+reserve-x7"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x9 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X9 < %t %s
// CHECK-FIXED-X9: "-target-feature" "+reserve-x9"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x10 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X10 < %t %s
// CHECK-FIXED-X10: "-target-feature" "+reserve-x10"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x11 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X11 < %t %s
// CHECK-FIXED-X11: "-target-feature" "+reserve-x11"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x12 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X12 < %t %s
// CHECK-FIXED-X12: "-target-feature" "+reserve-x12"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x13 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X13 < %t %s
// CHECK-FIXED-X13: "-target-feature" "+reserve-x13"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x14 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X14 < %t %s
// CHECK-FIXED-X14: "-target-feature" "+reserve-x14"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x15 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X15 < %t %s
// CHECK-FIXED-X15: "-target-feature" "+reserve-x15"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x18 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X18 < %t %s
// CHECK-FIXED-X18: "-target-feature" "+reserve-x18"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x20 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X20 < %t %s
// CHECK-FIXED-X20: "-target-feature" "+reserve-x20"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x21 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X21 < %t %s
// CHECK-FIXED-X21: "-target-feature" "+reserve-x21"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x22 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X22 < %t %s
// CHECK-FIXED-X22: "-target-feature" "+reserve-x22"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x23 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X23 < %t %s
// CHECK-FIXED-X23: "-target-feature" "+reserve-x23"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x24 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X24 < %t %s
// CHECK-FIXED-X24: "-target-feature" "+reserve-x24"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x25 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X25 < %t %s
// CHECK-FIXED-X25: "-target-feature" "+reserve-x25"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x26 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X26 < %t %s
// CHECK-FIXED-X26: "-target-feature" "+reserve-x26"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x27 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X27 < %t %s
// CHECK-FIXED-X27: "-target-feature" "+reserve-x27"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x28 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X28 < %t %s
// CHECK-FIXED-X28: "-target-feature" "+reserve-x28"

// RUN: %clang --target=aarch64-none-gnu -ffixed-x30 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-X30 < %t %s
// CHECK-FIXED-X30: "-target-feature" "+reserve-x30"

// Test multiple of reserve-x# options together.
// RUN: %clang --target=aarch64-none-gnu \
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
// RUN: %clang --target=aarch64-none-gnu \
// RUN: -ffixed-x1 \
// RUN: -ffixed-x2 \
// RUN: -ffixed-x3 \
// RUN: -ffixed-x4 \
// RUN: -ffixed-x5 \
// RUN: -ffixed-x6 \
// RUN: -ffixed-x7 \
// RUN: -ffixed-x9 \
// RUN: -ffixed-x10 \
// RUN: -ffixed-x11 \
// RUN: -ffixed-x12 \
// RUN: -ffixed-x13 \
// RUN: -ffixed-x14 \
// RUN: -ffixed-x15 \
// RUN: -ffixed-x18 \
// RUN: -ffixed-x20 \
// RUN: -ffixed-x21 \
// RUN: -ffixed-x22 \
// RUN: -ffixed-x23 \
// RUN: -ffixed-x24 \
// RUN: -ffixed-x25 \
// RUN: -ffixed-x26 \
// RUN: -ffixed-x27 \
// RUN: -ffixed-x28 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-X1 \
// RUN: --check-prefix=CHECK-FIXED-X2 \
// RUN: --check-prefix=CHECK-FIXED-X3 \
// RUN: --check-prefix=CHECK-FIXED-X4 \
// RUN: --check-prefix=CHECK-FIXED-X5 \
// RUN: --check-prefix=CHECK-FIXED-X6 \
// RUN: --check-prefix=CHECK-FIXED-X7 \
// RUN: --check-prefix=CHECK-FIXED-X9 \
// RUN: --check-prefix=CHECK-FIXED-X10 \
// RUN: --check-prefix=CHECK-FIXED-X11 \
// RUN: --check-prefix=CHECK-FIXED-X12 \
// RUN: --check-prefix=CHECK-FIXED-X13 \
// RUN: --check-prefix=CHECK-FIXED-X14 \
// RUN: --check-prefix=CHECK-FIXED-X15 \
// RUN: --check-prefix=CHECK-FIXED-X18 \
// RUN: --check-prefix=CHECK-FIXED-X20 \
// RUN: --check-prefix=CHECK-FIXED-X21 \
// RUN: --check-prefix=CHECK-FIXED-X22 \
// RUN: --check-prefix=CHECK-FIXED-X23 \
// RUN: --check-prefix=CHECK-FIXED-X24 \
// RUN: --check-prefix=CHECK-FIXED-X25 \
// RUN: --check-prefix=CHECK-FIXED-X26 \
// RUN: --check-prefix=CHECK-FIXED-X27 \
// RUN: --check-prefix=CHECK-FIXED-X28 \
// RUN: < %t %s

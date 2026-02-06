// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r8 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R8 < %t %s
// CHECK-FIXED-R8: "-target-feature" "+reserve-r8"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r9 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R9 < %t %s
// CHECK-FIXED-R9: "-target-feature" "+reserve-r9"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r10 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R10 < %t %s
// CHECK-FIXED-R10: "-target-feature" "+reserve-r10"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r11 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R11 < %t %s
// CHECK-FIXED-R11: "-target-feature" "+reserve-r11"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r12 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R12 < %t %s
// CHECK-FIXED-R12: "-target-feature" "+reserve-r12"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r13 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R13 < %t %s
// CHECK-FIXED-R13: "-target-feature" "+reserve-r13"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r14 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R14 < %t %s
// CHECK-FIXED-R14: "-target-feature" "+reserve-r14"

// RUN: %clang --target=x86_64-unknown-linux-gnu -ffixed-r15 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R15 < %t %s
// CHECK-FIXED-R15: "-target-feature" "+reserve-r15"

// Test multiple of reserve-r# options together.
// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN: -ffixed-r8 \
// RUN: -ffixed-r9 \
// RUN: -ffixed-r15 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-R8 \
// RUN: --check-prefix=CHECK-FIXED-R9 \
// RUN: --check-prefix=CHECK-FIXED-R15 \
// RUN: < %t %s

// Test all reserve-r# options together.
// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN: -ffixed-r8 \
// RUN: -ffixed-r9 \
// RUN: -ffixed-r10 \
// RUN: -ffixed-r11 \
// RUN: -ffixed-r12 \
// RUN: -ffixed-r13 \
// RUN: -ffixed-r14 \
// RUN: -ffixed-r15 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-R8 \
// RUN: --check-prefix=CHECK-FIXED-R9 \
// RUN: --check-prefix=CHECK-FIXED-R10 \
// RUN: --check-prefix=CHECK-FIXED-R11 \
// RUN: --check-prefix=CHECK-FIXED-R12 \
// RUN: --check-prefix=CHECK-FIXED-R13 \
// RUN: --check-prefix=CHECK-FIXED-R14 \
// RUN: --check-prefix=CHECK-FIXED-R15 \
// RUN: < %t %s

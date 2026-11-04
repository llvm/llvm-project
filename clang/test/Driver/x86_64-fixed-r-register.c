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

// RUN: %clang --target=x86_64-unknown-linux-gnu -mapx-features=egpr -ffixed-r16 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R16 < %t %s
// CHECK-FIXED-R16: "-target-feature" "+reserve-r16"

// RUN: %clang --target=x86_64-unknown-linux-gnu -mapx-features=egpr -ffixed-r31 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-R31 < %t %s
// CHECK-FIXED-R31: "-target-feature" "+reserve-r31"

// RUN: not %clang --target=x86_64-unknown-linux-gnu -ffixed-r16 -### %s 2>&1 | FileCheck --check-prefix=CHECK-NO-APX %s
// CHECK-NO-APX: error: unsupported option '-ffixed-r16' for target 'x86_64-unknown-linux-gnu'

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

// Test all reserve-r# options together with APX EGPR.
// RUN: %clang --target=x86_64-unknown-linux-gnu \
// RUN: -mapx-features=egpr \
// RUN: -ffixed-r16 \
// RUN: -ffixed-r17 \
// RUN: -ffixed-r18 \
// RUN: -ffixed-r19 \
// RUN: -ffixed-r20 \
// RUN: -ffixed-r21 \
// RUN: -ffixed-r22 \
// RUN: -ffixed-r23 \
// RUN: -ffixed-r24 \
// RUN: -ffixed-r25 \
// RUN: -ffixed-r26 \
// RUN: -ffixed-r27 \
// RUN: -ffixed-r28 \
// RUN: -ffixed-r29 \
// RUN: -ffixed-r30 \
// RUN: -ffixed-r31 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-ALL-EGPR \
// RUN: < %t %s
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r17"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r18"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r19"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r20"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r21"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r22"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r23"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r24"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r25"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r26"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r27"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r28"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r29"
// CHECK-ALL-EGPR: "-target-feature" "+reserve-r30"

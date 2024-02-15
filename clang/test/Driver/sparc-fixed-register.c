// RUN: %clang --target=sparc-none-gnu -ffixed-g1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G1 < %t %s
// CHECK-FIXED-G1: "-target-feature" "+reserve-g1"

// RUN: %clang --target=sparc-none-gnu -ffixed-g2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G2 < %t %s
// CHECK-FIXED-G2: "-target-feature" "+reserve-g2"

// RUN: %clang --target=sparc-none-gnu -ffixed-g3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G3 < %t %s
// CHECK-FIXED-G3: "-target-feature" "+reserve-g3"

// RUN: %clang --target=sparc-none-gnu -ffixed-g4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G4 < %t %s
// CHECK-FIXED-G4: "-target-feature" "+reserve-g4"

// RUN: %clang --target=sparc-none-gnu -ffixed-g5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G5 < %t %s
// CHECK-FIXED-G5: "-target-feature" "+reserve-g5"

// RUN: %clang --target=sparc-none-gnu -ffixed-g6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G6 < %t %s
// CHECK-FIXED-G6: "-target-feature" "+reserve-g6"

// RUN: %clang --target=sparc-none-gnu -ffixed-g7 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-G7 < %t %s
// CHECK-FIXED-G7: "-target-feature" "+reserve-g7"

// RUN: %clang --target=sparc-none-gnu -ffixed-o0 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O0 < %t %s
// CHECK-FIXED-O0: "-target-feature" "+reserve-o0"

// RUN: %clang --target=sparc-none-gnu -ffixed-o1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O1 < %t %s
// CHECK-FIXED-O1: "-target-feature" "+reserve-o1"

// RUN: %clang --target=sparc-none-gnu -ffixed-o2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O2 < %t %s
// CHECK-FIXED-O2: "-target-feature" "+reserve-o2"

// RUN: %clang --target=sparc-none-gnu -ffixed-o3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O3 < %t %s
// CHECK-FIXED-O3: "-target-feature" "+reserve-o3"

// RUN: %clang --target=sparc-none-gnu -ffixed-o4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O4 < %t %s
// CHECK-FIXED-O4: "-target-feature" "+reserve-o4"

// RUN: %clang --target=sparc-none-gnu -ffixed-o5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-O5 < %t %s
// CHECK-FIXED-O5: "-target-feature" "+reserve-o5"

// RUN: %clang --target=sparc-none-gnu -ffixed-l0 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L0 < %t %s
// CHECK-FIXED-L0: "-target-feature" "+reserve-l0"

// RUN: %clang --target=sparc-none-gnu -ffixed-l1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L1 < %t %s
// CHECK-FIXED-L1: "-target-feature" "+reserve-l1"

// RUN: %clang --target=sparc-none-gnu -ffixed-l2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L2 < %t %s
// CHECK-FIXED-L2: "-target-feature" "+reserve-l2"

// RUN: %clang --target=sparc-none-gnu -ffixed-l3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L3 < %t %s
// CHECK-FIXED-L3: "-target-feature" "+reserve-l3"

// RUN: %clang --target=sparc-none-gnu -ffixed-l4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L4 < %t %s
// CHECK-FIXED-L4: "-target-feature" "+reserve-l4"

// RUN: %clang --target=sparc-none-gnu -ffixed-l5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L5 < %t %s
// CHECK-FIXED-L5: "-target-feature" "+reserve-l5"

// RUN: %clang --target=sparc-none-gnu -ffixed-l6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L6 < %t %s
// CHECK-FIXED-L6: "-target-feature" "+reserve-l6"

// RUN: %clang --target=sparc-none-gnu -ffixed-l7 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-L7 < %t %s
// CHECK-FIXED-L7: "-target-feature" "+reserve-l7"

// RUN: %clang --target=sparc-none-gnu -ffixed-i0 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I0 < %t %s
// CHECK-FIXED-I0: "-target-feature" "+reserve-i0"

// RUN: %clang --target=sparc-none-gnu -ffixed-i1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I1 < %t %s
// CHECK-FIXED-I1: "-target-feature" "+reserve-i1"

// RUN: %clang --target=sparc-none-gnu -ffixed-i2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I2 < %t %s
// CHECK-FIXED-I2: "-target-feature" "+reserve-i2"

// RUN: %clang --target=sparc-none-gnu -ffixed-i3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I3 < %t %s
// CHECK-FIXED-I3: "-target-feature" "+reserve-i3"

// RUN: %clang --target=sparc-none-gnu -ffixed-i4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I4 < %t %s
// CHECK-FIXED-I4: "-target-feature" "+reserve-i4"

// RUN: %clang --target=sparc-none-gnu -ffixed-i5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-I5 < %t %s
// CHECK-FIXED-I5: "-target-feature" "+reserve-i5"

// Test multiple of reserve-* options together.
// RUN: %clang --target=sparc-none-gnu \
// RUN: -ffixed-g1 \
// RUN: -ffixed-o2 \
// RUN: -ffixed-l3 \
// RUN: -ffixed-i4 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-G1 \
// RUN: --check-prefix=CHECK-FIXED-O2 \
// RUN: --check-prefix=CHECK-FIXED-L3 \
// RUN: --check-prefix=CHECK-FIXED-I4 \
// RUN: < %t %s

// Test all reserve-* options together.
// RUN: %clang --target=sparc-none-gnu \
// RUN: -ffixed-g1 \
// RUN: -ffixed-g2 \
// RUN: -ffixed-g3 \
// RUN: -ffixed-g4 \
// RUN: -ffixed-g5 \
// RUN: -ffixed-g6 \
// RUN: -ffixed-g7 \
// RUN: -ffixed-o0 \
// RUN: -ffixed-o1 \
// RUN: -ffixed-o2 \
// RUN: -ffixed-o3 \
// RUN: -ffixed-o4 \
// RUN: -ffixed-o5 \
// RUN: -ffixed-l0 \
// RUN: -ffixed-l1 \
// RUN: -ffixed-l2 \
// RUN: -ffixed-l3 \
// RUN: -ffixed-l4 \
// RUN: -ffixed-l5 \
// RUN: -ffixed-l6 \
// RUN: -ffixed-l7 \
// RUN: -ffixed-i0 \
// RUN: -ffixed-i1 \
// RUN: -ffixed-i2 \
// RUN: -ffixed-i3 \
// RUN: -ffixed-i4 \
// RUN: -ffixed-i5 \
// RUN: -### %s 2> %t
// RUN: FileCheck \
// RUN: --check-prefix=CHECK-FIXED-G1 \
// RUN: --check-prefix=CHECK-FIXED-G2 \
// RUN: --check-prefix=CHECK-FIXED-G3 \
// RUN: --check-prefix=CHECK-FIXED-G4 \
// RUN: --check-prefix=CHECK-FIXED-G5 \
// RUN: --check-prefix=CHECK-FIXED-G6 \
// RUN: --check-prefix=CHECK-FIXED-G7 \
// RUN: --check-prefix=CHECK-FIXED-O0 \
// RUN: --check-prefix=CHECK-FIXED-O1 \
// RUN: --check-prefix=CHECK-FIXED-O2 \
// RUN: --check-prefix=CHECK-FIXED-O3 \
// RUN: --check-prefix=CHECK-FIXED-O4 \
// RUN: --check-prefix=CHECK-FIXED-O5 \
// RUN: --check-prefix=CHECK-FIXED-L0 \
// RUN: --check-prefix=CHECK-FIXED-L1 \
// RUN: --check-prefix=CHECK-FIXED-L2 \
// RUN: --check-prefix=CHECK-FIXED-L3 \
// RUN: --check-prefix=CHECK-FIXED-L4 \
// RUN: --check-prefix=CHECK-FIXED-L5 \
// RUN: --check-prefix=CHECK-FIXED-L6 \
// RUN: --check-prefix=CHECK-FIXED-L7 \
// RUN: --check-prefix=CHECK-FIXED-I0 \
// RUN: --check-prefix=CHECK-FIXED-I1 \
// RUN: --check-prefix=CHECK-FIXED-I2 \
// RUN: --check-prefix=CHECK-FIXED-I3 \
// RUN: --check-prefix=CHECK-FIXED-I4 \
// RUN: --check-prefix=CHECK-FIXED-I5 \
// RUN: < %t %s

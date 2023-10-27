// REQUIRES: m68k-registered-target
// RUN: %clang -target m68k -ffixed-a0 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A0 < %t %s
// CHECK-FIXED-A0: "-target-feature" "+reserve-a0"

// RUN: %clang -target m68k -ffixed-a1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A1 < %t %s
// CHECK-FIXED-A1: "-target-feature" "+reserve-a1"

// RUN: %clang -target m68k -ffixed-a2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A2 < %t %s
// CHECK-FIXED-A2: "-target-feature" "+reserve-a2"

// RUN: %clang -target m68k -ffixed-a3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A3 < %t %s
// CHECK-FIXED-A3: "-target-feature" "+reserve-a3"

// RUN: %clang -target m68k -ffixed-a4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A4 < %t %s
// CHECK-FIXED-A4: "-target-feature" "+reserve-a4"

// RUN: %clang -target m68k -ffixed-a5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A5 < %t %s
// CHECK-FIXED-A5: "-target-feature" "+reserve-a5"

// RUN: %clang -target m68k -ffixed-a6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-A6 < %t %s
// CHECK-FIXED-A6: "-target-feature" "+reserve-a6"

// RUN: %clang -target m68k -ffixed-d0 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D0 < %t %s
// CHECK-FIXED-D0: "-target-feature" "+reserve-d0"

// RUN: %clang -target m68k -ffixed-d1 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D1 < %t %s
// CHECK-FIXED-D1: "-target-feature" "+reserve-d1"

// RUN: %clang -target m68k -ffixed-d2 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D2 < %t %s
// CHECK-FIXED-D2: "-target-feature" "+reserve-d2"

// RUN: %clang -target m68k -ffixed-d3 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D3 < %t %s
// CHECK-FIXED-D3: "-target-feature" "+reserve-d3"

// RUN: %clang -target m68k -ffixed-d4 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D4 < %t %s
// CHECK-FIXED-D4: "-target-feature" "+reserve-d4"

// RUN: %clang -target m68k -ffixed-d5 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D5 < %t %s
// CHECK-FIXED-D5: "-target-feature" "+reserve-d5"

// RUN: %clang -target m68k -ffixed-d6 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D6 < %t %s
// CHECK-FIXED-D6: "-target-feature" "+reserve-d6"

// RUN: %clang -target m68k -ffixed-d7 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-D7 < %t %s
// CHECK-FIXED-D7: "-target-feature" "+reserve-d7"

// ==== Floating point ====
// RUN: %clang -target m68k -m68000 -mhard-float -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX881 < %t %s
// RUN: %clang -target m68k -m68000 -m68881 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX881 < %t %s

// RUN: %clang -target m68k -m68010 -mhard-float -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX881 < %t %s
// RUN: %clang -target m68k -m68010 -m68881 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX881 < %t %s

// RUN: %clang -target m68k -m68020 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX881 < %t %s

// RUN: %clang -target m68k -m68030 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MX882 < %t %s

// CHECK-MX881: "-target-feature" "+isa-68881"
// CHECK-MX882: "-target-feature" "+isa-68882"


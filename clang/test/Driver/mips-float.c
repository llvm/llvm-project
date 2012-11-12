// REQUIRES: mips-registered-target
// Check handling -mhard-float / -msoft-float / -mfloat-abi options
// when build for MIPS platforms.
//
// Default
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// CHECK-DEF: "-mfloat-abi" "hard"
//
// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD: "-mfloat-abi" "hard"
//
// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: "-msoft-float"
// CHECK-SOFT: "-mfloat-abi" "soft"
// CHECK-SOFT: "-target-feature" "+soft-float"
//
// -mfloat-abi=hard
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-HARD %s
// CHECK-ABI-HARD: "-mfloat-abi" "hard"
//
// -mfloat-abi=soft
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mfloat-abi=soft \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-SOFT %s
// CHECK-ABI-SOFT: "-msoft-float"
// CHECK-ABI-SOFT: "-mfloat-abi" "soft"
// CHECK-ABI-SOFT: "-target-feature" "+soft-float"
//
// -mfloat-abi=single
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mfloat-abi=single \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-SINGLE %s
// CHECK-ABI-SINGLE: "-target-feature" "+single-float"

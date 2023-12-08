// Check handling -mhard-float / -msoft-float / -mfloat-abi options
// when build for MIPS platforms.
//
// Default
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// CHECK-DEF: "-mfloat-abi" "hard"
//
// Default on FreeBSD
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-freebsd12 \
// RUN:   | FileCheck --check-prefix=DEF-FREEBSD %s
// DEF-FREEBSD: "-target-feature" "+soft-float"
// DEF-FREEBSD: "-msoft-float"
// DEF-FREEBSD: "-mfloat-abi" "soft"
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
// CHECK-SOFT: "-target-feature" "+soft-float"
// CHECK-SOFT: "-msoft-float"
// CHECK-SOFT: "-mfloat-abi" "soft"
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
// CHECK-ABI-SOFT: "-target-feature" "+soft-float"
// CHECK-ABI-SOFT: "-msoft-float"
// CHECK-ABI-SOFT: "-mfloat-abi" "soft"
//
// -mdouble-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msingle-float -mdouble-float \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-DOUBLE %s
// CHECK-ABI-DOUBLE: "-mfloat-abi" "hard"
// CHECK-ABI-DOUBLE-NOT: "+single-float"
//
// -msingle-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mdouble-float -msingle-float \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-SINGLE %s
// CHECK-ABI-SINGLE: "-target-feature" "+single-float"
// CHECK-ABI-SINGLE: "-mfloat-abi" "hard"
//
// -msoft-float -msingle-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -msingle-float \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-SOFT-SINGLE %s
// CHECK-ABI-SOFT-SINGLE: "-target-feature" "+single-float"
// CHECK-ABI-SOFT-SINGLE: "-mfloat-abi" "soft"
//
// Default -mips16
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mips16 \
// RUN:   | FileCheck --check-prefix=CHECK-DEF-MIPS16 %s
// CHECK-DEF-MIPS16: "-target-feature" "+mips16"
// CHECK-DEF-MIPS16: "-mfloat-abi" "hard"
//
// -mhard-float -mips16
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mhard-float -mips16 \
// RUN:   | FileCheck --check-prefix=CHECK-HARD-MIPS16 %s
// CHECK-HARD-MIPS16: "-target-feature" "+mips16"
// CHECK-HARD-MIPS16: "-mfloat-abi" "hard"
//
// -msoft-float -mips16
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -mips16 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-MIPS16 %s
// CHECK-SOFT-MIPS16: "-target-feature" "+soft-float"
// CHECK-SOFT-MIPS16: "-target-feature" "+mips16"
// CHECK-SOFT-MIPS16: "-msoft-float"
// CHECK-SOFT-MIPS16: "-mfloat-abi" "soft"
//
// -mfloat-abi=hard -mips16
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mfloat-abi=hard -mips16 \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-HARD-MIPS16 %s
// CHECK-ABI-HARD-MIPS16: "-target-feature" "+mips16"
// CHECK-ABI-HARD-MIPS16: "-mfloat-abi" "hard"
//
// -mfloat-abi=soft -mips16
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mfloat-abi=soft -mips16 \
// RUN:   | FileCheck --check-prefix=CHECK-ABI-SOFT-MIPS16 %s
// CHECK-ABI-SOFT-MIPS16: "-target-feature" "+soft-float"
// CHECK-ABI-SOFT-MIPS16: "-target-feature" "+mips16"
// CHECK-ABI-SOFT-MIPS16: "-msoft-float"
// CHECK-ABI-SOFT-MIPS16: "-mfloat-abi" "soft"

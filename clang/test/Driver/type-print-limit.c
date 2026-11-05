// RUN: %clang -### -ftype-print-limit=25 %s 2>&1 | FileCheck %s --check-prefix=SET
// RUN: %clang -### -ftype-print-limit=0 %s 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: %clang -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULT

// SET: "-ftype-print-limit=25"
// ZERO: "-ftype-print-limit=0"
// DEFAULT-NOT: -ftype-print-limit

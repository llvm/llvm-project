// RUN: %clang -### -S -foperator-names -fno-operator-names %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-1 %s
// CHECK-1: "-fno-operator-names"

// RUN: %clang -### -S -fno-operator-names -foperator-names %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CHECK-2 %s
// CHECK-2-NOT: "-fno-operator-names"

// RUN: %clang --target=hexagon -### -mmemops %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-MEMOPS

// RUN: %clang --target=hexagon -### -mno-memops %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-MEMOPS

// CHECK-MEMOPS: "-target-feature" "+memops"

// CHECK-NO-MEMOPS: "-target-feature" "-memops"


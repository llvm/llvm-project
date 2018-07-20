// RUN: %clang -target hexagon -### -mnvj %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NVJ

// RUN: %clang -target hexagon -### -mno-nvj %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-NVJ

// CHECK-NVJ: "-target-feature" "+nvj"

// CHECK-NO-NVJ: "-target-feature" "-nvj"


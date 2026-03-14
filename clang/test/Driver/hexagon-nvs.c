// RUN: %clang --target=hexagon -### -mnvs %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NVS

// RUN: %clang --target=hexagon -### -mno-nvs %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-NVS

// CHECK-NVS: "-target-feature" "+nvs"

// CHECK-NO-NVS: "-target-feature" "-nvs"


// RUN: %clang -target hexagon -### -mpackets %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-PACKETS

// RUN: %clang -target hexagon -### -mno-packets %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-PACKETS

// CHECK-PACKETS: "-target-feature" "+packets"

// CHECK-NO-PACKETS: "-target-feature" "-packets"


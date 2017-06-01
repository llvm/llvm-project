// RUN: %clang -target hexagon -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target hexagon -### -mlong-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LONG-CALLS

// RUN: %clang -target hexagon -### -mlong-calls -mno-long-calls %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-LONG-CALLS

// CHECK-DEFAULT-NOT: "-target-feature" "+long-calls"

// CHECK-LONG-CALLS: "-target-feature" "+long-calls"

// CHECK-NO-LONG-CALLS-NOT: "-target-feature" "+long-calls"


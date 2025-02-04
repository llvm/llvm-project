// RUN: %clang -target aarch64 -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target aarch64 -### -mexecute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target aarch64 -### -mexecute-only -mno-execute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY


// -mpure-code flag for GCC compatibility
// RUN: %clang -target aarch64 -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target aarch64 -### -mpure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target aarch64 -### -mpure-code -mno-pure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// CHECK-NO-EXECUTE-ONLY-NOT: "+execute-only"
// CHECK-EXECUTE-ONLY: "+execute-only"

void a() {}

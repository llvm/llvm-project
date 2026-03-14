// RUN: %clang -target armv6t2-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv6t2-eabi -### -mexecute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv6t2-eabi -### -mexecute-only -mno-execute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### -mexecute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### -mexecute-only -mno-execute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### -mexecute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### -mexecute-only -mno-execute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### -mexecute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### -mexecute-only -mno-execute-only %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY


// -mpure-code flag for GCC compatibility
// RUN: %clang -target armv6t2-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv6t2-eabi -### -mpure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv6t2-eabi -### -mpure-code -mno-pure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### -mpure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv7m-eabi -### -mpure-code -mno-pure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### -mpure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv8m.base-eabi -### -mpure-code -mno-pure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### -mpure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-EXECUTE-ONLY

// RUN: %clang -target armv8m.main-eabi -### -mpure-code -mno-pure-code %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-EXECUTE-ONLY

// CHECK-NO-EXECUTE-ONLY-NOT: "+execute-only"
// CHECK-EXECUTE-ONLY: "+execute-only"

void a() {}

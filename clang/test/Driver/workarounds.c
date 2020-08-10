// Check that Driver workarounds are enabled.
//
// RUN: %clang %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-unknown %s -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-apple-darwin %s -W -### 2>&1 | FileCheck %s
// CHECK: "-cc1"
// Note: Add CHECK-SAME checks after this note for each workaround.
// CHECK-SAME: "-Wno-elaborated-enum-base"


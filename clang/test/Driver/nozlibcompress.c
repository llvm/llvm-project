// REQUIRES: !zlib

// RUN: %clang -### -fintegrated-as -gz -c %s 2>&1 | FileCheck %s -check-prefix CHECK-WARN
// RUN: %clang -### -fintegrated-as -gz=none -c %s 2>&1 | FileCheck -allow-empty -check-prefix CHECK-NOWARN %s

// CHECK-WARN: warning: cannot compress debug sections (zlib not installed)
// CHECK-NOWARN-NOT: warning: cannot compress debug sections (zlib not installed)

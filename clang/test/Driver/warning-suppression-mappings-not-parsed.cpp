// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo '[unknown-warning]' > %t/foo.txt
// RUN: %clang -fdriver-only --warning-suppression-mappings=%t/foo.txt %s | FileCheck -allow-empty %s
// CHECK-NOT: unknown warning option 'unknown-warning'

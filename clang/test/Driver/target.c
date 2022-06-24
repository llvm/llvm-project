// RUN: %clang --target=unknown-unknown-unknown -c %s \
// RUN:   -### 2>&1 | FileCheck %s

// Ensure we get a crazy triple here as we asked for one.
// CHECK: Target: unknown-unknown-unknown

// Also check that the legacy spelling works.
// RUN: %clang --target=unknown-unknown-unknown -c %s \
// RUN:   -### 2>&1 | FileCheck %s

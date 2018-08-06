// RUN: not %clang %s --hedonism -### 2>&1 | \
// RUN: FileCheck %s
// RUN: not %clang %s --hell -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN

// CHECK: error: unsupported option '--hedonism'
// DID-YOU-MEAN: error: unsupported option '--hell', did you mean '--help'?

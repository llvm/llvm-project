// RUN: not %clang_cc1 %s -E --helium 2>&1 | \
// RUN: FileCheck %s
// RUN: not %clang_cc1 %s -E --hel[ 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN
// RUN: not %clang %s -E -Xclang --hel[ 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN

// CHECK: error: unknown argument: '--helium'
// DID-YOU-MEAN: error: unknown argument '--hel[', did you mean '--help'?

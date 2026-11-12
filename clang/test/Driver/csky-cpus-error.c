// Check failed cases

// RUN: not %clang -target csky -c %s 2>&1 -mcpu=generic1 | FileCheck -check-prefix=FAIL-MCPU-NAME %s
// FAIL-MCPU-NAME: error: the clang compiler does not support '-mcpu=generic1'

// RUN: not %clang -target csky -c %s 2>&1 -mcpu=generic -march=ck860 | FileCheck -check-prefix=MISMATCH-ARCH %s
// MISMATCH-ARCH: error: the clang compiler does not support '-mcpu=generic'

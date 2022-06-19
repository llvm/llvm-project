// RUN: not %clang -target csky-unknown-elf -march=csky %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CSKY %s
// CSKY: error: invalid arch name '-march=csky'

// RUN: not %clang -target csky-unknown-elf -march=CK810 %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CSKY-UPPER %s
// CSKY-UPPER: error: invalid arch name '-march=CK810'

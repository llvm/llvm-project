// RUN: not %clang --target=loongarch64 -march=loongarch -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LOONGARCH %s
// LOONGARCH: error: invalid arch name '-march=loongarch'

// RUN: not %clang --target=loongarch64 -march=LA464 -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck --check-prefix=LA464-UPPER %s
// LA464-UPPER: error: invalid arch name '-march=LA464'

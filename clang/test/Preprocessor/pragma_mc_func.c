// RUN: not %clang --target=powerpc64-ibm-aix -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang --target=powerpc64-ibm-aix -ferr-pragma-mc-func-aix -fsyntax-only \
// RUN:   %s 2>&1 | FileCheck %s
#pragma mc_func asm_barrier {"60000000"}

// CHECK:  error: #pragma mc_func is not supported

// Cases where no errors occur.
// RUN: %clang --target=powerpc64-ibm-aix -fno-err-pragma-mc-func-aix -fsyntax-only %s
// RUN: %clang --target=powerpc64-ibm-aix -ferr-pragma-mc-func-aix -fsyntax-only \
// RUN:    -fno-err-pragma-mc-func-aix %s
// RUN: %clang --target=powerpc64-ibm-aix -Werror=unknown-pragmas \
// RUN:   -fno-err-pragma-mc-func-aix -fsyntax-only %s

// Cases on a non-AIX target.
// RUN: not %clang --target=powerpc64le-unknown-linux-gnu \
// RUN:   -Werror=unknown-pragmas -fno-err-pragma-mc-func-aix -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck --check-prefix=UNUSED %s
// RUN: %clang --target=powerpc64le-unknown-linux-gnu \
// RUN:   -fno-err-pragma-mc-func-aix -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck --check-prefix=UNUSED %s

// UNUSED: clang: warning: argument unused during compilation: '-fno-err-pragma-mc-func-aix' [-Wunused-command-line-argument]

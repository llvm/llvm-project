// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=return %s | FileCheck %s --check-prefix=RETURN
// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=branch %s | FileCheck %s --check-prefix=BRANCH
// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - -fcf-protection=full %s   | FileCheck %s --check-prefix=FULL

// RETURN: #define __CET__ 2
// BRANCH: #define __CET__ 1
// FULL: #define __CET__ 3
void foo() {}

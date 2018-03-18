// RUN: not %clang_cc1 -fsyntax-only -S -o %t -triple i386-unknown-unknown -fcf-protection=return %s 2>&1 | FileCheck %s --check-prefix=RETURN
// RUN: not %clang_cc1 -fsyntax-only -S -o %t -triple i386-unknown-unknown -fcf-protection=branch %s 2>&1 | FileCheck %s --check-prefix=BRANCH

// RETURN: error: option 'cf-protection=return' cannot be specified without '-mshstk'
// BRANCH: error: option 'cf-protection=branch' cannot be specified without '-mibt'
void foo() {}

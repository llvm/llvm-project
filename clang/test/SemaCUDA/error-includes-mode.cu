// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck --check-prefix HOST %s
// RUN: not %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_35 \
// RUN:   -fcuda-is-device -fsyntax-only %s 2>&1 | FileCheck --check-prefix SM35 %s

// HOST: 1 error generated when compiling for host
// SM35: 1 error generated when compiling for sm_35
error;

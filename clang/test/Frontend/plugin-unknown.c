// RUN: not %clang_cc1 -plugin asdf %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -add-plugin asdf %s 2>&1 | FileCheck --check-prefix=ADD %s

// CHECK: unable to find plugin 'asdf'
// ADD: unable to find plugin 'asdf'

// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple powerpc-unknown-aix -maix-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix -msvr4-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -maix-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -msvr4-struct-return \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK: unsupported option

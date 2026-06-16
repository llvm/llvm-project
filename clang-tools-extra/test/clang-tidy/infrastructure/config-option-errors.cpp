// RUN: not clang-tidy --config='not valid yaml: ][' \
// RUN:     -checks='misc-explicit-constructor' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-INVALID
// CHECK-INVALID: Error: invalid configuration specified.

// RUN: not clang-tidy --config='{}' --config-file=%S/Inputs/config-file/config-file %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-MUTUAL
// CHECK-MUTUAL: Error: --config-file and --config are mutually exclusive.

// RUN: not clang-tidy --config-file=%t.nonexistent \
// RUN:     -checks='misc-explicit-constructor' %s -- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-NOT-FOUND
// CHECK-NOT-FOUND: Error: can't read config-file '{{.*}}.nonexistent':

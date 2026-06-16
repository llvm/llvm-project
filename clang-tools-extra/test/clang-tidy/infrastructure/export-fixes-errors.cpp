// RUN: not clang-tidy \
// RUN:     --export-fixes=%t.no-such-directory/fixes.yaml \
// RUN:     -checks='misc-explicit-constructor' %s -- 2>&1 \
// RUN:   | FileCheck %s

// CHECK: Error opening output file:

class A { A(int) {} };

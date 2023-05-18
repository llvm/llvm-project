__attribute__((overloadable))
void testFixedPoint(_Accum);
// CHECK: USR: c:@F@testFixedPoint#@BT@Accum#

// RUN: c-index-test -index-file %s -ffixed-point | FileCheck %s

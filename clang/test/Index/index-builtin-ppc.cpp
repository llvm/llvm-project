void testPpc(__vector_quad);
// CHECK: USR: c:@F@testPpc#@BT@__vector_quad#

void testIBM(__ibm128);
// CHECK: USR: c:@F@testIBM#@BT@__ibm128#
//
// RUN: c-index-test -index-file %s --target=powerpc64 -target-cpu pwr10 | FileCheck %s

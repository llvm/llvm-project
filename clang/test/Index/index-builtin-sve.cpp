void testSve(__SVInt8_t sve);
// CHECK: USR: c:@F@testSve#@BT@__SVInt8_t#

void testBf16(__bf16);
// CHECK: USR: c:@F@testBf16#@BT@__bf16#

// RUN: c-index-test -index-file %s --target=aarch64 -target-feature +bf16 -target-feature +sve -std=c++11 | FileCheck %s

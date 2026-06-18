// RUN: not %clang_cc1 -fnative-half-type -fnative-int16-type -std=hlsl202x -triple dxilv1.0-unknown-shadermodel6.0-compute \
// RUN:  -finclude-default-header -S -o - %s 2>&1 | FileCheck %s -DTEST_TYPE=half
// RUN: not %clang_cc1 -fnative-half-type -fnative-int16-type -std=hlsl202x -triple dxilv1.0-unknown-shadermodel6.0-compute \
// RUN:  -finclude-default-header -S -o - %s 2>&1 | FileCheck %s -DTEST_TYPE=int16_t
// RUN: not %clang_cc1 -fnative-half-type -fnative-int16-type -std=hlsl202x -triple dxilv1.0-unknown-shadermodel6.0-compute \
// RUN:  -finclude-default-header -S -o - %s 2>&1 | FileCheck %s -DTEST_TYPE=uint16_t

// check we error on 16 bit type if shader model is too old
// CHECK: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl202x' and shader model is '6.0'
vector<TEST_TYPE,3> test_error(vector<TEST_TYPE,3> p0, TEST_TYPE p1) {
  return min(p0, p1);
}

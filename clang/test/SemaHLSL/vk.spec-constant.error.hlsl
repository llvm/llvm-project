// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.8-compute -verify %s

#ifndef __spirv__
// expected-warning@+2{{'vk::constant_id' attribute ignored}}
#endif
[[vk::constant_id(0)]]
const bool sc0 = true;

#ifdef __spirv__
// expected-error@+2{{variable with 'vk::constant_id' attribute must be a const int/float/enum/bool and be initialized with a literal}}
[[vk::constant_id(1)]]
const bool sc1 = sc0; // error

// expected-warning@+1{{'vk::constant_id' attribute only applies to external global variables}}
[[vk::constant_id(2)]]
static const bool sc2 = false; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be a const int/float/enum/bool and be initialized with a literal}}
[[vk::constant_id(3)]]
const bool sc3; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be a const int/float/enum/bool and be initialized with a literal}}
[[vk::constant_id(4)]]
bool sc4 = false; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be a const int/float/enum/bool and be initialized with a literal}}
[[vk::constant_id(5)]]
const int2 sc5 = {0,0}; // error

[numthreads(1,1,1)]
void main() {
  // expected-warning@+1{{'vk::constant_id' attribute only applies to external global variables}}
  [[vk::constant_id(6)]]
  const bool sc6 = false; // error
}
#endif

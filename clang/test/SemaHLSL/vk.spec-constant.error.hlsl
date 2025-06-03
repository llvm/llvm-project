// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-compute -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.8-compute -verify %s

#ifndef __spirv__
// expected-warning@+2{{'constant_id' attribute ignored}}
#endif
[[vk::constant_id(0)]]
const bool sc0 = true;

#ifdef __spirv__
// expected-error@+2{{variable with 'vk::constant_id' attribute cannot have an initializer that is not a constexpr}}
[[vk::constant_id(1)]]
const bool sc1 = sc0; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be externally visible}}
[[vk::constant_id(2)]]
static const bool sc2 = false; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must have an initializer}}
[[vk::constant_id(3)]]
const bool sc3; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be const}}
[[vk::constant_id(4)]]
bool sc4 = false; // error

// expected-error@+2{{variable with 'vk::constant_id' attribute must be an enum, bool, integer, or floating point value}}
[[vk::constant_id(5)]]
const int2 sc5 = {0,0}; // error

[numthreads(1,1,1)]
void main() {
  // expected-error@+2{{variable with 'vk::constant_id' attribute must be externally visible}}
  [[vk::constant_id(6)]]
  const bool sc6 = false; // error
}
#endif

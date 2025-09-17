// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint64_t5x5 mat;
// expected-error@-1  {{unknown type name 'uint64_t5x5'}}

// Note: this one only fails because -fnative-half-type is not set
uint16_t4x4 mat;
// expected-error@-1  {{unknown type name 'uint16_t4x4'}}


// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

uint64_t5x5 mat;
// expected-error@-1  {{unknown type name 'uint64_t5x5'}}

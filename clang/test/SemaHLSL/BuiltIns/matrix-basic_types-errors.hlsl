// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint64_t5x5 mat;
// expected-error@-1  {{unknown type name 'uint64_t5x5'}}

// Note: this one only fails because -fnative-half-type is not set
uint16_t4x4 mat2;
// expected-error@-1  {{unknown type name 'uint16_t4x4'}}

matrix<int, 5, 5> mat3;
// expected-error@-1 {{constraints not satisfied for alias template 'matrix' [with element = int, rows_count = 5, cols_count = 5]}}
// expected-note@* {{because '5 <= 4' (5 <= 4) evaluated to false}}

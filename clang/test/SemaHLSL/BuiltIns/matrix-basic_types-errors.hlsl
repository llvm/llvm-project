// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint64_t5x5 mat;
// expected-error@-1  {{unknown type name 'uint64_t5x5'}}

// Note: this one only fails because -fnative-half-type is not set
uint16_t4x4 mat2;
// expected-error@-1  {{unknown type name 'uint16_t4x4'}}

matrix<int, 5, 5> mat3;
// expected-error@-1 {{constraints not satisfied for alias template 'matrix' [with element = int, rows_count = 5, cols_count = 5]}}
// expected-note@* {{because '5 <= 4' (5 <= 4) evaluated to false}}

using float8x4 = __attribute__((matrix_type(8,4))) float;
// expected-error@-1 {{matrix row size too large}}

using float4x8 = __attribute__((matrix_type(4,8))) float;
// expected-error@-1 {{matrix column size too large}}

using float8x8 = __attribute__((matrix_type(8,8))) float;
// expected-error@-1 {{matrix row and column size too large}}

using floatNeg1x4 = __attribute__((matrix_type(-1,4))) float;
// expected-error@-1 {{matrix row size too large}}
using float4xNeg1 = __attribute__((matrix_type(4,-1))) float;
// expected-error@-1 {{matrix column size too large}}
using floatNeg1xNeg1 = __attribute__((matrix_type(-1,-1))) float; 
// expected-error@-1 {{matrix row and column size too large}}

using float0x4 = __attribute__((matrix_type(0,4))) float;
// expected-error@-1 {{zero matrix size}}
using float4x0 = __attribute__((matrix_type(4,0))) float;
// expected-error@-1 {{zero matrix size}}
using float0x0 = __attribute__((matrix_type(0,0))) float;
// expected-error@-1 {{zero matrix size}}

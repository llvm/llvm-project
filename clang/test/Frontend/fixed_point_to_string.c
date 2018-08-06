// RUN: %clang_cc1 -ast-dump -ffixed-point %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -ffixed-point -fpadding-on-unsigned-fixed-point %s | FileCheck %s

/**
 * Check the same values are printed in the AST regardless of if unsigned types
 * have the same number of fractional bits as signed types.
 */

unsigned short _Accum u_short_accum = 0.5uhk;
unsigned _Accum u_accum = 0.5uk;
unsigned long _Accum u_long_accum = 0.5ulk;
unsigned short _Fract u_short_fract = 0.5uhr;
unsigned _Fract u_fract = 0.5ur;
unsigned long _Fract u_long_fract = 0.5ulr;

//CHECK: FixedPointLiteral {{.*}} 'unsigned short _Accum' 0.5
//CHECK: FixedPointLiteral {{.*}} 'unsigned _Accum' 0.5
//CHECK: FixedPointLiteral {{.*}} 'unsigned long _Accum' 0.5
//CHECK: FixedPointLiteral {{.*}} 'unsigned short _Fract' 0.5
//CHECK: FixedPointLiteral {{.*}} 'unsigned _Fract' 0.5
//CHECK: FixedPointLiteral {{.*}} 'unsigned long _Fract' 0.5

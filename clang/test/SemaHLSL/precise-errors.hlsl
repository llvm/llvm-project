// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -verify %s

// expected-error@+1 {{'precise' attribute only applies to variables, non-static data members, and functions}}
typedef precise float PreciseFloat;

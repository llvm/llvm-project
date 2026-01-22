// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

void SplatOfMattoMat(int4x3 N){
    int4x4 M = N;
    // expected-error@-1 {{cannot initialize a variable of type 'matrix<[2 * ...], 4>' with an lvalue of type 'matrix<[2 * ...], 3>'}}
}

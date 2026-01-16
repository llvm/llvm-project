// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

void SplatOfVectortoMat(int4 V){
    int2x2 M = V;
    // expected-error@-1 {{cannot initialize a variable of type 'int2x2' (aka 'matrix<int, 2, 2>') with an lvalue of type 'int4' (aka 'vector<int, 4>')}}
}

void SplatOfMattoMat(int4x3 N){
    int4x4 M = N;
    // expected-error@-1 {{cannot initialize a variable of type 'matrix<[2 * ...], 4>' with an lvalue of type 'matrix<[2 * ...], 3>'}}
}

// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

void SplatOfUndersizedVectortoMat(int3 V){
    int2x2 M = V;
    // expected-error@-1 {{too few initializers in list for type 'int2x2' (aka 'matrix<int, 2, 2>') (expected 4 but found 3)}}
}

void SplatOfOversizedVectortoMat(int3 V){
    int1x2 M = V;
    // expected-error@-1 {{too many initializers in list for type 'int1x2' (aka 'matrix<int, 1, 2>') (expected 2 but found 3)}}
}

void SplatOfMattoMat(int4x3 N){
    int4x4 M = N;
    // expected-error@-1 {{cannot initialize a variable of type 'matrix<[2 * ...], 4>' with an lvalue of type 'matrix<[2 * ...], 3>'}}
}

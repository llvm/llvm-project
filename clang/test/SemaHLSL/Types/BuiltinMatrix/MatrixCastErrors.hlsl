// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

// Note column is too large
export int3x2 shape_cast_error(float2x3 f23) {
    int3x2 i32 = (int3x2)f23;
    // expected-error@-1 {{conversion between matrix types 'int3x2' (aka 'matrix<int, 3, 2>') and 'matrix<float, 2, 3>' of different size is not allowed}}
    return i32;
}
// Note row is too large
export int2x3 shape_cast_error2(float3x2 f32) {
    int2x3 i23 = (int2x3)f32;
    // expected-error@-1 {{conversion between matrix types 'int2x3' (aka 'matrix<int, 2, 3>') and 'matrix<float, 3, 2>' of different size is not allowed}}
    return i23;
}

// Note do the type change independent of the shape should still error
export int2x3 shape_cast_error3(float3x2 f32) {
    int2x3 i23 = (int3x2)f32;
    // expected-error@-1 {{cannot initialize a variable of type 'matrix<[...], 2, 3>' with an rvalue of type 'matrix<[...], 3, 2>}}
    return i23;
}

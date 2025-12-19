// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -verify %s

export int3x4 trunc_cast(int4x4 i44) {
    int3x4 i34 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 3, 4>'}}
    return i34;
}

export int4x3 trunc_cast0(int4x4 i44) {
    int4x3 i43 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 4, 3>'}}
    return i43;
}

export int3x3 trunc_cast1(int4x4 i44) {
    int3x3 i33 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 3, 3>'}}
    return i33;
}

export int3x2 trunc_cast2(int4x4 i44) {
    int3x2 i32 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 3, 2>'}}
    return i32;
}

export int2x3 trunc_cast3(int4x4 i44) {
    int2x3 i23 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 2, 3>'}}
    return i23;
}

export int2x2 trunc_cast4(int4x4 i44) {
    int2x2 i22 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 2, 2>'}}
    return i22;
}

export int2x1 trunc_cast5(int4x4 i44) {
    int2x1 i21 = i44;
    // expected-warning@-1{{implicit conversion truncates matrix: 'int4x4' (aka 'matrix<int, 4, 4>') to 'matrix<int, 2, 1>'}}
    return i21;
}

export int trunc_scalar_cast6(int4x4 i44) {
    int i1 = i44;
    // expected-warning@-1{{implicit conversion turns matrix to scalar: 'int4x4' (aka 'matrix<int, 4, 4>') to 'int'}}
    return i1;
}


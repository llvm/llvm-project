// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -verify %s

void foo() {
    float3x3 A;
    float r = A._m00;      // read is ok
    float good1 = A._11;    
    float good2 = A._33;

    float bad0 = A._m44;    // expected-error {{row subscript is out of bounds of zero based indexing}} expected-error {{column subscript is out of bounds of zero based indexing}}
    float bad1 = A._m33;    // expected-error {{row subscripted is out of bounds}} expected-error {{column subscripted is out of bounds}}
    float bad2 = A._mA2;    // expected-error {{matrix subscript is not an integer}}
    float bad3 = A._m2F;    // expected-error {{matrix subscript is not an integer}}

    float bad4 = A._00;    // expected-error {{row subscript is out of bounds of one based indexing}} expected-error {{column subscript is out of bounds of one based indexing}}
    float bad5 = A._44;    // expected-error {{row subscripted is out of bounds}} expected-error {{column subscripted is out of bounds}}
    float bad6 = A._55;    // expected-error {{row subscript is out of bounds of one based indexing}} expected-error {{column subscript is out of bounds of one based indexing}}

    
    A._m12 = 3.14;         // write is OK
}

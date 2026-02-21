// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -finclude-default-header -verify %s

typedef vector<float, 5> float5;

void foo() {
    float3x3 A;
    float r = A._m00;         // read is ok
    float good1 = A._11;
    float good2 = A._33;

    float bad0 = A._m44;      // expected-error {{matrix row element accessor is out of bounds of zero based indexing}} expected-error {{matrix column element accessor is out of bounds of zero based indexing}}
    float bad1 = A._m33;      // expected-error {{matrix row index 3 is out of bounds of rows size 3}} expected-error {{matrix column index 3 is out of bounds of columns size 3}}
    float bad2 = A._mA2;      // expected-error {{invalid matrix member 'A' expected row as integer}}
    float bad3 = A._m2F;      // expected-error {{invalid matrix member 'F' expected column as integer}}

    float bad4 = A._00;       // expected-error {{matrix row element accessor is out of bounds of one based indexing}} expected-error {{matrix column element accessor is out of bounds of one based indexing}}
    float bad5 = A._44;       // expected-error {{matrix row index 3 is out of bounds of rows size 3}} expected-error {{matrix column index 3 is out of bounds of columns size 3}}
    float bad6 = A._55;       // expected-error {{matrix row element accessor is out of bounds of one based indexing}} expected-error {{matrix column element accessor is out of bounds of one based indexing}}
    float bad7 = A.foo;       // expected-error {{invalid matrix member 'foo' expected zero based: '_mRC' or one-based: '_RC' accessor}}
    float2 bad8 = A._m00_33;  // expected-error {{invalid matrix member '_m00_33' expected zero based: '_mRC' accessor}}
    float2 bad9 = A._11_m33;  // expected-error {{invalid matrix member '_11_m33' expected one-based: '_RC' accessor}}
    float bad10 = A._m0000;   // expected-error {{invalid matrix member '_m0000' expected zero based: '_mRC' accessor}}
    float bad11 = A._m1;      // expected-error {{invalid matrix member '_m1' expected zero based: '_mRC' accessor}}
    float bad12 = A._m;       // expected-error {{invalid matrix member '_m' expected zero based: '_mRC' accessor}}
    float bad13 = A._1;       // expected-error {{invalid matrix member '_1' expected one-based: '_RC' accessor}}
    float bad14 = A.m;        // expected-error {{invalid matrix member 'm' expected length 4 for zero based: '_mRC' or length 3 for one-based: '_RC' accessor}}
    float bad15 = A._;        // expected-error {{invalid matrix member '_' expected length 4 for zero based: '_mRC' or length 3 for one-based: '_RC' accessor}}
    float bad16 = A._m00_m;   // expected-error {{invalid matrix member '_m00_m' expected zero based: '_mRC' accessor}}
    float bad17 = A._m11_m2;  // expected-error {{invalid matrix member '_m11_m2' expected zero based: '_mRC' accessor}}
    float bad18 = A._m11_mAF; // expected-error {{invalid matrix member 'A' expected row as integer}} // expected-error {{invalid matrix member 'F' expected column as integer}}

    A._m12 = 3.14;           // write is OK
    A._m00_m00 = 1.xx;       // expected-error {{matrix is not assignable (contains duplicate components)}}

    float4x4 B;
    float5 vec5;
    B._m00_m01_m02_m03_m10 = vec5;  // expected-error {{matrix swizzle length must be between 1 and 4 but is 5}}
    float5 badVec5 = B._m00_m01_m02_m03_m10; // expected-error {{matrix swizzle length must be between 1 and 4 but is 5}}
}

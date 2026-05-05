// RUN: %clang_cc1 -triple dxilv1.7-unknown-shadermodel6.7-library -finclude-default-header -verify -emit-llvm -o - -x hlsl %s

// expected-warning@+1{{declaration does not declare anything}}
RWStructuredBuffer<float>;
RWStructuredBuffer<uint> a; 
RWStructuredBuffer<float> b;

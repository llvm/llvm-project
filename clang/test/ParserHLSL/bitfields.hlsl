// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -o - %s -verify

// expected-no-diagnostics

struct MyBitFields {
    unsigned int field1 : 3; // 3 bits for field1
    unsigned int field2 : 4; // 4 bits for field2
    int field3 : 5;          // 5 bits for field3 (signed)
};



[numthreads(1,1,1)]
void main() {
  MyBitFields m;
  m.field1 = 4;
  m.field2 = m.field1*2;
}
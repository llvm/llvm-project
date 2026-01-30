// RUN: %clang_dxc -T lib_6_7 -Wmissing-declarations %s -### %s -Xclang -verify

// expected-warning@+1{{declaration does not declare anything}}
RWStructuredBuffer<float>;
RWStructuredBuffer<uint> a; 
RWStructuredBuffer<float> b;

void run() {
    a[0] = asuint(b[0]);
}

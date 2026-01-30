// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7  %s -Xclang -verify \
// RUN:   -O3 %s -Xclang -verify

// expected-warning@+1{{declaration does not declare anything}}
RWStructuredBuffer<float>;
RWStructuredBuffer<uint> a; 
RWStructuredBuffer<float> b;

void run() {
    a[0] = asuint(b[0]);
}

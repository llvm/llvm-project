// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=RWBuffer
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=Buffer
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=RasterizerOrderedBuffer
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=StructuredBuffer
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=RWStructuredBuffer
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=AppendStructuredBuffer -DUSE_APPEND=1
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=ConsumeStructuredBuffer -DUSE_CONSUME=1
// RUN: %clang_dxc -Wmissing-declarations -T lib_6_7 \
// RUN:  -O3 %s -Xclang -verify -DRES=RasterizerOrderedStructuredBuffer

// expected-warning@+1{{declaration does not declare anything}}
RES<float>;
RES<uint> a; 
RES<float> b;

RWStructuredBuffer<uint> Out;

void run() {
    #ifdef USE_APPEND
    a.Append(Out[0]);
    b.Append(asfloat(Out[0]));
    #elif USE_CONSUME
    Out[0] = a.Consume() + asuint(b.Consume());
    #else
    Out[0] = a[0] + asuint(b[0]);
    #endif
}

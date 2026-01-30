// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=RWBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=Buffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=RasterizerOrderedBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=StructuredBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=RWStructuredBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=AppendStructuredBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=ConsumeStructuredBuffer
// RUN: %clang -Wmissing-declarations --driver-mode=dxc -Zi -E CSMain -T cs_6_2 \
// RUN:  -enable-16bit-types -O3 %s -Xclang -verify -DRES=RasterizerOrderedStructuredBuffer

// expected-warning@+1{{declaration does not declare anything}}
RES<float16_t>;
RES<uint> a; 
RES<float16_t> b;

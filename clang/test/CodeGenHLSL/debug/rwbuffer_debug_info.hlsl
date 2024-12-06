// RUN: %clang  --driver-mode=dxc -Zi -Fc out.s -T cs_6_3 %s

RWBuffer<float4> Out : register(u7, space4);

[numthreads(8,1,1)]
void main(uint GI : SV_GroupIndex) {
  Out[GI] = 0;
}

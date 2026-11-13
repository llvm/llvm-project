// RUN: not %clang_dxc -T cs_6_0 -E two_globals -Vd -Fo %t.dxil %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLOC
// RUN: not %clang_dxc -T cs_6_0 -E two_globals -Vd -Zi -Fo %t.dxil %s 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBALS-LOC
// RUN: not %clang_dxc -T cs_6_0 -E global_and_array -Vd -Fo %t.dxil %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NOLOC
// RUN: not %clang_dxc -T cs_6_0 -E global_and_array -Vd -Zi -Fo %t.dxil %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ARRAY-LOC

// Test that a resource access that can't be mapped to a unique global resource
// is reported as a regular error rather than crashing the compiler.

// CHECK-NOT: PLEASE submit a bug report
// CHECK: warning: assignment of 'Out1' to local resource 'Out' is not to the same unique global resource
// CHECK: warning: assignment of 'OutArr[0]' to local resource 'Out' is not to the same unique global resource
// NOLOC: error: Resource access is not guaranteed to map to a unique global resource
// GLOBALS-LOC: [[@LINE+18]]:{{[0-9]+}}: error: Resource access is not guaranteed to map to a unique global resource
// ARRAY-LOC: [[@LINE+25]]:{{[0-9]+}}: error: Resource access is not guaranteed to map to a unique global resource
// CHECK-NOT: PLEASE submit a bug report

RWBuffer<uint> In : register(u0);
RWStructuredBuffer<uint> Out0 : register(u1);
RWStructuredBuffer<uint> Out1 : register(u2);
RWStructuredBuffer<uint> OutArr[];

cbuffer c {
  bool cond;
};

[numthreads(64, 1, 1)]
void two_globals(uint3 tid : SV_DispatchThreadID) {
  RWStructuredBuffer<uint> Out = Out0;
  if (cond)
    Out = Out1;
  Out[tid.x] = In[tid.x];
}

[numthreads(64, 1, 1)]
void global_and_array(uint3 tid : SV_DispatchThreadID) {
  RWStructuredBuffer<uint> Out = Out0;
  if (cond)
    Out = OutArr[0];
  Out[tid.x] = In[tid.x];
}


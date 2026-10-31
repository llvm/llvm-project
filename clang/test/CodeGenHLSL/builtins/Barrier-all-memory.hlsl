// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.8-vertex -hlsl-entry main -DVERTEX %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck \
// RUN:   --check-prefix=VERTEX %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.8-compute -hlsl-entry main %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck \
// RUN:   --check-prefix=COMPUTE %s

#ifdef VERTEX
[shader("vertex")]
float4 main() : SV_Position {
  // VERTEX: call void @llvm.dx.barrier.by.memory.type(i32 1, i32 4)
  Barrier(ALL_MEMORY, DEVICE_SCOPE);
  return 0;
}
#else
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  // COMPUTE: call void @llvm.dx.barrier.by.memory.type(i32 3, i32 4)
  Barrier(ALL_MEMORY, DEVICE_SCOPE);
}
#endif

// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  float x;
  float y;
};

cbuffer CB {
  float a;
  float b[4];
  S s;
}

// CHECK-LABEL: define{{.*}}@_Z13cbuffer_loadsi(
// CHECK: load float, ptr addrspace(2) @a, align 4, !invariant.load [[INVARIANT:![0-9]+]]
// CHECK: load float, ptr addrspace(2) %cbufferidx, align 16, !invariant.load [[INVARIANT]]
export float cbuffer_loads(int i) {
  return a + b[i];
}

// CHECK-LABEL: define{{.*}}@_Z13struct_memberv(
// CHECK: load float, ptr addrspace(2) getelementptr {{.*}}@s{{.*}}, align 4, !invariant.load [[INVARIANT]]
export float struct_member() {
  return s.y;
}

// CHECK: [[INVARIANT]] = !{}

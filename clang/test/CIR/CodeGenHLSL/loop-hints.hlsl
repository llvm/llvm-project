// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void use(int);

[numthreads(1, 1, 1)] void main() {
  int n = 4;
  [unroll] for (int i = 0; i < n; ++i) use(i);
  [unroll(4)] for (int i = 0; i < n; ++i) use(i);
  [loop] for (int i = 0; i < n; ++i) use(i);
}

// CHECK-LABEL: cir.func {{.*}} @_Z4mainv
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = enable>
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 4>
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable>

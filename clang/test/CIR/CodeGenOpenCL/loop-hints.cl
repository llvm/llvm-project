// RUN: %clang_cc1 -x cl -triple spir64 -cl-std=CL2.0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void use(int);

kernel void k(int n) {
  __attribute__((opencl_unroll_hint)) for (int i = 0; i < n; ++i)
    use(i);
  __attribute__((opencl_unroll_hint(1))) for (int i = 0; i < n; ++i)
    use(i);
  __attribute__((opencl_unroll_hint(8))) for (int i = 0; i < n; ++i)
    use(i);
}

// CHECK-LABEL: cir.func {{.*}} @k
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = enable>
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = disable>
// CHECK: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll_count = 8>
